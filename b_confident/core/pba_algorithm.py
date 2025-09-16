"""
Core PBA (Perplexity-Based Adjacency) Algorithm Implementation

This module implements Algorithm 1 from the paper: PBA Uncertainty Estimation
Based on the methodology described in "Perplexity-Based Adjacency for Uncertainty
Quantification in Large Language Models" by Javier Marin.

Key Formula: UPBA(s) = 1/n * Σ f(perplexity(si|s<i))
where f(p) = 1 - exp(-β·p)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import logging

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class PBAConfig:
    """
    Configuration for PBA uncertainty quantification.

    Parameters validated through systematic analysis in the paper:
    - α = 0.9 provides optimal balance between coverage and efficiency
    - β = 0.5 provides appropriate sensitivity without oversensitivity
    """

    # Core PBA parameters (validated in paper)
    alpha: float = 0.9  # Probability mass threshold (optimal from paper)
    beta: float = 0.5   # Sensitivity parameter (optimal from paper)

    # Computational parameters
    temperature: float = 1.0  # Temperature scaling for logits
    device: Optional[str] = None  # Computing device (auto-detected if None)
    dtype: torch.dtype = torch.float32  # Computation precision

    # Memory optimization
    batch_processing: bool = True  # Process tokens in batches
    max_batch_size: int = 32  # Maximum batch size for processing

    # Validation parameters
    validate_inputs: bool = True  # Validate input tensors
    numerical_stability: bool = True  # Apply numerical stability measures

    def __post_init__(self):
        """Validate configuration parameters"""
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")

        if self.beta <= 0.0:
            raise ValueError(f"beta must be positive, got {self.beta}")

        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

        if self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {self.max_batch_size}")


class PBAUncertainty:
    """
    PBA Uncertainty Estimation Implementation

    Implements Algorithm 1 from the paper with optimizations for production use.

    The algorithm computes uncertainty as:
    1. Forward pass to get logits
    2. Convert to probabilities via softmax
    3. Define adjacent possible P(c) based on probability threshold
    4. Calculate entropy over adjacent possible
    5. Convert to perplexity: perplexity = 2^entropy
    6. Apply sensitivity function: f(p) = 1 - exp(-β·p)

    Example:
        >>> config = PBAConfig(alpha=0.9, beta=0.5)
        >>> pba = PBAUncertainty(config)
        >>> uncertainty = pba.calculate_uncertainty(logits)
    """

    def __init__(self, config: Optional[PBAConfig] = None):
        """
        Initialize PBA uncertainty calculator.

        Args:
            config: PBA configuration. If None, uses default optimized parameters.
        """
        self.config = config or PBAConfig()

        # Auto-detect device if not specified
        if self.config.device is None:
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initialized PBA with α={self.config.alpha}, β={self.config.beta}")

    def _validate_logits(self, logits: torch.Tensor) -> None:
        """Validate input logits tensor"""
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"logits must be torch.Tensor, got {type(logits)}")

        if logits.dim() < 1:
            raise ValueError(f"logits must be at least 1D, got shape {logits.shape}")

        if torch.isnan(logits).any():
            raise ValueError("logits contains NaN values")

        if torch.isinf(logits).any():
            raise ValueError("logits contains infinite values")

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits"""
        if self.config.temperature != 1.0:
            return logits / self.config.temperature
        return logits

    def _calculate_adjacent_possible_threshold(self, probs: torch.Tensor) -> float:
        """
        Calculate threshold τ for adjacent possible definition.

        Implements Equation 3: τ = inf{θ: Σ P(t|c) ≥ α}
        where the sum is over tokens t with P(t|c) ≥ θ
        """
        # Sort probabilities in descending order
        sorted_probs, _ = torch.sort(probs, descending=True)

        # Find cumulative sum
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # Find the index where cumsum >= alpha
        mask = cumsum >= self.config.alpha

        if not mask.any():
            # If no tokens reach alpha threshold, use minimum probability
            return sorted_probs.min().item()

        # Get the threshold probability
        idx = mask.nonzero(as_tuple=True)[0][0]
        threshold = sorted_probs[idx].item()

        return threshold

    def _calculate_entropy_over_adjacent_possible(
        self,
        probs: torch.Tensor,
        threshold: float
    ) -> float:
        """
        Calculate entropy over the adjacent possible.

        Implements line 13 of Algorithm 1:
        entropy = -Σ P(t|c) log P(t|c) for t ∈ P(c)
        """
        # Create mask for adjacent possible
        adjacent_mask = probs >= threshold

        if not adjacent_mask.any():
            # If no tokens in adjacent possible, return maximum entropy
            return np.log2(len(probs))

        # Filter probabilities to adjacent possible
        adjacent_probs = probs[adjacent_mask]

        # Renormalize probabilities over adjacent possible
        adjacent_probs = adjacent_probs / adjacent_probs.sum()

        # Calculate entropy (using log2 for perplexity calculation)
        entropy = -(adjacent_probs * torch.log2(adjacent_probs + 1e-12)).sum()

        return entropy.item()

    def _perplexity_to_uncertainty(self, perplexity: float) -> float:
        """
        Convert perplexity to uncertainty using sensitivity function.

        Implements f(p) = 1 - exp(-β·p) from Equation 5
        """
        return 1.0 - np.exp(-self.config.beta * perplexity)

    def calculate_token_uncertainty(self, logits: torch.Tensor, actual_token_id: Optional[int] = None) -> float:
        """
        Calculate uncertainty for a single token prediction.

        Uses simplified PBA approach aligned with paper implementation:
        - If actual_token_id provided: calculate perplexity for that token
        - Otherwise: use entropy-based approach over adjacent possible

        Args:
            logits: Raw model logits for next token prediction [vocab_size]
            actual_token_id: If provided, calculate uncertainty for this specific token

        Returns:
            PBA uncertainty score in [0, 1]
        """
        if self.config.validate_inputs:
            self._validate_logits(logits)

        # Ensure 1D tensor
        if logits.dim() > 1:
            logits = logits.squeeze()

        # Apply temperature scaling
        scaled_logits = self._apply_temperature(logits)

        if actual_token_id is not None:
            # Paper-aligned approach: calculate perplexity for specific token
            log_probs = F.log_softmax(scaled_logits, dim=-1)
            token_log_prob = log_probs[actual_token_id]
            perplexity = torch.exp(-token_log_prob).item()
        else:
            # Fallback: use adjacent possible approach
            # Convert to probabilities
            probs = F.softmax(scaled_logits, dim=-1)

            # Calculate threshold for adjacent possible
            threshold = self._calculate_adjacent_possible_threshold(probs)

            # Calculate entropy over adjacent possible
            entropy = self._calculate_entropy_over_adjacent_possible(probs, threshold)

            # Convert entropy to perplexity
            perplexity = 2 ** entropy

        # Apply sensitivity function
        uncertainty = self._perplexity_to_uncertainty(perplexity)

        return uncertainty

    def calculate_sequence_uncertainty(
        self,
        logits_sequence: List[torch.Tensor]
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Calculate uncertainty for a sequence of token predictions.

        Implements full Algorithm 1: UPBA(s) = 1/n * Σ f(perplexity(si|s<i))

        Args:
            logits_sequence: List of logits tensors, one per token position

        Returns:
            Dictionary containing:
            - 'sequence_uncertainty': Average uncertainty over sequence
            - 'token_uncertainties': Per-token uncertainty scores
            - 'sequence_length': Number of tokens
            - 'metadata': Additional information
        """
        if not logits_sequence:
            raise ValueError("logits_sequence cannot be empty")

        token_uncertainties = []

        # Calculate uncertainty for each token
        for i, token_logits in enumerate(logits_sequence):
            try:
                uncertainty = self.calculate_token_uncertainty(token_logits)
                token_uncertainties.append(uncertainty)
            except Exception as e:
                logger.warning(f"Error calculating uncertainty for token {i}: {e}")
                # Use maximum uncertainty as fallback
                token_uncertainties.append(1.0)

        # Calculate sequence-level uncertainty as mean
        sequence_uncertainty = np.mean(token_uncertainties)

        return {
            'sequence_uncertainty': sequence_uncertainty,
            'token_uncertainties': token_uncertainties,
            'sequence_length': len(logits_sequence),
            'metadata': {
                'config': self.config,
                'max_token_uncertainty': max(token_uncertainties),
                'min_token_uncertainty': min(token_uncertainties),
                'std_token_uncertainty': np.std(token_uncertainties)
            }
        }

    def calculate_batch_uncertainty(
        self,
        batch_logits: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Calculate uncertainty for a batch of predictions efficiently.

        Args:
            batch_logits: Tensor of shape [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]

        Returns:
            Dictionary with batch uncertainty results
        """
        if self.config.validate_inputs:
            self._validate_logits(batch_logits)

        original_shape = batch_logits.shape

        # Reshape to [batch_size * seq_len, vocab_size] if needed
        if batch_logits.dim() == 3:
            batch_size, seq_len, vocab_size = batch_logits.shape
            batch_logits = batch_logits.view(-1, vocab_size)
        else:
            batch_size = batch_logits.shape[0]
            seq_len = 1

        uncertainties = []

        # Process in chunks to manage memory
        chunk_size = self.config.max_batch_size
        for i in range(0, len(batch_logits), chunk_size):
            chunk = batch_logits[i:i+chunk_size]
            chunk_uncertainties = []

            for logits in chunk:
                uncertainty = self.calculate_token_uncertainty(logits)
                chunk_uncertainties.append(uncertainty)

            uncertainties.extend(chunk_uncertainties)

        # Reshape uncertainties back to original batch structure
        uncertainties = torch.tensor(uncertainties, dtype=self.config.dtype)
        if seq_len > 1:
            uncertainties = uncertainties.view(batch_size, seq_len)

        # Calculate batch statistics
        batch_stats = {
            'mean_uncertainty': uncertainties.mean().item(),
            'std_uncertainty': uncertainties.std().item(),
            'max_uncertainty': uncertainties.max().item(),
            'min_uncertainty': uncertainties.min().item(),
        }

        return {
            'uncertainties': uncertainties,
            'batch_stats': batch_stats,
            'original_shape': original_shape,
            'config': self.config
        }

    def get_computational_overhead(self) -> Dict[str, str]:
        """
        Get information about computational overhead.

        Based on paper results: 19% overhead over standard forward pass.
        """
        return {
            'expected_overhead': '19%',
            'memory_overhead': 'Minimal - reuses forward pass computations',
            'comparison_to_ensembles': '~15x faster than ensemble methods',
            'comparison_to_monte_carlo': '~8x faster than Monte Carlo dropout',
            'scalability': 'Linear with sequence length'
        }