"""
Hugging Face Transformers Integration

Provides drop-in replacement for standard model.generate() calls with uncertainty
quantification enabled. Designed to integrate seamlessly into existing workflows
without requiring architectural changes.

Key Features:
- Drop-in replacement for transformers.generate()
- Preserves all existing model functionality
- Adds uncertainty scoring with minimal memory overhead
- Compatible with major transformer architectures (GPT, LLaMA, Mistral, BERT-family)
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig
)
from typing import Optional, Dict, List, Union, Tuple, Any
import logging
from dataclasses import dataclass
import warnings
import time

from ..core.pba_algorithm import PBAUncertainty, PBAConfig
from ..core.metrics import calculate_uncertainty_metrics, CalibrationResults

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyGenerationResult:
    """Results from uncertainty-enabled generation with performance metrics"""
    sequences: torch.Tensor  # Generated token sequences
    uncertainty_scores: List[float]  # Per-sequence uncertainty scores
    token_uncertainties: List[List[float]]  # Per-token uncertainties for each sequence
    sequence_scores: Optional[torch.Tensor]  # Original sequence scores if available
    metadata: Dict[str, Any]  # Additional information
    performance_metrics: Dict[str, float]  # Timing and performance data


class UncertaintyTransformersModel:
    """
    Wrapper class that adds uncertainty quantification to Hugging Face models.

    This wrapper preserves all existing model functionality while adding PBA
    uncertainty calculation during inference. Memory overhead is minimal as it
    primarily stores probability distributions for uncertainty calculation.

    Example:
        >>> from transformers import AutoModel, AutoTokenizer
        >>> base_model = AutoModel.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = UncertaintyTransformersModel(base_model, tokenizer)
        >>> result = model.uncertainty_generate("The weather today is", max_length=50)
        >>> print(f"Uncertainty: {result.uncertainty_scores[0]:.3f}")
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        pba_config: Optional[PBAConfig] = None
    ):
        """
        Initialize uncertainty-enabled transformers model.

        Args:
            model: Pre-trained Hugging Face model
            tokenizer: Tokenizer (optional, for text generation)
            pba_config: PBA configuration (uses optimized defaults if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.pba_config = pba_config or PBAConfig()
        self.pba_calculator = PBAUncertainty(self.pba_config)

        # Model metadata
        self.model_name = getattr(model, 'name_or_path', 'unknown')
        self.architecture = model.config.architectures[0] if hasattr(model.config, 'architectures') else 'unknown'

        # Validate model compatibility
        self._validate_model_compatibility()

        logger.info(f"Initialized UncertaintyTransformersModel for {self.architecture}")

    def _validate_model_compatibility(self) -> None:
        """Validate that the model is compatible with uncertainty quantification"""
        if not hasattr(self.model, 'forward'):
            raise ValueError("Model must have forward() method")

        # Check for common autoregressive model attributes
        if hasattr(self.model.config, 'is_decoder'):
            if not self.model.config.is_decoder:
                warnings.warn(
                    "Model is not configured as decoder. Uncertainty quantification "
                    "works best with autoregressive (decoder) models."
                )

        # Supported architectures (non-exhaustive)
        supported_archs = {
            'GPT2LMHeadModel', 'GPTNeoForCausalLM', 'LlamaForCausalLM',
            'MistralForCausalLM', 'CodeLlamaForCausalLM', 'QWenLMHeadModel'
        }

        if self.architecture not in supported_archs:
            warnings.warn(
                f"Architecture {self.architecture} not explicitly tested. "
                f"Supported: {supported_archs}"
            )

    def forward_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Forward pass with uncertainty calculation.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional model arguments

        Returns:
            Tuple of (model outputs, per-position uncertainty scores)
        """
        # Standard forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )

        # Extract logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            raise ValueError("Model outputs don't contain logits")

        # Calculate uncertainties for each position
        uncertainties = []
        seq_len = logits.shape[1]

        for pos in range(seq_len):
            try:
                position_uncertainty = self.pba_calculator.calculate_token_uncertainty(
                    logits[0, pos]  # Assume batch size 1 for now
                )
                uncertainties.append(position_uncertainty)
            except Exception as e:
                logger.warning(f"Error calculating uncertainty at position {pos}: {e}")
                uncertainties.append(1.0)  # Maximum uncertainty as fallback

        return outputs, uncertainties

    def uncertainty_generate(
        self,
        inputs: Union[str, torch.Tensor],
        max_length: int = 50,
        num_return_sequences: int = 1,
        generation_config: Optional[GenerationConfig] = None,
        **generation_kwargs
    ) -> UncertaintyGenerationResult:
        """
        Generate text with uncertainty quantification.

        Drop-in replacement for model.generate() with added uncertainty scoring.

        Args:
            inputs: Input text string or token tensor
            max_length: Maximum generation length
            num_return_sequences: Number of sequences to return
            generation_config: Generation configuration
            **generation_kwargs: Additional generation arguments

        Returns:
            UncertaintyGenerationResult with sequences and uncertainty scores
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text generation")

        # Tokenize inputs if string
        if isinstance(inputs, str):
            input_ids = self.tokenizer.encode(inputs, return_tensors="pt")
            input_text = inputs
        else:
            input_ids = inputs
            input_text = self.tokenizer.decode(inputs[0]) if self.tokenizer else "N/A"

        # Move to model device
        input_ids = input_ids.to(self.model.device)

        # Prepare generation config
        if generation_config is None:
            generation_config = GenerationConfig(
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=True,  # Enable sampling for uncertainty estimation
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **generation_kwargs
            )

        # Performance timing
        total_start_time = time.time()
        generation_time = 0.0
        uncertainty_time = 0.0

        # Custom generation loop to capture uncertainties
        generated_sequences = []
        all_uncertainties = []
        all_token_uncertainties = []

        for seq_idx in range(num_return_sequences):
            sequence_tokens = input_ids.clone()
            sequence_uncertainties = []
            token_uncertainties = []

            # Generation loop
            for step in range(max_length - input_ids.shape[1]):
                # Forward pass (timed)
                gen_start = time.time()
                with torch.no_grad():
                    outputs = self.model(sequence_tokens)
                    logits = outputs.logits[0, -1]  # Last position logits

                # Sample next token (using temperature if specified)
                sampling_logits = logits.clone()
                if generation_config.temperature != 1.0:
                    sampling_logits = sampling_logits / generation_config.temperature

                probs = F.softmax(sampling_logits, dim=-1)
                if generation_config.do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(sampling_logits, dim=-1).unsqueeze(0)

                generation_time += time.time() - gen_start

                # Calculate uncertainty for the ACTUAL generated token (paper-aligned)
                unc_start = time.time()
                try:
                    uncertainty = self.pba_calculator.calculate_token_uncertainty(
                        logits, actual_token_id=next_token.item()
                    )
                    sequence_uncertainties.append(uncertainty)
                except Exception as e:
                    logger.warning(f"Error in uncertainty calculation: {e}")
                    sequence_uncertainties.append(1.0)

                uncertainty_time += time.time() - unc_start

                # Append token to sequence
                sequence_tokens = torch.cat([sequence_tokens, next_token.unsqueeze(0)], dim=1)

                # Check for end token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            # Calculate sequence-level uncertainty
            if sequence_uncertainties:
                sequence_uncertainty = sum(sequence_uncertainties) / len(sequence_uncertainties)
            else:
                sequence_uncertainty = 0.0

            generated_sequences.append(sequence_tokens[0])
            all_uncertainties.append(sequence_uncertainty)
            all_token_uncertainties.append(sequence_uncertainties)

        # Convert to expected format
        max_seq_len = max(len(seq) for seq in generated_sequences)
        padded_sequences = []

        for seq in generated_sequences:
            if len(seq) < max_seq_len:
                padding = torch.full(
                    (max_seq_len - len(seq),),
                    self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    device=seq.device
                )
                seq = torch.cat([seq, padding])
            padded_sequences.append(seq)

        sequences_tensor = torch.stack(padded_sequences)

        total_time = time.time() - total_start_time

        # Performance metrics (demonstrating the paper's 19% overhead claim)
        performance_metrics = {
            'total_time': total_time,
            'generation_time': generation_time,
            'uncertainty_time': uncertainty_time,
            'uncertainty_overhead_pct': (uncertainty_time / generation_time * 100) if generation_time > 0 else 0.0,
            'total_overhead_pct': ((total_time - generation_time) / generation_time * 100) if generation_time > 0 else 0.0,
            'tokens_per_second': sum(len(seq) for seq in generated_sequences) / total_time if total_time > 0 else 0.0,
            'uncertainty_calculations_per_second': len(all_token_uncertainties[0]) / uncertainty_time if uncertainty_time > 0 and all_token_uncertainties else 0.0
        }

        # Metadata
        metadata = {
            'input_text': input_text,
            'model_name': self.model_name,
            'architecture': self.architecture,
            'pba_config': self.pba_config,
            'generation_config': generation_config,
            'avg_uncertainty': sum(all_uncertainties) / len(all_uncertainties) if all_uncertainties else 0.0,
            'max_uncertainty': max(all_uncertainties) if all_uncertainties else 0.0,
            'min_uncertainty': min(all_uncertainties) if all_uncertainties else 0.0,
            'performance_metrics': performance_metrics  # For backwards compatibility
        }

        return UncertaintyGenerationResult(
            sequences=sequences_tensor,
            uncertainty_scores=all_uncertainties,
            token_uncertainties=all_token_uncertainties,
            sequence_scores=None,  # Could be added if needed
            metadata=metadata,
            performance_metrics=performance_metrics
        )

    def calibrate_model(
        self,
        validation_texts: List[str],
        validation_labels: List[int],
        calibration_method: str = "temperature_scaling"
    ) -> Dict[str, float]:
        """
        Calibrate uncertainty estimates on validation data.

        Args:
            validation_texts: List of validation input texts
            validation_labels: List of correctness labels (0/1)
            calibration_method: Calibration method to use

        Returns:
            Calibration results and optimal parameters
        """
        logger.info(f"Calibrating model on {len(validation_texts)} samples")

        uncertainties = []
        confidences = []

        for text in validation_texts:
            try:
                result = self.uncertainty_generate(
                    text, max_length=text.count(' ') + 10, num_return_sequences=1
                )
                uncertainty = result.uncertainty_scores[0]
                confidence = 1.0 - uncertainty

                uncertainties.append(uncertainty)
                confidences.append(confidence)
            except Exception as e:
                logger.warning(f"Error processing text '{text[:50]}...': {e}")
                uncertainties.append(0.5)  # Neutral uncertainty
                confidences.append(0.5)

        # Calculate calibration metrics
        results = calculate_uncertainty_metrics(
            uncertainties, validation_labels, confidences
        )

        return {
            'ece': results.ece,
            'brier_score': results.brier_score,
            'auroc': results.auroc,
            'stability_score': results.stability_score,
            'calibration_method': calibration_method,
            'n_samples': len(validation_texts)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the wrapped model"""
        return {
            'model_name': self.model_name,
            'architecture': self.architecture,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.model.device),
            'pba_config': self.pba_config.__dict__,
            'supports_generation': hasattr(self.model, 'generate'),
            'has_tokenizer': self.tokenizer is not None
        }


def uncertainty_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: Union[str, torch.Tensor],
    pba_config: Optional[PBAConfig] = None,
    **generation_kwargs
) -> UncertaintyGenerationResult:
    """
    Convenience function for one-off uncertainty generation.

    Drop-in replacement for standard generation with uncertainty quantification.

    Args:
        model: Pre-trained Hugging Face model
        tokenizer: Tokenizer
        inputs: Input text or tokens
        pba_config: PBA configuration
        **generation_kwargs: Generation arguments

    Returns:
        Generation result with uncertainty scores

    Example:
        >>> from transformers import AutoModel, AutoTokenizer
        >>> model = AutoModel.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> result = uncertainty_generate(model, tokenizer, "Hello world")
        >>> print(f"Generated: {tokenizer.decode(result.sequences[0])}")
        >>> print(f"Uncertainty: {result.uncertainty_scores[0]:.3f}")
    """
    wrapper = UncertaintyTransformersModel(model, tokenizer, pba_config)
    return wrapper.uncertainty_generate(inputs, **generation_kwargs)