#!/usr/bin/env python3
"""
Simplified PBA Uncertainty Implementation for HuggingFace Space
Real Perplexity-Based Adjacency uncertainty quantification - Python 3.7+ compatible
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import logging

logger = logging.getLogger(__name__)

class PBAConfig:
    """Configuration for PBA uncertainty quantification"""
    def __init__(self, alpha=0.9, beta=0.5, temperature=1.0, device=None):
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.device = device

class UncertaintyResult:
    """Results from uncertainty quantification"""
    def __init__(self, generated_texts, uncertainty_scores, token_perplexities, token_uncertainties):
        self.generated_texts = generated_texts
        self.uncertainty_scores = uncertainty_scores
        self.token_perplexities = token_perplexities
        self.token_uncertainties = token_uncertainties

class PBAUncertaintyQuantifier:
    """Real PBA uncertainty quantification implementation"""

    def __init__(self, model_name, config=None):
        self.config = config or PBAConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading model: " + str(model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name

    def calculate_token_perplexity(self, logits, target_token_id):
        """Calculate perplexity for a specific token"""
        scaled_logits = logits / self.config.temperature
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        token_log_prob = log_probs[target_token_id].item()
        perplexity = math.exp(-token_log_prob)
        return perplexity

    def calculate_pba_uncertainty(self, perplexities):
        """Calculate PBA uncertainty from token perplexities"""
        if not perplexities:
            return 0.5

        transformed_perplexities = []
        for p in perplexities:
            transformed = 1.0 - math.exp(-self.config.beta * p)
            transformed_perplexities.append(transformed)

        uncertainty = np.mean(transformed_perplexities)
        uncertainty = max(0.0, min(1.0, uncertainty))
        return uncertainty

    def generate_with_uncertainty(self, input_text, max_length=50, num_return_sequences=1, do_sample=True, temperature=1.0):
        """Generate text with uncertainty quantification"""

        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        input_length = inputs.shape[1]

        all_generated_texts = []
        all_uncertainty_scores = []
        all_token_perplexities = []
        all_token_uncertainties = []

        with torch.no_grad():
            for _ in range(num_return_sequences):
                generated_ids = inputs.clone()
                token_perplexities = []
                token_uncertainties = []

                for step in range(max_length - input_length):
                    outputs = self.model(generated_ids)
                    logits = outputs.logits[0, -1, :]

                    if do_sample:
                        scaled_logits = logits / temperature
                        probs = F.softmax(scaled_logits, dim=-1)
                        next_token_id = torch.multinomial(probs, 1).item()
                    else:
                        next_token_id = torch.argmax(logits).item()

                    perplexity = self.calculate_token_perplexity(logits, next_token_id)
                    token_perplexities.append(perplexity)

                    token_uncertainty = 1.0 - math.exp(-self.config.beta * perplexity)
                    token_uncertainties.append(token_uncertainty)

                    next_token = torch.tensor([[next_token_id]], device=self.device)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

                    if next_token_id == self.tokenizer.eos_token_id:
                        break

                generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                uncertainty_score = self.calculate_pba_uncertainty(token_perplexities)

                all_generated_texts.append(generated_text)
                all_uncertainty_scores.append(uncertainty_score)
                all_token_perplexities.append(token_perplexities)
                all_token_uncertainties.append(token_uncertainties)

        return UncertaintyResult(
            generated_texts=all_generated_texts,
            uncertainty_scores=all_uncertainty_scores,
            token_perplexities=all_token_perplexities,
            token_uncertainties=all_token_uncertainties
        )

def uncertainty_generate(model_name, input_text, max_length=50, num_return_sequences=1, pba_config=None, **kwargs):
    """Main function for uncertainty generation"""
    quantifier = PBAUncertaintyQuantifier(model_name, pba_config)
    return quantifier.generate_with_uncertainty(
        input_text=input_text,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        **kwargs
    )

def calculate_uncertainty_metrics(uncertainty_scores, correctness_labels, confidence_scores=None):
    """Calculate uncertainty calibration metrics"""

    if confidence_scores is None:
        confidence_scores = [1.0 - u for u in uncertainty_scores]

    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = [(c >= bin_lower) and (c < bin_upper) for c in confidence_scores]
        prop_in_bin = sum(in_bin) / len(in_bin) if in_bin else 0

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean([correctness_labels[i] for i, in_b in enumerate(in_bin) if in_b])
            avg_confidence_in_bin = np.mean([confidence_scores[i] for i, in_b in enumerate(in_bin) if in_b])
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    # Brier Score
    brier_score = np.mean([(c - l)**2 for c, l in zip(confidence_scores, correctness_labels)])

    # Simple AUROC calculation
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score([1-l for l in correctness_labels], uncertainty_scores)
    except:
        auroc = 0.5

    return {
        'ece': ece,
        'brier_score': brier_score,
        'auroc': auroc,
        'accuracy': np.mean(correctness_labels),
        'avg_uncertainty': np.mean(uncertainty_scores)
    }

def calibrate_model(model_name, validation_texts, validation_labels, pba_config=None):
    """Calibrate model and return metrics"""

    quantifier = PBAUncertaintyQuantifier(model_name, pba_config)
    uncertainties = []

    logger.info("Calibrating on " + str(len(validation_texts)) + " samples")

    for i, text in enumerate(validation_texts):
        try:
            result = quantifier.generate_with_uncertainty(
                input_text=text,
                max_length=min(50, len(text.split()) + 20),
                num_return_sequences=1
            )
            uncertainties.append(result.uncertainty_scores[0])
        except Exception as e:
            logger.warning("Error processing sample " + str(i) + ": " + str(e))
            uncertainties.append(0.5)

    metrics = calculate_uncertainty_metrics(uncertainties, validation_labels)

    return {
        'uncertainties': uncertainties,
        'validation_labels': validation_labels,
        'metrics': metrics,
        'model_name': model_name,
        'n_samples': len(validation_texts)
    }

# Test function
def test_pba_implementation():
    """Test the PBA implementation with a simple example"""
    print("Testing PBA Uncertainty Implementation...")

    result = uncertainty_generate(
        model_name="gpt2",
        input_text="The capital of France is",
        max_length=20,
        num_return_sequences=1
    )

    print("Generated: " + result.generated_texts[0])
    print("Uncertainty: " + str(round(result.uncertainty_scores[0], 4)))
    print("Success!")

if __name__ == "__main__":
    test_pba_implementation()