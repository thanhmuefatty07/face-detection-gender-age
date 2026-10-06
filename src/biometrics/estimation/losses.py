"""
Biometric Multi-Task Loss Functions
====================================
Unified loss landscape combining:
1. Cramér Distance (Squared Earth Mover's Distance on ordered 101-bin age CDFs)
2. Soft-KL Divergence on continuous Gaussian simplex distributions
3. Soft-L1 MAE Expectation penalty on continuous scalar age
4. Masked Dimorphic BCE modulated by craniofacial sexual dimorphism weight
5. Aleatoric Heteroscedastic Uncertainty loss (Kendall & Gal formulation)
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def cramer_distance_loss(pred_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """
    Cramér Distance (Squared Energy Mover's Distance / 1-Wasserstein on ordered CDFs).
    Penalizes prediction distance along the ordered ordinal age axis.
    
    Args:
        pred_probs: (B, K) predicted probability simplex
        target_probs: (B, K) target probability simplex
    Returns:
        Scalar tensor: mean squared CDF difference
    """
    cdf_pred = torch.cumsum(pred_probs, dim=-1)
    cdf_target = torch.cumsum(target_probs, dim=-1)
    # Sum of squared CDF differences across bins, averaged across batch
    diff_sq = (cdf_pred - cdf_target) ** 2
    return torch.mean(torch.sum(diff_sq, dim=-1))


def soft_kl_divergence_loss(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Kullback-Leibler divergence between predicted simplex and smooth continuous target simplex.
    
    Args:
        pred_probs: (B, K) predicted probabilities
        target_probs: (B, K) continuous target distribution (e.g. Gaussian LDL)
    Returns:
        Scalar tensor: KL divergence
    """
    log_pred = torch.log(torch.clamp(pred_probs, min=eps))
    # F.kl_div expects log-probabilities as input and probabilities as target
    return F.kl_div(log_pred, target_probs, reduction="batchmean")


def soft_l1_mae_loss(expected_age: torch.Tensor, target_age: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error on continuous expected age prediction.
    
    Args:
        expected_age: (B,) expected age scalar sum(p * bin)
        target_age: (B,) ground-truth continuous biological/apparent age
    Returns:
        Scalar tensor: MAE loss
    """
    return torch.mean(torch.abs(expected_age - target_age))


def masked_dimorphic_bce_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    gender_mask: torch.Tensor,
    dimorphism_weights: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Masked Binary Cross Entropy loss modulated by craniofacial sexual dimorphism weight.
    Guarantees that unlabelled samples (gender_mask == 0.0) produce zero gradient and
    do not corrupt gender classification, while sexually ambiguous ages (children < 12,
    elderly > 65) have attenuated loss penalties.
    
    Args:
        pred_logits: (B,) unnormalized gender logit (positive = Female)
        targets: (B,) target gender (1.0 = Female, 0.0 = Male)
        gender_mask: (B,) validity mask (1.0 = valid, 0.0 = missing/unlabelled)
        dimorphism_weights: (B,) attenuation weight in [0.2, 1.0]
    Returns:
        Scalar tensor: weighted BCE loss
    """
    bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="none")
    effective_weights = gender_mask * dimorphism_weights
    total_weight = torch.sum(effective_weights)

    if total_weight <= eps:
        # If all items in batch are masked out, return zero loss while preserving graph
        return torch.sum(bce * effective_weights)

    return torch.sum(bce * effective_weights) / (total_weight + eps)


def aleatoric_heteroscedastic_loss(
    expected_age: torch.Tensor,
    target_age: torch.Tensor,
    log_variance: torch.Tensor
) -> torch.Tensor:
    """
    Aleatoric heteroscedastic uncertainty loss (Kendall & Gal 2017).
    L = 0.5 * exp(-s) * (y - y_hat)^2 + 0.5 * s
    where s = log(sigma^2).
    
    Prevents degenerate sigma -> 0 or sigma -> inf solutions while allowing the
    network to learn dynamic observation noise for noisy, low-quality or ambiguous faces.
    
    Args:
        expected_age: (B,) continuous prediction
        target_age: (B,) ground-truth age
        log_variance: (B,) predicted log-variance s = log(sigma^2)
    Returns:
        Scalar tensor: heteroscedastic loss
    """
    residual_sq = (expected_age - target_age) ** 2
    # Clamping log_variance to avoid numerical overflow in exp(-s)
    s = torch.clamp(log_variance, min=-10.0, max=10.0)
    precision = torch.exp(-s)
    loss = 0.5 * precision * residual_sq + 0.5 * s
    return torch.mean(loss)


class UnifiedBiometricLoss(nn.Module):
    """
    Unified multi-task loss combining:
    - Cramér Distance (squared EMD)
    - Soft-KL Divergence
    - Soft-L1 MAE
    - Masked Dimorphic BCE
    - Aleatoric Heteroscedastic Uncertainty
    """

    def __init__(
        self,
        weight_cramer: float = 1.0,
        weight_kl: float = 0.5,
        weight_mae: float = 0.2,
        weight_gender: float = 1.0,
        weight_aleatoric: float = 0.1,
    ):
        super().__init__()
        self.weight_cramer = weight_cramer
        self.weight_kl = weight_kl
        self.weight_mae = weight_mae
        self.weight_gender = weight_gender
        self.weight_aleatoric = weight_aleatoric

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward multi-task loss computation.
        
        Args:
            preds: dictionary with keys:
                - 'age_probs': (B, 101)
                - 'expected_age': (B,)
                - 'gender_logit': (B,)
                - 'log_variance': (B,)
            targets: dictionary with keys:
                - 'ldl_101': (B, 101)
                - 'age_continuous': (B,)
                - 'gender': (B,)
                - 'gender_mask': (B,)
                - 'dimorphism_weight': (B,)
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # 1. Cramér Distance
        l_cramer = cramer_distance_loss(preds["age_probs"], targets["ldl_101"])

        # 2. Soft-KL Divergence
        l_kl = soft_kl_divergence_loss(preds["age_probs"], targets["ldl_101"])

        # 3. Soft-L1 MAE
        l_mae = soft_l1_mae_loss(preds["expected_age"], targets["age_continuous"])

        # 4. Masked Dimorphic BCE
        gender_mask = targets.get("gender_mask", torch.ones_like(targets["gender"]))
        dimorphism_w = targets.get("dimorphism_weight", torch.ones_like(targets["gender"]))
        l_gender = masked_dimorphic_bce_loss(
            preds["gender_logit"],
            targets["gender"],
            gender_mask,
            dimorphism_w
        )

        # 5. Aleatoric Heteroscedastic Loss
        l_aleatoric = aleatoric_heteroscedastic_loss(
            preds["expected_age"],
            targets["age_continuous"],
            preds["log_variance"]
        )

        # Weighted total
        total_loss = (
            self.weight_cramer * l_cramer
            + self.weight_kl * l_kl
            + self.weight_mae * l_mae
            + self.weight_gender * l_gender
            + self.weight_aleatoric * l_aleatoric
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_cramer": float(l_cramer.item()),
            "loss_kl": float(l_kl.item()),
            "loss_mae": float(l_mae.item()),
            "loss_gender": float(l_gender.item()),
            "loss_aleatoric": float(l_aleatoric.item()),
        }

        return total_loss, metrics
