"""
Biometric Multi-Task Loss Functions
====================================
Unified loss landscape combining:
1. Cramér Distance (Squared Earth Mover's Distance on ordered 101-bin age CDFs)
2. Soft-KL Divergence on continuous Gaussian simplex distributions (FP16 Underflow Protected)
3. Soft-L1 MAE Expectation penalty on continuous scalar age
4. Masked Dimorphic BCE modulated by craniofacial sexual dimorphism weight (Sample Count Normalized)
5. Aleatoric Heteroscedastic Uncertainty loss (L1-Detached Laplace formulation)
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
    p = pred_probs.float() / (torch.sum(pred_probs.float(), dim=-1, keepdim=True) + 1e-12)
    q = target_probs.float() / (torch.sum(target_probs.float(), dim=-1, keepdim=True) + 1e-12)
    cdf_pred = torch.cumsum(p, dim=-1)
    cdf_target = torch.cumsum(q, dim=-1)
    diff_sq = (cdf_pred - cdf_target) ** 2
    return torch.mean(torch.sum(diff_sq, dim=-1))


def soft_kl_divergence_loss(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    pred_logits: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Kullback-Leibler divergence between predicted simplex and smooth continuous target simplex.
    Prevents FP16 underflow NaN by using F.log_softmax on float32 logits (when available)
    or computing log probabilities strictly in float32.
    
    Args:
        pred_probs: (B, K) predicted probabilities
        target_probs: (B, K) continuous target distribution (e.g. Gaussian LDL)
        pred_logits: Optional (B, K) unnormalized logits for maximum numerical stability
        eps: numerical epsilon
    Returns:
        Scalar tensor: KL divergence
    """
    if pred_logits is not None:
        log_p = F.log_softmax(pred_logits.float(), dim=-1)
    else:
        log_p = torch.log(torch.clamp(pred_probs.float(), min=eps))
    return F.kl_div(log_p, target_probs.float(), reduction="batchmean")


def soft_l1_mae_loss(expected_age: torch.Tensor, target_age: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error on continuous expected age prediction.
    
    Args:
        expected_age: (B,) expected age scalar sum(p * bin)
        target_age: (B,) ground-truth continuous biological/apparent age
    Returns:
        Scalar tensor: MAE loss
    """
    exp_flat = expected_age.float().view(-1)
    tgt_flat = target_age.float().view(-1)
    return torch.mean(torch.abs(exp_flat - tgt_flat))


def masked_dimorphic_bce_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    gender_mask: torch.Tensor,
    dimorphism_weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Masked Binary Cross Entropy loss modulated by craniofacial sexual dimorphism weight.
    Guarantees that unlabelled samples (gender_mask == 0.0) produce zero gradient and
    do not corrupt gender classification, while sexually ambiguous ages (children < 12,
    elderly > 65) have attenuated loss penalties.
    
    Denominator is strictly normalized by valid_count (sum of mask), avoiding unintended
    gradient inflation from small dimorphism weights.
    
    Args:
        pred_logits: (B,) or (B, 1) unnormalized gender logit (positive = Female)
        targets: (B,) or (B, 1) target gender (1.0 = Female, 0.0 = Male)
        gender_mask: (B,) or (B, 1) validity mask (1.0 = valid, 0.0 = missing/unlabelled)
        dimorphism_weights: (B,) or (B, 1) attenuation weight in [0.2, 1.0]
        eps: epsilon for division stability
    Returns:
        Scalar tensor: weighted BCE loss
    """
    logits_flat = pred_logits.float().view(-1)
    targets_flat = targets.float().view(-1)
    mask_flat = gender_mask.float().view(-1)
    dim_flat = dimorphism_weights.float().view(-1)

    bce = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction="none")
    valid_count = torch.sum(mask_flat)

    if valid_count <= eps:
        # If all items in batch are masked out, return zero loss while preserving graph
        return torch.sum(bce * 0.0)

    return torch.sum(bce * mask_flat * dim_flat) / (valid_count + eps)


def aleatoric_heteroscedastic_loss(
    expected_age: torch.Tensor,
    target_age: torch.Tensor,
    log_variance: torch.Tensor,
) -> torch.Tensor:
    """
    L1-Detached Laplace Aleatoric Uncertainty Loss.
    L = exp(-s) * |y - y_hat.detach()| + s
    where s = log_variance.
    
    Detaches predicted continuous age y_hat so that uncertainty learning cannot exploit
    variance to bias age expectations or cause gradient explosion (||grad|| > 7.5).
    Formulates residual in L1 (Laplace distribution) for robustness to outliers.
    
    Args:
        expected_age: (B,) or (B, 1) continuous prediction
        target_age: (B,) or (B, 1) ground-truth age
        log_variance: (B,) or (B, 1) predicted log-variance s = log(sigma^2)
    Returns:
        Scalar tensor: heteroscedastic loss
    """
    pred_mu = expected_age.float().view(-1)
    tgt_flat = target_age.float().view(-1)
    log_var_flat = log_variance.float().view(-1)

    abs_residual = torch.abs(pred_mu.detach() - tgt_flat)
    s = torch.clamp(log_var_flat, min=-10.0, max=10.0)
    loss_unc = torch.mean(torch.exp(-s) * abs_residual + s)
    return loss_unc


class UnifiedBiometricLoss(nn.Module):
    """
    Unified multi-task loss combining:
    - Cramér Distance (squared EMD)
    - Soft-KL Divergence (FP16 protected)
    - Soft-L1 MAE
    - Masked Dimorphic BCE (valid-sample normalized)
    - Aleatoric Heteroscedastic Uncertainty (L1-Detached Laplace)
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
                - 'age_logits': Optional (B, 101)
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

        # 2. Soft-KL Divergence (FP16 numerical stability via log_softmax)
        age_logits = preds.get("age_logits")
        l_kl = soft_kl_divergence_loss(
            preds["age_probs"],
            targets["ldl_101"],
            pred_logits=age_logits
        )

        # 3. Soft-L1 MAE
        l_mae = soft_l1_mae_loss(preds["expected_age"], targets["age_continuous"])

        # 4. Masked Dimorphic BCE (strictly sample-count normalized)
        gender_mask = targets.get("gender_mask", torch.ones_like(targets["gender"]))
        dimorphism_w = targets.get("dimorphism_weight", torch.ones_like(targets["gender"]))
        l_gender = masked_dimorphic_bce_loss(
            preds["gender_logit"],
            targets["gender"],
            gender_mask,
            dimorphism_w
        )

        # 5. Aleatoric Heteroscedastic Loss (L1-Detached Laplace)
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
