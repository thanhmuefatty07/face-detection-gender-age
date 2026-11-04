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


def _prepare_sample_weights(
    batch_size: int,
    device: torch.device,
    mask: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes per-sample numerator weights w_i = m_i * alpha_i and denominator valid mask count sum(m_i).
    Preserves exact scalar scaling when alpha_i == c across the batch while zeroing out gradients
    for individual samples where m_i == 0 or alpha_i == 0.
    """
    if mask is not None:
        m = torch.clamp(
            torch.nan_to_num(mask.float().view(-1), nan=0.0, posinf=1.0, neginf=0.0),
            min=0.0,
        )
        if m.numel() == 1 and batch_size > 1:
            m = m.expand(batch_size)
    else:
        m = torch.ones(batch_size, device=device, dtype=torch.float32)

    if sample_weights is not None:
        alpha = torch.clamp(
            torch.nan_to_num(sample_weights.float().view(-1), nan=0.0, posinf=1.0, neginf=0.0),
            min=0.0,
        )
        if alpha.numel() == 1 and batch_size > 1:
            alpha = alpha.expand(batch_size)
        w = m * alpha
    else:
        w = m

    return w, torch.sum(m)


def cramer_distance_loss(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    age_mask: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Cramér Distance (Squared Energy Mover's Distance / 1-Wasserstein on ordered CDFs).
    Penalizes prediction distance along the ordered ordinal age axis.
    
    Args:
        pred_probs: (B, K) predicted probability simplex
        target_probs: (B, K) target probability simplex
        age_mask: Optional (B,) validity mask (1.0 = valid age, 0.0 = missing/binary)
        sample_weights: Optional (B,) per-sample confidence weight alpha_i
    Returns:
        Scalar tensor: mean squared CDF difference
    """
    p = pred_probs.float() / (torch.sum(pred_probs.float(), dim=-1, keepdim=True) + 1e-12)
    q = target_probs.float() / (torch.sum(target_probs.float(), dim=-1, keepdim=True) + 1e-12)
    cdf_pred = torch.cumsum(p, dim=-1)
    cdf_target = torch.cumsum(q, dim=-1)
    per_sample = torch.sum((cdf_pred - cdf_target) ** 2, dim=-1)
    if age_mask is not None or sample_weights is not None:
        w, valid = _prepare_sample_weights(per_sample.shape[0], per_sample.device, age_mask, sample_weights)
        if valid <= 1e-8:
            return torch.sum(per_sample * 0.0)
        return torch.sum(per_sample * w) / (valid + 1e-8)
    return torch.mean(per_sample)


def soft_kl_divergence_loss(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    pred_logits: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    age_mask: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
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
        age_mask: Optional (B,) validity mask (1.0 = valid age, 0.0 = missing/binary)
        sample_weights: Optional (B,) per-sample confidence weight alpha_i
    Returns:
        Scalar tensor: KL divergence
    """
    if pred_logits is not None:
        log_p = F.log_softmax(pred_logits.float(), dim=-1)
    else:
        log_p = torch.log(torch.clamp(pred_probs.float(), min=eps))
    if age_mask is not None or sample_weights is not None:
        per_sample = torch.sum(F.kl_div(log_p, target_probs.float(), reduction="none"), dim=-1)
        w, valid = _prepare_sample_weights(per_sample.shape[0], per_sample.device, age_mask, sample_weights)
        if valid <= eps:
            return torch.sum(per_sample * 0.0)
        return torch.sum(per_sample * w) / (valid + eps)
    return F.kl_div(log_p, target_probs.float(), reduction="batchmean")


def soft_l1_mae_loss(
    expected_age: torch.Tensor,
    target_age: torch.Tensor,
    age_mask: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mean Absolute Error on continuous expected age prediction.
    
    Args:
        expected_age: (B,) expected age scalar sum(p * bin)
        target_age: (B,) ground-truth continuous biological/apparent age
        age_mask: Optional (B,) validity mask (1.0 = valid age, 0.0 = missing/binary)
        sample_weights: Optional (B,) per-sample confidence weight alpha_i
    Returns:
        Scalar tensor: MAE loss
    """
    exp_flat = expected_age.float().view(-1)
    tgt_flat = target_age.float().view(-1)
    per_sample = torch.abs(exp_flat - tgt_flat)
    if age_mask is not None or sample_weights is not None:
        w, valid = _prepare_sample_weights(per_sample.shape[0], per_sample.device, age_mask, sample_weights)
        if valid <= 1e-8:
            return torch.sum(per_sample * 0.0)
        return torch.sum(per_sample * w) / (valid + 1e-8)
    return torch.mean(per_sample)


def interval_censored_ordinal_loss(
    pred_probs: torch.Tensor,
    age_min: torch.Tensor,
    age_max: torch.Tensor,
    expected_age: Optional[torch.Tensor] = None,
    age_mask: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Interval-Censored Ordinal Likelihood + Hinge-Interval Loss for 3-5 Year Age Brackets.

    Instead of penalizing deviations from an arbitrary point midpoint inside an age bracket
    [a_min, a_max], this loss:
    1. Maximizes the integrated probability mass inside the true interval:
           L_mass = -log( sum_{k = floor(a_min)}^{ceil(a_max)} p_k + eps )
    2. Applies a zero-inside-bracket Hinge L1 penalty outside [a_min, a_max]:
           L_hinge = ReLU(a_min - y_hat) + ReLU(y_hat - a_max)

    Args:
        pred_probs: (B, K) predicted age distribution over K=101 bins [0..100]
        age_min: (B,) lower interval bound in years
        age_max: (B,) upper interval bound in years
        expected_age: Optional (B,) expected age scalar
        age_mask: Optional (B,) validity mask (1.0 = valid age/interval, 0.0 = missing)
        sample_weights: Optional (B,) per-sample confidence weight alpha_i
        eps: numerical stability floor
    Returns:
        Scalar tensor: mean interval-censored loss
    """
    p = pred_probs.float()
    if p.ndim == 1:
        p = p.unsqueeze(0)
    if p.shape[0] == 0:
        return torch.sum(p * 0.0)
    p = p / (torch.sum(p, dim=-1, keepdim=True) + 1e-12)
    B, K = p.shape
    device = p.device

    bins = torch.arange(K, device=device, dtype=torch.float32).unsqueeze(0)  # (1, K)
    clean_min = torch.nan_to_num(age_min.float().view(B, 1), nan=0.0, posinf=float(K - 1), neginf=0.0)
    clean_max = torch.nan_to_num(age_max.float().view(B, 1), nan=float(K - 1), posinf=float(K - 1), neginf=0.0)
    low = torch.clamp(torch.floor(clean_min), min=0.0, max=float(K - 1))
    high = torch.clamp(torch.ceil(clean_max), min=0.0, max=float(K - 1))
    low, high = torch.minimum(low, high), torch.maximum(low, high)

    in_bracket_mask = ((bins >= low) & (bins <= high)).float()
    interval_mass = torch.sum(p * in_bracket_mask, dim=-1)  # (B,)
    nll_interval = -torch.log(torch.clamp(interval_mass, min=eps, max=1.0))

    if expected_age is None:
        mu = torch.sum(p * bins, dim=-1)
    else:
        mu = torch.nan_to_num(expected_age.float().view(-1), nan=0.0, posinf=float(K - 1), neginf=0.0)

    low_1d = low.view(-1)
    high_1d = high.view(-1)
    hinge_l1 = F.relu(low_1d - mu) + F.relu(mu - high_1d)
    per_sample = nll_interval + 0.25 * hinge_l1

    if age_mask is not None or sample_weights is not None:
        w, valid = _prepare_sample_weights(B, device, age_mask, sample_weights)
        if valid <= eps:
            return torch.sum(p * 0.0)
        masked_loss = torch.where(w > 0.0, per_sample * w, torch.zeros_like(per_sample))
        return torch.sum(masked_loss) / (valid + eps)

    return torch.mean(per_sample)


def masked_dimorphic_bce_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    gender_mask: torch.Tensor,
    dimorphism_weights: torch.Tensor,
    eps: float = 1e-8,
    sample_weights: Optional[torch.Tensor] = None,
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
        sample_weights: Optional (B,) per-sample confidence weight alpha_i
    Returns:
        Scalar tensor: weighted BCE loss
    """
    logits_flat = pred_logits.float().view(-1)
    targets_flat = targets.float().view(-1)
    dim_flat = dimorphism_weights.float().view(-1)

    bce = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction="none")
    w, valid_count = _prepare_sample_weights(
        logits_flat.shape[0], logits_flat.device, gender_mask, sample_weights
    )

    if valid_count <= eps:
        # If all items in batch are masked out, return zero loss while preserving graph
        return torch.sum(bce * 0.0)

    return torch.sum(bce * w * dim_flat) / (valid_count + eps)


def aleatoric_heteroscedastic_loss(
    expected_age: torch.Tensor,
    target_age: torch.Tensor,
    log_variance: torch.Tensor,
    age_mask: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
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
        age_mask: Optional (B,) validity mask (1.0 = valid age, 0.0 = missing/binary)
        sample_weights: Optional (B,) per-sample confidence weight alpha_i
    Returns:
        Scalar tensor: heteroscedastic loss
    """
    pred_mu = expected_age.float().view(-1)
    tgt_flat = target_age.float().view(-1)
    log_var_flat = log_variance.float().view(-1)

    abs_residual = torch.abs(pred_mu.detach() - tgt_flat)
    s = torch.clamp(log_var_flat, min=-10.0, max=10.0)
    per_sample = torch.exp(-s) * abs_residual + s
    if age_mask is not None or sample_weights is not None:
        w, valid = _prepare_sample_weights(per_sample.shape[0], per_sample.device, age_mask, sample_weights)
        if valid <= 1e-8:
            return torch.sum(per_sample * 0.0)
        return torch.sum(per_sample * w) / (valid + 1e-8)
    return torch.mean(per_sample)


class CoupledHeteroscedasticLaplaceLoss(nn.Module):
    """
    Coupled Heteroscedastic Laplace Loss with Huber Bounds.
    L = exp(-s) * Huber_delta(y_hat - y) + s
    where s = log_variance (or log-scale parameter).
    
    Provides:
    1. Smooth C1 quadratic region near zero (|r| <= delta) preventing gradient oscillations.
    2. Linear Huber region (|r| > delta) preventing gradient explosion.
    3. Smooth gradient scaling: for corrupt or occluded faces with high predicted uncertainty s,
       exp(-s) smoothly scales down gradients into the feature backbone, preventing representation corruption.
    """

    def __init__(
        self,
        delta: float = 1.0,
        min_log_var: float = -10.0,
        max_log_var: float = 10.0,
        detach_mean: bool = False,
    ):
        super().__init__()
        self.delta = float(delta)
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)
        self.detach_mean = detach_mean

    def forward(
        self,
        expected_age: torch.Tensor,
        target_age: torch.Tensor,
        log_variance: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        age_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mu = expected_age.float().view(-1)
        if self.detach_mean:
            mu = mu.detach()
        target = target_age.float().view(-1)
        s = torch.clamp(log_variance.float().view(-1), min=self.min_log_var, max=self.max_log_var)

        huber = F.huber_loss(mu, target, delta=self.delta, reduction="none")
        loss = torch.exp(-s) * huber + s

        if age_mask is not None or sample_weights is not None:
            w, valid = _prepare_sample_weights(loss.shape[0], loss.device, age_mask, sample_weights)
            if valid <= 1e-8:
                return torch.sum(loss * 0.0)
            return torch.sum(loss * w) / (valid + 1e-8)

        return torch.mean(loss)


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon - Khosla et al., NeurIPS 2020).
    Encourages normalized Re-ID embeddings of the same identity to pull together
    while pushing distinct identities apart on the unit hypersphere.
    
    L_supcon = - sum_{i} (1 / |P(i)|) * sum_{p in P(i)} log( exp(z_i . z_p / tau) / sum_{a != i} exp(z_i . z_a / tau) )
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) feature embeddings (will be L2-normalized)
            labels: (B,) identity / cluster label tensor
            sample_weights: Optional (B,) per-sample confidence weight alpha_i
        Returns:
            Scalar contrastive loss tensor
        """
        device = embeddings.device
        B = embeddings.shape[0]
        if B <= 1 or labels is None:
            return torch.sum(embeddings * 0.0)

        labels_1d = labels.contiguous().view(-1)
        if sample_weights is not None:
            alpha = torch.clamp(
                torch.nan_to_num(sample_weights.float().view(-1), nan=0.0, posinf=1.0, neginf=0.0),
                min=0.0,
            )
            if alpha.numel() == 1 and B > 1:
                alpha = alpha.expand(B)
            keep = alpha > 0.0
            if int(keep.sum().item()) <= 1:
                return torch.sum(embeddings * 0.0)
            embeddings = embeddings[keep]
            labels_1d = labels_1d[keep]
            alpha = alpha[keep]
            B = embeddings.shape[0]
        else:
            alpha = torch.ones(B, device=device, dtype=torch.float32)

        # Ensure float32 precision for numerical stability across CPU/GPU
        norm_emb = F.normalize(embeddings.float(), p=2, dim=-1)

        labels_col = labels_1d.view(-1, 1)
        # Pairwise label equality mask: (B, B), only valid non-negative labels can pair
        valid_labels = (labels_col >= 0).float()
        label_mask = torch.eq(labels_col, labels_col.T).float() * (valid_labels @ valid_labels.T)

        # Mask out self-contrast (diagonal)
        diag_mask = torch.scatter(
            torch.ones_like(label_mask),
            1,
            torch.arange(B, device=device).view(-1, 1),
            0.0
        )
        pos_mask = label_mask * diag_mask

        # Compute dot product similarities scaled by temperature in FP32
        sim = torch.div(torch.matmul(norm_emb, norm_emb.T), self.temperature)

        # Subtract max over valid off-diagonal pairs for exact numerical stability
        sim_off_diag = sim - (1.0 - diag_mask) * 1e9
        sim_max, _ = torch.max(sim_off_diag, dim=1, keepdim=True)
        logits = sim - sim_max.detach()

        # Denominator over all pairs except self
        exp_logits = torch.exp(logits) * diag_mask
        denom = torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        log_prob = logits - denom

        # Mean over positive pairs per anchor
        pos_counts = pos_mask.sum(1)
        has_positives = pos_counts > 0

        if not has_positives.any():
            return torch.sum(norm_emb * 0.0)

        # Average log-probability over positive pairs, weighted by anchor alpha_i
        mean_log_prob_pos = (pos_mask * log_prob).sum(1)[has_positives] / (pos_counts[has_positives] + 1e-12)
        loss = -torch.mean(mean_log_prob_pos * alpha[has_positives])
        return loss


def fine_interval_ldl_loss(
    pred_interval_probs: torch.Tensor,
    target_interval_probs: torch.Tensor,
    age_mask: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    23-Bracket (3-5 Year) Fine Age Interval Label Distribution Loss.
    Combines:
    1. Ordinal Cramér Distance across the 23 ordered age intervals:
       sum_{m=0}^{22} (CDF_pred(m) - CDF_target(m))^2
    2. KL Divergence on the 23-bracket simplex:
       KL(q_23 || p_23)
    """
    p = pred_interval_probs.float()
    q = target_interval_probs.float()
    if p.ndim == 1:
        p = p.unsqueeze(0)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if p.shape[0] == 0:
        return torch.sum(p * 0.0)

    p = p / (torch.sum(p, dim=-1, keepdim=True) + 1e-12)
    q = q / (torch.sum(q, dim=-1, keepdim=True) + 1e-12)

    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    cramer_23 = torch.sum((cdf_p - cdf_q) ** 2, dim=-1)

    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    kl_23 = torch.clamp(torch.sum(q * (torch.log(q_safe) - torch.log(p_safe)), dim=-1), min=0.0)

    per_sample = cramer_23 + 0.5 * kl_23
    if age_mask is not None or sample_weights is not None:
        w, valid = _prepare_sample_weights(per_sample.shape[0], per_sample.device, age_mask, sample_weights)
        if valid <= eps:
            return torch.sum(p * 0.0)
        masked_loss = torch.where(w > 0.0, per_sample * w, torch.zeros_like(per_sample))
        return torch.sum(masked_loss) / (valid + eps)
    return torch.mean(per_sample)


def ldl_tikhonov_smoothness_loss(
    pred_probs: torch.Tensor,
    age_mask: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Second-Order Finite-Difference Tikhonov Smoothness Regularization on the 101-bin Age Simplex.
    Penalizes jagged bin-to-bin oscillations:
        L_smooth(p) = mean_{k=1..K-2} (p_{k+1} - 2*p_k + p_{k-1})^2 * K
    Enforces smooth, biologically unimodal/bell-shaped age distributions under extreme noise.
    """
    p = pred_probs.float()
    if p.ndim == 1:
        p = p.unsqueeze(0)
    if p.shape[0] == 0 or p.shape[-1] < 3:
        return torch.sum(p * 0.0)

    p = p / (torch.sum(p, dim=-1, keepdim=True) + 1e-12)
    K = float(p.shape[-1])
    d2 = p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2]
    per_sample = torch.mean(d2 ** 2, dim=-1) * K

    if age_mask is not None or sample_weights is not None:
        w, valid = _prepare_sample_weights(per_sample.shape[0], per_sample.device, age_mask, sample_weights)
        if valid <= eps:
            return torch.sum(p * 0.0)
        masked_loss = torch.where(w > 0.0, per_sample * w, torch.zeros_like(per_sample))
        return torch.sum(masked_loss) / (valid + eps)
    return torch.mean(per_sample)


def photometric_consistency_loss(
    preds_clean: Dict[str, torch.Tensor],
    preds_corrupted: Dict[str, torch.Tensor],
    age_mask: Optional[torch.Tensor] = None,
    gender_mask: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Photometric, Pose & Extreme-Noise Dual-View Invariance Loss.
    Enforces consistency between predictions on a clean face view and an extreme
    lighting/blur/noise-corrupted view of the same face:
    1. Symmetric Jensen-Shannon + Cramér consistency on the 101-bin age distribution.
    2. Sigmoid probability consistency on the craniofacial dimorphic gender head.
    3. Unit-hypersphere cosine alignment on the 64-D Re-ID identity embedding.
    """
    p_clean = preds_clean["age_probs"].float().detach()
    p_corr = preds_corrupted["age_probs"].float()
    p_clean = p_clean / (torch.sum(p_clean, dim=-1, keepdim=True) + 1e-12)
    p_corr = p_corr / (torch.sum(p_corr, dim=-1, keepdim=True) + 1e-12)

    m_dist = 0.5 * (p_clean + p_corr)
    p_clean_safe = torch.clamp(p_clean, min=eps)
    p_corr_safe = torch.clamp(p_corr, min=eps)
    log_m = torch.log(torch.clamp(m_dist, min=eps))
    kl_clean_m = torch.sum(p_clean * (torch.log(p_clean_safe) - log_m), dim=-1)
    kl_corr_m = torch.sum(p_corr * (torch.log(p_corr_safe) - log_m), dim=-1)
    jsd_age = torch.clamp(0.5 * (kl_clean_m + kl_corr_m), min=0.0)
    cramer_cons = torch.sum((torch.cumsum(p_corr, dim=-1) - torch.cumsum(p_clean, dim=-1)) ** 2, dim=-1)
    age_cons_per = jsd_age + 0.5 * cramer_cons

    B = p_clean.shape[0]
    device = p_clean.device
    w_age, valid_age = _prepare_sample_weights(B, device, age_mask, sample_weights)
    if valid_age > eps:
        loss_age_cons = torch.sum(torch.where(w_age > 0.0, age_cons_per * w_age, torch.zeros_like(age_cons_per))) / (valid_age + eps)
    else:
        loss_age_cons = torch.sum(p_corr * 0.0)

    if "gender_logit" in preds_clean and "gender_logit" in preds_corrupted:
        g_clean = torch.sigmoid(preds_clean["gender_logit"].float().view(-1)).detach()
        g_corr = torch.sigmoid(preds_corrupted["gender_logit"].float().view(-1))
        g_diff = (g_corr - g_clean) ** 2
        w_gen, valid_gen = _prepare_sample_weights(B, device, gender_mask, sample_weights)
        if valid_gen > eps:
            loss_gen_cons = torch.sum(torch.where(w_gen > 0.0, g_diff * w_gen, torch.zeros_like(g_diff))) / (valid_gen + eps)
        else:
            loss_gen_cons = torch.sum(g_corr * 0.0)
    else:
        loss_gen_cons = torch.sum(p_corr * 0.0)

    if "reid_embedding" in preds_clean and "reid_embedding" in preds_corrupted:
        r_clean = F.normalize(preds_clean["reid_embedding"].float(), p=2, dim=-1).detach()
        r_corr = F.normalize(preds_corrupted["reid_embedding"].float(), p=2, dim=-1)
        cos_dist = 1.0 - torch.sum(r_clean * r_corr, dim=-1)
        w_all, valid_all = _prepare_sample_weights(B, device, None, sample_weights)
        if valid_all > eps:
            loss_reid_cons = torch.sum(torch.where(w_all > 0.0, cos_dist * w_all, torch.zeros_like(cos_dist))) / (valid_all + eps)
        else:
            loss_reid_cons = torch.sum(r_corr * 0.0)
    else:
        loss_reid_cons = torch.sum(p_corr * 0.0)

    return loss_age_cons + 0.5 * loss_gen_cons + 0.25 * loss_reid_cons


class UnifiedBiometricLoss(nn.Module):
    """
    Unified multi-task loss combining:
    - Cramér Distance (squared EMD on 101-bin age CDFs)
    - Soft-KL Divergence (FP16 protected)
    - Soft-L1 MAE on expected scalar age
    - Interval-Censored Ordinal Likelihood + 23-Bracket Fine Interval LDL Loss
    - Tikhonov Second-Order Distribution Smoothness Regularization
    - Masked Dimorphic BCE (valid-sample normalized)
    - Aleatoric Heteroscedastic Uncertainty (L1-Detached Laplace or Coupled Huber)
    - Supervised Contrastive Loss for 64-D Re-ID Head (weight 0.15)
    """

    def __init__(
        self,
        weight_cramer: float = 1.0,
        weight_kl: float = 0.5,
        weight_mae: float = 0.2,
        weight_gender: float = 1.0,
        weight_aleatoric: float = 0.1,
        weight_reid: float = 0.15,
        weight_interval: float = 0.25,
        weight_fine_23: float = 0.35,
        weight_smoothness: float = 0.0,
        reid_temperature: float = 0.07,
        use_coupled_laplace: bool = False,
    ):
        super().__init__()
        self.weight_cramer = weight_cramer
        self.weight_kl = weight_kl
        self.weight_mae = weight_mae
        self.weight_gender = weight_gender
        self.weight_aleatoric = weight_aleatoric
        self.weight_reid = weight_reid
        self.weight_interval = weight_interval
        self.weight_fine_23 = weight_fine_23
        self.weight_smoothness = weight_smoothness
        self.supcon_loss = SupervisedContrastiveLoss(temperature=reid_temperature)
        self.coupled_laplace_loss = CoupledHeteroscedasticLaplaceLoss()
        self.use_coupled_laplace = use_coupled_laplace

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward multi-task loss computation.
        """
        age_mask = targets.get("age_mask", targets.get("has_valid_age"))
        alpha_w = targets.get("alpha_sample_weight", None)

        # 1. Cramér Distance (per-sample weighted by age_mask and alpha_sample_weight)
        l_cramer = cramer_distance_loss(
            preds["age_probs"],
            targets["ldl_101"],
            age_mask=age_mask,
            sample_weights=alpha_w,
        )

        # 2. Soft-KL Divergence (FP16 numerical stability via log_softmax)
        age_logits = preds.get("age_logits")
        l_kl = soft_kl_divergence_loss(
            preds["age_probs"],
            targets["ldl_101"],
            pred_logits=age_logits,
            age_mask=age_mask,
            sample_weights=alpha_w,
        )

        # 3. Soft-L1 MAE
        l_mae = soft_l1_mae_loss(
            preds["expected_age"],
            targets["age_continuous"],
            age_mask=age_mask,
            sample_weights=alpha_w,
        )

        # 3b. Interval-Censored Ordinal Loss (active when age_min and age_max are provided)
        if "age_min" in targets and "age_max" in targets and self.weight_interval > 0.0:
            l_interval = interval_censored_ordinal_loss(
                preds["age_probs"],
                targets["age_min"],
                targets["age_max"],
                expected_age=preds.get("expected_age"),
                age_mask=age_mask,
                sample_weights=alpha_w,
            )
        else:
            l_interval = torch.sum(preds["age_probs"] * 0.0)

        # 3c. 23-Bracket Fine Interval LDL Loss (active when ldl_fine_23 is provided in targets)
        if "ldl_fine_23" in targets and self.weight_fine_23 > 0.0:
            pred_iv_probs = preds.get("age_interval_probs")
            if pred_iv_probs is None and "age_probs" in preds and preds["age_probs"].shape[-1] == 101:
                from .intervals import aggregate_101_to_fine_intervals_torch
                pred_iv_probs = aggregate_101_to_fine_intervals_torch(preds["age_probs"])

            fine_iv_logits = preds.get("fine_interval_logits", preds.get("age_interval_logits"))
            l_fine_terms = []
            if fine_iv_logits is not None:
                l_fine_terms.append(
                    fine_interval_ldl_loss(
                        F.softmax(fine_iv_logits.float(), dim=-1),
                        targets["ldl_fine_23"],
                        age_mask=age_mask,
                        sample_weights=alpha_w,
                    )
                )
            if pred_iv_probs is not None:
                l_fine_terms.append(
                    fine_interval_ldl_loss(
                        pred_iv_probs,
                        targets["ldl_fine_23"],
                        age_mask=age_mask,
                        sample_weights=alpha_w,
                    )
                )
            if len(l_fine_terms) == 2:
                l_fine_23 = 0.5 * (l_fine_terms[0] + l_fine_terms[1])
            elif len(l_fine_terms) == 1:
                l_fine_23 = l_fine_terms[0]
            else:
                l_fine_23 = torch.sum(preds["age_probs"] * 0.0)
        else:
            l_fine_23 = torch.sum(preds["age_probs"] * 0.0)

        # 3d. Tikhonov Distribution Smoothness Regularization
        if self.weight_smoothness > 0.0:
            l_smooth = ldl_tikhonov_smoothness_loss(
                preds["age_probs"],
                age_mask=age_mask,
                sample_weights=alpha_w,
            )
        else:
            l_smooth = torch.sum(preds["age_probs"] * 0.0)

        # 4. Masked Dimorphic BCE (strictly sample-count normalized, per-sample alpha weighted)
        gender_mask = targets.get("gender_mask", torch.ones_like(targets["gender"]))
        dimorphism_w = targets.get("dimorphism_weight", torch.ones_like(targets["gender"]))
        l_gender = masked_dimorphic_bce_loss(
            preds["gender_logit"],
            targets["gender"],
            gender_mask,
            dimorphism_w,
            sample_weights=alpha_w,
        )

        # 5. Aleatoric Heteroscedastic Loss
        if self.use_coupled_laplace:
            l_aleatoric = self.coupled_laplace_loss(
                preds["expected_age"],
                targets["age_continuous"],
                preds["log_variance"],
                sample_weights=alpha_w,
                age_mask=age_mask,
            )
        else:
            l_aleatoric = aleatoric_heteroscedastic_loss(
                preds["expected_age"],
                targets["age_continuous"],
                preds["log_variance"],
                age_mask=age_mask,
                sample_weights=alpha_w,
            )

        # 6. Re-ID Supervised Contrastive Loss (strictly genuine non-negative subject identities only)
        reid_emb = preds.get("reid_embedding")
        if reid_emb is not None:
            id_labels = targets.get(
                "identity",
                targets.get("reid_label", targets.get("labels", targets.get("subject_id", None)))
            )
            if id_labels is not None:
                l_reid = self.supcon_loss(reid_emb, id_labels, sample_weights=alpha_w)
            else:
                l_reid = torch.sum(reid_emb * 0.0)
        else:
            l_reid = torch.tensor(0.0, device=preds["age_probs"].device if "age_probs" in preds else "cpu")

        # Weighted total (per-sample alpha_sample_weight already applied in each head)
        total_loss = (
            self.weight_cramer * l_cramer
            + self.weight_kl * l_kl
            + self.weight_mae * l_mae
            + self.weight_interval * l_interval
            + self.weight_fine_23 * l_fine_23
            + self.weight_smoothness * l_smooth
            + self.weight_gender * l_gender
            + self.weight_aleatoric * l_aleatoric
            + self.weight_reid * l_reid
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_cramer": float(l_cramer.item()),
            "loss_kl": float(l_kl.item()),
            "loss_mae": float(l_mae.item()),
            "loss_interval": float(l_interval.item()),
            "loss_fine_23": float(l_fine_23.item()),
            "loss_smoothness": float(l_smooth.item()),
            "loss_gender": float(l_gender.item()),
            "loss_aleatoric": float(l_aleatoric.item()),
            "loss_reid": float(l_reid.item()),
        }

        return total_loss, metrics


def refine_interval_ldl_with_teacher(
    target_ldl_101: torch.Tensor,
    teacher_age_probs: torch.Tensor,
    age_min: Optional[torch.Tensor] = None,
    age_max: Optional[torch.Tensor] = None,
    blend_weight: float = 0.45,
    margin_years: float = 1.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Interval-Constrained Teacher Label Distribution Refinement.

    For interval-annotated or noisy apparent-age samples ([age_min, age_max]), projects
    the Teacher's 101-bin age distribution onto a smooth window around [age_min, age_max]
    so the Teacher resolves the fine intra-bracket continuous age peak without violating
    ground-truth bracket boundaries.
    """
    q_gt = target_ldl_101.float()
    p_t = teacher_age_probs.detach().float()
    q_gt = q_gt / (torch.sum(q_gt, dim=-1, keepdim=True) + 1e-12)
    p_t = p_t / (torch.sum(p_t, dim=-1, keepdim=True) + 1e-12)

    B, K = q_gt.shape
    device = q_gt.device
    beta = float(max(0.0, min(1.0, blend_weight)))

    if age_min is not None and age_max is not None:
        bins = torch.arange(K, device=device, dtype=torch.float32).unsqueeze(0)  # (1, K)
        low = torch.clamp(
            torch.nan_to_num(age_min.float().view(B, 1), nan=0.0, posinf=float(K - 1), neginf=0.0),
            min=0.0,
            max=float(K - 1),
        )
        high = torch.clamp(
            torch.nan_to_num(age_max.float().view(B, 1), nan=float(K - 1), posinf=float(K - 1), neginf=0.0),
            min=0.0,
            max=float(K - 1),
        )
        low, high = torch.minimum(low, high), torch.maximum(low, high)
        dist_out = F.relu(low - bins) + F.relu(bins - high)
        window = torch.exp(-0.5 * (dist_out / max(0.5, float(margin_years))) ** 2)
        p_t_proj = p_t * window
        p_t_sum = torch.sum(p_t_proj, dim=-1, keepdim=True)
        p_t_proj = torch.where(p_t_sum > eps, p_t_proj / (p_t_sum + 1e-12), q_gt)
        eff_beta = beta * ((high - low) >= 2.0).float()
    else:
        p_t_proj = p_t
        eff_beta = beta

    refined = (1.0 - eff_beta) * q_gt + eff_beta * p_t_proj
    refined = refined / (torch.sum(refined, dim=-1, keepdim=True) + 1e-12)
    return refined


class BiometricDistillationLoss(nn.Module):
    """
    Multi-Task Knowledge Distillation & Missing-Label Imputation Engine for Facial Biometrics:
    1. Temperature-Scaled 101-Bin Age Simplex KL + Ordinal Cramér CDF Distillation.
    2. 23-Bracket Fine Age Interval Logit & Simplex Distillation.
    3. Confidence-Gated Missing-Gender Pseudo-Label Imputation (rescues unlabelled gender rows
       where gender_mask == 0 when Teacher confidence |2*p_T - 1| >= confidence_threshold)
       + Soft Dimorphic Gender Distillation on labelled rows.
    4. Unit-Hypersphere Cosine & Pairwise Relational Similarity Matrix Distillation on 64-D Re-ID.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        gender_temperature: float = 1.5,
        missing_gender_conf_thresh: float = 0.65,
        weight_age_kd: float = 1.0,
        weight_interval_kd: float = 0.5,
        weight_gender_kd: float = 0.6,
        weight_reid_kd: float = 0.35,
    ):
        super().__init__()
        self.temperature = float(max(0.1, temperature))
        self.gender_temperature = float(max(0.1, gender_temperature))
        self.missing_gender_conf_thresh = float(max(0.0, min(0.99, missing_gender_conf_thresh)))
        self.weight_age_kd = float(weight_age_kd)
        self.weight_interval_kd = float(weight_interval_kd)
        self.weight_gender_kd = float(weight_gender_kd)
        self.weight_reid_kd = float(weight_reid_kd)

    def forward(
        self,
        student_preds: Dict[str, torch.Tensor],
        teacher_preds: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        targets = targets or {}
        age_mask = targets.get("age_mask", targets.get("has_valid_age"))
        alpha_w = targets.get("alpha_sample_weight")
        tau = self.temperature

        # 1. 101-Bin Age Logit & Cramér CDF Distillation
        s_age_probs = student_preds["age_probs"].float()
        t_age_probs = teacher_preds["age_probs"].detach().float()
        s_age_probs = s_age_probs / (torch.sum(s_age_probs, dim=-1, keepdim=True) + 1e-12)
        t_age_probs = t_age_probs / (torch.sum(t_age_probs, dim=-1, keepdim=True) + 1e-12)

        B = s_age_probs.shape[0]
        device = s_age_probs.device

        if "age_logits" in student_preds and "age_logits" in teacher_preds:
            s_log_p_tau = F.log_softmax(student_preds["age_logits"].float() / tau, dim=-1)
            t_p_tau = F.softmax(teacher_preds["age_logits"].detach().float() / tau, dim=-1)
            kl_tau = (tau ** 2) * torch.sum(F.kl_div(s_log_p_tau, t_p_tau, reduction="none"), dim=-1)
        else:
            s_log_p = torch.log(torch.clamp(s_age_probs, min=eps))
            kl_tau = torch.sum(F.kl_div(s_log_p, t_age_probs, reduction="none"), dim=-1)

        cramer_kd = torch.sum((torch.cumsum(s_age_probs, dim=-1) - torch.cumsum(t_age_probs, dim=-1)) ** 2, dim=-1)
        age_kd_per = 0.6 * kl_tau + 0.4 * cramer_kd
        w_age, valid_age = _prepare_sample_weights(B, device, age_mask, alpha_w)
        if valid_age > eps:
            loss_age_kd = torch.sum(torch.where(w_age > 0.0, age_kd_per * w_age, torch.zeros_like(age_kd_per))) / (valid_age + eps)
        else:
            loss_age_kd = torch.sum(s_age_probs * 0.0)

        # 2. 23-Bracket Fine Interval Distillation
        s_iv_logits = student_preds.get("fine_interval_logits", student_preds.get("age_interval_logits"))
        t_iv_logits = teacher_preds.get("fine_interval_logits", teacher_preds.get("age_interval_logits"))
        if s_iv_logits is not None and t_iv_logits is not None:
            s_iv_log_p = F.log_softmax(s_iv_logits.float() / tau, dim=-1)
            t_iv_p = F.softmax(t_iv_logits.detach().float() / tau, dim=-1)
            iv_kl = (tau ** 2) * torch.sum(F.kl_div(s_iv_log_p, t_iv_p, reduction="none"), dim=-1)
            s_iv_p1 = F.softmax(s_iv_logits.float(), dim=-1)
            t_iv_p1 = F.softmax(t_iv_logits.detach().float(), dim=-1)
            iv_cramer = torch.sum((torch.cumsum(s_iv_p1, dim=-1) - torch.cumsum(t_iv_p1, dim=-1)) ** 2, dim=-1)
            iv_per = 0.5 * iv_kl + 0.5 * iv_cramer
            if valid_age > eps:
                loss_iv_kd = torch.sum(torch.where(w_age > 0.0, iv_per * w_age, torch.zeros_like(iv_per))) / (valid_age + eps)
            else:
                loss_iv_kd = torch.sum(s_iv_logits * 0.0)
        else:
            loss_iv_kd = torch.sum(s_age_probs * 0.0)

        # 3. Gender Distillation + Confidence-Gated Missing-Gender Imputation
        if "gender_logit" in student_preds and "gender_logit" in teacher_preds:
            tau_g = self.gender_temperature
            s_g_logit = student_preds["gender_logit"].float().view(-1)
            t_g_logit = teacher_preds["gender_logit"].detach().float().view(-1)
            t_g_prob = torch.sigmoid(t_g_logit / tau_g)
            t_g_hard_prob = torch.sigmoid(t_g_logit)
            t_g_conf = torch.abs(2.0 * t_g_hard_prob - 1.0)  # in [0, 1]

            dim_w = targets.get("dimorphism_weight", torch.ones_like(s_g_logit)).float().view(-1)
            if "gender_mask" in targets:
                g_mask = targets["gender_mask"].float().view(-1)
            else:
                g_mask = torch.ones_like(s_g_logit)

            # Impute missing gender rows (g_mask == 0) when Teacher is confident >= threshold
            imputed_mask = ((g_mask <= 0.0) & (t_g_conf >= self.missing_gender_conf_thresh)).float() * t_g_conf
            effective_g_mask = torch.clamp(g_mask + 0.75 * imputed_mask, min=0.0, max=1.0)

            bce_kd = (tau_g ** 2) * F.binary_cross_entropy_with_logits(
                s_g_logit / tau_g, t_g_prob, reduction="none"
            )
            w_gen, valid_gen = _prepare_sample_weights(B, device, effective_g_mask, alpha_w)
            if valid_gen > eps:
                loss_gender_kd = torch.sum(bce_kd * w_gen * dim_w) / (valid_gen + eps)
            else:
                loss_gender_kd = torch.sum(s_g_logit * 0.0)
            imputed_count = float(torch.sum(imputed_mask > 0.0).item())
        else:
            loss_gender_kd = torch.sum(s_age_probs * 0.0)
            imputed_count = 0.0

        # 4. Unit-Hypersphere Cosine & Pairwise Relational Re-ID Distillation
        if "reid_embedding" in student_preds and "reid_embedding" in teacher_preds:
            s_reid = F.normalize(student_preds["reid_embedding"].float(), p=2, dim=-1)
            t_reid = F.normalize(teacher_preds["reid_embedding"].detach().float(), p=2, dim=-1)
            cos_loss = torch.mean(1.0 - torch.sum(s_reid * t_reid, dim=-1))
            if B > 1:
                sim_s = torch.matmul(s_reid, s_reid.T)
                sim_t = torch.matmul(t_reid, t_reid.T)
                rel_loss = F.mse_loss(sim_s, sim_t)
            else:
                rel_loss = torch.sum(s_reid * 0.0)
            loss_reid_kd = cos_loss + 0.5 * rel_loss
        else:
            loss_reid_kd = torch.sum(s_age_probs * 0.0)

        total_kd = (
            self.weight_age_kd * loss_age_kd
            + self.weight_interval_kd * loss_iv_kd
            + self.weight_gender_kd * loss_gender_kd
            + self.weight_reid_kd * loss_reid_kd
        )
        kd_metrics = {
            "loss_kd_total": float(total_kd.item()),
            "loss_kd_age": float(loss_age_kd.item()),
            "loss_kd_interval": float(loss_iv_kd.item()),
            "loss_kd_gender": float(loss_gender_kd.item()),
            "loss_kd_reid": float(loss_reid_kd.item()),
            "kd_imputed_gender_count": imputed_count,
        }
        return total_kd, kd_metrics


