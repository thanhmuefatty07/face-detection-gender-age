"""
Biometric Multi-Task Deep Neural Architectures
=============================================
Provides modular PyTorch multi-task backbones:
1. ConvNeXt-V2 (Nano/Tiny/Base): 7x7 depthwise conv captures fine micro-epidermal texture.
2. MobileNetV4 (Conv-Medium): Hardware-friendly Universal Inverted Bottleneck (UIB).
Heads:
- Head 1: 101-bin Softmax continuous Age LDL distribution.
- Head 2: Dimorphism-weighted Gender logit.
- Head 3: Aleatoric uncertainty sigma log-variance.
- Head 4: 64-D normalized Re-ID facial identity embedding.
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None


class ConvNeXtV2BiometricHead(nn.Module):
    """Multi-task projection head for ConvNeXt representations."""

    def __init__(self, in_features: int = 512, num_age_bins: int = 101, reid_dim: int = 64):
        super().__init__()
        self.age_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_age_bins)
        )
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.reid_head = nn.Sequential(
            nn.Linear(in_features, reid_dim),
            nn.LayerNorm(reid_dim)
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        age_logits = self.age_head(feat)
        gender_logit = self.gender_head(feat)
        log_var = self.uncertainty_head(feat)
        reid_emb = F.normalize(self.reid_head(feat), p=2, dim=-1)
        return age_logits, gender_logit, log_var, reid_emb


class MobileNetV4BiometricHead(nn.Module):
    """Lightweight projection head optimized for mobile/CPU inference."""

    def __init__(self, in_features: int = 256, num_age_bins: int = 101, reid_dim: int = 64):
        super().__init__()
        self.age_head = nn.Linear(in_features, num_age_bins)
        self.gender_head = nn.Linear(in_features, 1)
        self.uncertainty_head = nn.Linear(in_features, 1)
        self.reid_head = nn.Sequential(
            nn.Linear(in_features, reid_dim),
            nn.LayerNorm(reid_dim)
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        age_logits = self.age_head(feat)
        gender_logit = self.gender_head(feat)
        log_var = self.uncertainty_head(feat)
        reid_emb = F.normalize(self.reid_head(feat), p=2, dim=-1)
        return age_logits, gender_logit, log_var, reid_emb


class FallbackStem(nn.Module):
    """Lightweight convolutional feature extractor fallback if timm is unavailable."""

    def __init__(self, out_features: int = 256):
        super().__init__()
        self.num_features = out_features
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 80x80
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 40x40
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 20x20
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, out_features, kernel_size=3, stride=2, padding=1), # 10x10
            nn.BatchNorm2d(out_features),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BiometricMultiTaskModel(nn.Module):
    """
    Unified multi-task biometric model.
    Accepts 160x160 aligned face tensor: (B, 3, 160, 160).
    Extracts:
    1. 101-bin age probability simplex & expected continuous age.
    2. Gender logit and probability.
    3. Aleatoric uncertainty log-variance & standard deviation.
    4. 64-D unit-sphere normalized Re-ID identity embedding.
    """

    def __init__(
        self,
        backbone_name: str = "convnextv2_nano",
        num_age_bins: int = 101,
        reid_dim: int = 64,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_age_bins = num_age_bins
        self.reid_dim = reid_dim

        # Map generic shorthand names
        actual_name = backbone_name
        if backbone_name == "mobilenetv4":
            actual_name = "mobilenetv4_conv_medium"
        elif backbone_name == "convnextv2":
            actual_name = "convnextv2_nano"

        # Initialize backbone
        if timm is not None:
            try:
                self.backbone = timm.create_model(actual_name, pretrained=pretrained, num_classes=0)
                # Probe actual output dimension
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 160, 160)
                    out_probe = self.backbone(dummy)
                    if out_probe.ndim > 2:
                        out_probe = torch.flatten(F.adaptive_avg_pool2d(out_probe, (1, 1)), 1)
                    in_features = out_probe.shape[-1]
            except Exception:
                self.backbone = FallbackStem(out_features=256)
                in_features = 256
        else:
            self.backbone = FallbackStem(out_features=256)
            in_features = 256

        self.in_features = in_features

        # Head 1: 101-bin continuous Age LDL Head
        self.age_ldl_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Linear(256, num_age_bins)
        )

        # Head 2: Gender Classification Head
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

        # Head 3: Aleatoric Heteroscedastic Uncertainty Head: log(sigma^2)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # Head 4: 64-D Re-ID Identity Appearance Embedding Head
        self.reid_head = nn.Sequential(
            nn.Linear(in_features, reid_dim),
            nn.LayerNorm(reid_dim)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Aligned face tensor (B, 3, 160, 160)
        Returns:
            Dictionary containing predictions:
            - 'age_logits': (B, 101)
            - 'age_probs': (B, 101) probability simplex
            - 'expected_age': (B,) scalar continuous expectation
            - 'age_expected': (B,) alias for backward compatibility
            - 'gender_logit': (B,) logit (positive = Female)
            - 'gender_prob': (B,) probability of Female
            - 'log_variance': (B,) log(sigma^2)
            - 'aleatoric_std': (B,) sigma
            - 'reid_embedding': (B, 64) L2-normalized unit sphere embedding
            - 'features': (B, in_features)
        """
        feat = self.backbone(x)
        if feat.ndim > 2:
            feat = torch.flatten(F.adaptive_avg_pool2d(feat, (1, 1)), 1)

        age_logits = self.age_ldl_head(feat)
        age_probs = F.softmax(age_logits, dim=-1)

        gender_logit = self.gender_head(feat).squeeze(-1)
        gender_prob = torch.sigmoid(gender_logit)

        log_var = self.uncertainty_head(feat).squeeze(-1)
        aleatoric_std = torch.exp(0.5 * torch.clamp(log_var, min=-10.0, max=10.0))

        # Re-ID normalized embedding on unit sphere
        reid_raw = self.reid_head(feat)
        reid_emb = F.normalize(reid_raw, p=2, dim=-1)

        # Continuous age expectation
        age_bins = torch.arange(self.num_age_bins, device=x.device, dtype=torch.float32)
        expected_age = torch.sum(age_probs * age_bins, dim=-1)

        return {
            "age_logits": age_logits,
            "age_probs": age_probs,
            "expected_age": expected_age,
            "age_expected": expected_age,
            "gender_logit": gender_logit,
            "gender_prob": gender_prob,
            "log_variance": log_var,
            "aleatoric_std": aleatoric_std,
            "reid_embedding": reid_emb,
            "features": feat
        }
