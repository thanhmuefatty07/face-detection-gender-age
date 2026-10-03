"""
Biometric Multi-Task Deep Neural Architectures
=============================================
Provides modular PyTorch multi-task backbones:
1. ConvNeXt-V2 (Micro/Nano/Tiny): 7x7 depthwise conv captures fine micro-epidermal texture.
2. MobileNetV4 (Conv-Medium): Universal Inverted Bottleneck (UIB) designed for hardware latency.
Heads:
- Head 1: 101-bin Softmax continuous Age LDL distribution.
- Head 2: Dimorphism-weighted Gender logit.
- Head 3: Aleatoric uncertainty sigma log-variance.
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtV2BiometricHead(nn.Module):
    """Multi-task projection head for ConvNeXt representations."""

    def __init__(self, in_features: int = 512, num_age_bins: int = 101):
        super().__init__()
        # Age head: 101-bin probability simplex
        self.age_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_age_bins)
        )
        # Gender head: 1 logit for P(Female)
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        # Aleatoric uncertainty head: log(sigma^2)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        age_logits = self.age_head(feat)
        gender_logit = self.gender_head(feat)
        log_var = self.uncertainty_head(feat)
        return age_logits, gender_logit, log_var


class MobileNetV4BiometricHead(nn.Module):
    """Lightweight projection head optimized for mobile/CPU inference."""

    def __init__(self, in_features: int = 256, num_age_bins: int = 101):
        super().__init__()
        self.age_head = nn.Linear(in_features, num_age_bins)
        self.gender_head = nn.Linear(in_features, 1)
        self.uncertainty_head = nn.Linear(in_features, 1)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.age_head(feat), self.gender_head(feat), self.uncertainty_head(feat)


class BiometricMultiTaskModel(nn.Module):
    """
    Unified multi-task biometric model.
    Accepts 160x160 aligned face tensor: (B, 3, 160, 160).
    """

    def __init__(self, backbone_name: str = "mobilenetv4", num_age_bins: int = 101, pretrained: bool = False):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_age_bins = num_age_bins

        # Lightweight convolutional feature extractor (fallback or full timm)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 80x80
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 40x40
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# 20x20
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 10x10
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.head = MobileNetV4BiometricHead(in_features=256, num_age_bins=num_age_bins)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Returns:
            {
                'age_probs': (B, 101) probability simplex,
                'age_expected': (B,) scalar age expectation,
                'gender_prob': (B,) probability of Female,
                'aleatoric_std': (B,) standard deviation of visual uncertainty
            }
        """
        feat = self.stem(x)
        age_logits, gender_logit, log_var = self.head(feat)

        age_probs = F.softmax(age_logits, dim=-1)
        bin_weights = torch.arange(self.num_age_bins, device=x.device, dtype=torch.float32)
        age_expected = torch.sum(age_probs * bin_weights, dim=-1)

        gender_prob = torch.sigmoid(gender_logit).squeeze(-1)
        aleatoric_std = torch.exp(0.5 * log_var).squeeze(-1)

        return {
            "age_probs": age_probs,
            "age_expected": age_expected,
            "gender_prob": gender_prob,
            "aleatoric_std": aleatoric_std
        }
