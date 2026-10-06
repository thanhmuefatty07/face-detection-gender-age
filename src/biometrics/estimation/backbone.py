"""
Biometric Multi-Task Deep Neural Architectures
=============================================
Provides modular PyTorch multi-task backbones:
1. ConvNeXt-V2-Tiny (Pareto Sweet Spot): Multi-scale hierarchical feature extraction.
2. MultiScaleBiometricFusion:
   - Stage 2 (20x20): micro-epidermal texture, wrinkles, pores.
   - Stage 3 (10x10): mesoscopic facial contours, landmarks.
   - Stage 4 (5x5): macro craniofacial bone structure.
   - 1x1 Conv -> 128-ch projection + Mixed Dual-Pooling (GAP || GMP) -> 256-D per stage.
   - Concatenated [z2 || z3 || z4] (768-D) -> Linear + LayerNorm -> z_fused (512-D).
3. Head Routing Separation:
   - Head 1 (Age LDL): 512-D z_fused -> 101-bin Softmax continuous distribution.
   - Head 2 (Gender): 256-D z4 ONLY -> dimorphism logit (cranial macro-structure invariance).
   - Head 3 (Aleatoric Uncertainty): 512-D z_fused -> log-variance.
   - Head 4 (Re-ID Identity): 256-D z4 ONLY -> 64-D unit-sphere normalized vector.
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for Efficient Mobile Network Design (Hou et al., CVPR 2021).
    Encodes channel relationships and long-range dependencies with precise positional
    information along horizontal (X) and vertical (Y) coordinate directions.
    
    Particularly effective for localizing fine facial landmarks and micro-wrinkle
    patterns (e.g., forehead furrows, crow's feet, nasolabial folds) on Stage 2
    without diluting directional spatial cues prior to pooling.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing():
            if x.shape[2] <= 1 or x.shape[3] <= 1:
                return x

        identity = x
        b, c, h, w = x.shape

        # 1D direction-aware pooling along X and Y axes
        x_h = torch.mean(x, dim=3, keepdim=True)                        # (B, C, H, 1)
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)    # (B, C, W, 1)

        # Concatenate along spatial coordinate axis
        y = torch.cat([x_h, x_w], dim=2)                                # (B, C, H + W, 1)
        y = self.act(self.bn1(self.conv1(y)))

        # Split back to H and W components
        x_h_out = y[:, :, :h, :]                                        # (B, mip, H, 1)
        x_w_out = y[:, :, h:, :].permute(0, 1, 3, 2)                    # (B, mip, 1, W)

        a_h = torch.sigmoid(self.conv_h(x_h_out))                       # (B, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(x_w_out))                       # (B, C, 1, W)

        return identity * a_h * a_w


class MultiScaleBiometricFusion(nn.Module):
    """
    Multi-Scale Biometric Feature Fusion with Stage 2 Coordinate Attention.
    
    Extracts and fuses hierarchical spatial features:
    - Stage 2: (B, C2, 20, 20) -> Coordinate Attention micro-epidermal textures, wrinkles, and pores
    - Stage 3: (B, C3, 10, 10) -> mesoscopic facial contours and landmarks
    - Stage 4: (B, C4, 5, 5)   -> macro craniofacial bone structure
    
    Processing:
    For stage k in {2, 3, 4}:
      1. Stage 2 CoordinateAttention (direction-aware X/Y localization)
      2. 1x1 Conv projection (C_k -> 128)
      3. Mixed Dual-Pooling: [AdaptiveAvgPool2d((1, 1)) || AdaptiveMaxPool2d((1, 1))] -> z_k in R^256
    
    Fusion:
      [z2 || z3 || z4] in R^768 -> Linear(768, 512) + LayerNorm(512) -> z_fused in R^512
    
    Head Routing:
      - z_fused (512-D) routes to Age LDL & Aleatoric Uncertainty heads
      - z4 (256-D) routes ONLY to Gender & Re-ID heads (macro craniofacial invariance)
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int] = (192, 384, 768),
        proj_dim: int = 128,
        fused_dim: int = 512,
        use_coord_att: bool = True,
    ):
        super().__init__()
        c2, c3, c4 = in_channels
        self.proj_dim = proj_dim
        self.fused_dim = fused_dim
        self.macro_dim = proj_dim * 2  # 256
        self.use_coord_att = use_coord_att

        self.coord_att2 = CoordinateAttention(c2) if use_coord_att else nn.Identity()
        self.proj2 = nn.Conv2d(c2, proj_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(c3, proj_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(c4, proj_dim, kernel_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 2 * 3, fused_dim),
            nn.LayerNorm(fused_dim),
        )

    def _pool_stage(self, feat: torch.Tensor, proj_layer: nn.Module) -> torch.Tensor:
        if feat.ndim == 2:
            feat = feat.unsqueeze(-1).unsqueeze(-1)
        p = proj_layer(feat)
        avg_p = torch.flatten(self.avg_pool(p), 1)
        max_p = torch.flatten(self.max_pool(p), 1)
        return torch.cat([avg_p, max_p], dim=-1)

    def forward(
        self, stage2: torch.Tensor, stage3: torch.Tensor, stage4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stage2_enhanced = self.coord_att2(stage2)
        z2 = self._pool_stage(stage2_enhanced, self.proj2)  # (B, 256)
        z3 = self._pool_stage(stage3, self.proj3)          # (B, 256)
        z4 = self._pool_stage(stage4, self.proj4)          # (B, 256)

        concat_z = torch.cat([z2, z3, z4], dim=-1)          # (B, 768)
        z_fused = self.fusion(concat_z)                      # (B, 512)
        return z_fused, z4


class ConvNeXtV2BiometricHead(nn.Module):
    """Multi-task projection head for ConvNeXt representations."""

    def __init__(self, in_features: int = 512, in_macro: int = 256, num_age_bins: int = 101, reid_dim: int = 64):
        super().__init__()
        self.in_features = in_features
        self.in_macro = in_macro
        self.age_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_age_bins)
        )
        self.gender_head = nn.Sequential(
            nn.Linear(in_macro, 128),
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
            nn.Linear(in_macro, reid_dim),
            nn.LayerNorm(reid_dim)
        )

    def forward(self, feat: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(feat, (tuple, list)):
            z_fused, z4 = feat[0], feat[1]
        else:
            z_fused = feat
            z4 = feat[:, :self.in_macro] if feat.shape[-1] >= self.in_macro else feat

        age_logits = self.age_head(z_fused)
        gender_logit = self.gender_head(z4)
        log_var = self.uncertainty_head(z_fused)
        reid_emb = F.normalize(self.reid_head(z4), p=2, dim=-1)
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
    """Lightweight multi-stage convolutional feature extractor fallback if timm is unavailable."""

    def __init__(
        self,
        out_features: int = 256,
        stage_channels: Tuple[int, int, int] = (192, 384, 768)
    ):
        super().__init__()
        self.num_features = out_features
        self.stage_channels = stage_channels
        c2, c3, c4 = stage_channels

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1),   # 80x80
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),  # 40x40
            nn.BatchNorm2d(96),
            nn.SiLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(96, c2, kernel_size=3, stride=2, padding=1),  # 20x20
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),  # 10x10
            nn.BatchNorm2d(c3),
            nn.SiLU(),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1),  # 5x5
            nn.BatchNorm2d(c4),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return s2, s3, s4


class BiometricMultiTaskModel(nn.Module):
    """
    Unified multi-task biometric model with MultiScaleBiometricFusion.
    Accepts 160x160 aligned face tensor: (B, 3, 160, 160).
    Extracts:
    1. 101-bin age probability simplex & expected continuous age (via z_fused: Stage 2+3+4).
    2. Gender logit and probability (via z4: Stage 4 macro craniofacial invariance).
    3. Aleatoric uncertainty log-variance & standard deviation (via z_fused).
    4. 64-D unit-sphere normalized Re-ID identity embedding (via z4: Stage 4).
    """

    def __init__(
        self,
        backbone_name: str = "convnextv2_tiny",
        num_age_bins: int = 101,
        reid_dim: int = 64,
        pretrained: bool = False,
        drop_path_rate: float = 0.0,
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
            actual_name = "convnextv2_tiny"

        # Initialize backbone with multi-stage feature extraction
        stage_channels: Tuple[int, int, int] = (192, 384, 768)
        if timm is not None:
            try:
                timm_kwargs: Dict[str, Any] = {"pretrained": pretrained, "features_only": True}
                if drop_path_rate > 0.0:
                    timm_kwargs["drop_path_rate"] = drop_path_rate
                self.backbone = timm.create_model(actual_name, **timm_kwargs)
                ch = self.backbone.feature_info.channels()
                if len(ch) >= 3:
                    stage_channels = (ch[-3], ch[-2], ch[-1])
                elif len(ch) == 2:
                    stage_channels = (ch[0], ch[0], ch[1])
                else:
                    stage_channels = (ch[0], ch[0], ch[0])
            except Exception:
                self.backbone = FallbackStem(stage_channels=stage_channels)
        else:
            self.backbone = FallbackStem(stage_channels=stage_channels)

        self.stage_channels = stage_channels
        self.fusion = MultiScaleBiometricFusion(
            in_channels=stage_channels,
            proj_dim=128,
            fused_dim=512
        )
        self.in_features = 512
        self.macro_features = 256

        # Head 1: 101-bin continuous Age LDL Head (receives z_fused: 512-D)
        self.age_ldl_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, num_age_bins)
        )

        # Head 2: Gender Classification Head (receives ONLY z4: 256-D macro invariance)
        self.gender_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

        # Head 3: Aleatoric Heteroscedastic Uncertainty Head: log(sigma^2) (receives z_fused: 512-D)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # Head 4: 64-D Re-ID Identity Appearance Embedding Head (receives ONLY z4: 256-D)
        self.reid_head = nn.Sequential(
            nn.Linear(256, reid_dim),
            nn.LayerNorm(reid_dim)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-scale fusion and head routing separation.
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
            - 'features': (B, 512) fused feature embedding z_fused
            - 'z4': (B, 256) macro Stage 4 feature embedding
        """
        if isinstance(self.backbone, FallbackStem):
            s2, s3, s4 = self.backbone(x)
        else:
            feats = self.backbone(x)
            if isinstance(feats, (list, tuple)):
                if len(feats) >= 3:
                    s2, s3, s4 = feats[-3], feats[-2], feats[-1]
                elif len(feats) == 2:
                    s2, s3, s4 = feats[0], feats[0], feats[1]
                else:
                    s2 = s3 = s4 = feats[0]
            else:
                s2 = s3 = s4 = feats

        # Multi-Scale Biometric Fusion
        z_fused, z4 = self.fusion(s2, s3, s4)

        # Head 1: Age LDL (z_fused: micro + meso + macro)
        age_logits = self.age_ldl_head(z_fused)
        age_probs = F.softmax(age_logits, dim=-1)

        # Head 2: Gender (z4 ONLY: macro craniofacial invariance)
        gender_logit = self.gender_head(z4).squeeze(-1)
        gender_prob = torch.sigmoid(gender_logit)

        # Head 3: Aleatoric Uncertainty (z_fused)
        log_var = self.uncertainty_head(z_fused).squeeze(-1)
        aleatoric_std = torch.exp(0.5 * torch.clamp(log_var, min=-10.0, max=10.0))

        # Head 4: Re-ID Identity Embedding (z4 ONLY: unit-sphere normalized)
        reid_raw = self.reid_head(z4)
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
            "features": z_fused,
            "z4": z4
        }
