"""
Biometric Multi-Task Deep Neural Architectures (ConvNeXt-V2 Biometric Edition)
==============================================================================
Provides a mathematically, biologically, and photometrically grounded ConvNeXt-V2
multi-task architecture engineered for 23-bracket (3-5 year) age intervals,
continuous 101-bin Label Distribution Learning (LDL), craniofacial dimorphic gender,
heteroscedastic aleatoric uncertainty, and 64-D unit-hypersphere Re-ID:

1. PhotometricRetinexStem (Lambertian-Specular Dichromatic Decomposition):
   - Multi-scale homomorphic log-reflectance extraction (Weber-Fechner / Retinex)
     canceling directional raking illumination gradients and cast shadows.
   - Differentiable luminance reliability envelope w(Y) = 4*Y*(1-Y) suppressing
     specular highlight bloom and low-light read-noise floors.
   - ConvNeXt-V2 Global Response Normalization (GRN) residual gating.
2. ConvNeXt-V2 Backbone + MultiScaleBiometricFusion with GRN & Bilateral Symmetry:
   - Stage 2 (20x20): micro-epidermal rhytids, wrinkles, pores (CoordinateAttention + Bilateral Symmetry).
   - Stage 3 (10x10): mesoscopic facial contours, landmarks (CoordinateAttention + Bilateral Symmetry).
   - Stage 4 (5x5): macro craniofacial bone structure (strictly isolated for Gender & Re-ID).
   - 1x1 Conv -> 128-ch projection + Mixed Dual-Pooling (GAP || GMP) -> 256-D per stage.
   - Concatenated [z2 || z3 || z4] (768-D) -> Linear + LayerNorm + GRN1D residual -> z_fused (512-D).
3. Hierarchical Coarse-to-Fine Head Routing Separation:
   - Head 1A (23-Bracket Fine Interval Prior): 512-D z_fused -> 23 coarse-to-fine biological stage logits.
   - Head 1B (101-Bin Continuous Age LDL): 512-D z_fused -> 101 fine logits coupled with 23-bracket prior.
   - Head 2 (Gender): 256-D z4 ONLY -> dimorphism logit (macro craniofacial bone invariance).
   - Head 3 (Aleatoric Uncertainty): 512-D z_fused -> log-variance log(sigma^2).
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

from .intervals import FINE_AGE_INTERVALS, NUM_FINE_AGE_INTERVALS


def _build_fine_interval_partition_matrix(num_age_bins: int = 101) -> torch.Tensor:
    """
    Constructs the exact binary partition matrix M in {0, 1}^{K x 23}
    mapping K integer age bins [0..K-1] to the 23 fine 3-5 year age intervals.
    """
    mat = torch.zeros((num_age_bins, NUM_FINE_AGE_INTERVALS), dtype=torch.float32)
    for iv_idx, (_, low, high) in enumerate(FINE_AGE_INTERVALS):
        if int(low) < num_age_bins:
            l_clamped = max(0, int(low))
            h_clamped = min(num_age_bins - 1, int(high))
            if h_clamped >= l_clamped:
                mat[l_clamped : h_clamped + 1, iv_idx] = 1.0
    return mat


class GlobalResponseNorm2d(nn.Module):
    """
    ConvNeXt-V2 Global Response Normalization for 2D feature maps (Woo et al., CVPR 2023).
    Enhances inter-channel feature competition and prevents channel collapse under
    extreme illumination contrast or sensor noise:
        g_c = ||X_c||_2,  nu_c = g_c / (mean_j(g_j) + eps)
        GRN(X)_c = X_c + gamma_c * (X_c * nu_c) + beta_c
    Computes spatial L2 norm in float32 to prevent FP16 overflow on 160x160 maps.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        gx = torch.sqrt(torch.mean(x_f * x_f, dim=(2, 3), keepdim=True) + self.eps)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        out = x_f + self.gamma.float() * (x_f * nx) + self.beta.float()
        return out.to(dtype=orig_dtype)


class GlobalResponseNorm1d(nn.Module):
    """
    ConvNeXt-V2 Global Response Normalization for 1D feature embeddings (B, D).
    Promotes neuron diversity across multi-scale fused biometric representations.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        gx = torch.abs(x_f)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        out = x_f + self.gamma.float() * (x_f * nx) + self.beta.float()
        return out.to(dtype=orig_dtype)


class PhotometricRetinexStem(nn.Module):
    """
    Physical Dichromatic & Multi-Scale Retinex (MSR) Illumination-Invariant Stem Adapter.

    Models Lambertian + Specular surface optics:
        I(x, y) = R_albedo(x, y) * L_illumination(x, y) + S_specular(x, y)

    1. Extracts Multi-Scale Retinex log-reflectance at fine epidermal scale (sigma=1.2)
       and mesoscopic facial-plane scale (sigma=2.5), canceling multiplicative lighting
       gradients (raking side-light, harsh shadows, exposure shifts).
    2. Applies a differentiable photometric reliability gate w(Y) = 4*Y*(1-Y) on CIE luminance Y
       to attenuate saturated specular glare (Y -> 1) and dark read-noise floors (Y -> 0).
    3. Refines intrinsic reflectance via a depthwise ConvNeXt-V2 GRN block and injects it into
       the input via a bounded LayerScale residual gate tanh(residual_gate).
    """

    def __init__(
        self,
        channels: int = 3,
        hidden_channels: int = 24,
        hidden_dim: Optional[int] = None,
        eps: float = 1e-4,
    ):
        super().__init__()
        if hidden_dim is not None:
            hidden_channels = int(hidden_dim)
        self.channels = int(channels)
        self.eps = float(eps)
        self.register_buffer("kernel_fine", self._build_gaussian_kernel(kernel_size=5, sigma=1.2))
        self.register_buffer("kernel_coarse", self._build_gaussian_kernel(kernel_size=9, sigma=2.5))
        self.register_buffer(
            "luma_weights",
            torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.dw_conv = nn.Conv2d(10, 10, kernel_size=5, padding=2, groups=10, bias=False)
        self.norm = nn.GroupNorm(num_groups=2, num_channels=10)
        self.pw_expand = nn.Conv2d(10, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm2d(hidden_channels)
        self.pw_proj = nn.Conv2d(hidden_channels, 3, kernel_size=1)
        self.residual_gate = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))

    @staticmethod
    def _build_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        g1d = torch.exp(-(coords ** 2) / (2.0 * (sigma ** 2)))
        g2d = torch.outer(g1d, g1d)
        g2d = g2d / torch.sum(g2d)
        return g2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing() and (x.ndim != 4 or x.shape[1] != 3):
            return x
        orig_dtype = x.dtype
        x_f = x.float()
        img_01 = torch.clamp((x_f + 1.0) * 0.5, min=self.eps, max=1.0)
        log_i = torch.log(img_01 + self.eps)
        pad_f = self.kernel_fine.shape[-1] // 2
        pad_c = self.kernel_coarse.shape[-1] // 2
        if torch.jit.is_tracing():
            mode_f = "reflect"
            mode_c = "reflect"
        else:
            h_in, w_in = img_01.shape[2], img_01.shape[3]
            mode_f = "reflect" if (h_in > pad_f and w_in > pad_f) else "replicate"
            mode_c = "reflect" if (h_in > pad_c and w_in > pad_c) else "replicate"
        illum_fine = F.conv2d(
            F.pad(img_01, (pad_f, pad_f, pad_f, pad_f), mode=mode_f),
            self.kernel_fine.float(),
            groups=3,
        )
        r_fine = log_i - torch.log(torch.clamp(illum_fine, min=self.eps))
        illum_coarse = F.conv2d(
            F.pad(img_01, (pad_c, pad_c, pad_c, pad_c), mode=mode_c),
            self.kernel_coarse.float(),
            groups=3,
        )
        r_coarse = log_i - torch.log(torch.clamp(illum_coarse, min=self.eps))
        luma = torch.sum(img_01 * self.luma_weights.float(), dim=1, keepdim=True)
        exposure_weight = 4.0 * luma * (1.0 - luma)
        feat_in = torch.cat([x_f, r_fine, r_coarse, exposure_weight], dim=1).to(dtype=self.dw_conv.weight.dtype)
        delta = self.pw_proj(self.grn(self.act(self.pw_expand(self.norm(self.dw_conv(feat_in)))))).float()
        out = torch.clamp(x_f + torch.tanh(self.residual_gate.float()) * delta, min=-2.5, max=2.5)
        return out.to(dtype=orig_dtype)


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
    Multi-Scale Biometric Feature Fusion with Coordinate Attention, Bilateral Craniofacial
    Symmetry Compensation, and ConvNeXt-V2 Global Response Normalization (GRN).
    
    Extracts and fuses hierarchical spatial features:
    - Stage 2: (B, C2, 20, 20) -> Coordinate Attention + Bilateral Symmetry micro-epidermal textures, wrinkles, pores
    - Stage 3: (B, C3, 10, 10) -> Coordinate Attention + Bilateral Symmetry mesoscopic facial contours and landmarks
    - Stage 4: (B, C4, 5, 5)   -> macro craniofacial bone structure
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

        self.coord_att2 = CoordinateAttention(c2, reduction=16) if use_coord_att else nn.Identity()
        self.coord_att3 = CoordinateAttention(c3, reduction=16) if use_coord_att else nn.Identity()

        # Learnable bilateral sagittal symmetry gates for Stage 2 & Stage 3
        self.sym_gate2 = nn.Parameter(torch.zeros(1, c2, 1, 1))
        self.sym_gate3 = nn.Parameter(torch.zeros(1, c3, 1, 1))

        self.proj2 = nn.Conv2d(c2, proj_dim, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(c3, proj_dim, kernel_size=1, bias=False)
        self.proj4 = nn.Conv2d(c4, proj_dim, kernel_size=1, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 2 * 3, fused_dim),
            nn.LayerNorm(fused_dim),
        )
        # ConvNeXt-V2 Global Response Normalization (GRN) residual block on fused representation
        self.fusion_grn_block = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            GlobalResponseNorm1d(fused_dim),
            nn.Linear(fused_dim, fused_dim),
        )
        self.fusion_norm = nn.LayerNorm(fused_dim)

    @staticmethod
    def _apply_bilateral_symmetry(feat: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing():
            if feat.ndim != 4 or feat.shape[3] <= 1:
                return feat
        feat_mirror = torch.flip(feat, dims=[3])
        diff = 0.5 * (feat_mirror - feat)
        scale = torch.abs(feat) + 0.25
        cauchy_w = 1.0 / (1.0 + (diff / scale) ** 2)
        return feat + torch.tanh(gate.to(dtype=feat.dtype)) * (diff * cauchy_w)

    def _pool_stage(self, feat: torch.Tensor, proj_layer: nn.Module) -> torch.Tensor:
        if not torch.jit.is_tracing() and feat.ndim == 2:
            feat = feat.unsqueeze(-1).unsqueeze(-1)
        p = proj_layer(feat.contiguous())
        avg_p = torch.flatten(self.avg_pool(p), 1)
        max_p = torch.flatten(self.max_pool(p), 1)
        return torch.cat([avg_p, max_p], dim=-1)

    def forward(
        self, stage2: torch.Tensor, stage3: torch.Tensor, stage4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stage2_enhanced = self._apply_bilateral_symmetry(self.coord_att2(stage2), self.sym_gate2)
        stage3_enhanced = self._apply_bilateral_symmetry(self.coord_att3(stage3), self.sym_gate3)

        z2 = self._pool_stage(stage2_enhanced, self.proj2)  # (B, 256)
        z3 = self._pool_stage(stage3_enhanced, self.proj3)  # (B, 256)
        z4 = self._pool_stage(stage4, self.proj4)           # (B, 256) - Strictly isolated Stage 4

        concat_z = torch.cat([z2, z3, z4], dim=-1)          # (B, 768)
        z_base = self.fusion(concat_z)                      # (B, 512)
        z_fused = self.fusion_norm(z_base + 0.25 * self.fusion_grn_block(z_base))
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
    """Lightweight multi-stage ConvNeXt-V2-style feature extractor fallback if timm is unavailable."""

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
            GlobalResponseNorm2d(c2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),  # 10x10
            nn.BatchNorm2d(c3),
            nn.SiLU(),
            GlobalResponseNorm2d(c3),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1),  # 5x5
            nn.BatchNorm2d(c4),
            nn.SiLU(),
            GlobalResponseNorm2d(c4),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return s2, s3, s4


class BiometricMultiTaskModel(nn.Module):
    """
    Unified ConvNeXt-V2 Multi-Task Biometric Model with:
    - PhotometricRetinexStem (Lambertian-Specular illumination invariance)
    - MultiScaleBiometricFusion (CoordinateAttention + Bilateral Symmetry + ConvNeXt-V2 GRN)
    - Hierarchical Coarse-to-Fine 23-Interval + 101-Bin Continuous Age LDL Head
    - Macro-Isolated Craniofacial Dimorphic Gender & 64-D Unit-Hypersphere Re-ID Heads

    Accepts 160x160 aligned face tensor: (B, 3, 160, 160).
    Extracts:
    1. 23-bracket (3-5 yr) age interval probabilities & 101-bin age probability simplex
       & expected continuous age (via z_fused: Stage 2+3+4).
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
        use_photometric_stem: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_age_bins = num_age_bins
        self.num_age_intervals = NUM_FINE_AGE_INTERVALS
        self.reid_dim = reid_dim
        self.use_photometric_stem = use_photometric_stem

        # Photometric Dichromatic Retinex Stem for extreme lighting / glare / shadow invariance
        self.photometric_stem = PhotometricRetinexStem(channels=3) if use_photometric_stem else nn.Identity()

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

        # Register partition matrix (K, 23) and bin vector (K,) as buffers for zero-overhead GPU execution
        self.register_buffer(
            "age_bins_vec",
            torch.arange(num_age_bins, dtype=torch.float32),
        )
        partition_mat = _build_fine_interval_partition_matrix(num_age_bins)
        self.register_buffer("fine_interval_partition", partition_mat)

        # Head 1A: Coarse-to-Fine 23-Bracket Biological Age Interval Head (receives z_fused: 512-D)
        self.age_interval_head = nn.Sequential(
            nn.Linear(512, 192),
            nn.SiLU(),
            GlobalResponseNorm1d(192),
            nn.Linear(192, NUM_FINE_AGE_INTERVALS),
        )
        self.hier_interval_scale = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

        # Head 1B: 101-bin continuous Age LDL Head (receives z_fused: 512-D)
        self.age_ldl_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            GlobalResponseNorm1d(256),
            nn.Linear(256, num_age_bins)
        )

        # Head 2: Gender Classification Head (receives ONLY z4: 256-D macro invariance)
        self.gender_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            GlobalResponseNorm1d(128),
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

    def forward(
        self,
        x: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with Photometric Retinex Stem, Multi-Scale Biometric Fusion,
        Hierarchical 23-Interval + 101-Bin Age Coupling, and Head Routing Separation.
        """
        x_photo = self.photometric_stem(x)

        if isinstance(self.backbone, FallbackStem):
            s2, s3, s4 = self.backbone(x_photo)
        else:
            feats = self.backbone(x_photo)
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

        # Head 1: Hierarchical 23-Interval + 101-Bin Age LDL (z_fused: micro + meso + macro)
        fine_age_logits = self.age_ldl_head(z_fused)
        interval_logits = self.age_interval_head(z_fused)
        if self.num_age_bins == 101:
            fine_interval_log_prior = F.log_softmax(interval_logits, dim=-1)
            part_mat = self.fine_interval_partition.to(dtype=fine_interval_log_prior.dtype)
            lifted_interval_prior = torch.matmul(fine_interval_log_prior, part_mat.t())
            scale = F.softplus(self.hier_interval_scale.to(dtype=fine_age_logits.dtype))
            age_logits = fine_age_logits + scale * lifted_interval_prior
        else:
            age_logits = fine_age_logits
        age_probs = F.softmax(age_logits, dim=-1)

        # Head 2: Gender (z4 ONLY: macro craniofacial invariance)
        gender_logit = self.gender_head(z4).squeeze(-1)
        gender_prob = torch.sigmoid(gender_logit)

        # Head 3: Aleatoric Uncertainty (z_fused)
        log_var = torch.clamp(self.uncertainty_head(z_fused).squeeze(-1), min=-6.0, max=4.0)
        aleatoric_std = torch.exp(0.5 * log_var)

        # Head 4: Re-ID Identity Embedding (z4 ONLY: unit-sphere normalized)
        reid_raw = self.reid_head(z4)
        reid_emb = F.normalize(reid_raw, p=2, dim=-1)

        # Continuous age expectation
        age_bins_cast = self.age_bins_vec.to(dtype=age_probs.dtype)
        expected_age = torch.sum(age_probs * age_bins_cast, dim=-1)

        # Export compatibility: return tuple for TorchScript and ONNX
        if torch.jit.is_tracing() or return_dict is False:
            return age_logits, gender_logit, log_var, reid_emb

        part_mat_probs = self.fine_interval_partition.to(dtype=age_probs.dtype)
        gauss_focus = torch.exp(
            -0.5 * ((age_bins_cast.unsqueeze(0) - expected_age.unsqueeze(-1)) / 1.6) ** 2
        )
        focused_probs = age_probs * (0.20 + 0.80 * gauss_focus)
        age_interval_probs = torch.matmul(focused_probs, part_mat_probs)
        if self.num_age_bins == 101:
            iv_head_probs = F.softmax(interval_logits, dim=-1)
            focus_iv = torch.matmul(gauss_focus, part_mat_probs)
            age_interval_probs = 0.85 * age_interval_probs + 0.15 * (iv_head_probs * focus_iv)
        age_interval_probs = age_interval_probs / torch.clamp(
            age_interval_probs.sum(dim=-1, keepdim=True), min=1e-12
        )
        age_interval_idx = torch.argmax(age_interval_probs, dim=-1)

        return {
            "age_logits": age_logits,
            "age_probs": age_probs,
            "age_ldl": age_probs,
            "fine_interval_logits": interval_logits,
            "age_interval_logits": interval_logits,
            "age_interval_probs": age_interval_probs,
            "age_interval_idx": age_interval_idx,
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

    @torch.no_grad()
    def predict_smooth(
        self,
        x: torch.Tensor,
        enable_flip_tta: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Smooth, illumination-and-pose-compensated inference via single-batch Bilateral Flip
        Test-Time Ensembling (Flip-TTA). Executes [x, flip(x)] in one fused GPU/CPU batch
        to cancel directional lighting shadows and out-of-plane yaw bias.
        """
        if not enable_flip_tta:
            out_single = self.forward(x, return_dict=True)
            assert isinstance(out_single, dict)
            out_single["epistemic_std"] = torch.zeros_like(out_single["aleatoric_std"])
            out_single["total_std"] = out_single["aleatoric_std"]
            return out_single

        B = x.shape[0]
        x_flip = torch.flip(x, dims=[3])
        x_cat = torch.cat([x, x_flip], dim=0)
        out_cat = self.forward(x_cat, return_dict=True)
        assert isinstance(out_cat, dict)

        age_probs = 0.5 * (out_cat["age_probs"][:B] + out_cat["age_probs"][B:])
        age_probs = age_probs / torch.clamp(age_probs.sum(dim=-1, keepdim=True), min=1e-12)
        age_logits = torch.log(torch.clamp(age_probs, min=1e-8))
        age_bins_cast = self.age_bins_vec.to(dtype=age_probs.dtype)
        expected_age = torch.sum(age_probs * age_bins_cast, dim=-1)

        fine_iv_logits = 0.5 * (out_cat["age_interval_logits"][:B] + out_cat["age_interval_logits"][B:])
        part_mat_probs = self.fine_interval_partition.to(dtype=age_probs.dtype)
        gauss_focus = torch.exp(
            -0.5 * ((age_bins_cast.unsqueeze(0) - expected_age.unsqueeze(-1)) / 1.6) ** 2
        )
        focused_probs = age_probs * (0.20 + 0.80 * gauss_focus)
        age_interval_probs = torch.matmul(focused_probs, part_mat_probs)
        if self.num_age_bins == 101:
            iv_head_probs = F.softmax(fine_iv_logits, dim=-1)
            focus_iv = torch.matmul(gauss_focus, part_mat_probs)
            age_interval_probs = 0.85 * age_interval_probs + 0.15 * (iv_head_probs * focus_iv)
        age_interval_probs = age_interval_probs / torch.clamp(
            age_interval_probs.sum(dim=-1, keepdim=True), min=1e-12
        )
        age_interval_idx = torch.argmax(age_interval_probs, dim=-1)

        gender_logit = 0.5 * (out_cat["gender_logit"][:B] + out_cat["gender_logit"][B:])
        gender_prob = torch.sigmoid(gender_logit)

        # Combine aleatoric variance with bilateral epistemic disagreement + hemiface asymmetry variance
        var_orig = out_cat["aleatoric_std"][:B] ** 2
        var_flip = out_cat["aleatoric_std"][B:] ** 2
        aleatoric_std = 0.5 * (out_cat["aleatoric_std"][:B] + out_cat["aleatoric_std"][B:])
        epistemic_age_diff = 0.5 * torch.abs(out_cat["expected_age"][:B] - out_cat["expected_age"][B:])
        w_half = x.shape[3] // 2
        if w_half > 0:
            left_half = x[:, :, :, :w_half]
            right_half = torch.flip(x[:, :, :, -w_half:], dims=[3])
            hemiface_var_gap = torch.abs(
                torch.var(left_half, dim=(1, 2, 3), unbiased=False)
                - torch.var(right_half, dim=(1, 2, 3), unbiased=False)
            )
            bilateral_penalty = 0.50 * hemiface_var_gap
        else:
            bilateral_penalty = torch.zeros_like(aleatoric_std)
        epistemic_std = torch.sqrt(epistemic_age_diff ** 2 + bilateral_penalty ** 2)
        gender_disagreement = 0.05 * ((out_cat["gender_logit"][:B] - out_cat["gender_logit"][B:]) ** 2)
        disagreement_var = (epistemic_std ** 2) + gender_disagreement
        total_var = 0.5 * (var_orig + var_flip) + disagreement_var
        total_std = torch.sqrt(torch.clamp(total_var, min=1e-6))
        log_var = torch.log(torch.clamp(total_var, min=1e-6))

        reid_emb = F.normalize(
            0.5 * (out_cat["reid_embedding"][:B] + out_cat["reid_embedding"][B:]),
            p=2,
            dim=-1,
        )

        return {
            "age_logits": age_logits,
            "age_probs": age_probs,
            "age_ldl": age_probs,
            "fine_interval_logits": fine_iv_logits,
            "age_interval_logits": fine_iv_logits,
            "age_interval_probs": age_interval_probs,
            "age_interval_idx": age_interval_idx,
            "expected_age": expected_age,
            "age_expected": expected_age,
            "gender_logit": gender_logit,
            "gender_prob": gender_prob,
            "log_variance": log_var,
            "aleatoric_std": aleatoric_std,
            "epistemic_std": epistemic_std,
            "total_std": total_std,
            "reid_embedding": reid_emb,
            "features": 0.5 * (out_cat["features"][:B] + out_cat["features"][B:]),
            "z4": 0.5 * (out_cat["z4"][:B] + out_cat["z4"][B:]),
        }



# Alias for explicit ConvNeXt-V2 Biometric naming
ConvNeXtV2BiometricModel = BiometricMultiTaskModel


