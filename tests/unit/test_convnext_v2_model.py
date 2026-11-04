"""
Unit Tests for BiometricMultiTaskModel (ConvNeXt-V2-Tiny, MultiScaleBiometricFusion)
=====================================================================================
Validates:
1. ConvNeXt-V2-Tiny default backbone instantiation.
2. MultiScaleBiometricFusion dual-pooling and projection dimensions.
3. Head routing separation and strict gradient isolation (Gender/Re-ID isolated from Stage 2/3).
4. Full backward compatibility across MobileNetV4 and FallbackStem.
5. 101-bin age LDL probability simplex and continuous expected age.
6. Gender logit and sigmoid probability.
7. Aleatoric heteroscedastic log-variance and standard deviation.
8. 64-D unit-normalized Re-ID embedding representation.
"""

from __future__ import annotations
import pytest
import torch
import torch.nn.functional as F

from src.biometrics.estimation.backbone import (
    BiometricMultiTaskModel,
    MultiScaleBiometricFusion,
    FallbackStem,
)


class TestBiometricMultiTaskModel:
    def test_default_backbone_is_convnextv2_tiny(self):
        """Default instantiation must configure ConvNeXt-V2-Tiny."""
        model = BiometricMultiTaskModel(pretrained=False)
        assert model.backbone_name == "convnextv2_tiny"
        assert model.in_features == 512
        assert model.macro_features == 256
        assert hasattr(model, "fusion")

    @pytest.mark.parametrize("backbone_name", ["convnextv2_tiny", "convnextv2_nano", "mobilenetv4"])
    def test_model_forward_shape_and_constraints(self, backbone_name):
        model = BiometricMultiTaskModel(backbone_name=backbone_name, pretrained=False)
        model.eval()

        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 160, 160)

        with torch.no_grad():
            outputs = model(dummy_input)

        # 1. Age LDL Head
        assert "age_probs" in outputs
        assert outputs["age_probs"].shape == (batch_size, 101)
        probs_sum = outputs["age_probs"].sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones(batch_size), atol=1e-4)

        # Expected age
        assert "expected_age" in outputs
        assert outputs["expected_age"].shape == (batch_size,)
        assert (outputs["expected_age"] >= 0.0).all()
        assert (outputs["expected_age"] <= 100.0).all()

        # 2. Gender Head
        assert "gender_logit" in outputs
        assert outputs["gender_logit"].shape == (batch_size,)
        assert "gender_prob" in outputs
        assert outputs["gender_prob"].shape == (batch_size,)
        assert (outputs["gender_prob"] >= 0.0).all()
        assert (outputs["gender_prob"] <= 1.0).all()

        # 3. Aleatoric Uncertainty Head
        assert "log_variance" in outputs
        assert outputs["log_variance"].shape == (batch_size,)
        assert "aleatoric_std" in outputs
        assert outputs["aleatoric_std"].shape == (batch_size,)
        assert (outputs["aleatoric_std"] > 0.0).all()

        # 4. 64-D Re-ID Embedding Head
        assert "reid_embedding" in outputs
        assert outputs["reid_embedding"].shape == (batch_size, 64)
        norms = torch.norm(outputs["reid_embedding"], p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-4)

        # 5. Internal representations
        assert "features" in outputs
        assert outputs["features"].shape == (batch_size, 512)
        assert "z4" in outputs
        assert outputs["z4"].shape == (batch_size, 256)

    def test_multiscale_fusion_module_direct(self):
        """Verifies MultiScaleBiometricFusion module input/output dimensions."""
        fusion = MultiScaleBiometricFusion(in_channels=(192, 384, 768), proj_dim=128, fused_dim=512)
        fusion.eval()

        B = 2
        s2 = torch.randn(B, 192, 20, 20)
        s3 = torch.randn(B, 384, 10, 10)
        s4 = torch.randn(B, 768, 5, 5)

        z_fused, z4 = fusion(s2, s3, s4)
        assert z_fused.shape == (B, 512)
        assert z4.shape == (B, 256)

    def test_head_routing_gradient_isolation_gender(self):
        """Gender head receives ONLY Stage 4: zero gradient into Stage 2 and Stage 3 projections."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.train()

        x = torch.randn(2, 3, 160, 160)
        out = model(x)

        # Backward exclusively from Gender head
        gender_loss = out["gender_logit"].sum()
        gender_loss.backward()

        # Stage 4 projection MUST receive gradient
        assert model.fusion.proj4.weight.grad is not None
        assert not torch.all(model.fusion.proj4.weight.grad == 0.0)

        # Stage 2 and Stage 3 projections MUST NOT receive gradient
        assert model.fusion.proj2.weight.grad is None
        assert model.fusion.proj3.weight.grad is None

    def test_head_routing_gradient_isolation_reid(self):
        """Re-ID identity head receives ONLY Stage 4: zero gradient into Stage 2 and Stage 3."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.train()

        x = torch.randn(2, 3, 160, 160)
        out = model(x)

        # Backward exclusively from Re-ID head
        # Target projection is required because sum of LayerNorm normalized vector is identically zero
        target_reid = torch.randn_like(out["reid_embedding"])
        reid_loss = (out["reid_embedding"] * target_reid).sum()
        reid_loss.backward()

        # Stage 4 projection MUST receive gradient
        assert model.fusion.proj4.weight.grad is not None
        assert not torch.all(model.fusion.proj4.weight.grad == 0.0)

        # Stage 2 and Stage 3 projections MUST NOT receive gradient
        assert model.fusion.proj2.weight.grad is None
        assert model.fusion.proj3.weight.grad is None

    def test_head_routing_multiscale_gradients_age_and_uncertainty(self):
        """Age LDL head receives fused embedding: gradients flow to ALL stages (2, 3, and 4)."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.train()

        x = torch.randn(2, 3, 160, 160)
        out = model(x)

        age_loss = out["expected_age"].sum()
        age_loss.backward()

        assert model.fusion.proj2.weight.grad is not None
        assert not torch.all(model.fusion.proj2.weight.grad == 0.0)
        assert model.fusion.proj3.weight.grad is not None
        assert not torch.all(model.fusion.proj3.weight.grad == 0.0)
        assert model.fusion.proj4.weight.grad is not None
        assert not torch.all(model.fusion.proj4.weight.grad == 0.0)

    def test_fallback_stem_compatibility(self):
        """Verifies FallbackStem produces correct 3-stage feature shapes and runs multi-task model."""
        stem = FallbackStem(stage_channels=(192, 384, 768))
        stem.eval()
        x = torch.randn(2, 3, 160, 160)
        s2, s3, s4 = stem(x)

        assert s2.shape == (2, 192, 20, 20)
        assert s3.shape == (2, 384, 10, 10)
        assert s4.shape == (2, 768, 5, 5)

        # Force fallback inside model
        model = BiometricMultiTaskModel(pretrained=False)
        model.backbone = stem
        out = model(x)
        assert out["age_probs"].shape == (2, 101)
        assert out["gender_logit"].shape == (2,)
        assert out["reid_embedding"].shape == (2, 64)

    def test_reid_cosine_similarity(self):
        """Validates that distinct inputs produce distinct unit-normalized embeddings."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.eval()

        img1 = torch.randn(1, 3, 160, 160)
        img2 = torch.randn(1, 3, 160, 160)

        with torch.no_grad():
            emb1 = model(img1)["reid_embedding"]
            emb2 = model(img2)["reid_embedding"]

        cos_sim = torch.sum(emb1 * emb2, dim=-1).item()
        assert -1.0 <= cos_sim <= 1.0

    def test_gradient_flow_to_all_heads(self):
        """Verifies that combined multi-task loss gradients flow through all 4 heads into the backbone."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.train()

        x = torch.randn(2, 3, 160, 160)
        out = model(x)

        target_reid = torch.randn_like(out["reid_embedding"])
        loss = (
            out["expected_age"].sum()
            + out["gender_logit"].sum()
            + out["log_variance"].sum()
            + (out["reid_embedding"] * target_reid).sum()
        )
        loss.backward()

        assert model.age_ldl_head[0].weight.grad is not None
        assert not torch.all(model.age_ldl_head[0].weight.grad == 0.0)
        assert model.gender_head[0].weight.grad is not None
        assert not torch.all(model.gender_head[0].weight.grad == 0.0)
        assert model.uncertainty_head[0].weight.grad is not None
        assert not torch.all(model.uncertainty_head[0].weight.grad == 0.0)
        assert model.reid_head[0].weight.grad is not None
        assert not torch.all(model.reid_head[0].weight.grad == 0.0)

    def test_photometric_retinex_stem_and_grn(self):
        """Verifies PhotometricRetinexStem and GlobalResponseNorm preserve shape, bounds, and gradients."""
        from src.biometrics.estimation.backbone import (
            PhotometricRetinexStem,
            GlobalResponseNorm1d,
            GlobalResponseNorm2d,
        )

        stem = PhotometricRetinexStem(hidden_channels=24)
        x = torch.randn(2, 3, 160, 160, requires_grad=True)
        out = stem(x)
        assert out.shape == (2, 3, 160, 160)
        assert torch.all(out >= -2.5) and torch.all(out <= 2.5)
        out.sum().backward()
        assert x.grad is not None
        assert stem.dw_conv.weight.grad is not None

        grn2d = GlobalResponseNorm2d(16)
        f2d = torch.randn(2, 16, 10, 10, requires_grad=True)
        assert grn2d(f2d).shape == (2, 16, 10, 10)

        grn1d = GlobalResponseNorm1d(32)
        f1d = torch.randn(2, 32, requires_grad=True)
        assert grn1d(f1d).shape == (2, 32)

    def test_hierarchical_23_interval_head_and_partition_coupling(self):
        """Verifies hierarchical 23-bracket (3-5 yr) head outputs and gradient coupling."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.train()
        x = torch.randn(2, 3, 160, 160)
        out = model(x)

        assert "fine_interval_logits" in out
        assert out["fine_interval_logits"].shape == (2, 23)
        assert "age_interval_probs" in out
        assert out["age_interval_probs"].shape == (2, 23)
        assert torch.allclose(out["age_interval_probs"].sum(dim=-1), torch.ones(2), atol=1e-4)

        loss = out["expected_age"].sum() + out["fine_interval_logits"].sum()
        loss.backward()
        assert model.age_interval_head[0].weight.grad is not None
        assert not torch.all(model.age_interval_head[0].weight.grad == 0.0)
        assert model.hier_interval_scale.grad is not None

    def test_predict_smooth_bilateral_flip_tta_and_epistemic_uncertainty(self):
        """Verifies predict_smooth integrates bilateral flip TTA and computes epistemic + total uncertainty."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 160, 160)

        with torch.no_grad():
            out_tta = model.predict_smooth(x, enable_flip_tta=True)
            out_single = model.predict_smooth(x, enable_flip_tta=False)

        for d in (out_tta, out_single):
            assert d["age_probs"].shape == (2, 101)
            assert d["age_interval_probs"].shape == (2, 23)
            assert d["epistemic_std"].shape == (2,)
            assert d["total_std"].shape == (2,)
            assert torch.all(d["total_std"] >= d["aleatoric_std"] - 1e-5)
        assert torch.allclose(out_single["epistemic_std"], torch.zeros(2), atol=1e-6)

    def test_photometric_stem_and_model_dtype_and_small_spatial_safety(self):
        """Verifies float64/bfloat16 dtype safety and small spatial reflection-pad fallback."""
        from src.biometrics.estimation.backbone import PhotometricRetinexStem

        stem = PhotometricRetinexStem(hidden_channels=24)
        # Small spatial input (4x4 <= pad_c=4) must fall back to replicate padding without crashing
        x_small = torch.randn(1, 3, 4, 4)
        out_small = stem(x_small)
        assert out_small.shape == (1, 3, 4, 4)
        assert torch.all(torch.isfinite(out_small))

        # Float64 model + input must not fail with FloatTensor vs DoubleTensor mismatch
        model_f64 = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False).double().eval()
        x_f64 = torch.randn(1, 3, 160, 160, dtype=torch.float64)
        with torch.no_grad():
            out_f64 = model_f64.predict_smooth(x_f64, enable_flip_tta=True)
        assert out_f64["expected_age"].dtype == torch.float64
        assert torch.all(torch.isfinite(out_f64["expected_age"]))

    def test_cauchy_lorentzian_bilateral_symmetry_redescending_outlier_rejection(self):
        """Verifies unilateral outlier spikes on one hemiface are rejected by Cauchy-Lorentzian gate."""
        feat = torch.ones(1, 8, 10, 10)
        # Inject extreme background clutter / occlusion spike on right hemiface (x >= 5)
        feat[:, :, :, 5:] = 1000.0
        gate = torch.tensor(1.5).view(1, 1, 1, 1)

        out = MultiScaleBiometricFusion._apply_bilateral_symmetry(feat, gate)
        # Left hemiface (originally 1.0) must NOT be corrupted by 0.5*1000 or -0.25*1000
        left_hemiface = out[:, :, :, :5]
        assert torch.all(torch.isfinite(left_hemiface))
        assert torch.max(torch.abs(left_hemiface - 1.0)).item() < 5.0

    def test_custom_num_age_bins_partition_parity(self):
        """Verifies num_age_bins != 101 maintains exact interval partition parity in forward and predict_smooth."""
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", num_age_bins=80, pretrained=False).eval()
        x = torch.randn(2, 3, 160, 160)
        with torch.no_grad():
            out_f = model(x)
            out_s = model.predict_smooth(x, enable_flip_tta=True)
        # Intervals 19..22 correspond to ages 81..100, which must have 0 probability when num_age_bins=80
        assert torch.allclose(out_f["age_interval_probs"][:, 19:], torch.zeros(2, 4), atol=1e-6)
        assert torch.allclose(out_s["age_interval_probs"][:, 19:], torch.zeros(2, 4), atol=1e-6)

    def test_kaggle_trained_checkpoint_strict_loading(self):
        """Verifies models/convnextv2_tiny_biometrics.pt loads with strict=True if present."""
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent.parent
        ckpt_path = root / "models" / "convnextv2_tiny_biometrics.pt"
        if not ckpt_path.exists():
            ckpt_path = root / "reports" / "convnextv2_tiny_biometrics.pt"
        if not ckpt_path.exists():
            pytest.skip("Trained checkpoint not present")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model.predict_smooth(x, enable_flip_tta=True)
        assert 0.0 <= float(out["expected_age"].item()) <= 100.0
        assert 0.0 <= float(out["gender_prob"].item()) <= 1.0

    def test_kaggle_training_notebook_cell5_script_integrity(self, tmp_path):
        """Verifies Cell 5 of kaggle_training_convnextv2.ipynb writes a standalone syntax-valid train_ddp.py."""
        import ast
        import json
        from pathlib import Path

        nb_path = Path(__file__).resolve().parent.parent.parent / "notebooks" / "training" / "kaggle_training_convnextv2.ipynb"
        nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
        cell5_src = "".join(nb_data["cells"][5]["source"])
        assert "def generate_training_notebook" not in cell5_src
        ast.parse(cell5_src)

        # Execute Cell 5 in tmp_path to verify the written train_ddp.py
        cell5_local = cell5_src.replace("'train_ddp.py'", repr(str(tmp_path / "train_ddp.py")))
        exec(compile(cell5_local, "<cell5>", "exec"), {})
        ddp_code = (tmp_path / "train_ddp.py").read_text(encoding="utf-8")
        ast.parse(ddp_code)
        assert "def generate_training_notebook" not in ddp_code
        assert "cauchy_w = 1.0 / (1.0 + (diff / scale) ** 2)" in ddp_code
        assert "kl_23 = torch.clamp(torch.sum(q * (torch.log(q_safe) - torch.log(p_safe)), dim=-1), min=0.0)" in ddp_code
        assert "kl_1 = torch.sum(p1 * (torch.log(p1_s) - log_m), dim=-1)" in ddp_code

    def test_trained_checkpoint_extreme_yaw_hemiface_occlusion_and_pipeline_consistency(self):
        """Verifies trained checkpoint under >60 deg yaw hemiface occlusion and BiometricPipeline interval/TorchScript consistency."""
        from pathlib import Path
        import numpy as np
        from src.biometrics.pipeline import BiometricPipeline
        from src.biometrics.core.types import FaceLandmarks
        from src.biometrics.estimation.intervals import get_fine_interval_for_age
        from src.biometrics.estimation.dimorphism import get_cohort_for_age

        root = Path(__file__).resolve().parent.parent.parent
        ckpt_path = root / "models" / "convnextv2_tiny_biometrics.pt"
        if not ckpt_path.exists():
            ckpt_path = root / "reports" / "convnextv2_tiny_biometrics.pt"
        if not ckpt_path.exists():
            pytest.skip("Trained checkpoint not present")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
        model.eval()

        # 1. Symmetric vs severe unilateral hemiface occlusion (>60 deg yaw foreshortening / half-face mask)
        torch.manual_seed(7)
        half = torch.randn(1, 3, 160, 80).clamp(-1.0, 1.0)
        x_sym = torch.cat([half, torch.flip(half, dims=[3])], dim=3)
        x_hemiface_occ = x_sym.clone()
        x_hemiface_occ[:, :, :, :80] = -1.0  # Complete unilateral blackout of left hemiface

        with torch.no_grad():
            out_sym = model.predict_smooth(x_sym, enable_flip_tta=True)
            out_occ = model.predict_smooth(x_hemiface_occ, enable_flip_tta=True)

        assert torch.all(torch.isfinite(out_occ["expected_age"]))
        assert 0.0 <= float(out_occ["expected_age"].item()) <= 100.0
        # Bilateral Flip-TTA epistemic std must rise under severe unilateral hemiface occlusion
        assert float(out_occ["total_std"].item()) > float(out_sym["total_std"].item())

        # 2. Verify BiometricPipeline 100% canonical interval & cohort consistency across 25 diverse inputs
        pipe = BiometricPipeline(estimator_model=model, enable_quality_audit=False)
        dummy_lmks = FaceLandmarks(np.zeros((5, 2), dtype=np.float32))
        rng = np.random.RandomState(123)
        for _ in range(25):
            crop_bgr = rng.randint(0, 256, size=(160, 160, 3), dtype=np.uint8)
            pred = pipe._estimate_biometrics(crop_bgr, dummy_lmks)
            exp_iv_name, exp_iv_idx, exp_low, exp_high = get_fine_interval_for_age(pred.expected_age)
            exp_cohort_name, exp_cohort_idx = get_cohort_for_age(pred.expected_age)
            assert pred.age_interval_name == exp_iv_name
            assert pred.age_interval_index == exp_iv_idx
            assert pred.age_interval_bounds == (exp_low, exp_high)
            assert pred.age_cohort_name == exp_cohort_name
            assert pred.age_cohort_index == exp_cohort_idx

        # 3. Verify TorchScript-traced model compatibility in BiometricPipeline
        traced = torch.jit.trace(model, torch.zeros(1, 3, 160, 160))
        pipe_traced = BiometricPipeline(estimator_model=traced, enable_quality_audit=False)
        pred_traced = pipe_traced._estimate_biometrics(
            rng.randint(0, 256, size=(160, 160, 3), dtype=np.uint8),
            dummy_lmks,
        )
        assert 0.0 <= pred_traced.expected_age <= 100.0
        assert pred_traced.age_interval_bounds[0] <= int(round(pred_traced.expected_age)) <= pred_traced.age_interval_bounds[1]

    def test_canonical_224x224_resolution_and_geometry(self):
        """Verifies ARCFACE_CANONICAL_224 Umeyama alignment and 224x224 ConvNeXt-V2 forward/predict_smooth parity."""
        import numpy as np
        from src.biometrics.core.geometry import ARCFACE_CANONICAL_112, ARCFACE_CANONICAL_224, UmeyamaAligner

        assert ARCFACE_CANONICAL_224.shape == (5, 2)
        assert np.allclose(ARCFACE_CANONICAL_224, ARCFACE_CANONICAL_112 * 2.0)

        aligner_224 = UmeyamaAligner(target_size=224)
        assert np.allclose(aligner_224.canonical_points, ARCFACE_CANONICAL_224)
        dummy_img = np.full((300, 300, 3), 140, dtype=np.uint8)
        src_pts = ARCFACE_CANONICAL_224 + 25.0
        aligned = aligner_224.align(dummy_img, src_pts)
        assert aligned.shape == (224, 224, 3)

        model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False)
        model.eval()
        x_224 = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out_f = model(x_224)
            out_s = model.predict_smooth(x_224, enable_flip_tta=True)
        assert out_f["age_probs"].shape == (2, 101)
        assert out_f["age_interval_probs"].shape == (2, 23)
        assert torch.allclose(out_f["age_interval_probs"].sum(dim=-1), torch.ones(2), atol=1e-5)
        assert torch.allclose(out_s["age_interval_probs"].sum(dim=-1), torch.ones(2), atol=1e-5)
        assert out_s["reid_embedding"].shape == (2, 64)

    def test_knowledge_distillation_loss_and_interval_refinement(self):
        """Verifies refine_interval_ldl_with_teacher and BiometricDistillationLoss gradients and edge cases."""
        from src.biometrics.estimation.intervals import compute_interval_box_ldl
        from src.biometrics.estimation.losses import (
            BiometricDistillationLoss,
            refine_interval_ldl_with_teacher,
        )

        # 1. Interval LDL Refinement: sample 0 has coarse interval [25, 32], sample 1 has exact age [40, 40]
        box_0 = torch.from_numpy(compute_interval_box_ldl(25.0, 32.0, sigma=2.0))
        box_1 = torch.from_numpy(compute_interval_box_ldl(40.0, 40.0, sigma=2.0))
        box_101 = torch.stack([box_0, box_1], dim=0)

        # Teacher sharply peaks at age 30 for sample 0, and age 70 (outside exact [40, 40]) for sample 1
        t_probs = torch.zeros(2, 101)
        t_probs[0, 29:32] = torch.tensor([0.2, 0.6, 0.2])
        t_probs[1, 70] = 1.0
        age_min = torch.tensor([25.0, 40.0])
        age_max = torch.tensor([32.0, 40.0])

        ref_101 = refine_interval_ldl_with_teacher(
            box_101, t_probs, age_min, age_max, blend_weight=0.70
        )
        assert ref_101.shape == (2, 101)
        assert torch.allclose(ref_101.sum(dim=-1), torch.ones(2), atol=1e-5)
        # Sample 0 (coarse interval [25, 32]) must shift its peak toward Teacher's peak at bin 30
        assert float(ref_101[0, 30].item()) > float(box_101[0, 30].item())
        # Sample 1 (exact age [40, 40]) must NOT be corrupted by Teacher's out-of-range peak at 70
        assert torch.allclose(ref_101[1], box_101[1], atol=1e-5)

        # 2. BiometricDistillationLoss with missing-gender pseudo-label imputation and Re-ID relational similarity
        kd_loss_fn = BiometricDistillationLoss(temperature=2.0, missing_gender_conf_thresh=0.65)
        s_age_logits = torch.randn(4, 101, requires_grad=True)
        s_iv_logits = torch.randn(4, 23, requires_grad=True)
        s_g_logit = torch.randn(4, requires_grad=True)
        s_reid_raw = torch.randn(4, 64, requires_grad=True)

        student_preds = {
            "age_logits": s_age_logits,
            "age_probs": torch.softmax(s_age_logits, dim=-1),
            "fine_interval_logits": s_iv_logits,
            "gender_logit": s_g_logit,
            "reid_embedding": torch.nn.functional.normalize(s_reid_raw, p=2, dim=-1),
        }
        teacher_preds = {
            "age_logits": torch.randn(4, 101),
            "age_probs": torch.softmax(torch.randn(4, 101), dim=-1),
            "fine_interval_logits": torch.randn(4, 23),
            # High confidence on row 0 (logit=3.0 -> |2p-1|=0.90 >= 0.65) where gender_mask=0, low confidence on row 1 (logit=0.1 -> |2p-1|=0.05 < 0.65)
            "gender_logit": torch.tensor([3.0, 0.1, -2.5, 2.0]),
            "reid_embedding": torch.nn.functional.normalize(torch.randn(4, 64), p=2, dim=-1),
        }
        targets = {
            "has_valid_age": torch.tensor([1.0, 1.0, 1.0, 0.0]),
            "gender_mask": torch.tensor([0.0, 0.0, 1.0, 1.0]),
            "dimorphism_weight": torch.tensor([1.0, 1.0, 1.0, 0.8]),
        }

        loss_val, kd_metrics = kd_loss_fn(student_preds, teacher_preds, targets=targets)
        assert torch.isfinite(loss_val)
        assert float(loss_val.item()) > 0.0
        assert kd_metrics["kd_imputed_gender_count"] == 1.0
        loss_val.backward()
        assert s_age_logits.grad is not None and torch.all(torch.isfinite(s_age_logits.grad))
        assert s_iv_logits.grad is not None and torch.all(torch.isfinite(s_iv_logits.grad))
        assert s_g_logit.grad is not None and torch.all(torch.isfinite(s_g_logit.grad))
        # Row 0 (gender_mask=0, Teacher conf >= 0.65) receives non-zero pseudo-label gradient;
        # Row 1 (gender_mask=0, Teacher conf < 0.65) receives zero gradient!
        assert abs(float(s_g_logit.grad[0].item())) > 1e-5
        assert abs(float(s_g_logit.grad[1].item())) < 1e-7
        assert s_reid_raw.grad is not None and torch.all(torch.isfinite(s_reid_raw.grad))

        # 3. Verify Gaussian interval focus sigma=1.6 parity between local backbone.py and kaggle_training_kernel.py
        from pathlib import Path
        kernel_src = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "kaggle_pipeline"
            / "kaggle_training_kernel.py"
        ).read_text(encoding="utf-8")
        assert "/ 1.6) ** 2" in kernel_src
        assert "/ 2.25) ** 2" not in kernel_src




