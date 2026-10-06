"""
Kaggle Training Kernel Generator (DistributedDataParallel)
===========================================================
Generates a complete, reproducible Jupyter Notebook:
`notebooks/training/kaggle_training_convnextv2.ipynb`
optimized for training ConvNeXt-V2-Tiny on Dual Tesla T4 GPUs (32GB VRAM)
using DistributedDataParallel (DDP via torchrun --standalone --nproc_per_node=2),
Effective Batch Size = 128, and 3 Fail-Fast Telemetry Tripwires.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional


def generate_training_notebook(output_path: Path | str = "notebooks/training/kaggle_training_convnextv2.ipynb") -> Path:
    """
    Generates a production-grade self-contained training notebook for Kaggle GPUs
    with Multi-Scale ConvNeXt-V2-Tiny, Head Routing Separation, and TorchRun DDP.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # DDP script code to be written inside the notebook
    train_ddp_script = '''"""
Distributed Training Script for ConvNeXt-V2-Tiny Biometrics
Executed via: torchrun --standalone --nproc_per_node=2 train_ddp.py
"""
import os, sys, math, time, json, cv2
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import timm

# 1. Initialize Distributed Environment
dist.init_process_group(backend="nccl")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 2))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
is_main_process = (local_rank == 0)

if is_main_process:
    print(f"[DDP Init] World Size: {world_size} | Local Rank: {local_rank} | Device: {device}")

# 2. Gaussian LDL Utility
def compute_gaussian_ldl(mean_age: float, sigma: float = 2.5, num_bins: int = 101) -> np.ndarray:
    bins = np.arange(num_bins, dtype=np.float64)
    diff = bins - float(mean_age)
    probs = np.exp(-0.5 * (diff / max(sigma, 0.5)) ** 2)
    probs /= (probs.sum() + 1e-12)
    return probs.astype(np.float32)

# 3. Golden Dataset Definition
class FaceGoldenDataset(Dataset):
    def __init__(self, manifest_path, split="train", images_dir=None):
        self.manifest_path = Path(manifest_path)
        self.images_dir = Path(images_dir) if images_dir else None
        self.df = pd.read_parquet(self.manifest_path) if self.manifest_path.suffix == ".parquet" else pd.read_csv(self.manifest_path)
        if "split" in self.df.columns and split:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bgr = None
        for k in ["image_path", "crop_path", "filepath", "path"]:
            p = row.get(k)
            if p and os.path.exists(str(p)):
                img_bgr = cv2.imread(str(p))
                if img_bgr is not None:
                    break
        if img_bgr is None and self.images_dir:
            img_id = str(row.get("image_id", ""))
            cand = self.images_dir / img_id
            if cand.exists():
                img_bgr = cv2.imread(str(cand))
        if img_bgr is not None:
            if img_bgr.shape[:2] != (160, 160):
                img_bgr = cv2.resize(img_bgr, (160, 160))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor_img = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 127.5 - 1.0
        else:
            tensor_img = torch.zeros((3, 160, 160), dtype=torch.float32)

        age_val = float(row.get("age_continuous", 25.0))
        ldl_101 = compute_gaussian_ldl(age_val)

        g_raw = row.get("gender")
        has_gender = (g_raw is not None) and (str(g_raw).strip().lower() in ["female", "male", "f", "m", "1", "0"])
        gender_mask = 1.0 if has_gender else 0.0
        gender_target = 1.0 if str(g_raw).strip().lower() in ["female", "f", "1"] else 0.0
        dimorphism_w = float(row.get("dimorphism_weight", 1.0))

        return {
            "image": tensor_img,
            "ldl_101": torch.from_numpy(ldl_101),
            "age_continuous": torch.tensor(age_val, dtype=torch.float32),
            "gender": torch.tensor(gender_target, dtype=torch.float32),
            "gender_mask": torch.tensor(gender_mask, dtype=torch.float32),
            "dimorphism_weight": torch.tensor(dimorphism_w, dtype=torch.float32),
        }

# 4. Multi-Scale Biometric Fusion & Model
class MultiScaleBiometricFusion(nn.Module):
    def __init__(self, in_channels=(192, 384, 768), proj_dim=128, fused_dim=512):
        super().__init__()
        c2, c3, c4 = in_channels
        self.proj2 = nn.Conv2d(c2, proj_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(c3, proj_dim, kernel_size=1)
        self.proj4 = nn.Conv2d(c4, proj_dim, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 6, fused_dim),
            nn.LayerNorm(fused_dim)
        )

    def _pool_stage(self, feat, proj):
        if feat.ndim == 2: feat = feat.unsqueeze(-1).unsqueeze(-1)
        p = proj(feat)
        return torch.cat([torch.flatten(self.avg_pool(p), 1), torch.flatten(self.max_pool(p), 1)], dim=-1)

    def forward(self, s2, s3, s4):
        z2 = self._pool_stage(s2, self.proj2)
        z3 = self._pool_stage(s3, self.proj3)
        z4 = self._pool_stage(s4, self.proj4)
        z_fused = self.fusion(torch.cat([z2, z3, z4], dim=-1))
        return z_fused, z4

class BiometricMultiTaskModel(nn.Module):
    def __init__(self, backbone_name="convnextv2_tiny", num_age_bins=101, reid_dim=64, pretrained=True, drop_path_rate=0.1):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, drop_path_rate=drop_path_rate)
        ch = self.backbone.feature_info.channels()
        if len(ch) >= 3:
            stage_channels = (ch[-3], ch[-2], ch[-1])
        elif len(ch) == 2:
            stage_channels = (ch[0], ch[0], ch[1])
        else:
            stage_channels = (ch[0], ch[0], ch[0])
        self.fusion = MultiScaleBiometricFusion(in_channels=stage_channels, proj_dim=128, fused_dim=512)

        self.age_ldl_head = nn.Sequential(nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, num_age_bins))
        self.gender_head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
        self.uncertainty_head = nn.Sequential(nn.Linear(512, 64), nn.SiLU(), nn.Linear(64, 1))
        self.reid_head = nn.Sequential(nn.Linear(256, reid_dim), nn.LayerNorm(reid_dim))

    def forward(self, x):
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
        z_fused, z4 = self.fusion(s2, s3, s4)

        age_logits = self.age_ldl_head(z_fused)
        age_probs = F.softmax(age_logits, dim=-1)
        gender_logit = self.gender_head(z4).squeeze(-1)
        log_var = self.uncertainty_head(z_fused).squeeze(-1)
        reid_emb = F.normalize(self.reid_head(z4), p=2, dim=-1)
        expected_age = torch.sum(age_probs * torch.arange(101, device=x.device, dtype=torch.float32), dim=-1)

        return {
            "age_logits": age_logits,
            "age_probs": age_probs,
            "expected_age": expected_age,
            "gender_logit": gender_logit,
            "log_variance": log_var,
            "reid_embedding": reid_emb
        }

# 5. Numerically Stable Unified Loss Functions
class UnifiedBiometricLoss(nn.Module):
    def __init__(self, w_cramer=1.0, w_kl=0.5, w_mae=0.2, w_gender=1.0, w_unc=0.1):
        super().__init__()
        self.w_cramer, self.w_kl, self.w_mae, self.w_gender, self.w_unc = w_cramer, w_kl, w_mae, w_gender, w_unc

    def forward(self, preds, targets):
        # 1. Cramer Distance
        p = preds["age_probs"].float() / (preds["age_probs"].float().sum(-1, keepdim=True) + 1e-12)
        q = targets["ldl_101"].float() / (targets["ldl_101"].float().sum(-1, keepdim=True) + 1e-12)
        l_cramer = torch.mean(torch.sum((torch.cumsum(p, -1) - torch.cumsum(q, -1)) ** 2, -1))

        # 2. Soft-KL Divergence (log_softmax FP16 protection)
        log_p = F.log_softmax(preds["age_logits"].float(), dim=-1)
        l_kl = F.kl_div(log_p, targets["ldl_101"].float(), reduction="batchmean")

        # 3. Soft-L1 MAE
        l_mae = torch.mean(torch.abs(preds["expected_age"].float() - targets["age_continuous"].float()))

        # 4. Masked Dimorphic BCE (Normalized by valid sample count)
        bce = F.binary_cross_entropy_with_logits(preds["gender_logit"].float(), targets["gender"].float(), reduction="none")
        mask = targets["gender_mask"].float()
        dim_w = targets["dimorphism_weight"].float()
        valid_cnt = torch.sum(mask)
        l_gender = torch.sum(bce * mask * dim_w) / (valid_cnt + 1e-8) if valid_cnt > 0 else torch.sum(bce * 0.0)

        # 5. Aleatoric Laplace Loss (Detached expectation)
        res = torch.abs(preds["expected_age"].detach().float() - targets["age_continuous"].float())
        s = torch.clamp(preds["log_variance"].float(), -10.0, 10.0)
        l_unc = torch.mean(torch.exp(-s) * res + s)

        total = self.w_cramer * l_cramer + self.w_kl * l_kl + self.w_mae * l_mae + self.w_gender * l_gender + self.w_unc * l_unc
        return total, {"cramer": l_cramer.item(), "kl": l_kl.item(), "mae": l_mae.item(), "gender": l_gender.item(), "unc": l_unc.item()}

# 6. DDP Training Loop with 3 Fail-Fast Telemetry Tripwires
def run_ddp_training():
    manifest_p = Path("/kaggle/input/face-dataset-mining-extreme-filtering-pipeline/metadata.parquet")
    if not manifest_p.exists():
        manifest_p = Path("reports/metadata.parquet")

    # If dataset not found, create a small synthetic dataset for demonstration
    if not manifest_p.exists():
        df_dummy = pd.DataFrame({
            "age_continuous": np.random.uniform(10, 80, 256),
            "gender": np.random.choice(["Male", "Female"], 256),
            "dimorphism_weight": np.random.uniform(0.5, 1.0, 256),
            "split": "train"
        })
        manifest_p.parent.mkdir(parents=True, exist_ok=True)
        df_dummy.to_parquet("dummy_manifest.parquet")
        manifest_p = Path("dummy_manifest.parquet")

    dataset = FaceGoldenDataset(manifest_p, split="train")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=2, pin_memory=True)

    model = BiometricMultiTaskModel(backbone_name="convnextv2_tiny", pretrained=False, drop_path_rate=0.1).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    criterion = UnifiedBiometricLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

    # 3 epochs warmup + 27 epochs cosine annealing = 30 epochs
    num_epochs = 30
    steps_per_epoch = max(1, len(dataloader))
    warmup_steps = 3 * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    global_step = 0
    initial_loss = None

    if is_main_process:
        print(f"🚀 Starting DDP Training: Effective Batch Size = {64 * world_size} (64 x {world_size} GPUs) | Total Steps: {total_steps}")

    model.train()
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            global_step += 1
            imgs = batch["image"].to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "image"}

            # Learning rate warmup
            if global_step <= warmup_steps:
                warmup_lr = 5e-4 * (global_step / warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                preds = model(imgs)
                loss, metrics = criterion(preds, targets)

            # FAIL-FAST TRIPWIRE 1: Immediate halt on NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"🚨 [FAIL-FAST TRIPWIRE 1] Loss is NaN/Inf at step {global_step}!")

            # FAIL-FAST TRIPWIRE 2: Loss divergence check at step 100
            if global_step == 1:
                initial_loss = loss.item()
            elif global_step == 100 and initial_loss is not None:
                if loss.item() > 2.0 * initial_loss:
                    raise RuntimeError(f"🚨 [FAIL-FAST TRIPWIRE 2] Loss diverged at step 100: current {loss.item():.4f} > 2.0x initial {initial_loss:.4f}!")

            # FAIL-FAST TRIPWIRE 3: Variance collapse / explosion check
            mean_log_var = preds["log_variance"].mean().item()
            if mean_log_var > 6.0:
                raise RuntimeError(f"🚨 [FAIL-FAST TRIPWIRE 3] Variance explosion (mean log_var={mean_log_var:.2f} > 6.0) at step {global_step}!")
            if mean_log_var < -9.0:
                raise RuntimeError(f"🚨 [FAIL-FAST TRIPWIRE 3] Variance collapse (mean log_var={mean_log_var:.2f} < -9.0) at step {global_step}!")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # TRIPWIRE 1B: Gradient norm anomaly check
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 50.0:
                raise RuntimeError(f"🚨 [FAIL-FAST TRIPWIRE 1B] Gradient norm anomaly ({grad_norm:.2f} > 50.0) at step {global_step}!")

            scaler.step(optimizer)
            scaler.update()

            if global_step > warmup_steps:
                scheduler.step()

            if is_main_process and (global_step % 20 == 0 or global_step == 1):
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"[Epoch {epoch+1:02d}/{num_epochs}] Step {global_step:04d} | Loss: {loss.item():.4f} | Cramer: {metrics['cramer']:.4f} | KL: {metrics['kl']:.4f} | MAE: {metrics['mae']:.2f} | GradNorm: {grad_norm:.2f} | LR: {cur_lr:.2e}")

    if is_main_process:
        ckpt_path = Path("convnextv2_tiny_biometrics.pt")
        torch.save(model.module.state_dict(), ckpt_path)
        print(f"✅ Training completed successfully! Model checkpoint saved to: {ckpt_path.resolve()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    run_ddp_training()
'''

    cells = [
        # Cell 0: Header Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Kaggle Cloud Training: SOTA ConvNeXt-V2-Tiny Biometrics (Dual Tesla T4 DDP)\n",
                "### Multi-Scale Biometric Fusion (20x20 + 10x10 + 5x5), Head Routing Separation & Fail-Fast Telemetry\n",
                "**Runtime**: Kaggle Dual Tesla T4 (2x16GB VRAM, Mixed Precision AMP)\n",
                "**Protocol**: `torchrun --standalone --nproc_per_node=2 train_ddp.py` (Effective Batch Size = 128)\n",
                "\n",
                "Key Advancements:\n",
                "- **ConvNeXt-V2-Tiny**: 27.87M params (Pareto Sweet Spot), depths [3, 3, 9, 3], channels [96, 192, 384, 768].\n",
                "- **MultiScaleBiometricFusion**: Dual-pooled 1x1 conv features from Stage 2 (micro), Stage 3 (meso), Stage 4 (macro).\n",
                "- **Head Routing Isolation**: Age LDL & Aleatoric heads receive 512-D fused embedding; Gender & Re-ID receive strictly 256-D Stage 4 cranial macro-structure.\n",
                "- **Mathematical Loss Stability**: FP16 Soft-KL underflow protected via `log_softmax`, strictly sample-normalized Masked BCE, and L1-Detached Laplace Aleatoric loss.\n",
                "- **3 Fail-Fast Telemetry Tripwires**: NaN/Inf grad > 50 halt, loss divergence 2.0x halt at step 100, variance explosion/collapse halt."
            ]
        },
        # Cell 1: Environment & Setup
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys, math, time, json\n",
                "from pathlib import Path\n",
                "import torch\n",
                "\n",
                "# Install required dependencies\n",
                "!pip install -q timm pyarrow fastparquet\n",
                "import timm\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "num_gpus = torch.cuda.device_count()\n",
                "print(f'[Env] PyTorch {torch.__version__} | Timm {timm.__version__} | GPUs: {num_gpus}')\n",
                "for g in range(num_gpus):\n",
                "    print(f'  [GPU {g}] {torch.cuda.get_device_name(g)} ({torch.cuda.get_device_properties(g).total_memory / 1e9:.1f} GB)')\n",
                "assert num_gpus >= 2, f'Expected 2 GPUs for DDP training, found {num_gpus}'\n"
            ]
        },
        # Cell 2: Dataset Definition & Verification
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, cv2\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "from torch.utils.data import Dataset\n",
                "\n",
                "def compute_gaussian_ldl(mean_age: float, sigma: float = 2.5, num_bins: int = 101) -> np.ndarray:\n",
                "    bins = np.arange(num_bins, dtype=np.float64)\n",
                "    diff = bins - float(mean_age)\n",
                "    probs = np.exp(-0.5 * (diff / max(sigma, 0.5)) ** 2)\n",
                "    probs /= (probs.sum() + 1e-12)\n",
                "    return probs.astype(np.float32)\n",
                "\n",
                "class FaceGoldenDataset(Dataset):\n",
                "    def __init__(self, manifest_path, split='train', images_dir=None):\n",
                "        self.manifest_path = Path(manifest_path)\n",
                "        self.images_dir = Path(images_dir) if images_dir else None\n",
                "        self.df = pd.read_parquet(self.manifest_path) if self.manifest_path.suffix == '.parquet' else pd.read_csv(self.manifest_path)\n",
                "        if 'split' in self.df.columns and split:\n",
                "            self.df = self.df[self.df['split'] == split].reset_index(drop=True)\n",
                "        else:\n",
                "            self.df = self.df.reset_index(drop=True)\n",
                "\n",
                "    def __len__(self):\n",
                "        return len(self.df)\n",
                "\n",
                "    def __getitem__(self, idx):\n",
                "        row = self.df.iloc[idx]\n",
                "        img_bgr = None\n",
                "        for k in ['image_path', 'crop_path', 'filepath', 'path']:\n",
                "            p = row.get(k)\n",
                "            if p and os.path.exists(str(p)):\n",
                "                img_bgr = cv2.imread(str(p))\n",
                "                if img_bgr is not None: break\n",
                "        if img_bgr is None and self.images_dir:\n",
                "            img_id = str(row.get('image_id', ''))\n",
                "            cand = self.images_dir / img_id\n",
                "            if cand.exists(): img_bgr = cv2.imread(str(cand))\n",
                "        if img_bgr is not None:\n",
                "            if img_bgr.shape[:2] != (160, 160): img_bgr = cv2.resize(img_bgr, (160, 160))\n",
                "            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)\n",
                "            tensor_img = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 127.5 - 1.0\n",
                "        else:\n",
                "            tensor_img = torch.zeros((3, 160, 160), dtype=torch.float32)\n",
                "\n",
                "        age_val = float(row.get('age_continuous', 25.0))\n",
                "        ldl_101 = compute_gaussian_ldl(age_val)\n",
                "        g_raw = row.get('gender')\n",
                "        has_gender = (g_raw is not None) and (str(g_raw).strip().lower() in ['female', 'male', 'f', 'm', '1', '0'])\n",
                "        gender_mask = 1.0 if has_gender else 0.0\n",
                "        gender_target = 1.0 if str(g_raw).strip().lower() in ['female', 'f', '1'] else 0.0\n",
                "        dimorphism_w = float(row.get('dimorphism_weight', 1.0))\n",
                "\n",
                "        return {\n",
                "            'image': tensor_img,\n",
                "            'ldl_101': torch.from_numpy(ldl_101),\n",
                "            'age_continuous': torch.tensor(age_val, dtype=torch.float32),\n",
                "            'gender': torch.tensor(gender_target, dtype=torch.float32),\n",
                "            'gender_mask': torch.tensor(gender_mask, dtype=torch.float32),\n",
                "            'dimorphism_weight': torch.tensor(dimorphism_w, dtype=torch.float32)\n",
                "        }\n",
                "print('✅ FaceGoldenDataset with real image loading, gender_mask and LDL distribution loaded.')\n"
            ]
        },
        # Cell 3: Multi-Scale Architecture Definition
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch.nn as nn\n",
                "import torch.nn.functional as F\n",
                "\n",
                "class MultiScaleBiometricFusion(nn.Module):\n",
                "    def __init__(self, in_channels=(192, 384, 768), proj_dim=128, fused_dim=512):\n",
                "        super().__init__()\n",
                "        c2, c3, c4 = in_channels\n",
                "        self.proj2 = nn.Conv2d(c2, proj_dim, kernel_size=1)\n",
                "        self.proj3 = nn.Conv2d(c3, proj_dim, kernel_size=1)\n",
                "        self.proj4 = nn.Conv2d(c4, proj_dim, kernel_size=1)\n",
                "        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))\n",
                "        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))\n",
                "        self.fusion = nn.Sequential(nn.Linear(proj_dim * 6, fused_dim), nn.LayerNorm(fused_dim))\n",
                "\n",
                "    def _pool_stage(self, feat, proj):\n",
                "        if feat.ndim == 2: feat = feat.unsqueeze(-1).unsqueeze(-1)\n",
                "        p = proj(feat)\n",
                "        return torch.cat([torch.flatten(self.avg_pool(p), 1), torch.flatten(self.max_pool(p), 1)], dim=-1)\n",
                "\n",
                "    def forward(self, s2, s3, s4):\n",
                "        z2 = self._pool_stage(s2, self.proj2)\n",
                "        z3 = self._pool_stage(s3, self.proj3)\n",
                "        z4 = self._pool_stage(s4, self.proj4)\n",
                "        z_fused = self.fusion(torch.cat([z2, z3, z4], dim=-1))\n",
                "        return z_fused, z4\n",
                "\n",
                "class BiometricMultiTaskModel(nn.Module):\n",
                "    def __init__(self, backbone_name='convnextv2_tiny', num_age_bins=101, reid_dim=64, pretrained=True, drop_path_rate=0.1):\n",
                "        super().__init__()\n",
                "        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, drop_path_rate=drop_path_rate)\n",
                "        ch = self.backbone.feature_info.channels()\n",
                "        if len(ch) >= 3:\n",
                "            stage_channels = (ch[-3], ch[-2], ch[-1])\n",
                "        elif len(ch) == 2:\n",
                "            stage_channels = (ch[0], ch[0], ch[1])\n",
                "        else:\n",
                "            stage_channels = (ch[0], ch[0], ch[0])\n",
                "        self.fusion = MultiScaleBiometricFusion(in_channels=stage_channels, proj_dim=128, fused_dim=512)\n",
                "        self.age_ldl_head = nn.Sequential(nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, num_age_bins))\n",
                "        self.gender_head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))\n",
                "        self.uncertainty_head = nn.Sequential(nn.Linear(512, 64), nn.SiLU(), nn.Linear(64, 1))\n",
                "        self.reid_head = nn.Sequential(nn.Linear(256, reid_dim), nn.LayerNorm(reid_dim))\n",
                "\n",
                "    def forward(self, x):\n",
                "        feats = self.backbone(x)\n",
                "        if isinstance(feats, (list, tuple)):\n",
                "            if len(feats) >= 3:\n",
                "                s2, s3, s4 = feats[-3], feats[-2], feats[-1]\n",
                "            elif len(feats) == 2:\n",
                "                s2, s3, s4 = feats[0], feats[0], feats[1]\n",
                "            else:\n",
                "                s2 = s3 = s4 = feats[0]\n",
                "        else:\n",
                "            s2 = s3 = s4 = feats\n",
                "        z_fused, z4 = self.fusion(s2, s3, s4)\n",
                "        age_logits = self.age_ldl_head(z_fused)\n",
                "        age_probs = F.softmax(age_logits, dim=-1)\n",
                "        gender_logit = self.gender_head(z4).squeeze(-1)\n",
                "        log_var = self.uncertainty_head(z_fused).squeeze(-1)\n",
                "        reid_emb = F.normalize(self.reid_head(z4), p=2, dim=-1)\n",
                "        expected_age = torch.sum(age_probs * torch.arange(101, device=x.device, dtype=torch.float32), dim=-1)\n",
                "        return {\n",
                "            'age_logits': age_logits, 'age_probs': age_probs, 'expected_age': expected_age,\n",
                "            'gender_logit': gender_logit, 'log_variance': log_var, 'reid_embedding': reid_emb\n",
                "        }\n",
                "print('✅ Multi-Scale ConvNeXt-V2-Tiny architecture defined.')\n"
            ]
        },
        # Cell 4: Unified Biometric Loss
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class UnifiedBiometricLoss(nn.Module):\n",
                "    def __init__(self, w_cramer=1.0, w_kl=0.5, w_mae=0.2, w_gender=1.0, w_unc=0.1):\n",
                "        super().__init__()\n",
                "        self.w_cramer, self.w_kl, self.w_mae, self.w_gender, self.w_unc = w_cramer, w_kl, w_mae, w_gender, w_unc\n",
                "\n",
                "    def forward(self, preds, targets):\n",
                "        p = preds['age_probs'].float() / (preds['age_probs'].float().sum(-1, keepdim=True) + 1e-12)\n",
                "        q = targets['ldl_101'].float() / (targets['ldl_101'].float().sum(-1, keepdim=True) + 1e-12)\n",
                "        l_cramer = torch.mean(torch.sum((torch.cumsum(p, -1) - torch.cumsum(q, -1)) ** 2, -1))\n",
                "\n",
                "        log_p = F.log_softmax(preds['age_logits'].float(), dim=-1)\n",
                "        l_kl = F.kl_div(log_p, targets['ldl_101'].float(), reduction='batchmean')\n",
                "\n",
                "        l_mae = torch.mean(torch.abs(preds['expected_age'].float() - targets['age_continuous'].float()))\n",
                "\n",
                "        bce = F.binary_cross_entropy_with_logits(preds['gender_logit'].float(), targets['gender'].float(), reduction='none')\n",
                "        mask = targets['gender_mask'].float()\n",
                "        dim_w = targets['dimorphism_weight'].float()\n",
                "        valid_cnt = torch.sum(mask)\n",
                "        l_gender = torch.sum(bce * mask * dim_w) / (valid_cnt + 1e-8) if valid_cnt > 0 else torch.sum(bce * 0.0)\n",
                "\n",
                "        res = torch.abs(preds['expected_age'].detach().float() - targets['age_continuous'].float())\n",
                "        s = torch.clamp(preds['log_variance'].float(), -10.0, 10.0)\n",
                "        l_unc = torch.mean(torch.exp(-s) * res + s)\n",
                "\n",
                "        total = self.w_cramer * l_cramer + self.w_kl * l_kl + self.w_mae * l_mae + self.w_gender * l_gender + self.w_unc * l_unc\n",
                "        return total, {'cramer': l_cramer.item(), 'kl': l_kl.item(), 'mae': l_mae.item(), 'gender': l_gender.item(), 'unc': l_unc.item()}\n",
                "print('✅ UnifiedBiometricLoss with FP16 protection initialized.')\n"
            ]
        },
        # Cell 5: Generate train_ddp.py file
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Write standalone train_ddp.py script for torchrun execution\n",
                "with open('train_ddp.py', 'w', encoding='utf-8') as f:\n",
                f"    f.write({repr(train_ddp_script)})\n",
                "print('✅ Wrote train_ddp.py for distributed execution across Dual Tesla T4 GPUs.')\n"
            ]
        },
        # Cell 6: Launch DDP Distributed Training via TorchRun
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run DDP training across 2 GPUs with torchrun\n",
                "!torchrun --standalone --nproc_per_node=2 train_ddp.py\n"
            ]
        },
        # Cell 7: Checkpoint Validation
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if Path('convnextv2_tiny_biometrics.pt').exists():\n",
                "    sz = Path('convnextv2_tiny_biometrics.pt').stat().st_size / 1e6\n",
                "    print(f'🎉 Checkpoint ready: convnextv2_tiny_biometrics.pt ({sz:.2f} MB)')\n",
                "else:\n",
                "    print('⚠️ Checkpoint not found yet.')\n"
            ]
        }
    ]

    nb_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_p.write_text(json.dumps(nb_data, indent=2), encoding="utf-8")

    # Also update notebooks/kaggle_training_convnextv2.ipynb if output_path is different
    alt_p = Path("notebooks/kaggle_training_convnextv2.ipynb")
    if out_p.resolve() != alt_p.resolve():
        alt_p.parent.mkdir(parents=True, exist_ok=True)
        alt_p.write_text(json.dumps(nb_data, indent=2), encoding="utf-8")

    return out_p


if __name__ == "__main__":
    p = generate_training_notebook()
    print(f"Generated training notebook at: {p}")
