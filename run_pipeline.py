#!/usr/bin/env python3
"""
================================================================================
Investigating Contrastive Learning for Vision Transformer-Based Breast
Histopathological Image Classification under Limited Labeled Data
================================================================================
End-to-End Pipeline Script:
  Stage 1: Self-Supervised Contrastive Pretraining (SimCLR + NT-Xent)
  Stage 2: Full-Data Supervised Fine-Tuning with Early Stopping
  Stage 3: Controlled Label-Scarcity Benchmark (10%, 25%, 50%, 75%, 100%)
  Stage 4: Comprehensive Metrics, Confusion Matrices, ROC & Efficiency Curves

Author: Research Team
Target Platform: Kaggle GPU (Tesla T4 / P100) or Local PyTorch Environment
================================================================================
"""

import os
import gc
import sys
import copy
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler

# ------------------------------------------------------------------------------
# Graceful Imports for torchvision and timm (built-in on Kaggle)
# ------------------------------------------------------------------------------
try:
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)


# ==============================================================================
# 1. REPRODUCIBILITY & SYSTEM INSPECTION
# ==============================================================================
def set_seed(seed: int = 42):
    """Fix all random seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inspect_system(device: torch.device):
    """Logs system device, GPU model, and memory availability."""
    print("=" * 80)
    print("SYSTEM & ACCELERATOR INSPECTION")
    print("=" * 80)
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"TorchVision Ready : {HAS_TORCHVISION}")
    print(f"Timm Ready        : {HAS_TIMM}")
    print(f"Active Device     : {device}")
    if device.type == "cuda":
        print(f"GPU Model         : {torch.cuda.get_device_name(0)}")
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"VRAM Available    : {free_mem / 1e9:.2f} GB / {total_mem / 1e9:.2f} GB")
    print("=" * 80)


# ==============================================================================
# 2. DATASETS & MEMORY-SAFE LAZY HDF5 LOADER
# ==============================================================================
class PCamHDF5Dataset(Dataset):
    """
    Worker-safe lazy-loading PyTorch Dataset for PCam HDF5 files.
    Opens file pointers on-demand per DataLoader worker process to prevent deadlocks.
    """
    def __init__(self, x_path: str, y_path: str, transform=None):
        self.x_path = str(x_path)
        self.y_path = str(y_path)
        self.transform = transform

        # Probe dataset length and keys in an ephemeral context
        with h5py.File(self.x_path, 'r') as fx:
            self.x_key = 'x' if 'x' in fx else list(fx.keys())[0]
            self.length = fx[self.x_key].shape[0]

        with h5py.File(self.y_path, 'r') as fy:
            self.y_key = 'y' if 'y' in fy else list(fy.keys())[0]

        self.fx: Optional[h5py.File] = None
        self.fy: Optional[h5py.File] = None

    def _init_db(self):
        self.fx = h5py.File(self.x_path, 'r')
        self.fy = h5py.File(self.y_path, 'r')

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fx is None:
            self._init_db()

        img_np = self.fx[self.x_key][idx]
        label = int(np.squeeze(self.fy[self.y_key][idx]))

        img_pil = Image.fromarray(img_np)
        if self.transform is not None:
            img = self.transform(img_pil)
        else:
            img = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0

        return img, torch.tensor(label, dtype=torch.long)


class SyntheticPCamDataset(Dataset):
    """
    Synthetic in-memory dataset for local debugging and smoke testing
    when real multi-GB HDF5 files are not present.
    """
    def __init__(self, num_samples: int = 200, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        np.random.seed(42)
        self.images = np.random.randint(0, 256, (num_samples, 96, 96, 3), dtype=np.uint8)
        self.labels = np.random.randint(0, 2, num_samples, dtype=np.int64)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        img_pil = Image.fromarray(self.images[idx])
        if self.transform is not None:
            img = self.transform(img_pil)
        else:
            img = torch.from_numpy(self.images[idx].transpose((2, 0, 1))).float() / 255.0
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def resolve_pcam_paths(data_dir: Path) -> Dict[str, str]:
    """
    Scans data directory to locate train, validation, and test HDF5 files.
    Supports both official 'andrewmvd' structure and flat legacy structures.
    """
    paths = {}
    print(f"\nResolving PCam HDF5 files under: {data_dir}")

    # 1. Search for Training Images & Labels
    train_x = list(data_dir.rglob("*training_split.h5")) or list(data_dir.rglob("*train_x.h5"))
    train_y = list(data_dir.rglob("*train_y.h5"))
    if train_x and train_y:
        paths["train_x"] = str(train_x[0])
        paths["train_y"] = str(train_y[0])
        print(f"  -> Detected Train Images: {paths['train_x']}")
        print(f"  -> Detected Train Labels: {paths['train_y']}")

    # 2. Search for Validation Images & Labels
    val_x = list(data_dir.rglob("*validation_split.h5")) or list(data_dir.rglob("*valid_x.h5"))
    val_y = list(data_dir.rglob("*valid_y.h5"))
    if val_x and val_y:
        paths["val_x"] = str(val_x[0])
        paths["val_y"] = str(val_y[0])
        print(f"  -> Detected Val Images  : {paths['val_x']}")
        print(f"  -> Detected Val Labels  : {paths['val_y']}")

    # 3. Search for Test Images & Labels
    test_x = list(data_dir.rglob("*test_split.h5")) or list(data_dir.rglob("*test_x.h5"))
    test_y = list(data_dir.rglob("*test_y.h5"))
    if test_x and test_y:
        paths["test_x"] = str(test_x[0])
        paths["test_y"] = str(test_y[0])
        print(f"  -> Detected Test Images : {paths['test_x']}")
        print(f"  -> Detected Test Labels : {paths['test_y']}")

    return paths


# ==============================================================================
# 3. DATA AUGMENTATION & PREPROCESSING TRANSFORMS (PICKLE-SAFE)
# ==============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class FallbackPILTransforms:
    """Pure PIL/PyTorch transform fallbacks if torchvision is not installed."""
    @staticmethod
    def to_tensor_normalize(img: Image.Image, size: int = 224) -> torch.Tensor:
        img_resized = img.resize((size, size), Image.Resampling.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose((2, 0, 1)))
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std

    @staticmethod
    def random_augment(img: Image.Image, size: int = 224) -> torch.Tensor:
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        rotations = [0, 90, 180, 270]
        rot = random.choice(rotations)
        if rot > 0:
            img = img.rotate(rot)
        return FallbackPILTransforms.to_tensor_normalize(img, size=size)


class ContrastiveTwoViewTransform:
    """
    Applies two independent stochastic augmentations to generate positive pairs
    for self-supervised contrastive learning (SimCLR).
    """
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        if HAS_TORCHVISION:
            self.aug = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            self.aug = None

    def __call__(self, x: Image.Image):
        if self.aug is not None:
            return self.aug(x), self.aug(x)
        else:
            v1 = FallbackPILTransforms.random_augment(x, self.image_size)
            v2 = FallbackPILTransforms.random_augment(x, self.image_size)
            return v1, v2


class SupervisedTrainTransform:
    """Standard supervised augmentation preserving cellular morphology."""
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        if HAS_TORCHVISION:
            self.aug = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            self.aug = None

    def __call__(self, img: Image.Image):
        if self.aug is not None:
            return self.aug(img)
        return FallbackPILTransforms.random_augment(img, self.image_size)


class EvalTransform:
    """Deterministic preprocessing for validation and testing."""
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        if HAS_TORCHVISION:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            self.tf = None

    def __call__(self, img: Image.Image):
        if self.tf is not None:
            return self.tf(img)
        return FallbackPILTransforms.to_tensor_normalize(img, self.image_size)


# ==============================================================================
# 4. MODEL ARCHITECTURES & NT-XENT CONTRASTIVE LOSS
# ==============================================================================
class StandaloneViT(nn.Module):
    """
    Compact pure-PyTorch Vision Transformer backbone fallback if timm is not installed.
    Matches ViT-Tiny specifications: patch size 16, embed dim 192, 4 heads, 6 layers.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192, depth=6, num_heads=4):
        super().__init__()
        self.num_features = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]  # [CLS] token representation


def create_encoder_backbone(model_name: str, pretrained: bool = True) -> nn.Module:
    """Instantiates ViT backbone using timm or standalone fallback."""
    if HAS_TIMM:
        try:
            return timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception as e:
            print(f"Warning: timm model instantiation failed ({e}). Using StandaloneViT.")
            return StandaloneViT()
    else:
        return StandaloneViT()


class ProjectionHead(nn.Module):
    """2-layer non-linear MLP projection head mapping CLS representations."""
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRViT(nn.Module):
    """Complete Self-Supervised Contrastive ViT Assembly."""
    def __init__(self, model_name: str = "vit_tiny_patch16_224", pretrained: bool = True, proj_dim: int = 128):
        super().__init__()
        self.encoder = create_encoder_backbone(model_name, pretrained=pretrained)
        in_features = self.encoder.num_features
        self.projector = ProjectionHead(in_dim=in_features, hidden_dim=512, out_dim=proj_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        proj = F.normalize(self.projector(feat), dim=-1)
        return feat, proj


class ViTClassifier(nn.Module):
    """Downstream Binary Classifier combining ViT encoder and linear classification head."""
    def __init__(self, encoder: nn.Module, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        in_features = encoder.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.classifier(feat)


class NTXentLoss(nn.Module):
    """
    Vectorized Normalized Temperature-scaled Cross Entropy Loss (NT-Xent / InfoNCE).
    Safe for mixed-precision FP16/FP32 autocast.
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0) # [2*B, D]
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Mask diagonal self-similarity entries
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        min_val = torch.finfo(sim_matrix.dtype).min
        sim_matrix = sim_matrix.masked_fill(mask, min_val)

        # Targets: positive pair is i <-> i+B
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ], dim=0)

        return self.cross_entropy(sim_matrix, targets)


# ==============================================================================
# 5. TRAINING ENGINES WITH EARLY STOPPING & SCHEDULERS
# ==============================================================================
def train_contrastive_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device
) -> float:
    """Executes one epoch of self-supervised contrastive pretraining."""
    model.train()
    total_loss = 0.0

    for (v1, v2), _ in loader:
        v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = criterion(z1, z2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * v1.size(0)

    return total_loss / len(loader.dataset)


def train_supervised_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device
) -> Tuple[float, float]:
    """Executes one supervised classification training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0

    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(imgs)
            loss = criterion(outputs, lbls)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == lbls).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = correct / len(loader.dataset)
    return avg_loss, avg_acc


def validate_supervised_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluates classifier performance on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == lbls).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = correct / len(loader.dataset)
    return avg_loss, avg_acc


def train_classifier_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    desc: str = "Classifier"
) -> nn.Module:
    """
    Trains classifier with AdamW, CosineAnnealingLR, validation evaluation,
    and early stopping. Restores best model state.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    best_weights = copy.deepcopy(model.state_dict())

    print(f"\n--- Training {desc} (Max Epochs: {epochs}, Patience: {patience}) ---")
    for epoch in range(1, epochs + 1):
        t_start = time.time()
        t_loss, t_acc = train_supervised_epoch(model, train_loader, optimizer, criterion, scaler, device)
        v_loss, v_acc = validate_supervised_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t_start

        is_best = v_loss < best_val_loss
        if is_best:
            best_val_loss = v_loss
            best_val_acc = v_acc
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
            tag = "*"
        else:
            patience_counter += 1
            tag = " "

        print(f"  Epoch [{epoch:02d}/{epochs:02d}] {tag} "
              f"Train Loss: {t_loss:.4f} | Acc: {t_acc*100:.2f}% || "
              f"Val Loss: {v_loss:.4f} | Acc: {v_acc*100:.2f}% | Time: {elapsed:.1f}s")

        if patience_counter >= patience:
            print(f"  --> Early stopping triggered at Epoch {epoch}. Best Val Loss: {best_val_loss:.4f} (Acc: {best_val_acc*100:.2f}%)")
            break

    # Restore best validation weights
    model.load_state_dict(best_weights)
    return model


def evaluate_test_set(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    """Computes comprehensive clinical metrics on held-out test set."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(lbls.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5
    cm = confusion_matrix(all_targets, all_preds)

    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc,
        "cm": cm, "targets": all_targets, "preds": all_preds, "probs": all_probs
    }


# ==============================================================================
# 6. PIPELINE ORCHESTRATION ENGINE
# ==============================================================================
def run_project(args):
    """Executes the full experimental pipeline end-to-end."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    inspect_system(device)

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    for d in [checkpoints_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save execution config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # --------------------------------------------------------------------------
    # DATASET RESOLUTION & SETUP
    # --------------------------------------------------------------------------
    paths = resolve_pcam_paths(Path(args.data_dir))
    use_synthetic = args.debug or not ("train_x" in paths and "train_y" in paths)

    eval_tf = EvalTransform(args.image_size)
    sup_tf  = SupervisedTrainTransform(args.image_size)
    cl_tf   = ContrastiveTwoViewTransform(args.image_size)

    if use_synthetic:
        print("\n[!] Running in DEBUG / SYNTHETIC mode (using simulated image tensors).")
        sample_n = 240 if args.debug else 1000
        full_train_ds = SyntheticPCamDataset(sample_n, transform=eval_tf)
        val_ds        = SyntheticPCamDataset(int(sample_n * 0.2), transform=eval_tf)
        test_ds       = SyntheticPCamDataset(int(sample_n * 0.2), transform=eval_tf)
        all_train_labels = full_train_ds.labels
        raw_train_indices = np.arange(len(full_train_ds))
        contrastive_raw_ds = SyntheticPCamDataset(sample_n, transform=cl_tf)
        supervised_raw_ds  = SyntheticPCamDataset(sample_n, transform=sup_tf)
    else:
        # Full PCam Dataset
        full_train_ds = PCamHDF5Dataset(paths["train_x"], paths["train_y"], transform=eval_tf)

        # Load labels for stratification
        with h5py.File(paths["train_y"], 'r') as fy:
            yk = 'y' if 'y' in fy else list(fy.keys())[0]
            raw_labels = np.asarray(fy[yk][:]).ravel()

        # Fixed Canonical Training Pool (e.g., 16,000 samples)
        pool_size = min(args.canonical_train_size, len(full_train_ds))
        sss_pool = StratifiedShuffleSplit(n_splits=1, train_size=pool_size, random_state=args.seed)
        raw_train_indices, _ = next(sss_pool.split(np.zeros(len(raw_labels)), raw_labels))
        all_train_labels = raw_labels[raw_train_indices]

        # Validation Dataset
        if "val_x" in paths and "val_y" in paths:
            val_full = PCamHDF5Dataset(paths["val_x"], paths["val_y"], transform=eval_tf)
            val_indices = np.arange(min(args.canonical_val_size, len(val_full)))
            val_ds = Subset(val_full, val_indices)
        else:
            val_ds = Subset(full_train_ds, raw_train_indices[:int(pool_size * 0.1)])

        # Test Dataset (using official test split if available)
        if "test_x" in paths and "test_y" in paths:
            test_full = PCamHDF5Dataset(paths["test_x"], paths["test_y"], transform=eval_tf)
            test_indices = np.arange(min(args.canonical_test_size, len(test_full)))
            test_ds = Subset(test_full, test_indices)
        else:
            test_ds = Subset(full_train_ds, raw_train_indices[int(pool_size * 0.1):int(pool_size * 0.2)])

        contrastive_raw_ds = PCamHDF5Dataset(paths["train_x"], paths["train_y"], transform=cl_tf)
        supervised_raw_ds  = PCamHDF5Dataset(paths["train_x"], paths["train_y"], transform=sup_tf)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"\nDataset Initialization Complete:")
    print(f"  • Training Pool Size : {len(raw_train_indices)} samples")
    print(f"  • Validation Set Size: {len(val_ds)} samples")
    print(f"  • Test Set Size      : {len(test_ds)} samples")

    # --------------------------------------------------------------------------
    # STAGE 1: SELF-SUPERVISED CONTRASTIVE PRETRAINING (SimCLR)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 1: SELF-SUPERVISED CONTRASTIVE PRETRAINING (SimCLR)")
    print("=" * 80)

    contrastive_subset = Subset(contrastive_raw_ds, raw_train_indices)
    contrastive_loader = DataLoader(
        contrastive_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == "cuda")
    )

    contrastive_model = SimCLRViT(model_name=args.model_name, pretrained=(not args.debug), proj_dim=args.projection_dim).to(device)
    cl_optimizer = torch.optim.AdamW(contrastive_model.parameters(), lr=args.contrastive_lr, weight_decay=args.weight_decay)
    cl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cl_optimizer, T_max=args.contrastive_epochs, eta_min=1e-6)
    cl_criterion = NTXentLoss(temperature=args.temperature)
    cl_scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    cl_history = []
    cl_start = time.time()
    for epoch in range(1, args.contrastive_epochs + 1):
        e_start = time.time()
        ep_loss = train_contrastive_epoch(contrastive_model, contrastive_loader, cl_optimizer, cl_criterion, cl_scaler, device)
        cl_scheduler.step()
        cl_history.append({"epoch": epoch, "loss": ep_loss})
        print(f"Pretrain Epoch [{epoch:02d}/{args.contrastive_epochs:02d}] | NT-Xent Loss: {ep_loss:.4f} | Time: {time.time() - e_start:.1f}s")

    print(f"\nContrastive Pretraining complete in {(time.time() - cl_start)/60:.2f} mins.")

    # Save Pretrained Encoder Checkpoint
    pretrained_encoder_path = checkpoints_dir / "contrastive_vit_pretrained.pth"
    torch.save(contrastive_model.encoder.state_dict(), pretrained_encoder_path)
    print(f"Saved Pretrained Encoder Weights: {pretrained_encoder_path}")

    # --------------------------------------------------------------------------
    # STAGE 2: FULL-DATA SUPERVISED FINE-TUNING (100% Labels)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 2: FULL-DATA SUPERVISED FINE-TUNING (100% Reference)")
    print("=" * 80)

    train_100_subset = Subset(supervised_raw_ds, raw_train_indices)
    train_100_loader = DataLoader(train_100_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Initialize Fine-Tuned Model from Contrastive Pretrained Encoder
    finetune_encoder = create_encoder_backbone(args.model_name, pretrained=False)
    finetune_encoder.load_state_dict(torch.load(pretrained_encoder_path, map_location=device))
    finetuned_model = ViTClassifier(finetune_encoder, num_classes=2).to(device)

    finetuned_model = train_classifier_with_early_stopping(
        finetuned_model, train_100_loader, val_loader,
        epochs=args.finetune_epochs, lr=args.finetune_lr,
        weight_decay=args.weight_decay, patience=args.early_stopping_patience,
        device=device, desc="Contrastive ViT (100% Data)"
    )

    # Save Best 100% Checkpoint
    best_ft_path = checkpoints_dir / "best_contrastive_finetuned_vit.pth"
    torch.save(finetuned_model.state_dict(), best_ft_path)

    metrics_100 = evaluate_test_set(finetuned_model, test_loader, device)
    print("\n--- 100% Full-Data Reference Test Results ---")
    print(f"Accuracy  : {metrics_100['acc']:.4f}")
    print(f"F1-Score  : {metrics_100['f1']:.4f}")
    print(f"ROC-AUC   : {metrics_100['auc']:.4f}")
    print(f"Recall    : {metrics_100['rec']:.4f}")
    print(f"Precision : {metrics_100['prec']:.4f}")

    # --------------------------------------------------------------------------
    # STAGE 3: CONTROLLED LABEL-SCARCITY BENCHMARK (10%, 25%, 50%, 75%, 100%)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 3: CONTROLLED LABEL-SCARCITY BENCHMARK")
    print("=" * 80)
    print("Protocol: Testing Baseline ViT vs. Contrastive ViT on identical stratified subsets.")
    print("          Baseline starts from ImageNet weights; Contrastive starts from pretrained encoder.")

    fractions = [0.25, 1.0] if args.debug else args.fractions
    benchmark_records = []

    for frac in fractions:
        frac_pct = int(frac * 100)
        if frac == 1.0:
            frac_indices = raw_train_indices
        else:
            n_samples = max(int(len(raw_train_indices) * frac), 4)
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=args.seed)
            rel_idx, _ = next(sss.split(raw_train_indices, all_train_labels))
            frac_indices = raw_train_indices[rel_idx]

        frac_loader = DataLoader(
            Subset(supervised_raw_ds, frac_indices),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )

        print(f"\n>>> BENCHMARKING FRACTION: {frac_pct}% ({len(frac_indices)} samples) <<<")

        # 1. Baseline ViT: Trained from fresh ImageNet weights
        print(f"  [1/2] Training Baseline ViT on {frac_pct}% subset...")
        base_encoder = create_encoder_backbone(args.model_name, pretrained=(not args.debug))
        baseline_model = ViTClassifier(base_encoder, num_classes=2).to(device)
        baseline_model = train_classifier_with_early_stopping(
            baseline_model, frac_loader, val_loader,
            epochs=args.benchmark_epochs, lr=args.finetune_lr,
            weight_decay=args.weight_decay, patience=max(3, args.early_stopping_patience - 2),
            device=device, desc=f"Baseline ViT ({frac_pct}%)"
        )
        base_res = evaluate_test_set(baseline_model, test_loader, device)
        benchmark_records.append({
            "Model": "Baseline ViT",
            "Fraction": frac,
            "Label_Percentage": f"{frac_pct}%",
            "Samples": len(frac_indices),
            "Accuracy": round(base_res["acc"], 4),
            "Precision": round(base_res["prec"], 4),
            "Recall": round(base_res["rec"], 4),
            "F1_Score": round(base_res["f1"], 4),
            "ROC_AUC": round(base_res["auc"], 4)
        })

        # 2. Contrastive ViT: Fine-tuned from self-supervised encoder weights
        print(f"  [2/2] Training Contrastive ViT on {frac_pct}% subset...")
        cl_encoder = create_encoder_backbone(args.model_name, pretrained=False)
        cl_encoder.load_state_dict(torch.load(pretrained_encoder_path, map_location=device))
        cl_model = ViTClassifier(cl_encoder, num_classes=2).to(device)
        cl_model = train_classifier_with_early_stopping(
            cl_model, frac_loader, val_loader,
            epochs=args.benchmark_epochs, lr=args.finetune_lr,
            weight_decay=args.weight_decay, patience=max(3, args.early_stopping_patience - 2),
            device=device, desc=f"Contrastive ViT ({frac_pct}%)"
        )
        cl_res = evaluate_test_set(cl_model, test_loader, device)
        benchmark_records.append({
            "Model": "Contrastive ViT",
            "Fraction": frac,
            "Label_Percentage": f"{frac_pct}%",
            "Samples": len(frac_indices),
            "Accuracy": round(cl_res["acc"], 4),
            "Precision": round(cl_res["prec"], 4),
            "Recall": round(cl_res["rec"], 4),
            "F1_Score": round(cl_res["f1"], 4),
            "ROC_AUC": round(cl_res["auc"], 4)
        })

        print(f"  [Results at {frac_pct}% Data] "
              f"Baseline F1: {base_res['f1']:.4f} (AUC: {base_res['auc']:.4f}) | "
              f"Contrastive F1: {cl_res['f1']:.4f} (AUC: {cl_res['auc']:.4f})")

    # --------------------------------------------------------------------------
    # STAGE 4: EXPORT RESULTS, TABLES & COMPARATIVE PLOTS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 4: EXPORTING ARTIFACTS & RESEARCH PLOTS")
    print("=" * 80)

    df_bench = pd.DataFrame(benchmark_records)
    csv_path = results_dir / "benchmark_summary.csv"
    df_bench.to_csv(csv_path, index=False)
    print(f"Saved Benchmark Summary Table: {csv_path}")
    print("\n" + df_bench.to_string(index=False))

    # Plot F1-Score Efficiency Curve
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df_bench, x="Label_Percentage", y="F1_Score", hue="Model", marker="o", linewidth=2.5)
    plt.title("Data Efficiency: F1-Score vs. Available Labeled Data", fontsize=13, fontweight="bold")
    plt.xlabel("Labeled Training Fraction", fontsize=11)
    plt.ylabel("Test F1-Score", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(results_dir / "f1_efficiency_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot Accuracy Efficiency Curve
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df_bench, x="Label_Percentage", y="Accuracy", hue="Model", marker="s", linewidth=2.5)
    plt.title("Data Efficiency: Accuracy vs. Available Labeled Data", fontsize=13, fontweight="bold")
    plt.xlabel("Labeled Training Fraction", fontsize=11)
    plt.ylabel("Test Accuracy", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(results_dir / "accuracy_efficiency_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot Final Confusion Matrix (for 100% model)
    plt.figure(figsize=(5, 4))
    sns.heatmap(metrics_100["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative (Normal)", "Positive (Tumor)"],
                yticklabels=["Negative (Normal)", "Positive (Tumor)"])
    plt.title("Confusion Matrix (100% Fine-Tuned ViT)", fontsize=11, fontweight="bold")
    plt.ylabel("True Diagnosis")
    plt.xlabel("Predicted Diagnosis")
    plt.savefig(results_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(metrics_100["targets"], metrics_100["probs"])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {metrics_100['auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("ROC Characteristic Curve (100% ViT)", fontsize=11, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(results_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Generated and saved all comparative visual curves to: {results_dir}")
    print("\nPipeline execution complete successfully!")


# ==============================================================================
# 7. COMMAND-LINE INTERFACE (CLI)
# ==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run end-to-end Contrastive ViT pipeline for PatchCamelyon classification."
    )
    parser.add_argument("--data_dir", type=str, default="/kaggle/input",
                        help="Root directory where PCam HDF5 files or datasets are mounted.")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/pipeline_output",
                        help="Directory to save checkpoints, CSV summaries, and plot artifacts.")
    parser.add_argument("--model_name", type=str, default="vit_tiny_patch16_224",
                        help="Timm ViT model architecture (e.g. vit_tiny_patch16_224 or vit_base_patch16_224).")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Spatial resolution for ViT patch tokenization (default: 224).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--canonical_train_size", type=int, default=16000,
                        help="Total training pool size sampled from full PCam dataset.")
    parser.add_argument("--canonical_val_size", type=int, default=2000,
                        help="Validation set size.")
    parser.add_argument("--canonical_test_size", type=int, default=2000,
                        help="Held-out untouched test set size.")
    parser.add_argument("--contrastive_epochs", type=int, default=30,
                        help="Number of epochs for Stage 1 self-supervised pretraining.")
    parser.add_argument("--contrastive_lr", type=float, default=1e-4,
                        help="Learning rate for contrastive pretraining.")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature hyperparameter for NT-Xent loss.")
    parser.add_argument("--projection_dim", type=int, default=128,
                        help="Output dimensionality of the projection head.")
    parser.add_argument("--finetune_epochs", type=int, default=20,
                        help="Max epochs for Stage 2 full-data fine-tuning.")
    parser.add_argument("--finetune_lr", type=float, default=1e-4,
                        help="Learning rate for supervised fine-tuning.")
    parser.add_argument("--benchmark_epochs", type=int, default=15,
                        help="Max epochs per fraction in Stage 3 label-scarcity benchmark.")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Early stopping patience (epochs without validation loss improvement).")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW optimizer.")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75, 1.00],
                        help="List of label fractions to evaluate in benchmark.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable fast smoke test mode (simulated data, 1 epoch per stage).")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU execution even if CUDA is available.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.debug:
        args.contrastive_epochs = 1
        args.finetune_epochs = 1
        args.benchmark_epochs = 1
        args.early_stopping_patience = 2
        args.batch_size = 8
        args.canonical_train_size = 40
        args.canonical_val_size = 16
        args.canonical_test_size = 16
        args.num_workers = 0  # Avoid Windows spawn pickle overhead in debug

    run_project(args)
