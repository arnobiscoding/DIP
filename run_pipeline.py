#!/usr/bin/env python3
"""
================================================================================
Investigating Contrastive Learning for Vision Transformer-Based Breast
Histopathological Image Classification under Limited Labeled Data
================================================================================
End-to-End Pipeline Script (Advanced Metric Tracking & Scientific Evidence Suite):
  Stage 1: Self-Supervised Contrastive Pretraining (SimCLR + NT-Xent)
  Stage 2: Full-Data Supervised Fine-Tuning with Weak/Patient Early Stopping
  Stage 3: Controlled Label-Scarcity Benchmark (10%, 25%, 50%, 75%, 100%)
           with Zero Prior Supervised Contamination
  Stage 4: Comprehensive Clinical & Statistical Metrics, Prediction Persistence,
           Checkpoints for All Cases, Delta Gain Analysis, and Publication Plots

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
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, precision_recall_curve,
    matthews_corrcoef, brier_score_loss, confusion_matrix, roc_curve
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
    """Synthetic in-memory dataset for local debugging when HDF5 files are not present."""
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
    """Scans data directory to locate train, validation, and test HDF5 files."""
    paths = {}
    print(f"\nResolving PCam HDF5 files under: {data_dir}")

    train_x = list(data_dir.rglob("*training_split.h5")) or list(data_dir.rglob("*train_x.h5"))
    train_y = list(data_dir.rglob("*train_y.h5"))
    if train_x and train_y:
        paths["train_x"] = str(train_x[0])
        paths["train_y"] = str(train_y[0])
        print(f"  -> Detected Train Images: {paths['train_x']}")
        print(f"  -> Detected Train Labels: {paths['train_y']}")

    val_x = list(data_dir.rglob("*validation_split.h5")) or list(data_dir.rglob("*valid_x.h5"))
    val_y = list(data_dir.rglob("*valid_y.h5"))
    if val_x and val_y:
        paths["val_x"] = str(val_x[0])
        paths["val_y"] = str(val_y[0])
        print(f"  -> Detected Val Images  : {paths['val_x']}")
        print(f"  -> Detected Val Labels  : {paths['val_y']}")

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
    """Applies two independent stochastic augmentations to generate positive pairs."""
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
# 4. MODEL ARCHITECTURES & NT-XENT LOSS
# ==============================================================================
class StandaloneViT(nn.Module):
    """Compact pure-PyTorch Vision Transformer backbone fallback if timm is not installed."""
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
    """Vectorized Normalized Temperature-scaled Cross Entropy Loss (NT-Xent / InfoNCE)."""
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0) # [2*B, D]
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        min_val = torch.finfo(sim_matrix.dtype).min
        sim_matrix = sim_matrix.masked_fill(mask, min_val)

        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ], dim=0)

        return self.cross_entropy(sim_matrix, targets)


# ==============================================================================
# 5. COMPREHENSIVE METRIC EVALUATION ENGINE
# ==============================================================================
def evaluate_test_set(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    """
    Computes an exhaustive clinical, statistical, and probabilistic metric suite
    on the held-out test set.
    """
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

    # 1. Standard Classification Metrics
    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    # 2. Confusion Matrix & Clinical Ratios
    cm = confusion_matrix(all_targets, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # 3. Global Discriminative & Probabilistic Scores
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5
    try:
        pr_auc = average_precision_score(all_targets, all_probs)
    except Exception:
        pr_auc = 0.5
    try:
        mcc = matthews_corrcoef(all_targets, all_preds)
    except Exception:
        mcc = 0.0
    try:
        brier = brier_score_loss(all_targets, all_probs)
    except Exception:
        brier = 1.0

    return {
        "acc": acc, "bal_acc": bal_acc, "prec": prec, "rec": rec,
        "spec": spec, "npv": npv, "f1": f1, "macro_f1": macro_f1,
        "auc": auc, "pr_auc": pr_auc, "mcc": mcc, "brier": brier,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "cm": cm, "targets": all_targets, "preds": all_preds, "probs": all_probs
    }


# ==============================================================================
# 6. TRAINING ENGINES WITH RELAXED EARLY STOPPING & SCHEDULERS
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
) -> Tuple[float, float, float]:
    """Evaluates classifier performance on validation set (Loss, Accuracy, F1)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_lbls = [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_lbls.extend(lbls.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = accuracy_score(all_lbls, all_preds)
    avg_f1 = f1_score(all_lbls, all_preds, zero_division=0)
    return avg_loss, avg_acc, avg_f1


def train_classifier_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    monitor_metric: str,
    device: torch.device,
    desc: str = "Classifier"
) -> Tuple[nn.Module, int, int]:
    """
    Trains classifier with AdamW, CosineAnnealingLR, validation evaluation,
    and weakened/relaxed early stopping.
    Returns: (best_model, best_epoch, total_epochs_trained)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # Monitoring logic
    # monitor_metric in ['loss', 'f1', 'acc']
    if monitor_metric == "loss":
        best_score = float("inf")
    else:
        best_score = -float("inf")

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 1
    patience_counter = 0
    best_weights = copy.deepcopy(model.state_dict())
    total_epochs_trained = 0

    print(f"\n--- Training {desc} (Max Epochs: {epochs}, Patience: {patience}, Metric: {monitor_metric.upper()}) ---")
    for epoch in range(1, epochs + 1):
        total_epochs_trained = epoch
        t_start = time.time()
        t_loss, t_acc = train_supervised_epoch(model, train_loader, optimizer, criterion, scaler, device)
        v_loss, v_acc, v_f1 = validate_supervised_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t_start

        # Check improvement with min_delta
        if monitor_metric == "loss":
            improved = (best_score - v_loss) > min_delta
            current_metric = v_loss
        elif monitor_metric == "f1":
            improved = (v_f1 - best_score) > min_delta
            current_metric = v_f1
        else:  # 'acc'
            improved = (v_acc - best_score) > min_delta
            current_metric = v_acc

        if improved:
            best_score = current_metric
            best_val_loss = v_loss
            best_val_acc = v_acc
            best_val_f1 = v_f1
            best_epoch = epoch
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
            tag = "*"
        else:
            patience_counter += 1
            tag = " "

        print(f"  Epoch [{epoch:02d}/{epochs:02d}] {tag} "
              f"Train Loss: {t_loss:.4f} | Acc: {t_acc*100:.2f}% || "
              f"Val Loss: {v_loss:.4f} | Acc: {v_acc*100:.2f}% | F1: {v_f1:.4f} | Time: {elapsed:.1f}s")

        if patience_counter >= patience:
            print(f"  --> Early stopping triggered at Epoch {epoch} (Patience: {patience}). Best at Epoch {best_epoch} (Val F1: {best_val_f1:.4f}, Loss: {best_val_loss:.4f})")
            break

    model.load_state_dict(best_weights)
    return model, best_epoch, total_epochs_trained


# ==============================================================================
# 7. PIPELINE ORCHESTRATION ENGINE
# ==============================================================================
def run_project(args):
    """Executes the full experimental pipeline with extended metric tracking."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    inspect_system(device)

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    pred_dir = results_dir / "predictions"
    for d in [checkpoints_dir, results_dir, pred_dir]:
        d.mkdir(parents=True, exist_ok=True)

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
        eval_tf = EvalTransform(args.image_size)
        full_train_ds = PCamHDF5Dataset(paths["train_x"], paths["train_y"], transform=eval_tf)

        with h5py.File(paths["train_y"], 'r') as fy:
            yk = 'y' if 'y' in fy else list(fy.keys())[0]
            raw_labels = np.asarray(fy[yk][:]).ravel()

        pool_size = min(args.canonical_train_size, len(full_train_ds))
        sss_pool = StratifiedShuffleSplit(n_splits=1, train_size=pool_size, random_state=args.seed)
        raw_train_indices, _ = next(sss_pool.split(np.zeros(len(raw_labels)), raw_labels))
        all_train_labels = raw_labels[raw_train_indices]

        if "val_x" in paths and "val_y" in paths:
            val_full = PCamHDF5Dataset(paths["val_x"], paths["val_y"], transform=eval_tf)
            val_indices = np.arange(min(args.canonical_val_size, len(val_full)))
            val_ds = Subset(val_full, val_indices)
        else:
            val_ds = Subset(full_train_ds, raw_train_indices[:int(pool_size * 0.1)])

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

    cl_start = time.time()
    for epoch in range(1, args.contrastive_epochs + 1):
        e_start = time.time()
        ep_loss = train_contrastive_epoch(contrastive_model, contrastive_loader, cl_optimizer, cl_criterion, cl_scaler, device)
        cl_scheduler.step()
        print(f"Pretrain Epoch [{epoch:02d}/{args.contrastive_epochs:02d}] | NT-Xent Loss: {ep_loss:.4f} | Time: {time.time() - e_start:.1f}s")

    print(f"\nContrastive Pretraining complete in {(time.time() - cl_start)/60:.2f} mins.")

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

    finetune_encoder = create_encoder_backbone(args.model_name, pretrained=False)
    finetune_encoder.load_state_dict(torch.load(pretrained_encoder_path, map_location=device))
    finetuned_model = ViTClassifier(finetune_encoder, num_classes=2).to(device)

    finetuned_model, ft_best_ep, ft_tot_ep = train_classifier_with_early_stopping(
        finetuned_model, train_100_loader, val_loader,
        epochs=args.finetune_epochs, lr=args.finetune_lr,
        weight_decay=args.weight_decay, patience=args.early_stopping_patience,
        min_delta=args.min_delta, monitor_metric=args.early_stopping_metric,
        device=device, desc="Contrastive ViT (100% Reference)"
    )

    best_ft_path = checkpoints_dir / "best_contrastive_finetuned_vit.pth"
    torch.save(finetuned_model.state_dict(), best_ft_path)

    metrics_100 = evaluate_test_set(finetuned_model, test_loader, device)
    print("\n--- 100% Full-Data Reference Test Results ---")
    print(f"Accuracy  : {metrics_100['acc']:.4f} | Bal Acc  : {metrics_100['bal_acc']:.4f}")
    print(f"F1-Score  : {metrics_100['f1']:.4f} | Macro F1 : {metrics_100['macro_f1']:.4f}")
    print(f"ROC-AUC   : {metrics_100['auc']:.4f} | PR-AUC   : {metrics_100['pr_auc']:.4f}")
    print(f"Recall    : {metrics_100['rec']:.4f} | Precision: {metrics_100['prec']:.4f}")
    print(f"Spec      : {metrics_100['spec']:.4f} | NPV      : {metrics_100['npv']:.4f}")
    print(f"MCC       : {metrics_100['mcc']:.4f} | Brier    : {metrics_100['brier']:.4f}")

    # --------------------------------------------------------------------------
    # STAGE 3: CONTROLLED LABEL-SCARCITY BENCHMARK (10%, 25%, 50%, 75%, 100%)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 3: CONTROLLED LABEL-SCARCITY BENCHMARK")
    print("=" * 80)
    print("Protocol: Evaluates Baseline ViT vs. Contrastive ViT across all label fractions.")
    print("          Saves individual checkpoints, raw predictions, and extended metrics for all runs.")

    fractions = [0.25, 1.0] if args.debug else args.fractions
    benchmark_records = []
    roc_data_per_fraction = {}
    pr_data_per_fraction = {}
    confusion_matrices_dict = {}

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

        print(f"\n=======================================================")
        print(f">>> BENCHMARKING FRACTION: {frac_pct}% ({len(frac_indices)} samples) <<<")
        print(f"=======================================================")

        # ----------------------------------------------------------------------
        # 1. Baseline ViT: Trained from fresh ImageNet weights
        # ----------------------------------------------------------------------
        print(f"\n[1/2] Training Baseline ViT on {frac_pct}% subset...")
        t0 = time.time()
        base_encoder = create_encoder_backbone(args.model_name, pretrained=(not args.debug))
        baseline_model = ViTClassifier(base_encoder, num_classes=2).to(device)
        baseline_model, b_best_ep, b_tot_ep = train_classifier_with_early_stopping(
            baseline_model, frac_loader, val_loader,
            epochs=args.benchmark_epochs, lr=args.finetune_lr,
            weight_decay=args.weight_decay, patience=args.early_stopping_patience,
            min_delta=args.min_delta, monitor_metric=args.early_stopping_metric,
            device=device, desc=f"Baseline ViT ({frac_pct}%)"
        )
        base_train_time = time.time() - t0
        base_res = evaluate_test_set(baseline_model, test_loader, device)

        # Save individual checkpoint
        base_ckpt_path = checkpoints_dir / f"baseline_vit_frac_{frac_pct}.pth"
        torch.save(baseline_model.state_dict(), base_ckpt_path)

        # ----------------------------------------------------------------------
        # 2. Contrastive ViT: Fine-tuned from self-supervised encoder weights
        # ----------------------------------------------------------------------
        print(f"\n[2/2] Training Contrastive ViT on {frac_pct}% subset...")
        t1 = time.time()
        cl_encoder = create_encoder_backbone(args.model_name, pretrained=False)
        cl_encoder.load_state_dict(torch.load(pretrained_encoder_path, map_location=device))
        cl_model = ViTClassifier(cl_encoder, num_classes=2).to(device)
        cl_model, cl_best_ep, cl_tot_ep = train_classifier_with_early_stopping(
            cl_model, frac_loader, val_loader,
            epochs=args.benchmark_epochs, lr=args.finetune_lr,
            weight_decay=args.weight_decay, patience=args.early_stopping_patience,
            min_delta=args.min_delta, monitor_metric=args.early_stopping_metric,
            device=device, desc=f"Contrastive ViT ({frac_pct}%)"
        )
        cl_train_time = time.time() - t1
        cl_res = evaluate_test_set(cl_model, test_loader, device)

        # Save individual checkpoint
        cl_ckpt_path = checkpoints_dir / f"contrastive_vit_frac_{frac_pct}.pth"
        torch.save(cl_model.state_dict(), cl_ckpt_path)

        # ----------------------------------------------------------------------
        # 3. Save Raw Test Predictions CSV for this fraction
        # ----------------------------------------------------------------------
        pred_df = pd.DataFrame({
            "sample_idx": np.arange(len(base_res["targets"])),
            "true_label": base_res["targets"],
            "baseline_prob": base_res["probs"],
            "baseline_pred": base_res["preds"],
            "contrastive_prob": cl_res["probs"],
            "contrastive_pred": cl_res["preds"]
        })
        pred_csv_path = pred_dir / f"predictions_frac_{frac_pct}.csv"
        pred_df.to_csv(pred_csv_path, index=False)

        # Save ROC and PR curve data
        fpr_b, tpr_b, _ = roc_curve(base_res["targets"], base_res["probs"])
        fpr_c, tpr_c, _ = roc_curve(cl_res["targets"], cl_res["probs"])
        prec_b, rec_b, _ = precision_recall_curve(base_res["targets"], base_res["probs"])
        prec_c, rec_c, _ = precision_recall_curve(cl_res["targets"], cl_res["probs"])

        roc_data_per_fraction[frac_pct] = {"b": (fpr_b, tpr_b, base_res["auc"]), "c": (fpr_c, tpr_c, cl_res["auc"])}
        pr_data_per_fraction[frac_pct]  = {"b": (rec_b, prec_b, base_res["pr_auc"]), "c": (rec_c, prec_c, cl_res["pr_auc"])}
        confusion_matrices_dict[frac_pct] = {"b": base_res["cm"], "c": cl_res["cm"]}

        # ----------------------------------------------------------------------
        # 4. Record Metrics & Delta Gains
        # ----------------------------------------------------------------------
        delta_f1 = cl_res["f1"] - base_res["f1"]
        delta_auc = cl_res["auc"] - base_res["auc"]
        delta_rec = cl_res["rec"] - base_res["rec"]
        delta_acc = cl_res["acc"] - base_res["acc"]
        rel_f1_gain = (delta_f1 / base_res["f1"] * 100) if base_res["f1"] > 0 else 0.0

        benchmark_records.append({
            "Fraction": frac,
            "Label_Percentage": f"{frac_pct}%",
            "Samples": len(frac_indices),
            # Baseline
            "Baseline_Acc": round(base_res["acc"], 4),
            "Baseline_BalAcc": round(base_res["bal_acc"], 4),
            "Baseline_F1": round(base_res["f1"], 4),
            "Baseline_MacroF1": round(base_res["macro_f1"], 4),
            "Baseline_AUC": round(base_res["auc"], 4),
            "Baseline_PRAUC": round(base_res["pr_auc"], 4),
            "Baseline_Recall": round(base_res["rec"], 4),
            "Baseline_Precision": round(base_res["prec"], 4),
            "Baseline_Specificity": round(base_res["spec"], 4),
            "Baseline_MCC": round(base_res["mcc"], 4),
            "Baseline_Brier": round(base_res["brier"], 4),
            "Baseline_BestEpoch": b_best_ep,
            "Baseline_TotalEpochs": b_tot_ep,
            "Baseline_TimeSec": round(base_train_time, 1),
            # Contrastive
            "Contrastive_Acc": round(cl_res["acc"], 4),
            "Contrastive_BalAcc": round(cl_res["bal_acc"], 4),
            "Contrastive_F1": round(cl_res["f1"], 4),
            "Contrastive_MacroF1": round(cl_res["macro_f1"], 4),
            "Contrastive_AUC": round(cl_res["auc"], 4),
            "Contrastive_PRAUC": round(cl_res["pr_auc"], 4),
            "Contrastive_Recall": round(cl_res["rec"], 4),
            "Contrastive_Precision": round(cl_res["prec"], 4),
            "Contrastive_Specificity": round(cl_res["spec"], 4),
            "Contrastive_MCC": round(cl_res["mcc"], 4),
            "Contrastive_Brier": round(cl_res["brier"], 4),
            "Contrastive_BestEpoch": cl_best_ep,
            "Contrastive_TotalEpochs": cl_tot_ep,
            "Contrastive_TimeSec": round(cl_train_time, 1),
            # Contrastive Advantages (Deltas)
            "Delta_F1": round(delta_f1, 4),
            "Delta_AUC": round(delta_auc, 4),
            "Delta_Recall": round(delta_rec, 4),
            "Delta_Accuracy": round(delta_acc, 4),
            "Rel_F1_Gain_Pct": round(rel_f1_gain, 2)
        })

        print(f"\n>>> [Summary at {frac_pct}% Labeled Data] <<<")
        print(f"  • F1-Score   : Baseline = {base_res['f1']:.4f} | Contrastive = {cl_res['f1']:.4f} (Delta: {delta_f1:+.4f})")
        print(f"  • ROC-AUC    : Baseline = {base_res['auc']:.4f} | Contrastive = {cl_res['auc']:.4f} (Delta: {delta_auc:+.4f})")
        print(f"  • Recall/Sens: Baseline = {base_res['rec']:.4f} | Contrastive = {cl_res['rec']:.4f} (Delta: {delta_rec:+.4f})")
        print(f"  • Accuracy   : Baseline = {base_res['acc']:.4f} | Contrastive = {cl_res['acc']:.4f} (Delta: {delta_acc:+.4f})")

    # --------------------------------------------------------------------------
    # STAGE 4: EXPORT RICH ARTIFACTS & PUBLICATION-GRADE PLOTS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 4: EXPORTING SUMMARY TABLE & COMPARATIVE VISUALIZATIONS")
    print("=" * 80)

    df_bench = pd.DataFrame(benchmark_records)
    csv_path = results_dir / "benchmark_summary.csv"
    df_bench.to_csv(csv_path, index=False)
    print(f"Saved Comprehensive Benchmark Summary Table: {csv_path}")

    # Also save a melted tidy version for easy seaborn plotting
    tidy_rows = []
    for r in benchmark_records:
        tidy_rows.append({
            "Label_Percentage": r["Label_Percentage"], "Fraction": r["Fraction"], "Samples": r["Samples"],
            "Model": "Baseline ViT", "Accuracy": r["Baseline_Acc"], "F1_Score": r["Baseline_F1"],
            "ROC_AUC": r["Baseline_AUC"], "Recall": r["Baseline_Recall"], "Precision": r["Baseline_Precision"],
            "PR_AUC": r["Baseline_PRAUC"], "MCC": r["Baseline_MCC"]
        })
        tidy_rows.append({
            "Label_Percentage": r["Label_Percentage"], "Fraction": r["Fraction"], "Samples": r["Samples"],
            "Model": "Contrastive ViT", "Accuracy": r["Contrastive_Acc"], "F1_Score": r["Contrastive_F1"],
            "ROC_AUC": r["Contrastive_AUC"], "Recall": r["Contrastive_Recall"], "Precision": r["Contrastive_Precision"],
            "PR_AUC": r["Contrastive_PRAUC"], "MCC": r["Contrastive_MCC"]
        })
    df_tidy = pd.DataFrame(tidy_rows)
    df_tidy.to_csv(results_dir / "benchmark_tidy_metrics.csv", index=False)

    # 1. Multi-Metric 4-Panel Efficiency Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_configs = [
        ("F1_Score", "Test F1-Score (Macro)", axes[0, 0]),
        ("ROC_AUC", "Area Under ROC Curve (AUC)", axes[0, 1]),
        ("Recall", "Sensitivity / Recall (Tumor Detection)", axes[1, 0]),
        ("Accuracy", "Overall Diagnostic Accuracy", axes[1, 1])
    ]
    for col_name, title_name, ax in metric_configs:
        sns.lineplot(
            data=df_tidy, x="Label_Percentage", y=col_name, hue="Model",
            marker="o", linewidth=2.5, markersize=8, ax=ax,
            palette={"Baseline ViT": "#d95f02", "Contrastive ViT": "#1b9e77"}
        )
        ax.set_title(title_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Available Labeled Training Data", fontsize=10)
        ax.set_ylabel(col_name, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(results_dir / "multi_metric_efficiency.png", dpi=300)
    plt.close()

    # 2. Contrastive Advantage Delta Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    delta_cols = ["Delta_F1", "Delta_AUC", "Delta_Recall", "Delta_Accuracy"]
    df_melt_delta = df_bench.melt(
        id_vars=["Label_Percentage"], value_vars=delta_cols,
        var_name="Metric_Delta", value_name="Advantage"
    )
    sns.barplot(
        data=df_melt_delta, x="Label_Percentage", y="Advantage", hue="Metric_Delta",
        palette="viridis", ax=ax
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Contrastive Advantage (Margin over Baseline ViT)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Labeled Training Fraction", fontsize=11)
    ax.set_ylabel("Margin (Contrastive - Baseline)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(results_dir / "contrastive_advantage_deltas.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Comparative ROC Curves per Fraction
    n_fracs = len(roc_data_per_fraction)
    fig, axes = plt.subplots(1, n_fracs, figsize=(5 * n_fracs, 4.5), squeeze=False)
    for i, (frac_pct, curves) in enumerate(roc_data_per_fraction.items()):
        ax = axes[0, i]
        ax.plot(curves["b"][0], curves["b"][1], label=f"Baseline (AUC={curves['b'][2]:.3f})", color="#d95f02", lw=2)
        ax.plot(curves["c"][0], curves["c"][1], label=f"Contrastive (AUC={curves['c'][2]:.3f})", color="#1b9e77", lw=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(f"ROC Curve ({frac_pct}% Data)", fontsize=11, fontweight="bold")
        ax.set_xlabel("FPR (1 - Specificity)")
        ax.set_ylabel("TPR (Recall)")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / "comparative_roc_curves.png", dpi=300)
    plt.close()

    # 4. Comparative Precision-Recall Curves per Fraction
    fig, axes = plt.subplots(1, n_fracs, figsize=(5 * n_fracs, 4.5), squeeze=False)
    for i, (frac_pct, curves) in enumerate(pr_data_per_fraction.items()):
        ax = axes[0, i]
        ax.plot(curves["b"][0], curves["b"][1], label=f"Baseline (PR-AUC={curves['b'][2]:.3f})", color="#d95f02", lw=2)
        ax.plot(curves["c"][0], curves["c"][1], label=f"Contrastive (PR-AUC={curves['c'][2]:.3f})", color="#1b9e77", lw=2)
        ax.set_title(f"PR Curve ({frac_pct}% Data)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / "comparative_pr_curves.png", dpi=300)
    plt.close()

    # 5. Confusion Matrices Grid
    fig, axes = plt.subplots(n_fracs, 2, figsize=(8, 3.5 * n_fracs), squeeze=False)
    for i, (frac_pct, cms) in enumerate(confusion_matrices_dict.items()):
        # Baseline
        sns.heatmap(cms["b"], annot=True, fmt="d", cmap="Oranges", cbar=False, ax=axes[i, 0],
                    xticklabels=["Normal", "Tumor"], yticklabels=["Normal", "Tumor"])
        axes[i, 0].set_title(f"Baseline ViT ({frac_pct}% Data)", fontsize=10, fontweight="bold")
        axes[i, 0].set_ylabel("True Diagnosis")
        # Contrastive
        sns.heatmap(cms["c"], annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[i, 1],
                    xticklabels=["Normal", "Tumor"], yticklabels=["Normal", "Tumor"])
        axes[i, 1].set_title(f"Contrastive ViT ({frac_pct}% Data)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrices_grid.png", dpi=300)
    plt.close()

    print(f"All 5 publication plots exported to: {results_dir}")
    print("Individual checkpoints and prediction CSVs saved.")
    print("\nPipeline execution complete successfully!")


# ==============================================================================
# 8. COMMAND-LINE INTERFACE (CLI)
# ==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run end-to-end Contrastive ViT pipeline with exhaustive metrics & relaxed early stopping."
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
    parser.add_argument("--finetune_epochs", type=int, default=25,
                        help="Max epochs for Stage 2 full-data fine-tuning.")
    parser.add_argument("--finetune_lr", type=float, default=1e-4,
                        help="Learning rate for supervised fine-tuning.")
    parser.add_argument("--benchmark_epochs", type=int, default=20,
                        help="Max epochs per fraction in Stage 3 label-scarcity benchmark.")
    parser.add_argument("--early_stopping_patience", "--patience", dest="early_stopping_patience", type=int, default=10,
                        help="Early stopping patience (epochs without validation improvement). Default: 10.")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                        help="Minimum change threshold in validation metric to qualify as improvement.")
    parser.add_argument("--early_stopping_metric", type=str, default="loss", choices=["loss", "f1", "acc"],
                        help="Metric to monitor for early stopping ('loss', 'f1', or 'acc'). Default: 'loss'.")
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
        args.num_workers = 0

    run_project(args)
