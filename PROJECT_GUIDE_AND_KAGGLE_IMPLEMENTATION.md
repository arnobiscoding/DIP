# Project Guide: Investigating Contrastive Learning for Vision Transformer-Based Breast Histopathological Image Classification under Limited Labeled Data

---

## 1. Project Overview & Theoretical Foundation

### 1.1 Project at a Glance
Based on the research proposal and detailed methodology specifications:

| Component | Project Specification |
| :--- | :--- |
| **Domain** | Digital Image Processing (DIP) / Medical Image Analysis / Breast Histopathology |
| **Core Models** | Vision Transformer (ViT, e.g. Pretrained ViT-B/16 or ViT-Tiny) + Contrastive Learning (SimCLR) |
| **Dataset** | PatchCamelyon (PCam) — 327,680 histopathological image patches (96×96 RGB) |
| **Classification Task** | Binary Classification: Metastatic Tumor Tissue (Positive) vs. Normal Tissue (Negative) |
| **Main Comparison** | Standard Fine-Tuned ViT Baseline vs. Contrastive-Learning-Enhanced ViT |
| **Primary Variable** | Amount of Labeled Training Data: **100%, 50%, 25%, 10%** (and optional 75%) |
| **Evaluation Metrics** | Accuracy, Precision, Recall / Sensitivity, F1-Score, ROC-AUC, Confusion Matrix |
| **Target Implementation** | PyTorch on Kaggle GPU Environment (NVIDIA Tesla T4 or P100) |

---

### 1.2 Motivation & Clinical Context
* **The Histopathology Challenge:** Histopathology images contain rich morphological information essential for breast cancer diagnosis (e.g., detecting metastatic tissue in sentinel lymph nodes). However, whole-slide and patch-level images exhibit large variability due to tissue textures, nuclear pleomorphism, chemical staining protocols (Hematoxylin & Eosin), scanner magnifications, and local artifacts.
* **The Annotation Bottleneck:** High-quality pathological annotation requires exhaustive, manual evaluation by expert pathologists, making labeled medical datasets exceptionally costly and scarce.
* **Vision Transformers vs. Data Hunger:** While Vision Transformers (ViTs) excel at modeling long-range spatial relationships through multi-head self-attention, they lack the inductive biases (translation equivariance and locality) inherent to CNNs. Consequently, training ViTs under limited labeled data typically leads to severe overfitting or poor generalization.
* **Why Contrastive Learning?** Self-supervised contrastive learning (such as SimCLR) optimizes representations without ground-truth labels by mapping augmented views of the same image close together while pushing distinct image representations apart. By pretraining a ViT encoder on unlabeled histopathology patches, the model learns robust, stain- and rotation-invariant feature representations before fine-tuning on limited labeled data.

---

### 1.3 Research Questions & Gap Analysis
* **Main Research Question:**  
  *Can self-supervised contrastive pretraining improve the classification performance and label efficiency of a Vision Transformer for breast-cancer histopathological image classification?*
* **Supporting Research Questions:**
  1. How does a standard pretrained ViT perform on PCam when trained solely with supervised cross-entropy?
  2. How much does contrastive pretraining improve ViT performance when labeled training data is systematically reduced (100% $\to$ 50% $\to$ 25% $\to$ 10%)?
  3. Which evaluation metrics (e.g., Recall/Sensitivity vs. Specificity/Precision) benefit most from contrastive pretraining under severe label scarcity?
* **Working Research Gap:**  
  While contrastive learning and ViTs have been explored in medical imaging, existing studies evaluate disparate architectures, private datasets, or varied training protocols. This project establishes a **controlled, reproducible empirical benchmark** on the public PCam benchmark, holding splits, architectures, and evaluation protocols strictly identical to isolate the true effect of contrastive representation learning across label ratios.

---

### 1.4 Background Literature Reference
The project methodology positions itself relative to key recent literature:
1. **Shiri, Reddy & Sun (2024)** — *Supervised Contrastive Vision Transformer for Breast Histopathological Image Classification* (arXiv:2404.11052). Demonstrates contrastive ViT resilience on breast histopathology; acts as the primary methodological reference.
2. **Yao et al. (2025)** — *EVA-X: a foundation model for general chest X-ray analysis with self-supervised learning* (npj Digital Medicine). Highlights label-efficient representation learning combining contrastive objectives and masked image modeling.
3. **Rui et al. (2025)** — *Multi-modal Vision Pre-training for Medical Image Analysis* (CVPR 2025). Shows how contrastive pretraining captures invariant representations across imaging variations.
4. **He & Li (2025)** — *Vector Contrastive Learning for Pixel-Wise Pretraining in Medical Vision* (ICCV 2025). Addresses feature over-dispersion in medical contrastive objectives.
5. **Tastan et al. (2026)** — *MultiPersistence Topological Fusion with Vision Transformers for Skin Cancer Detection* (MIDL 2026). Confirms that ViT + contrastive learning remains a state-of-the-art paradigm in medical vision.

---

## 2. Dataset Specifications: PatchCamelyon (PCam)

### 2.1 Overview & Key Characteristics
* **Source:** Derived from the CAMELYON16 Challenge (lymph node histopathology scans).
* **Total Image Count:** 327,680 color patches.
* **Resolution:** $96 \times 96$ pixels, RGB (3 channels), stored as 8-bit unsigned integers (`uint8`) in HDF5 format.
* **Binary Task Definition:**
  * **Class 1 (Positive / Tumor):** Contains at least one pixel of metastatic tumor tissue in the central $32 \times 32$ pixel area.
  * **Class 0 (Negative / Non-Tumor):** No tumor cells within the central $32 \times 32$ region (may contain normal lymph node tissue, fat, stroma, or background).
  * *Note:* Tumor tissue outside the center $32 \times 32$ region does not influence the label, which makes the task clinically challenging and context-dependent.
* **Class Balance:** Perfectly balanced (50% positive, 50% negative) across official splits.

### 2.2 Official Splits & Data Leakage Prevention Rules
PCam defines official splits with **zero whole-slide image (WSI) overlap** between training, validation, and test splits:
* **Train Split:** 262,144 patches
* **Validation Split:** 32,768 patches
* **Test Split:** 32,768 patches

> [!IMPORTANT]
> **Cardinal Rules for Reliable Research:**
> 1. **Zero Data Leakage:** Never use validation or test images during contrastive pretraining or supervised training. The official test split must remain untouched until final evaluation.
> 2. **Identical Subsets:** In label-scarcity experiments (10%, 25%, 50%, 100%), the **exact same stratified random subsets** must be used for both the baseline ViT and the contrastive-enhanced ViT.
> 3. **Deterministic Evaluation:** All data augmentations (flips, crops, rotations) must be applied **only** to training streams. Validation and test evaluation must use purely deterministic preprocessing (resize + normalization).

---

## 3. End-to-End Methodology

The project follows a modular, 8-stage pipeline:

```
                      PCam HDF5 Dataset
              (Train / Validation / Test Splits)
                              │
                    [Stage 1: Preparation]
             Load HDF5, Verify Counts & Balances
                              │
                   [Stage 2: Preprocessing]
             Resize (224×224) + Normalization
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
  [Stage 4: ViT Baseline]               [Stage 3: Contrastive Augmentations]
  Pretrained ViT-B/16 / Tiny            Two-View Positive Pairs (View 1 & View 2)
  Direct Supervised Cross-Entropy                  │
  Trained on Labeled Subset                        ▼
         │                              [Stage 5: Contrastive Pretraining]
         │                              Shared ViT Encoder + MLP Projector
         │                              NT-Xent (InfoNCE) Loss (No Labels)
         │                                         │
         │                              [Stage 6: Fine-Tuning Stage]
         │                              Discard Projector, Add Linear Head
         │                              Fine-tune on Labeled Subset
         │                                         │
         └────────────────────┬────────────────────┘
                              │
                              ▼
            [Stage 7: Limited-Labeled-Data Experiments]
             Evaluate at 100%, 50%, 25%, 10% Labeled Data
                              │
                              ▼
                  [Stage 8: Final Evaluation]
        Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
                              │
                              ▼
            [Optional: Stage 9 - Ablation Study]
             Augmentation Set A vs. Augmentation Set B
```

---

### Stage 1: Dataset Preparation & Inspection
* Load PCam HDF5 files (`camelyonpatch_level_2_split_train_x.h5`, `train_y.h5`, etc.).
* Inspect image dimensions ($96 \times 96 \times 3$) and label distributions ($0$ and $1$).
* Display representative positive and negative patches to confirm visual integrity.
* Establish stratified sampling seeds to ensure 100% reproducibility.

### Stage 2: Digital Image Preprocessing
* Convert images from `uint8` arrays to PyTorch tensors.
* **Resize:** Upsample from $96 \times 96$ to the required ViT input resolution ($224 \times 224$) using bicubic or bilinear interpolation.
* **Normalization:** Standardize pixel values using ImageNet channel statistics:
  * Mean: $[0.485, 0.456, 0.406]$
  * Std: $[0.229, 0.224, 0.225]$

### Stage 3: Data Augmentation for Contrastive Learning
For each training image $x$, generate two stochastic views $(\tilde{x}_i, \tilde{x}_j)$ representing a positive pair:
* **Random Resized Crop:** Scale $(0.6 \text{ to } 1.0)$ resized to $224 \times 224$ (simulates scale and field-of-view shifts).
* **Horizontal & Vertical Flips:** $p = 0.5$ (tissue slices have no canonical top/bottom or left/right orientation).
* **Random Rotation:** $90^\circ, 180^\circ, 270^\circ$ or mild continuous rotation ($\pm 15^\circ$).
* **Color Jitter:** Mild brightness, contrast, and saturation variations ($\pm 0.2$) to mimic laboratory H&E staining variations.
* *Constraint:* Avoid aggressive distortions (e.g., severe solarization, color inversion, heavy blur) that destroy chromatin textures or alter tissue morphology.

### Stage 4: ViT Baseline Model (Supervised)
* **Backbone:** Standard pretrained Vision Transformer from `torchvision` or `timm` (e.g., `vit_tiny_patch16_224` or `vit_base_patch16_224`).
* **Classification Head:** Replace the original ImageNet 1,000-class head with a linear layer outputting 2 logits.
* **Loss Function:** Standard Cross-Entropy Loss.
* **Optimizer:** AdamW with cosine learning rate decay and weight decay $10^{-4}$.
* **Milestone:** Establish stable baseline validation curves and record baseline benchmark metrics.

### Stage 5: Self-Supervised Contrastive Pretraining (SimCLR)
* **Architecture:**
  * **Shared ViT Encoder $f(\cdot)$:** Processes augmented views $\tilde{x}_i, \tilde{x}_j$ to yield feature representations $h_i, h_j \in \mathbb{R}^D$ (the ViT `[CLS]` token embedding).
  * **Projection Head $g(\cdot)$:** A small 2-layer MLP (e.g., Linear $D \to 512 \to$ ReLU $\to$ BatchNorm $\to$ Linear $512 \to 128$) mapping $h$ to latent representations $z_i, z_j \in \mathbb{R}^{128}$.
* **Loss Formulation:** Normalized Temperature-scaled Cross Entropy ($\text{NT-Xent}$ / InfoNCE):
  $$\ell_{i,j} = -\log \frac{\exp\left(\text{sim}(z_i, z_j) / \tau\right)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq i]} \exp\left(\text{sim}(z_i, z_k) / \tau\right)}$$
  where $\text{sim}(u, v) = \frac{u^\top v}{\|u\|_2 \|v\|_2}$ and $\tau = 0.5$.
* **Note:** Ground truth tumor labels are **strictly omitted** during this stage.

### Stage 6: Supervised Fine-Tuning of Contrastive ViT
* Discard the projection head $g(\cdot)$.
* Attach a new linear classifier head to the frozen or unfrozen ViT encoder $f(\cdot)$.
* Train the classifier using Cross-Entropy Loss on labeled PCam subsets.
* Use a lower learning rate for the encoder ($10^{-5}$) and a standard rate for the classification head ($10^{-4}$).

### Stage 7: Controlled Label-Scarcity Experiments
Vary only the proportion of labeled training data while keeping the test set and evaluation code identical:
* **100% Labeled Data:** Upper performance bound.
* **50% Labeled Data:** Moderate label reduction.
* **25% Labeled Data:** Strong label scarcity.
* **10% Labeled Data:** Extreme low-label regime.
* *Protocol:* Subsets are sampled with stratified sampling (preserving 50/50 balance) and fixed random seeds. The **exact same indices** are provided to both the baseline and contrastive models.

### Stage 8: Evaluation & Comparative Analysis
Calculate comprehensive metrics on the untouched test split:
* **Accuracy:** Overall correctness across both classes.
* **Precision:** Fraction of positive predictions that are true metastatic tumor tissue.
* **Recall / Sensitivity:** Fraction of actual tumor cases correctly identified (critical clinical metric to minimize false negatives).
* **F1-Score:** Harmonic mean of precision and recall.
* **ROC-AUC:** Area Under the Receiver Operating Characteristic Curve (threshold-independent discriminative power).
* **Confusion Matrix:** True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).

---

## 4. Recommended Algorithm Stack

| Pipeline Stage | Recommended Technique | Purpose | Project Specification |
| :--- | :--- | :--- | :--- |
| **Data Ingestion** | `h5py` + PyTorch `Dataset` / `DataLoader` | Memory-efficient on-demand patch loading | Lazy file-handle opening per worker |
| **Preprocessing** | Bicubic Resize + Channel Normalization | Match ViT input specifications | $224 \times 224 \times 3$, ImageNet mean/std |
| **Augmentation** | RandomResizedCrop + Flips + Rotation + ColorJitter | Generate positive pairs without distortion | Medically sensible invariances |
| **Encoder Backbone** | Vision Transformer (`vit_tiny_patch16_224` or `vit_base_patch16_224`) | Global multi-head self-attention | Pretrained on ImageNet, patch size 16 |
| **Contrastive Objective** | SimCLR framework with NT-Xent / InfoNCE Loss | Unsupervised representation learning | Temperature $\tau = 0.5$, batch size 32–64 |
| **Projection Head** | 2-layer MLP (Input $\to$ 512 $\to$ ReLU $\to$ 128) | Isolate invariant features | L2-normalized outputs |
| **Fine-Tuning** | Linear Probe / End-to-End Fine-Tuning with Cross-Entropy | Downstream binary classification | 2 output logits |
| **Optimization** | AdamW with Cosine Annealing Learning Rate | Stable convergence and weight regularization | Learning rate $10^{-4}$, weight decay $10^{-4}$ |
| **Subsampling** | Stratified Random Subsampling | Controlled label-scarcity study | 100%, 50%, 25%, 10% fractions |

---

## 5. Kaggle Implementation Guide (Step-by-Step)

### 5.1 Kaggle Hardware & Environment Setup
1. **Create Notebook:** Start a new Kaggle notebook with Python 3 environment.
2. **GPU Selection:** In the right-hand panel under **Settings $\to$ Accelerator**, choose:
   * **GPU T4 x2** or **GPU P100** (provides 16 GB VRAM).
3. **Internet Access:** Set **Internet** to **On** (required to load initial weights from `timm`).
4. **Attach Dataset:**
   * Click **Add Data** in the top right.
   * Search for `pcam-dataset` (or upload the PCam HDF5 files to Kaggle Datasets).
   * Verify the dataset is mounted under `/kaggle/input/` (e.g. `/kaggle/input/pcam-dataset/`).

---

### 5.2 Complete Implementation Code

Below is the complete, self-contained, modular code designed to run smoothly on Kaggle without out-of-memory errors or worker deadlocks.

```python
# ==============================================================================
# 1. ENVIRONMENT CONFIGURATION & IMPORTS
# ==============================================================================
import os
import gc
import json
import random
import time
from pathlib import Path

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
from torchvision import transforms

import timm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# Fix random seeds for 100% reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

# Global Configuration Parameters
CONFIG = {
    "seed": 42,
    "model_name": "vit_tiny_patch16_224",  # Use 'vit_base_patch16_224' if VRAM permits
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 2,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "temperature": 0.5,
    "contrastive_epochs": 15,
    "finetune_epochs": 10,
    "label_fractions": [1.0, 0.50, 0.25, 0.10],
    "output_dir": "/kaggle/working/pcam_vit_results"
}

output_dir = Path(CONFIG["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)
```

```python
# ==============================================================================
# 2. MEMORY-SAFE LAZY HDF5 DATASET CLASS
# ==============================================================================
class PCamHDF5Dataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for PCam HDF5 files.
    Opens HDF5 file pointers on-demand per worker process to prevent deadlocks.
    """
    def __init__(self, x_path, y_path, transform=None):
        self.x_path = str(x_path)
        self.y_path = str(y_path)
        self.transform = transform
        
        # Read dataset length using a temporary context
        with h5py.File(self.x_path, 'r') as fx:
            key_x = 'x' if 'x' in fx else list(fx.keys())[0]
            self.length = fx[key_x].shape[0]
            
        with h5py.File(self.y_path, 'r') as fy:
            self.key_y = 'y' if 'y' in fy else list(fy.keys())[0]

        self.fx = None
        self.fy = None
        self.key_x = None

    def _init_db(self):
        self.fx = h5py.File(self.x_path, 'r')
        self.fy = h5py.File(self.y_path, 'r')
        self.key_x = 'x' if 'x' in self.fx else list(self.fx.keys())[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.fx is None:
            self._init_db()

        # Extract 96x96x3 uint8 array
        img_np = self.fx[self.key_x][idx]
        label = int(np.squeeze(self.fy[self.key_y][idx]))

        img_pil = Image.fromarray(img_np)
        if self.transform is not None:
            img = self.transform(img_pil)
        else:
            img = transforms.ToTensor()(img_pil)

        return img, torch.tensor(label, dtype=torch.long)
```

```python
# ==============================================================================
# 3. TRANSFORMS & DATA LOADERS
# ==============================================================================
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

# Deterministic evaluation transform (Validation & Test)
eval_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Standard supervised training transform
train_supervised_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Contrastive two-view generator (SimCLR)
class ContrastiveTwoViewTransform:
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)
```

```python
# ==============================================================================
# 4. CONTRASTIVE ViT MODEL & NT-XENT LOSS
# ==============================================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class SimCLRViT(nn.Module):
    def __init__(self, model_name=CONFIG["model_name"], pretrained=True, proj_dim=128):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.projector = ProjectionHead(self.encoder.num_features, 512, proj_dim)

    def forward(self, x):
        feat = self.encoder(x)
        proj = F.normalize(self.projector(feat), dim=-1)
        return feat, proj

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss for SimCLR."""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        reps = torch.cat([z_i, z_j], dim=0) # [2*B, D]
        sim_mat = torch.matmul(reps, reps.T) / self.temperature

        # Mask self-similarity entries on the diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        sim_mat.masked_fill_(mask, -1e9)

        # Ground truth pairs: i -> i+B, i+B -> i
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z_i.device),
            torch.arange(0, batch_size, device=z_i.device)
        ], dim=0)

        return F.cross_entropy(sim_mat, targets)
```

```python
# ==============================================================================
# 5. SUPERVISED CLASSIFIER ASSEMBLY
# ==============================================================================
class ViTClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.num_features, num_classes)
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat)

def build_baseline_vit(model_name=CONFIG["model_name"]):
    base = timm.create_model(model_name, pretrained=True, num_classes=0)
    return ViTClassifier(base, num_classes=2).to(device)

def build_contrastive_vit(pretrained_encoder_path, model_name=CONFIG["model_name"]):
    base = timm.create_model(model_name, pretrained=False, num_classes=0)
    ckpt = torch.load(pretrained_encoder_path, map_location=device)
    base.load_state_dict(ckpt, strict=False)
    return ViTClassifier(base, num_classes=2).to(device)
```

```python
# ==============================================================================
# 6. TRAINING & EVALUATION FUNCTIONS
# ==============================================================================
def train_contrastive_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    for (v1, v2), _ in loader:
        v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = criterion(z1, z2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * v1.size(0)

    return total_loss / len(loader.dataset)

def train_supervised_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = model(imgs)
            loss = criterion(outputs, lbls)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)

def evaluate_model(model, loader):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(lbls.numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    auc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc, "cm": cm}
```

```python
# ==============================================================================
# 7. MAIN EXPERIMENTAL EXECUTION LOOP (LABEL SCARCITY STUDY)
# ==============================================================================
# 1. Locate PCam Files under /kaggle/input
input_dir = Path("/kaggle/input")
train_x_paths = list(input_dir.rglob("*train_x.h5"))
train_y_paths = list(input_dir.rglob("*train_y.h5"))
test_x_paths  = list(input_dir.rglob("*test_x.h5"))
test_y_paths  = list(input_dir.rglob("*test_y.h5"))

if not train_x_paths or not train_y_paths:
    raise FileNotFoundError("Could not find PCam HDF5 files under /kaggle/input/. Ensure dataset is attached.")

train_dataset_full = PCamHDF5Dataset(train_x_paths[0], train_y_paths[0], transform=eval_transform)
test_dataset_full  = PCamHDF5Dataset(test_x_paths[0], test_y_paths[0], transform=eval_transform)

# 2. Extract Labels for Stratified Subsampling
with h5py.File(train_y_paths[0], 'r') as fy:
    k = 'y' if 'y' in fy else list(fy.keys())[0]
    all_train_labels = np.asarray(fy[k][:]).ravel()

# Fixed canonical subset for practical training (e.g. 16,000 train pool, 2,000 test)
CANONICAL_TRAIN_SIZE = 16000
CANONICAL_TEST_SIZE = 2000

sss_base = StratifiedShuffleSplit(n_splits=1, train_size=CANONICAL_TRAIN_SIZE, random_state=CONFIG["seed"])
base_train_indices, _ = next(sss_base.split(np.zeros(len(all_train_labels)), all_train_labels))

test_indices = np.arange(min(CANONICAL_TEST_SIZE, len(test_dataset_full)))
test_loader = DataLoader(
    Subset(test_dataset_full, test_indices),
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"]
)

# ------------------------------------------------------------------------------
# STEP A: SELF-SUPERVISED CONTRASTIVE PRETRAINING (ON FULL UNLABELED TRAIN POOL)
# ------------------------------------------------------------------------------
print("\n>>> STARTING CONTRASTIVE PRETRAINING (SIMCLR)...")
contrastive_dataset = PCamHDF5Dataset(
    train_x_paths[0], train_y_paths[0],
    transform=ContrastiveTwoViewTransform(CONFIG["image_size"])
)
contrastive_loader = DataLoader(
    Subset(contrastive_dataset, base_train_indices),
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    drop_last=True
)

contrastive_model = SimCLRViT(model_name=CONFIG["model_name"]).to(device)
cl_optimizer = torch.optim.AdamW(contrastive_model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
cl_criterion = NTXentLoss(temperature=CONFIG["temperature"])
cl_scaler = GradScaler()

for epoch in range(CONFIG["contrastive_epochs"]):
    ep_loss = train_contrastive_epoch(contrastive_model, contrastive_loader, cl_optimizer, cl_criterion, cl_scaler)
    print(f"Epoch [{epoch+1}/{CONFIG['contrastive_epochs']}] Contrastive Loss: {ep_loss:.4f}")

# Save pretrained encoder weights
encoder_ckpt_path = output_dir / "contrastive_vit_encoder.pth"
torch.save(contrastive_model.encoder.state_dict(), encoder_ckpt_path)
print(f"Pretrained encoder saved to {encoder_ckpt_path}")

# ------------------------------------------------------------------------------
# STEP B: CONTROLLED LABEL-SCARCITY BENCHMARK (10%, 25%, 50%, 100%)
# ------------------------------------------------------------------------------
benchmark_results = []
fractions = CONFIG["label_fractions"]

for frac in fractions:
    # Stratified subsampling from canonical training pool
    if frac == 1.0:
        sub_indices = base_train_indices
    else:
        n_sub = int(len(base_train_indices) * frac)
        sss_sub = StratifiedShuffleSplit(n_splits=1, train_size=n_sub, random_state=CONFIG["seed"])
        rel_idx, _ = next(sss_sub.split(base_train_indices, all_train_labels[base_train_indices]))
        sub_indices = base_train_indices[rel_idx]

    train_subset = PCamHDF5Dataset(train_x_paths[0], train_y_paths[0], transform=train_supervised_transform)
    frac_loader = DataLoader(
        Subset(train_subset, sub_indices),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"]
    )

    print(f"\n=======================================================")
    print(f"Benchmarking Label Fraction: {int(frac*100)}% ({len(sub_indices)} samples)")
    print(f"=======================================================")

    # 1. Train Baseline ViT
    baseline_model = build_baseline_vit()
    opt_b = torch.optim.AdamW(baseline_model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scaler_b = GradScaler()
    crit = nn.CrossEntropyLoss()

    for epoch in range(CONFIG["finetune_epochs"]):
        train_supervised_epoch(baseline_model, frac_loader, opt_b, crit, scaler_b)

    metrics_b = evaluate_model(baseline_model, test_loader)
    print(f"[Baseline ViT] Acc: {metrics_b['acc']:.4f} | F1: {metrics_b['f1']:.4f} | AUC: {metrics_b['auc']:.4f}")

    benchmark_results.append({
        "Model": "Baseline ViT",
        "Fraction": f"{int(frac*100)}%",
        "Samples": len(sub_indices),
        "Accuracy": metrics_b["acc"],
        "Precision": metrics_b["prec"],
        "Recall": metrics_b["rec"],
        "F1_Score": metrics_b["f1"],
        "ROC_AUC": metrics_b["auc"]
    })

    # 2. Train Contrastive-Enhanced ViT
    cl_finetuned_model = build_contrastive_vit(encoder_ckpt_path)
    opt_cl = torch.optim.AdamW(cl_finetuned_model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scaler_cl = GradScaler()

    for epoch in range(CONFIG["finetune_epochs"]):
        train_supervised_epoch(cl_finetuned_model, frac_loader, opt_cl, crit, scaler_cl)

    metrics_cl = evaluate_model(cl_finetuned_model, test_loader)
    print(f"[Contrastive ViT] Acc: {metrics_cl['acc']:.4f} | F1: {metrics_cl['f1']:.4f} | AUC: {metrics_cl['auc']:.4f}")

    benchmark_results.append({
        "Model": "Contrastive-Enhanced ViT",
        "Fraction": f"{int(frac*100)}%",
        "Samples": len(sub_indices),
        "Accuracy": metrics_cl["acc"],
        "Precision": metrics_cl["prec"],
        "Recall": metrics_cl["rec"],
        "F1_Score": metrics_cl["f1"],
        "ROC_AUC": metrics_cl["auc"]
    })

# ------------------------------------------------------------------------------
# STEP C: EXPORT SUMMARY TABLE & GENERATE PLOTS
# ------------------------------------------------------------------------------
df_results = pd.DataFrame(benchmark_results)
df_results.to_csv(output_dir / "label_scarcity_benchmark.csv", index=False)
print("\nBenchmark complete! Results saved to label_scarcity_benchmark.csv")
print(df_results.to_string(index=False))

# Plot comparative curves
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_results, x="Fraction", y="F1_Score", hue="Model", marker="o")
plt.title("Label Efficiency: F1-Score vs. Labeled Data Fraction")
plt.xlabel("Labeled Training Fraction")
plt.ylabel("Test F1-Score")
plt.grid(True)
plt.savefig(output_dir / "f1_efficiency_curve.png", bbox_inches="tight")
plt.show()
```

---

## 6. How to Run & Manage in Kaggle

### 6.1 Resource & Execution Constraints
1. **Execution Time Limits:** Kaggle notebooks have an interactive limit of 12 hours (and 9 hours for background "Save & Run All" commits). To ensure completion:
   * Keep contrastive pretraining epochs at 15–30.
   * Keep downstream fine-tuning epochs per fraction at 5–10 epochs.
2. **Kaggle Disk Space:** The `/kaggle/working/` directory has a 20 GB limit. Do not write extracted PNG images to disk; always load directly from HDF5 files on the fly.
3. **Out-of-Memory (OOM) Prevention:**
   * Use `torch.amp.autocast('cuda')` (mixed precision).
   * Call `gc.collect()` and `torch.cuda.empty_cache()` between fraction runs.
   * Use batch size 32 for `vit_tiny_patch16_224` or 16 for `vit_base_patch16_224`.

### 6.2 Expected Scientific Outcomes & Interpretation
* **Low-Label Regime (10% & 25%):** The Contrastive ViT is hypothesized to outperform the Baseline ViT significantly in F1-score and Recall/Sensitivity because unsupervised pretraining anchors feature representations around true tissue morphology rather than overfitting to small label counts.
* **Full-Data Regime (100%):** Performance difference typically narrows as the volume of supervised data becomes sufficient for the baseline ViT to compensate for lack of inductive biases.
* **Interpretation Honesty:** If contrastive learning provides only modest gains, this remains a valid scientific result indicating that standard transfer learning from large ImageNet checkpoints already imparts strong low-level edge/texture primitives.

---

## 7. Optional Ablation Study Protocol (Stage 13)

To explore the digital image processing design choices further, perform an ablation on the data augmentation policy:
* **Augmentation Policy A (Geometric only):** Random Resized Crop + Horizontal/Vertical Flips + 90° Rotations.
* **Augmentation Policy B (Geometric + Photometric):** Policy A + ColorJitter (brightness, contrast, saturation) + mild Gaussian Noise.
* **Research Question for Ablation:** Does enforcing photometric (staining) invariance yield superior transfer performance compared to purely geometric invariance in breast histopathology?
