# Methodology Verification & Audit Report: Notebook Implementations vs. Research Specifications

---

## 1. Executive Summary

This report presents a comprehensive technical audit of the current codebase ([`pipeline-2.ipynb`](file:///d:/DIP/pipeline-2.ipynb) and [`pcam-data-efficiency-benchmark.ipynb`](file:///d:/DIP/pcam-data-efficiency-benchmark.ipynb)) against the scientific and experimental requirements defined in the **Project Proposal** and **Detailed Methodology Guide**.

### 1.1 High-Level Audit Verdict
* **Core Contrastive Pipeline ([`pipeline-2.ipynb`](file:///d:/DIP/pipeline-2.ipynb)):** **90% COMPLIANT**. The self-supervised contrastive learning stage (SimCLR), two-view data augmentations, projection head, vectorized $\text{NT-Xent}$ loss, and supervised fine-tuning are well-implemented and mathematically sound.
* **Data Efficiency Benchmark ([`pcam-data-efficiency-benchmark.ipynb`](file:///d:/DIP/pcam-data-efficiency-benchmark.ipynb)):** **CRITICAL METHODOLOGICAL FLAW**. The benchmark loop loads model checkpoints that were **already trained on 100% of the labeled data**, contaminating the low-data evaluation (10%, 25%, 50%).
* **Dataset Splitting Protocol:** **PARTIALLY COMPLIANT**. All splits were carved out of `split_train_x.h5` rather than utilizing the official benchmark `split_test_x.h5`, bypassing the guaranteed patient-level Whole-Slide Image (WSI) separation.

---

## 2. Stage-by-Stage Verification Matrix

| Stage | Methodology Specification (PDFs) | Notebook Implementation | Compliance Status |
| :--- | :--- | :--- | :---: |
| **Stage 1: Dataset Preparation** | Load official PCam train/val/test splits; prevent data leakage across WSIs. | Train (16k), Val (2k), and Test (2k) carved from `split_train_x.h5`. Zero index leakage verified via assertions. | ⚠️ **Partial / Warning** |
| **Stage 2: Image Preprocessing** | Resize to $224 \times 224$; ImageNet mean/std normalization; deterministic val/test. | `Resize(224, 224)`, ImageNet normalization, deterministic evaluation transform. | ✅ **Pass** |
| **Stage 3: Contrastive Augmentations** | Two stochastic views per sample: crop, flip, rotation, mild color jitter. | `ContrastiveTransform` generates independent positive pairs with medically sound transforms. | ✅ **Pass** |
| **Stage 4: ViT Baseline Model** | Pretrained ViT (e.g., ViT-B/16 or ViT-Tiny) with 2-class linear classifier. | `vit_tiny_patch16_224` (timm) with `[CLS]` token output + 2-class head. | ✅ **Pass** |
| **Stage 5: Contrastive Pretraining** | SimCLR framework; 2-layer MLP projection head; NT-Xent / InfoNCE loss ($\tau = 0.5$); no labels. | 2-layer MLP ($192 \to 512 \to 128$ with GELU); vectorized NT-Xent with diagonal masking; 50 epochs. | ✅ **Pass** |
| **Stage 6: Supervised Fine-Tuning** | Discard projector; fine-tune encoder + linear classifier using Cross-Entropy & AdamW. | Projector removed; encoder loaded; fine-tuned for 50 epochs with CrossEntropyLoss and AdamW. | ✅ **Pass** |
| **Stage 7: Label-Scarcity Experiments** | Evaluate Baseline vs. Contrastive ViT on identical stratified subsets (10%, 25%, 50%, 100%). | Fractions tested with stratified sampling, but **models loaded were already trained on 100% labels**. | ❌ **Critical Flaw** |
| **Stage 8 & 9: Evaluation Metrics** | Accuracy, Precision, Recall/Sensitivity, F1-Score, ROC-AUC, Confusion Matrix. | All required metrics calculated; confusion matrices, ROC curves, and history plots generated. | ✅ **Pass** |

---

## 3. Detailed Audit: What Was Implemented Correctly

### 3.1 Memory-Safe Lazy HDF5 Loading ([`pipeline-2.ipynb`](file:///d:/DIP/pipeline-2.ipynb), Cell 7)
The custom `PCamHDF5Dataset` handles multi-process PyTorch `DataLoader` workers correctly:
* File handles (`self.fx`, `self.fy`) are lazily initialized per worker process in `__getitem__` rather than in `__init__`.
* This prevents process deadlocks, file corruption, and RAM exhaustion when reading large HDF5 datasets on Kaggle.

### 3.2 Medically Sound Contrastive Augmentations ([`pipeline-2.ipynb`](file:///d:/DIP/pipeline-2.ipynb), Cell 10)
The contrastive view generator implements:
```python
transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])
```
* **Medical Validity:** Incorporates rotational and flip invariance (histopathology tissue has no canonical orientation) and mild stain jitter, while correctly avoiding aggressive morphological distortions.

### 3.3 Vectorized NT-Xent Loss Formulation ([`pipeline-2.ipynb`](file:///d:/DIP/pipeline-2.ipynb), Cell 17)
The loss module correctly computes normalized temperature-scaled cosine similarities across the $2B \times 2B$ representation matrix:
* Embeddings $z_1, z_2$ are concatenated to shape $[2B, D]$.
* Pairwise dot products $\text{sim} = \frac{z z^\top}{\tau}$ are computed efficiently.
* Self-similarity diagonal elements are masked with $-9 \times 10^{15}$.
* Targets are mapped symmetrically: $i \to i+B$ and $i+B \to i$.

### 3.4 Evaluation Metrics & Visualization Suite ([`pipeline-2.ipynb`](file:///d:/DIP/pipeline-2.ipynb), Cells 31–36)
The evaluation script computes:
* **Accuracy**, **Precision**, **Recall (Sensitivity)**, **F1-Score**, and **ROC-AUC**.
* Visual artifacts: Confusion matrix heatmap, ROC curve, training/validation loss curves, and prediction CSV tables.

---

## 4. Critical Flaws & Discrepancies

### 4.1 Critical Flaw: Data Leakage via Model Checkpoints in Benchmark Loop
* **Location:** [`pcam-data-efficiency-benchmark.ipynb`](file:///d:/DIP/pcam-data-efficiency-benchmark.ipynb), Section 5 (Cell 5).
* **The Code in Question:**
  ```python
  MODELS_TO_BENCHMARK = {
      "Baseline ViT": {
          "arch": "vit_tiny_patch16_224",
          "ckpt_path": ".../baseline_vit/checkpoints/best_baseline_vit.pth"
      },
      "Contrastive Finetuned ViT": {
          "arch": "vit_tiny_patch16_224",
          "ckpt_path": ".../pipeline2_contrastive_vit/checkpoints/best_contrastive_finetuned_vit.pth"
      }
  }
  ```
* **The Methodological Problem:**
  1. `best_baseline_vit.pth` was **already trained on 100% (16,000) labeled training samples** in Notebook 1.
  2. `best_contrastive_finetuned_vit.pth` was **already fine-tuned on 100% (16,000) labeled training samples** in Pipeline 2.
  3. When the benchmark loop iterates through `frac = 0.10` (1,600 samples), it loads these fully trained models and runs **3 epochs of continual training** on the 10% subset.
* **Why This Invalidates the Scientific Experiment:**
  * The central research question is: *"Does contrastive pretraining help when labeled data is scarce?"*
  * By loading a baseline model that has already seen all 16,000 labels, the baseline model achieves **94.10% accuracy** at 10% data because it has already memorized/converged on the full dataset.
  * To measure true label efficiency, the Baseline ViT must start from the ImageNet-pretrained initialization and be trained **exclusively** on the 10% subset.

### 4.2 Discrepancy: Benchmark Training Duration (3 Epochs vs. Convergence)
* In Section 5 of the benchmark notebook, `epochs_per_fraction = 3`.
* For a fresh model trained on only 1,600 samples, 3 epochs (only 150 gradient steps at batch size 16) is insufficient for convergence.
* Meanwhile, the pre-converged checkpoints had already undergone 50 epochs of training. This creates an unfair comparison against models trained from scratch or linear probes.

### 4.3 Discrepancy: Dataset Slicing vs. Official PCam Split Files
* **Specification in PDF (Section 6 & Stage 1):**  
  *"Use the official train, validation, and test files... The official benchmark provides separate training, validation, and test splits with no overlap in whole-slide images between splits."*
* **Implementation in Notebooks:**  
  * `X_H5_PATH = ".../camelyonpatch_level_2_split_train_x.h5"`
  * Train (16,000), Val (2,000), and Test (2,000) are all indexed from within `split_train_x.h5`.
* **Scientific Implication:**  
  While the notebook verifies that `train_indices` and `test_indices` have zero index overlap, patches originating from the same patient slide (WSI) could appear in both train and test partitions. To guarantee true patient-level generalization, the test set must be loaded from `camelyonpatch_level_2_split_test_x.h5`.

---

## 5. Corrective Action Plan (How to Fix the Implementation)

To bring the codebase into 100% compliance with the research methodology, implement the following modifications:

### Fix 1: Modify Benchmark Model Initialization (`pcam-data-efficiency-benchmark.ipynb`)
Update the benchmark loop so that at each fraction, models are instantiated from their correct initializations rather than fully supervised checkpoints:

```python
# ==============================================================================
# CORRECTED BENCHMARK TRAINING INITIALIZATION
# ==============================================================================
for frac in FRACTIONS:
    sub_loader = get_fraction_loader(frac) # Stratified subset

    # 1. Baseline ViT: Must start from ImageNet pretrained weights (NEVER 100% PCam weights)
    baseline_model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=2).to(device)
    train_classifier(baseline_model, sub_loader, epochs=15, lr=1e-4)
    acc_base, f1_base, auc_base = evaluate_model(baseline_model, test_loader)

    # 2. Contrastive ViT: Must start from self-supervised checkpoint (NO prior labels seen)
    contrastive_model = build_vit_classifier(
        backbone_ckpt="contrastive_vit_pretrained.pth", # Output of SimCLR pretraining
        num_classes=2
    ).to(device)
    train_classifier(contrastive_model, sub_loader, epochs=15, lr=1e-4)
    acc_cl, f1_cl, auc_cl = evaluate_model(contrastive_model, test_loader)
```

### Fix 2: Utilize Official Test HDF5 File for Final Evaluation
Update the test data loader to point directly to the official benchmark test set:
```python
TEST_X_PATH = "/kaggle/input/datasets/tarequlislam8/pcam-dataset/camelyonpatch_level_2_split_test_x.h5"
TEST_Y_PATH = "/kaggle/input/datasets/tarequlislam8/pcam-dataset/camelyonpatch_level_2_split_test_y.h5"

test_dataset = PCamLazyDataset(TEST_X_PATH, TEST_Y_PATH, transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
```

---

## 6. Summary Comparison Table

| Feature / Protocol | Required by Research PDFs | Current Notebook Implementation | Action Required |
| :--- | :--- | :--- | :--- |
| **Contrastive Pretraining** | SimCLR + ViT + NT-Xent | Implemented in `pipeline-2.ipynb` | None (Compliant) |
| **Augmentation Policy** | 2-view positive pairs | Implemented in `ContrastiveTransform` | None (Compliant) |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1, AUC | Implemented in evaluation functions | None (Compliant) |
| **WSI Slide Separation** | Official `split_test_x.h5` file | Sliced from `split_train_x.h5` | Point test loader to `test_x.h5` |
| **10% Label Evaluation** | Fresh model trained only on 10% data | Fully trained (100%) model loaded + 3 epochs | Reset weights to ImageNet / Pretrained encoder before fraction training |
| **Fraction Epochs** | Full convergence on fraction subset | Fixed at 3 epochs | Increase to 10–15 epochs per fraction |

---

## 7. Conclusion

The core computer vision building blocks developed in [`pipeline-2.ipynb`](file:///d:/DIP/pipeline-2.ipynb) represent high-quality, professional PyTorch engineering that faithfully reflects the proposal's SimCLR + ViT architecture.

However, the empirical benchmark in [`pcam-data-efficiency-benchmark.ipynb`](file:///d:/DIP/pcam-data-efficiency-benchmark.ipynb) suffers from an experimental design flaw (evaluating models that were already trained on the full labeled dataset). Correcting this loading logic as detailed in Section 5 will allow the research team to accurately measure the true label-efficiency advantage of contrastive learning on breast cancer histopathology.
