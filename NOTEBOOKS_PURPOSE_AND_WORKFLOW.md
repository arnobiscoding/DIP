# Purpose & Workflow of the Project Notebooks

---

## 1. Executive Summary

The workspace contains two primary Jupyter Notebooks that form the core training and evaluation stages of the **Contrastive Vision Transformer (ViT) Breast Histopathology Project**:

1. **[`pipeline-2.ipynb`](file:///d:/DIP/pipeline-2.ipynb)** — **Contrastive Pretraining & Supervised Fine-Tuning Engine**  
   Pretrains the ViT encoder using self-supervised contrastive learning (SimCLR + NT-Xent) on unlabeled image patches, then fine-tunes it on the full labeled training set.
2. **[`pcam-data-efficiency-benchmark.ipynb`](file:///d:/DIP/pcam-data-efficiency-benchmark.ipynb)** — **Data Efficiency & Label-Scarcity Benchmark Engine**  
   Takes the models produced by the training pipelines and evaluates their performance across systematically reduced training label fractions (10%, 25%, 50%, 75%, 100%) on an identical held-out test set.

---

## 2. Notebook 1: `pipeline-2.ipynb`

### 2.1 Primary Goal
To execute **Stage 5 (Contrastive Pretraining)** and **Stage 6 (Supervised Fine-Tuning)** from the research methodology, producing a high-quality, contrastively trained Vision Transformer feature extractor and classifier.

### 2.2 Stage-by-Stage Workflow

```
                        16,000 Unlabeled Patches
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
        Augmented View 1                        Augmented View 2
               │                                       │
        ViT Encoder f(·)                        ViT Encoder f(·)  (Shared Weights)
               ▼                                       ▼
          Feature h_1                             Feature h_2
               │                                       │
        Projection g(·)                         Projection g(·)   (Shared Weights)
               ▼                                       ▼
         Embedding z_1                           Embedding z_2
               └───────────────────┬───────────────────┘
                                   ▼
                       NT-Xent Contrastive Loss
               (Optimizes encoder representations for 50 epochs)
                                   │
                                   ▼
             Output: 'contrastive_vit_pretrained.pth'
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        │  SUPERVISED FINE-TUNING (100% Labeled Training Set) │
        │  - Discard projection head g(·)                     │
        │  - Load pretrained encoder f(·)                     │
        │  - Attach 2-class Linear Head with Dropout          │
        │  - Train with Cross-Entropy Loss for 50 epochs      │
        └──────────────────────────┬──────────────────────────┘
                                   ▼
             Output: 'best_contrastive_finetuned_vit.pth'
```

### 2.3 Detailed Operations & Responsibilities
* **Data Ingestion:** Loads PatchCamelyon (PCam) images via a memory-safe, lazy-loading PyTorch `Dataset` using `h5py` to prevent multi-worker process deadlocks.
* **Contrastive Pretraining (SimCLR):**
  * Applies stochastic augmentations (random resized crop, flips, 90° rotations, mild color jitter) to generate positive image pairs $(\tilde{x}_i, \tilde{x}_j)$.
  * Passes views through a `vit_tiny_patch16_224` encoder to extract 192-dimensional `[CLS]` token representations.
  * Maps representations through a 2-layer MLP projection head ($192 \to 512 \to 128$ with GELU) to a normalized 128-dimensional latent hypersphere.
  * Calculates vectorized $\text{NT-Xent}$ (InfoNCE) loss with temperature $\tau = 0.5$ and diagonal masking, running for 50 epochs **without using class labels**.
* **Supervised Fine-Tuning:**
  * Discards the temporary projection head.
  * Attaches a 2-class linear classification head with Dropout (0.1).
  * Fine-tunes the network on 100% (16,000) labeled patches using Cross-Entropy Loss and AdamW optimizer.
* **Evaluation & Visualization:**
  * Evaluates on held-out validation and test sets.
  * Generates confusion matrices, ROC curves, and training history loss curves.

### 2.4 Artifacts Produced in `/kaggle/working/pipeline2_contrastive_vit/`
* `checkpoints/contrastive_vit_pretrained.pth` (Pretrained ViT encoder weights)
* `checkpoints/best_contrastive_finetuned_vit.pth` (Best fine-tuned classifier weights)
* `results/confusion_matrix.png` (Test set error distribution)
* `results/roc_curve.png` (ROC-AUC diagnostic plot)
* `results/metrics.json` & `predictions.csv` (Quantitative logs)

---

## 3. Notebook 2: `pcam-data-efficiency-benchmark.ipynb`

### 3.1 Primary Goal
To answer the core research question: **"Does contrastive learning help maintain Vision Transformer classification performance when labeled training data becomes scarce?"**

### 3.2 Stage-by-Stage Workflow

```
   Load Models to Benchmark:
     1. Baseline ViT (Supervised only)
     2. Contrastive Pretrained ViT (Direct probe / fresh head)
     3. Contrastive Finetuned ViT
                     │
                     ▼
   Loop through Training Data Fractions:
     [10% (1,600) | 25% (4,000) | 50% (8,000) | 75% (12,000) | 100% (16,000)]
                     │
     ┌───────────────┴───────────────┐
     ▼                               ▼
   Stratified Subsampling         Fine-Tune Model
   (Exact same sample indices      on Fraction Loader
    fed to all models)
                     │
                     ▼
   Evaluate on Identical Untouched Test Set (2,000 Samples)
   (Calculate Accuracy, F1-Score, ROC-AUC, Time)
                     │
                     ▼
   Export: 'benchmark_summary.csv' & Comparative Degradation Plots
```

### 3.3 Detailed Operations & Responsibilities
* **Dynamic Checkpoint Ingestion:** Automatically searches `/kaggle/input/` to detect and load the `.pth` weights exported by prior runs.
* **Stratified Subsampling:** Uses `StratifiedShuffleSplit` with fixed random seeds (`seed=42`) to create reproducible subsets (10%, 25%, 50%, 75%, 100%), ensuring the 50/50 tumor/normal class balance is preserved at every fraction.
* **Fair Model Comparison Protocol:** Feeds the **exact same image patches** to each model variant at every fraction step.
* **Standardized Evaluation:** Evaluates every model on the same 2,000-sample test set using identical deterministic preprocessing (no random augmentations).
* **Metric Aggregation & Plotting:** Logs accuracy, F1-score, and ROC-AUC per fraction and plots comparative learning curves to quantify how rapidly performance drops as labels are removed.

### 3.4 Artifacts Produced in `/kaggle/working/benchmark_results/`
* `benchmark_summary.csv` (Complete numerical table containing Accuracy, F1, ROC-AUC, and elapsed time per model and fraction).
* Comparative visualization plots of Test Accuracy / F1-Score vs. Labeled Data Fraction.

---

## 4. End-to-End Pipeline Architecture

The two notebooks function as an integrated two-stage research pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PIPELINE 1: Baseline Model (Initial Experiment / Notebook 1)                │
│  • Defines canonical 16k train / 2k val / 2k test splits                     │
│  • Trains pure supervised Baseline ViT on 100% data                         │
│  • Exports: best_baseline_vit.pth & split_indices.pt                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PIPELINE 2: Contrastive Engine (pipeline-2.ipynb)                           │
│  • Reads split indices from Pipeline 1                                      │
│  • Stage 1: Self-supervised SimCLR pretraining (NT-Xent Loss, 50 epochs)    │
│  • Stage 2: Supervised fine-tuning on 100% labels (50 epochs)               │
│  • Exports: contrastive_vit_pretrained.pth &                                │
│             best_contrastive_finetuned_vit.pth                              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PIPELINE 3: Empirical Benchmark (pcam-data-efficiency-benchmark.ipynb)      │
│  • Attaches checkpoints from Pipeline 1 and Pipeline 2                      │
│  • Tests all models across 10%, 25%, 50%, 75%, 100% labeled training data    │
│  • Evaluates on the fixed 2,000-sample test set                             │
│  • Exports: benchmark_summary.csv and comparative research plots            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Summary Table

| Attribute | `pipeline-2.ipynb` | `pcam-data-efficiency-benchmark.ipynb` |
| :--- | :--- | :--- |
| **Role in Project** | Model Training Engine (Contrastive + Full-Data Fine-tuning) | Comparative Benchmark & Evaluation Engine |
| **Input Data** | Full PCam Training Set (HDF5) + Split Indices | Trained Model Weights (`.pth`) + PCam Dataset |
| **Primary Method** | SimCLR Self-Supervised Learning + Supervised Fine-Tuning | Stratified Subsampling + Multi-Ratio Downstream Evaluation |
| **Loss Functions** | $\text{NT-Xent}$ (InfoNCE) Loss, Cross-Entropy Loss | Cross-Entropy Loss (during fraction fine-tuning) |
| **Key Output** | `contrastive_vit_pretrained.pth`, `best_contrastive_finetuned_vit.pth` | `benchmark_summary.csv`, Performance vs. Fraction Plots |
| **Target Question** | *How do we learn robust visual representations without labels?* | *Does contrastive pretraining maintain performance under label scarcity?* |
