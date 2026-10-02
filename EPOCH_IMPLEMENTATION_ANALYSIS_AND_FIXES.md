# Epoch Implementation Analysis & Required Corrections

---

## 1. Direct Answer

> **Is the epoch correctly implemented?**  
> **NO.** The epoch implementation across both [`pipeline-2.ipynb`](file:///d:/DIP/pipeline-2.ipynb) and [`pcam-data-efficiency-benchmark.ipynb`](file:///d:/DIP/pcam-data-efficiency-benchmark.ipynb) contains critical flaws, severe under-training in the benchmark, missing early stopping, and violation of the validation tuning guidelines specified in the research methodology.

---

## 2. Executive Comparison: Implemented vs. Required

| Dimension | What is Implemented in Notebooks | What the Research Methodology Requires (PDFs) | Severity |
| :--- | :--- | :--- | :---: |
| **Benchmark Epochs** | `epochs_per_fraction = 3` (Fixed 3 epochs) | Full convergence per fraction subset (15–25 epochs) | 🔴 **Critical** |
| **Benchmark Model State** | 3 continual epochs on models **already trained for 50 epochs** | Fresh model trained **only** on the fraction subset | 🔴 **Critical** |
| **Validation Monitoring** | Benchmark has **NO validation loop** during epochs; evaluates directly on Test | *"Tune on validation only. Keep validation separate from training."* | 🔴 **Critical** |
| **Early Stopping** | **None** in either notebook; runs full static loop | Early stopping to prevent overfitting and save GPU compute | 🟠 **High** |
| **Learning Rate Schedule** | Constant learning rate ($10^{-4}$) with no decay | Cosine Annealing or `ReduceLROnPlateau` | 🟠 **Medium** |
| **Pipeline-2 Fine-Tuning** | 50 static epochs (best validation accuracy reached at **Epoch 18**) | Early stopping should have terminated after Epoch 25 | 🟡 **Efficiency** |

---

## 3. The 5 Major Flaws in Detail

### Flaw 1: `epochs_per_fraction = 3` in the Benchmark is Severely Under-Trained
* **The Code in [`pcam-data-efficiency-benchmark.ipynb`](file:///d:/DIP/pcam-data-efficiency-benchmark.ipynb) (Cell 0 & 5):**
  ```python
  CONFIG = {
      "batch_size": 16,
      "epochs_per_fraction": 3,
      ...
  }
  for epoch in range(CONFIG["epochs_per_fraction"]): # ONLY 3 EPOCHS!
      model.train()
      for imgs, lbls in frac_loader:
          ...
  ```
* **Why This is Wrong:**
  * At a 10% label fraction (1,600 samples) with batch size 16, one epoch has only 100 batches ($1600 / 16$).
  * **3 epochs = only 300 gradient steps.**
  * A Vision Transformer (ViT) consists of multi-head self-attention layers and an MLP classification head. It is mathematically impossible for a ViT to properly adapt its weights in only 300 steps.
  * In the benchmark, `Contrastive Pretrained ViT` was given only these 3 epochs to learn a classification head from scratch, achieving a poor 89.90% accuracy. Meanwhile, `Baseline ViT` was loaded from an already-converged checkpoint (`best_baseline_vit.pth`) that had **already trained for 50 epochs** in Notebook 1, giving it an unfair 53-epoch advantage!

### Flaw 2: Benchmark Has No Validation Loop or Checkpoint Selection
* **The Code in [`pcam-data-efficiency-benchmark.ipynb`](file:///d:/DIP/pcam-data-efficiency-benchmark.ipynb) (Cell 5):**
  The benchmark trains on `frac_loader` for 3 epochs and then immediately calls:
  ```python
  # Evaluate directly on test_loader at the end of epoch 3
  acc, f1, auc = evaluate_model(model, test_loader)
  ```
* **Why This is Wrong:**
  * There is **no validation set evaluation** after each epoch.
  * You cannot verify whether the model is learning, oscillating, or overfitting.
  * You evaluate whatever random state the weights end up in at the final iteration, rather than selecting the best checkpoint (`best_val_loss`).
  * The PDF specifically instructs: *"Tune on validation only... Never train or tune on the test set."*

### Flaw 3: Absence of Early Stopping in `pipeline-2.ipynb` Wastes 64% of Compute
* **The Code in [`pipeline-2.ipynb`](file:///d:/DIP/pipeline-2.ipynb) (Cell 29):**
  Runs a rigid loop: `for epoch in range(1, 51):`
* **Execution Log Evidence:**
  * In Cell 30 of `pipeline-2.ipynb`, the log states:
    > `Successfully loaded model checkpoint from Epoch 18 (Validation Acc: 96.55%)`
  * The model converged and reached its peak validation accuracy at **Epoch 18**.
  * The notebook continued training for **32 additional unnecessary epochs** (Epochs 19 to 50), wasting ~25 minutes of GPU runtime and increasing the risk of Kaggle kernel timeouts.
* **Why Early Stopping is Required:**
  A simple patience counter of 7–10 epochs would have automatically stopped training around Epoch 25–28 and saved the optimal weights.

### Flaw 4: Missing Learning Rate Schedulers Across Epochs
* **The Code in Both Notebooks:**
  ```python
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
  ```
  Neither notebook employs a scheduler. The learning rate remains statically locked at $10^{-4}$ throughout all 50 epochs.
* **Why This is Wrong:**
  * ViT training requires an initial warm-up followed by cosine decay or step decay.
  * Without a scheduler, gradient updates remain large near the end of training, causing the optimizer to bounce around the local minimum instead of settling into the optimal basin.

---

## 4. What We Need to Change and Why

To fix the epoch implementation across both notebooks, apply the following 4 structural modifications:

---

### Change 1: Increase Benchmark Epochs & Add Validation Checkpointing
* **File:** [`pcam-data-efficiency-benchmark.ipynb`](file:///d:/DIP/pcam-data-efficiency-benchmark.ipynb)
* **What to change:** Increase `epochs_per_fraction` from 3 to **15 epochs** (with early stopping patience = 5), evaluate on `val_loader` after each epoch, and test only the best checkpoint.
* **Code Replacement:**
  ```python
  # REPLACE Cell 5 inner loop in pcam-data-efficiency-benchmark.ipynb:
  EPOCHS_PER_FRACTION = 15
  PATIENCE = 5

  best_val_loss = float("inf")
  patience_counter = 0
  best_model_state = None

  for epoch in range(1, EPOCHS_PER_FRACTION + 1):
      # Train 1 epoch
      model.train()
      for imgs, lbls in frac_loader:
          imgs, lbls = imgs.to(device), lbls.to(device)
          optimizer.zero_grad()
          loss = criterion(model(imgs), lbls)
          loss.backward()
          optimizer.step()

      # Validate 1 epoch
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
          for v_imgs, v_lbls in val_loader:
              v_imgs, v_lbls = v_imgs.to(device), v_lbls.to(device)
              val_loss += criterion(model(v_imgs), v_lbls).item() * v_imgs.size(0)
      val_loss /= len(val_loader.dataset)

      # Checkpoint based on Validation Loss
      if val_loss < best_val_loss:
          best_val_loss = val_loss
          patience_counter = 0
          best_model_state = copy.deepcopy(model.state_dict())
      else:
          patience_counter += 1
          if patience_counter >= PATIENCE:
              print(f"Early stopping triggered at epoch {epoch}")
              break

  # Load BEST validation checkpoint for the final test evaluation
  model.load_state_dict(best_model_state)
  acc, f1, auc = evaluate_model(model, test_loader)
  ```
* **Why:** Guarantees that models at 10% and 25% data have sufficient steps to adapt their weights, selects the best generalized model via validation loss, and prevents overfitting.

---

### Change 2: Fair Model Initialization in Benchmark (Zero Prior Supervised Exposure)
* **File:** [`pcam-data-efficiency-benchmark.ipynb`](file:///d:/DIP/pcam-data-efficiency-benchmark.ipynb)
* **What to change:** For each fraction, initialize:
  1. **Baseline ViT:** Fresh ImageNet pretrained `vit_tiny_patch16_224` + new 2-class head.
  2. **Contrastive ViT:** Load `contrastive_vit_pretrained.pth` encoder + new 2-class head.
* **Why:** Ensures that when evaluating 10% labeled data, neither model has already seen the remaining 90% of labels. This isolates the true representation benefit of contrastive learning.

---

### Change 3: Add Early Stopping to `pipeline-2.ipynb`
* **File:** [`pipeline-2.ipynb`](file:///d:/DIP/pipeline-2.ipynb), Cell 29 (Fine-Tuning Loop)
* **What to change:** Add early stopping with `patience = 8`:
  ```python
  PATIENCE = 8
  patience_counter = 0
  best_val_acc = 0.0

  for epoch in range(1, CONFIG["finetune_epochs"] + 1):
      t_loss, t_acc = train_supervised_epoch(...)
      v_loss, v_acc = validate_supervised_epoch(...)

      if v_acc > best_val_acc:
          best_val_acc = v_acc
          patience_counter = 0
          torch.save(finetune_model.state_dict(), best_ft_ckpt_path)
          print(f" --> Best Checkpoint Saved! (Val Acc: {best_val_acc*100:.2f}%)")
      else:
          patience_counter += 1
          if patience_counter >= PATIENCE:
              print(f"Early stopping triggered at epoch {epoch}. Terminating fine-tuning.")
              break
  ```
* **Why:** Saves ~20–30 minutes of Kaggle GPU quota per run and prevents validation overfitting.

---

### Change 4: Add Cosine Annealing Learning Rate Scheduler
* **File:** Both notebooks
* **What to change:** Wrap the AdamW optimizer with `torch.optim.lr_scheduler.CosineAnnealingLR`:
  ```python
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

  # Step the scheduler after each epoch:
  scheduler.step()
  ```
* **Why:** Smoothly anneals the learning rate towards zero as training progresses, allowing the ViT attention heads to settle cleanly into sharp decision boundaries.

---

## 5. Recommended Optimal Settings for Kaggle Runs

| Parameter | Recommended Setting | Rationale |
| :--- | :---: | :--- |
| **Contrastive Pretraining Epochs** | **30–50 epochs** | SimCLR requires longer pretraining for NT-Xent to separate negative clusters. |
| **Supervised Fine-Tuning Epochs (100% Data)** | **20–25 epochs** (with early stopping patience 7) | ViT fine-tuning on 16k samples converges by epoch 15–18. |
| **Benchmark Fraction Epochs (10%, 25%, 50%)** | **15 epochs** (with early stopping patience 5) | Provides enough gradient steps for 1,600 samples without excessive runtime. |
| **Batch Size** | **32** (or 16 if VRAM is constrained) | Higher batch sizes improve SimCLR negative sample quality. |
| **Scheduler** | `CosineAnnealingLR` | Mandatory for stable Vision Transformer convergence. |
