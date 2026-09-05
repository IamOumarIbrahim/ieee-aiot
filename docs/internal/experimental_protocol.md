# Experimental Protocol & Detector Training Configuration

## Overview

This document specifies the frozen experimental protocol, hardware-adapted training configurations, and methodological constraints for all detector benchmarking runs (YOLO11n, YOLO26n, D-FINE-N) on an **NVIDIA GeForce RTX 4060 Laptop/Desktop GPU (8 GB VRAM)**.

All runs adhere to a single controlled experimental variable: the **negative-frame ratio** (0%, 20%, 40%, 60%, 80%) in RQ1, and the negative curation strategy (random subsampling vs. hard-negative mining at matched size) in RQ2.

---

### Corrected & Frozen Configuration — RTX 4060 8GB

| Parameter                 |           YOLO11n |           YOLO26n |         D-FINE-N |
| ------------------------- | ----------------: | ----------------: | ---------------: |
| **Input Resolution**      |         640 × 640 |         640 × 640 |        640 × 640 |
| **Physical Batch Size**   |                16 |                16 |                4 |
| **Gradient Accumulation** |                 1 |                 1 |                8 |
| **Effective Batch Size**  |                16 |                16 |               32 |
| **Epochs**                |               100 |               100 |              160 |
| **Optimizer**             |            `auto` |            `auto` |            AdamW |
| **AMP**                   |                 ✓ |                 ✓ |                ✓ |
| **Seed**                  |  **1 fixed seed** |  **1 fixed seed** | **1 fixed seed** |
| **Native Augmentation**   |                 ✓ |                 ✓ |                ✓ |
| **Augmentation Stop**     | `close_mosaic=10` | `close_mosaic=10` | `stop_epoch=148` |
| **Backbone LR**           |                 — |                 — |       **0.0004** |
| **Head / Transformer LR** |                 — |                 — |       **0.0008** |
| **Weight Decay**          |            0.0005 |            0.0005 |           0.0001 |
| **EMA Restart Decay**     |                 — |                 — |           0.9999 |

---

### Key Methodological Statement

> The models were trained using their respective native optimization and augmentation frameworks. Hardware-constrained batch sizes were adjusted for the RTX 4060 (8 GB), with gradient accumulation applied only to D-FINE-N. Official learning rates, weight decay, augmentation schedules, and training budgets were retained where applicable. The same fixed random seed and training configuration were used across all negative-frame ratio configurations, with the negative-frame ratio as the controlled experimental variable.

---

### Protocol Guardrails & Design Rationale

* **Single Fixed Random Seed (`seed=42`):** A single deterministic random seed is used across all models and ratio configurations to manage computational budgets on single-GPU hardware while avoiding multi-seed averaging that could confound cross-ratio trends.
* **Stratified Frame-Level Partitioning (80/10/10):** Data partitioning adheres strictly to a stratified 80/10/10 random partition across the 15,723-frame driver monitoring corpus, ensuring balanced driver-cue distributions across splits without subject-disjoint leakage.
* **Framework-Native Optimizer Reporting:** Ultralytics `optimizer: auto` is retained as-is, recording the exact framework-selected optimizer (SGD/AdamW) for empirical reporting transparency.
* **Official D-FINE Optimization Rates:** Unscaled official learning rates (0.0004 backbone / 0.0008 head) are retained rather than artificially scaled down for reduced batch sizes.
* **Hardware Adaptation vs. Native Batch:** D-FINE-N's physical batch 4 + accumulation 8 is documented explicitly as an 8 GB VRAM hardware adaptation, avoiding any invalid claim of strict numerical equivalence to the official distributed batch size of 128.
* **Cross-Ratio Hyperparameter Invariance:** Training hyperparameters, augmentation cooldowns, and optimizer configurations remain strictly identical across all 5 negative-ratio levels within each detector lineage, isolating negative-frame prevalence as the sole experimental variable.

---

### Experimental Matrix & Ratio Configurations

All models are trained across 5 nested negative-frame ratio levels with a fixed positive core (2,401 frames) and evaluated on fixed, natural-distribution held-out validation and test sets (1,572 frames each, 80.9% negative prevalence):

| Split Identifier | Ratio | Positives | Negatives | Total Frames | YOLO Config Path | D-FINE Config Path |
|---|---|---|---|---|---|---|
| `train_00_pos_only` | 0% | 2,401 | 0 | 2,401 | `configs/yolo/yolo_00_pos_only.yaml` | `configs/dfine/dfine_00_pos_only.yml` |
| `train_20_low_neg` | 20% | 2,401 | 600 | 3,001 | `configs/yolo/yolo_20_low_neg.yaml` | `configs/dfine/dfine_20_low_neg.yml` |
| `train_40_mod_neg` | 40% | 2,401 | 1,600 | 4,001 | `configs/yolo/yolo_40_mod_neg.yaml` | `configs/dfine/dfine_40_mod_neg.yml` |
| `train_60_high_neg` | 60% | 2,401 | 3,602 | 6,003 | `configs/yolo/yolo_60_high_neg.yaml` | `configs/dfine/dfine_60_high_neg.yml` |
| `train_80_max_neg` | 80% | 2,401 | 9,604 | 12,005 | `configs/yolo/yolo_80_max_neg.yaml` | `configs/dfine/dfine_80_max_neg.yml` |

#### Total Training Run Budget

1. **RQ1 (Ratio Sweep):** 3 architectures × 5 ratio levels = **15 runs**
2. **RQ2 (Hard-Negative Curation):** 3 architectures × 2 curation conditions (random vs. hard-mined at best ratio) = **6 runs**
3. **Total Benchmark Scope:** **21 runs**

#### RQ2 Hard-Negative Mining Specification
* **Baseline Detector:** Checkpoint trained on `train_00_pos_only` (0% negative baseline).
* **Mining Candidate Pool:** Full training negative candidate pool (10,178 background frames).
* **Confidence Threshold ($\tau$):** $\tau = 0.25$ (frames yielding false-positive detections $\ge 0.25$ confidence are retained).
* **Sample Matching:** Exactly matches the negative-frame count of the respective architecture's best RQ1 ratio; if mined frames $< N_{\text{target}}$, backfill with random negative samples.

---

### Command Templates (RTX 4060 8GB)

#### YOLO11n & YOLO26n (Ultralytics)
```bash
# YOLO11n Example (20% negative split)
yolo detect train \
  data=configs/yolo/yolo_20_low_neg.yaml \
  model=yolo11n.pt \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  amp=True \
  seed=42 \
  close_mosaic=10 \
  weight_decay=0.0005 \
  optimizer=auto \
  device=0

# YOLO26n Example (20% negative split)
yolo detect train \
  data=configs/yolo/yolo_20_low_neg.yaml \
  model=yolo26n.pt \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  amp=True \
  seed=42 \
  close_mosaic=10 \
  weight_decay=0.0005 \
  optimizer=auto \
  device=0
```

#### D-FINE-N
```bash
# D-FINE-N Example (20% negative split: Physical batch 4, grad accum 8 -> effective batch 32)
python train.py \
  --config configs/dfine/dfine_20_low_neg.yml \
  --epochs 160 \
  --batch-size 4 \
  --accum-steps 8 \
  --amp \
  --seed 42 \
  --lr 0.0008 \
  --lr-backbone 0.0004 \
  --weight-decay 0.0001
```
