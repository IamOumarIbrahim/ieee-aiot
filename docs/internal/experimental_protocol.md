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

### Important Notes & Protocol Guardrails

* **One seed only.** Do not mention 13/37/73 anywhere in this paper. Training uses a single deterministic random seed (`seed=42`) across all models and ratio splits, yielding single point estimates.
* **No subject-disjoint split.** Do not import the 8/3/3 protocol from the other paper. Data partitioning uses the verified stratified 80/10/10 random partition across the 15,723-frame driver monitoring corpus.
* `optimizer: auto` should remain `auto`; report whatever optimizer Ultralytics actually selects (typically SGD or AdamW based on architecture defaults).
* D-FINE uses the official **0.0004 backbone / 0.0008 head** learning rates rather than artificially scaling them.
* D-FINE's batch 4 + accumulation 8 is a **hardware adaptation** for the 8 GB VRAM envelope, not an official native batch configuration.
* Don't claim effective batch 32 is equivalent to the official batch 128.
* Keep the configuration **identical across every negative-frame ratio**.

---

### Experimental Matrix & Ratio Configurations

All models are trained across 5 nested negative-frame ratio levels with a fixed positive core (2,401 frames) and evaluated on fixed, natural-distribution held-out validation and test sets (1,572 frames each, 80.9% negative prevalence):

| Split Identifier | Ratio | Positives | Negatives | Total Frames | Config Path |
|---|---|---|---|---|---|
| `train_00_pos_only` | 0% | 2,401 | 0 | 2,401 | `configs/yolo/yolo_00_pos_only.yaml` |
| `train_20_low_neg` | 20% | 2,401 | 600 | 3,001 | `configs/yolo/yolo_20_low_neg.yaml` |
| `train_40_mod_neg` | 40% | 2,401 | 1,600 | 4,001 | `configs/yolo/yolo_40_mod_neg.yaml` |
| `train_60_high_neg` | 60% | 2,401 | 3,602 | 6,003 | `configs/yolo/yolo_60_high_neg.yaml` |
| `train_80_max_neg` | 80% | 2,401 | 9,604 | 12,005 | `configs/yolo/yolo_80_max_neg.yaml` |

#### Total Training Run Budget

1. **RQ1 (Ratio Sweep):** 3 architectures × 5 ratio levels = **15 runs**
2. **RQ2 (Hard-Negative Curation):** 3 architectures × 2 curation conditions (random vs. hard-mined at best ratio) = **6 runs**
3. **Total Benchmark Scope:** **21 runs**

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
# D-FINE-N Example (Physical batch 4, grad accum 8 -> effective batch 32)
python train.py \
  --config configs/dfine/dfine_hgnetv2_n_coco.yml \
  --epochs 160 \
  --batch-size 4 \
  --accum-steps 8 \
  --amp \
  --seed 42 \
  --lr 0.0008 \
  --lr-backbone 0.0004 \
  --weight-decay 0.0001
```
