# Experimental Protocol & Detector Training Configuration

## Overview

This document specifies the frozen experimental protocol, hardware-adapted training configurations, and methodological constraints for all detector benchmarking runs (YOLO11n, YOLO26n, YOLO12n, YOLOv10n) on an **NVIDIA GeForce RTX 4060 Laptop/Desktop GPU (8 GB VRAM)**.

All runs adhere to a single controlled experimental variable: the **negative-frame ratio** (0%, 20%, 40%, 60%, 80%) in RQ1, and the negative curation strategy (random subsampling vs. hard-negative mining at matched cardinality $r_{\mathrm{ref}}=40\%$, $N=4,001$) in RQ2.

---

### Corrected & Frozen Configuration — RTX 4060 8GB

| Parameter | YOLO11n | YOLO26n | YOLO12n | YOLOv10n |
| :--- | :---: | :---: | :---: | :---: |
| **Input Resolution** | 640 × 640 | 640 × 640 | 640 × 640 | 640 × 640 |
| **Physical Batch Size** | 16 | 16 | 16 | 16 |
| **Gradient Accumulation** | 1 | 1 | 1 | 1 |
| **Effective Batch Size** | 16 | 16 | 16 | 16 |
| **Epochs** | 100 | 100 | 100 | 100 |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW |
| **Initial Learning Rate ($\mathrm{lr}_0$)** | 0.00125 | 0.00125 | 0.00125 | 0.00125 |
| **Weight Decay** | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
| **Precision** | FP32 (`amp=False`)† | FP32 (`amp=False`)† | FP32 (`amp=False`)† | FP32 (`amp=False`)† |
| **Augmentation Stop** | `close_mosaic=10` | `close_mosaic=10` | `close_mosaic=10` | `close_mosaic=10` |
| **Primary Random Seed** | 42 | 42 | 42 | 42 |
| **Replication Seed (Phase 4 Active)** | 43 | 43 | 43 | 43 |

---

### Key Methodological Statement

> All four lightweight models were trained using a frozen, unified optimization schedule across an NVIDIA RTX 4060 (8 GB VRAM) training environment. To isolate feature aggregation and head mechanics, identical hyperparameters (AdamW, $\mathrm{lr}_0=0.00125$, batch 16, 100 epochs, `close_mosaic=10`, FP32 precision) were held invariant across all five negative-frame ratio configurations and curation splits.

---

### Protocol Guardrails & Design Rationale

* **Frozen AdamW Optimization:** To circumvent instability and driver crashes with experimental optimizers (e.g., Muon optimizer crashing cuBLAS on large splits), all models are trained with AdamW ($\mathrm{lr}_0=0.00125$, weight decay 0.0005).
* **† FP32 Precision (Windows cuBLAS Fault Circumvention):** On Windows with PyTorch 2.6.0+cu124 on Ada Lovelace GPUs (RTX 4060), `cublasGemmStridedBatchedEx` encounters `CUBLAS_STATUS_INTERNAL_ERROR` under FP16/BF16 AMP. FP32 (`amp=False`) runs with numerical stability and consumes ~1.5 to 2.2 GB of VRAM at batch 16.
* **Stratified Frame-Level Partitioning (80/10/10):** Data partitioning adheres strictly to a stratified 80/10/10 random partition across the 15,723-frame corpus (`seed=42`): 1,572 val frames, 1,572 test frames (80.9% negative prevalence), and 2,401 positive training frames held invariant across all configurations.
* **Matched Model Capacity:** All four evaluated detectors belong to the lightweight "nano" family ($2.3$--$2.6\text{M}$ parameters, $6.2$--$7.6$ GFLOPs, and latency under $22$\,ms on the host edge GPU), isolating spatial feature aggregation from parameter scaling.
* **Multi-Seed Replication (Phase 4):** While initial empirical findings were established under a frozen deterministic seed (`seed=42`), Phase 4 (`src/training/run_phase4_second_seed.py`) replicates the full 20-run matrix under `seed=43` to establish empirical variance bounds (mean $\pm$ std).

---

### Experimental Matrix & Ratio Configurations

All models are trained across 5 nested negative-frame ratio levels with a fixed positive core (2,401 frames) and evaluated on fixed held-out validation and test sets (1,572 frames each, 80.9% negative prevalence):

| Split Identifier | Ratio | Positives | Negatives | Total Frames | Ultralytics Config Path |
|---|---|---|---|---|---|
| `train_00_pos_only` | 0% | 2,401 | 0 | 2,401 | `configs/yolo/yolo_00_pos_only.yaml` |
| `train_20_low_neg` | 20% | 2,401 | 600 | 3,001 | `configs/yolo/yolo_20_low_neg.yaml` |
| `train_40_mod_neg` | 40% | 2,401 | 1,600 | 4,001 | `configs/yolo/yolo_40_mod_neg.yaml` |
| `train_60_high_neg` | 60% | 2,401 | 3,602 | 6,003 | `configs/yolo/yolo_60_high_neg.yaml` |
| `train_80_max_neg` | 80% | 2,401 | 9,604 | 12,005 | `configs/yolo/yolo_80_max_neg.yaml` |

#### Completed Single-Seed Benchmark Scope (`seed=42`):
1. **RQ1 (Ratio Sweep):** 4 architectures × 5 ratio levels = **20 runs**
2. **RQ2 (Hard-Negative Curation):** 4 architectures × 2 curation conditions (random vs. hard-mined at $r_{\mathrm{ref}}=40\%$) = **8 runs**
3. **Total Completed Benchmark Runs:** **28 runs**

#### RQ2 Hard-Negative Mining Specification:
* **Baseline Detector:** Checkpoint trained on `train_00_pos_only` (0% negative baseline).
* **Mining Candidate Pool:** Full training negative candidate pool (10,178 background frames).
* **Confidence Threshold ($\tau$):** $\tau = 0.25$ (frames yielding false-positive detections $\ge 0.25$ confidence are retained).
* **Sample Matching:** Exactly matched cardinality ($N=4,001$ frames, $N_{\mathrm{target}}=1,600$ negatives); backfilled deterministically with remaining negative pool samples (`seed=42`).

---

### Active Ongoing Execution: Phase 4 Multi-Seed Robustness Sweep (`seed=43`)

* **Runner:** `src/training/run_phase4_second_seed.py --device 0`
* **Process Status:** Currently running on the NVIDIA RTX 4060 Laptop GPU.
* **Scope:** 4 architectures (YOLO11n, YOLO26n, YOLO12n, YOLOv10n) $\times$ 5 splits = **20 runs total**.
* **Aggregation Target:** Computes mean $\pm$ standard deviation across Seed 42 and Seed 43, outputting to `runs/multi_seed_rq1_summary.json` to ground confidence intervals for Table III.

---

### Execution Commands (PowerShell Syntax)

#### 1. Phase 4 Multi-Seed Runner (Currently Running)
```powershell
python src\training\run_phase4_second_seed.py --device 0
```

#### 2. Full Architecture Sweeps (Seed 42 Baseline)
```powershell
# YOLO11n ratio sweep
python src\training\train_yolo_sweep.py --model yolo11n.pt --device 0

# YOLO26n ratio sweep
python src\training\train_yolo_sweep.py --model yolo26n.pt --device 0

# YOLO12n ratio sweep
python src\training\train_yolo_sweep.py --model yolo12n.pt --device 0

# YOLOv10n ratio sweep
python src\training\train_yolo_sweep.py --model yolov10n.pt --device 0
```

#### 3. RQ2 Curated Sweep Execution
```powershell
python scripts\run_rq2_curated_sweep.py --device 0
```


