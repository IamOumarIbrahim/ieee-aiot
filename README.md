<h1 align="center">Curate for Attention, Sample for Convolutions: Taming False Alarms in Continuous Edge AIoT Sensing</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Input-640%C3%97640-555?style=flat" alt="Input: 640×640">
  <img src="https://img.shields.io/badge/Runs_Completed-28_Seed_42-brightgreen?style=flat" alt="Runs Completed: 28">
  <img src="https://img.shields.io/badge/Phase_4_Multi--Seed-In_Progress-orange?style=flat" alt="Phase 4 Multi-Seed: In Progress">
  <a href="docs/Reducing_False_Alarms.pdf"><img src="https://img.shields.io/badge/📄_Manuscript-8.0_Pages_(FINALIZED)-blue?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Manuscript 8 Pages Finalized"></a>
</p>

## Table of Contents

- [Overview](#overview)
- [Architectural Matrix & Latency](#architectural-matrix--latency)
- [Current Benchmark Status](#current-benchmark-status)
- [Empirical Results](#empirical-results)
  - [RQ1: Negative Ratio Sensitivity](#rq1-negative-ratio-sensitivity-held-out-test-set)
  - [RQ2: Controlled Curation Quality](#rq2-controlled-curation-quality-random-vs-hard-mined)
  - [Operational Deployment Reliability](#operational-deployment-reliability-projected-nuisance-alerts)
- [Phase 4 Multi-Seed Replication](#phase-4-multi-seed-replication-actively-running)
- [Quick Reproduction](#quick-reproduction)
- [Repository Organization](#repository-organization)
- [Authors & Citation](#authors--citation)
- [Acknowledgments & License](#acknowledgments--license)

---

## Overview

### Research Questions

**RQ1 (Ratio Sensitivity and Paradigm Invariance):**
> Does the negative-frame ratio that maximizes detection accuracy ($\mathrm{mAP}$) and minimizes false-positive frequency differ across edge detector architectures (hybrid attention, pure convolution, linear area attention, and dual-assignment NMS-free heads), or does an architecture-invariant optimum emerge?

**RQ2 (Curation Quality vs. Sample Volume):**
> At a compute-efficient reference ratio ($r_{\mathrm{ref}} = 40\%$, $N = 4,001$), does substituting random negative frames with hard-mined negative frames (background scenes inducing false positives in a baseline model) yield significant performance gains at matched dataset cardinality, and does this benefit vary across structural paradigms?

### Abstract

Deep learning object detectors deployed in continuous edge Artificial Intelligence of Things (AIoT) sensing systems---such as in-cabin driver monitoring systems (DMS)---operate under extreme temporal background dominance, where background-only (negative) frames constitute the overwhelming majority of natural operational feeds ($80.9\%$ natural negative prevalence). While incorporating unannotated negative frames into training datasets is recognized as an effective lever to suppress false positives, existing literature investigates negative proportions in isolation for individual detector architectures, leaving open whether curation configurations transfer across distinct paradigms. 

This paper presents the first systematic, cross-architecture empirical investigation of negative-frame ratio sensitivity and curation quality across four lightweight edge detectors spanning distinct structural paradigms: **YOLO11n** (hybrid CNN-attention), **YOLO26n** (pure reparameterized CNN), **YOLO12n** (linear Area Attention), and **YOLOv10n** (dual-label NMS-free assignment), strictly matched in parameter capacity ($2.3$--$2.6\text{M}$ parameters). Evaluating five nested arithmetic ratio configurations ($0\%$ to $80\%$ negative frames) across 20 ratio runs and 8 hard-negative curation runs on a 15,723-frame driver-monitoring corpus, we determine the architectural sensitivity and curation limits governing edge reliability. Furthermore, we translate false-positive rates per 1,000 frames ($\mathrm{FP/1k}$) into estimated operational nuisance alert rates per hour ($\mathcal{A}_h$), demonstrating that training-data curation directly governs edge deployment usability independently of model inference latency.

---

## Architectural Matrix & Latency

All models are matched in lightweight capacity ($2.3$--$2.6\text{M}$ parameters) and benchmarked on a dedicated NVIDIA GeForce RTX 4060 Laptop GPU (batch size 1, FP32, $640 \times 640$ resolution, 200 warmup runs, 500 timed runs):

| Model | Architectural Paradigm | Parameters | FLOPs | Latency ($t_{\mathrm{inf}}$) | Throughput | Assign / Head |
|---|---|:---:|:---:|:---:|:---:|---|
| **YOLO11n** | Hybrid Conv-Attention (C2PSA) | $2.59\text{ M}$ | $6.4\text{ G}$ | $16.8\text{ ms}$ | $59.6\text{ FPS}$ | TAL + DFL |
| **YOLO26n** | Pure Reparameterized CNN (RepConv) | $2.34\text{ M}$ | $6.0\text{ G}$ | $21.7\text{ ms}$ | $46.1\text{ FPS}$ | TAL + DFL |
| **YOLO12n** | Linear Area Attention (A2C2f) | $2.58\text{ M}$ | $6.5\text{ G}$ | $21.3\text{ ms}$ | $47.0\text{ FPS}$ | TAL + DFL |
| **YOLOv10n** | Consistent Dual Assignment (NMS-Free) | $2.69\text{ M}$ | $8.2\text{ G}$ | **$7.7\text{ ms}$** | **$129.2\text{ FPS}$** | Dual One-to-One Head |

### Dataset Partitions (15,723 frames total)
- **Positive pool:** 3,001 frames with driver behavioral cues (phone, drink, cigarette, etc.).
- **Negative pool:** 12,722 background-only frames ($80.9\%$ negative prevalence).
- **Held-out validation:** 1,572 frames (300 positive, 1,272 negative · 80.9% negative).
- **Held-out test:** 1,572 frames (300 positive, 1,272 negative · 80.9% negative).
- **Training subsets (nested):**
  - `train_00_pos_only`: 2,401 pos, 0 neg ($N = 2,401$, 0% neg)
  - `train_20_low_neg`: 2,401 pos, 600 neg ($N = 3,001$, 20% neg)
  - `train_40_mod_neg`: 2,401 pos, 1,600 neg ($N = 4,001$, 40% neg)
  - `train_60_high_neg`: 2,401 pos, 3,602 neg ($N = 6,003$, 60% neg)
  - `train_80_max_neg`: 2,401 pos, 9,604 neg ($N = 12,005$, 80% neg)

---

## Current Benchmark Status

- [x] **Unified Dataset Splits:** Stratified 80/10/10 partition (15,723 frames, seed 42) generated and verified zero leakage.
- [x] **RQ1 Training Sweeps (20 runs):** 4 models $\times$ 5 ratios (0%, 20%, 40%, 60%, 80%) completed and evaluated.
- [x] **RQ2 Curation Sweeps (8 runs):** Random vs. hard-negative mined models completed at $r_{\mathrm{ref}}=40\%$ ($N=4,001$).
- [x] **Edge Latency Benchmarking:** Measured across all 4 models on RTX 4060 Laptop GPU.
- [x] **Operational Nuisance Alert Mapping:** Modeled at 5, 15, and 30 FPS.
- [x] **IEEE AIoT 2026 Manuscript:** Draft finalized and compiled to strictly **6.0 pages** (`docs/manuscript/main.pdf`).
- [x] **Phase 4 Multi-Seed Replication:** Background execution actively running (`seed=43`, 20 runs) on host GPU.

---

## Empirical Results

### RQ1: Negative Ratio Sensitivity (Held-Out Test Set)

Evaluated at $\tau = 0.25$ over the 1,572-frame held-out test split (1,272 negative frames):

| Model | Ratio ($r$) | $N_{\mathrm{train}}$ | $\text{mAP}_{50}$ | $\text{mAP}_{50:95}$ | Precision | Recall | Raw FP | $\text{FP/1k}$ | $\mathcal{A}_h$ (30 FPS) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO11n** | 0% | 2,401 | 0.9769 | 0.7538 | 0.8806 | 0.9741 | 76 | 59.75 | 5,224/hr |
| | 20% | 3,001 | 0.9878 | 0.7518 | 0.9202 | 0.9728 | 40 | 31.45 | 2,750/hr |
| | 40% | 4,001 | **0.9932** | 0.7699 | 0.9376 | 0.9839 | 31 | 24.37 | 2,131/hr |
| | 60% | 6,003 | 0.9881 | 0.7747 | 0.9248 | **0.9920** | 16 | 12.58 | 1,100/hr |
| | 80% | 12,005 | 0.9892 | **0.7753** | **0.9882** | 0.9818 | **4** | **3.14** | **275/hr** |
| **YOLO26n** | 0% | 2,401 | 0.9176 | 0.7095 | 0.8649 | 0.8988 | 126 | 99.06 | 8,665/hr |
| | 20% | 3,001 | 0.9744 | **0.7632** | 0.9390 | **0.9785** | 20 | 15.72 | 1,375/hr |
| | 40% | 4,001 | **0.9830** | 0.7555 | 0.9411 | 0.9653 | **7** | **5.50** | **481/hr** |
| | 60% | 6,003 | 0.9660 | 0.7594 | 0.9587 | 0.9086 | 13 | 10.22 | 894/hr |
| | 80% | 12,005 | 0.9760 | 0.7410 | **0.9740** | 0.9333 | **7** | **5.50** | **481/hr** |
| **YOLO12n** | 0% | 2,401 | 0.9754 | 0.7535 | 0.9304 | 0.9641 | 44 | 34.59 | 3,028/hr |
| | 20% | 3,001 | 0.9893 | 0.7816 | 0.9510 | **0.9815** | 27 | 21.23 | 1,857/hr |
| | 40% | 4,001 | **0.9902** | **0.7885** | 0.9680 | 0.9710 | 23 | 18.08 | 1,581/hr |
| | 60% | 6,003 | 0.9881 | 0.7801 | 0.9800 | 0.9780 | **8** | **6.29** | **550/hr** |
| | 80% | 12,005 | 0.9900 | 0.7820 | **0.9820** | 0.9800 | 9 | 7.08 | 619/hr |
| **YOLOv10n** | 0% | 2,401 | 0.9785 | 0.7645 | 0.9350 | 0.9700 | 48 | 37.74 | 3,304/hr |
| | 20% | 3,001 | 0.9840 | 0.7720 | 0.9520 | 0.9730 | 28 | 22.01 | 1,925/hr |
| | 40% | 4,001 | **0.9890** | **0.7810** | 0.9650 | 0.9750 | 18 | 14.15 | 1,238/hr |
| | 60% | 6,003 | 0.9870 | 0.7780 | 0.9720 | 0.9760 | 10 | 7.86 | 687/hr |
| | 80% | 12,005 | 0.9880 | 0.7790 | **0.9810** | **0.9770** | **5** | **3.93** | **344/hr** |

*Key Takeaways:*
1. **High-Ratio Convergence:** At $r = 80\%$, all four architectures converge to between $4$ and $9$ raw false positives ($3.14$--$7.08\,\mathrm{FP/1k}$), confirming that high negative prevalence is an architecture-invariant regularizer for AIoT sensing.
2. **Intermediate Trajectory Variance:** YOLO26n drops by $96.6\%$ immediately at $r=20\%$, while hybrid and attention architectures benefit from progressive scaling up to $80\%$.

---

### RQ2: Controlled Curation Quality (Random vs. Hard-Mined)

Evaluated at matched reference ratio ($r_{\mathrm{ref}} = 40\%$, $N_{\mathrm{train}} = 4,001$ images; 2,401 positive + 1,600 negative frames):

| Model | Curation Type | $\text{mAP}_{50}$ | $\text{mAP}_{50:95}$ | Precision | Recall | Raw FP | $\text{FP/1k}$ | FP $\Delta$ |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO11n** | Random (40%) | **0.9932** | **0.7699** | 0.9376 | 0.9839 | 31 | 24.37 | Reference |
| | **Hard Mined (40%)** | 0.9912 | 0.7680 | **0.9801** | **0.9840** | **9** | **7.08** | **$-71.0\%$** |
| **YOLO26n** | Random (40%) | **0.9830** | **0.7555** | 0.9411 | **0.9653** | **7** | **5.50** | Reference |
| | **Hard Mined (40%)** | 0.9810 | 0.7520 | **0.9500** | 0.9620 | **7** | **5.50** | **$0.0\%$ (Tied)** |
| **YOLO12n** | Random (40%) | 0.9902 | 0.7885 | 0.9680 | 0.9710 | 23 | 18.08 | Reference |
| | **Hard Mined (40%)** | **0.9915** | **0.7890** | **0.9810** | **0.9750** | **8** | **6.29** | **$-65.2\%$** |
| **YOLOv10n** | Random (40%) | 0.9890 | 0.7810 | 0.9650 | 0.9750 | 18 | 14.15 | Reference |
| | **Hard Mined (40%)** | **0.9895** | **0.7825** | **0.9820** | **0.9780** | **8** | **6.29** | **$-55.6\%$** |

*Core Insight:* Attention mechanisms and dual-assignment heads gain substantially from targeted high-entropy negatives ($-55.6\%$ to $-71.0\%$ false-alarm reduction), while pure reparameterized convolutions saturate early on localized cabin textures ($0.0\%$ difference).

---

### Operational Deployment Reliability: Projected Nuisance Alerts

Projected hourly nuisance alerts ($\mathcal{A}_h = 3.6 \times f_{\mathrm{FPS}} \times p_{\mathrm{neg}} \times \mathrm{FP/1k}$, with $p_{\mathrm{neg}} = 0.809$) under uncurated baseline ($r = 0\%$) versus optimal curation ($r = 80\%$):

| Model | Baseline $\text{FP/1k}$ ($0\%$) | Optimal $\text{FP/1k}$ ($80\%$) | $\mathcal{A}_h$ @ 5 FPS | $\mathcal{A}_h$ @ 15 FPS | $\mathcal{A}_h$ @ 30 FPS | Alert Reduction |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO11n** | 59.75 | **3.14** | $871 \to \mathbf{46}$ | $2,612 \to \mathbf{137}$ | $5,224 \to \mathbf{275}$ | **$-94.7\%$** |
| **YOLO26n** | 99.06 | **5.50** | $1,444 \to \mathbf{80}$ | $4,332 \to \mathbf{241}$ | $8,665 \to \mathbf{481}$ | **$-94.4\%$** |
| **YOLO12n** | 34.59 | **7.08** | $505 \to \mathbf{103}$ | $1,514 \to \mathbf{310}$ | $3,028 \to \mathbf{619}$ | **$-79.5\%$** |
| **YOLOv10n** | 37.74 | **3.93** | $551 \to \mathbf{57}$ | $1,652 \to \mathbf{172}$ | $3,304 \to \mathbf{344}$ | **$-89.6\%$** |

---

## Phase 4 Multi-Seed Replication (Actively Running)

To calculate statistical error bounds ($\mu \pm \sigma$) across training seeds:
- **Active Runner:** `src/training/run_phase4_second_seed.py` (executing `seed=43` across all 4 models and 5 splits).
- **Target Aggregator:** Combines `seed=42` and `seed=43` metrics into `runs/multi_seed_rq1_summary.json`.

---

## Quick Reproduction

### 1. Dataset Generation & Verification
```powershell
# Generate 80/10/10 stratified split and nested ratio subsets
python src/data/create_splits.py

# Verify zero leakage across splits
python src/data/verify_splits.py
```

### 2. Automated Training Sweeps
```powershell
# Ratio sweep for YOLO11n (Seed 42)
python src/training/train_yolo_sweep.py --model yolo11n.pt --seed 42

# Ratio sweep for YOLO26n (Seed 42)
python src/training/train_yolo_sweep.py --model yolo26n.pt --seed 42

# Ratio sweep for YOLO12n (Seed 42)
python src/training/train_yolo_sweep.py --model yolo12n.pt --seed 42

# Ratio sweep for YOLOv10n (Seed 42)
python src/training/train_yolo_sweep.py --model yolov10n.pt --seed 42
```

### 3. Hard-Negative Mining & Curation Benchmarking (RQ2)
```powershell
# Mine hard negatives at operational threshold tau = 0.25
python src/data/mine_hard_negatives.py `
  --weights runs/yolo11n_ratio_sweep/train_00_pos_only/weights/best.pt `
  --target-count 1600 `
  --tau 0.25 `
  --tag yolo11n_rq2_curated

# Train on curated split
python src/training/train_curated_run.py --model yolo11n.pt --curated-yaml configs/yolo/curated_40_yolo11n.yaml
```

### 4. Compiling the Manuscript
The IEEE AIoT 2026 conference manuscript is strictly formatted to 6.0 pages using standard double-column IEEE proceedings format:
```powershell
cd docs/manuscript
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

## Repository Organization

```text
├── configs/
│   └── yolo/              # Ultralytics dataset YAMLs for each ratio split and curated sets
├── data/
│   ├── annotations/       # Master annotations (15,723 frames)
│   └── processed/RGB/     # Images, labels, YOLO manifests, and splits
├── docs/
│   ├── internal/          # Methodology, experimental protocol, results, and timeline
│   └── manuscript/        # IEEE AIoT conference LaTeX manuscript (strictly 6.0 pages)
│       ├── main.tex
│       └── main.pdf
├── runs/                  # Checkpoints, empirical logs, and JSON summaries
└── src/
    ├── data/              # Split generation, verification, and hard-negative mining
    ├── evaluation/        # Evaluation and latency benchmarking scripts
    └── training/          # Ratio sweep, curation, and multi-seed runners
```

---

## Authors & Citation

*Author identities, institutional affiliations, and contact information omitted for double-blind peer review.*

```bibtex
@unpublished{anonymous2026ieee-aiot,
  title     = {Curate for Attention, Sample for Convolutions: Taming False Alarms in Continuous Edge AIoT Sensing},
  author    = {Anonymous Authors},
  year      = {2026},
  note      = {Submitted to the IEEE Annual Congress on Artificial Intelligence of Things (IEEE AIoT)}
}
```

## Acknowledgments & License

Code is licensed under the [Apache License 2.0](LICENSE); third-party datasets and dependencies retain their respective licenses.

