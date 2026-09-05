<h1 align="center">Negative Frames Aren't Architecture-Agnostic: A Cross-Detector Study for Edge Driver Monitoring</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Input-640%C3%97640-555?style=flat" alt="Input: 640×640">

</p>

<p align="center">
  <img src="https://img.shields.io/badge/📄_Manuscript-In_Preparation-yellow?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Manuscript In Preparation">
</p>

## Table of Contents

- [Overview](#overview)
- [Current Benchmark Status](#current-benchmark-status)
- [Quick Reproduction](#quick-reproduction)
- [Repository Organization](#repository-organization)
- [Authors & Citation](#authors--citation)
- [Acknowledgments & License](#acknowledgments--license)

## Overview

### Research Questions

**RQ1 (Ratio Sensitivity and Paradigm Invariance):**
> Does the negative-frame ratio that maximizes detection accuracy ($\mathrm{mAP}$) and minimizes false-positive frequency differ across detector architectures (anchor-free CNN vs. real-time DETR), or does a uniform, architecture-invariant optimum emerge?

**RQ2 (Curation Quality vs. Sample Volume):**
> At each architecture's empirically identified best ratio, does substituting random negative frames with hard-mined negative frames (background scenes inducing false positives in a baseline model) yield further performance gains at matched dataset cardinality, and does this benefit vary between CNN and transformer architectures?

### Abstract

Deep learning object detectors deployed in continuous edge Artificial Intelligence of Things (AIoT) sensing systems---such as in-cabin driver monitoring systems (DMS)---operate under extreme temporal background dominance, where background-only (negative) frames constitute the overwhelming majority of natural operational feeds ($\approx 81\%$ negative prevalence). While incorporating unannotated negative frames into training datasets is recognized as an effective lever to suppress false positives, existing literature investigates negative-to-positive proportions in isolation for individual detector architectures, leaving open whether an optimal configuration transfers across architecturally distinct detection paradigms. This paper presents the first systematic, cross-architecture empirical investigation of negative-frame ratio sensitivity and curation quality across three lightweight edge detectors spanning distinct structural paradigms: YOLO11n (anchor-free CNN), YOLO26n (next-generation CNN with reparameterized convolutions), and D-FINE-N (real-time DETR transformer with fine-grained distribution refinement). Evaluating five nested arithmetic ratio configurations (0% to 80% negative frames) and an empirical hard-negative mining protocol at matched dataset cardinality on a 15,723-frame driver-monitoring corpus, we determine whether performance-maximizing negative-frame configurations are architecture-invariant or architecture-specific. Furthermore, we translate false-positive rates per 1,000 frames ($\mathrm{FP/1k}$) into estimated operational nuisance alert rates per hour ($\mathcal{A}_h$), demonstrating that training-data curation directly governs edge deployment usability independently of model inference latency.

### Core Contributions

1. **Cross-Architecture Negative-Ratio Benchmark:** The first controlled empirical sensitivity sweep of negative-frame training ratios across three lightweight edge detectors spanning CNN and transformer paradigms on a safety-critical AIoT in-cabin monitoring dataset, establishing whether curation configurations transfer across detector lineages.
2. **Controlled Hard-Negative Curation Protocol:** An empirical false-positive mining protocol benchmarked against uniform random sampling at strictly matched dataset cardinality, isolating the contribution of sample information entropy from sample volume across diverse detector architectures.
3. **Operational Deployment Cost Reframing:** An analytical mapping translating measured false positives per 1,000 frames ($\mathrm{FP/1k}$) into estimated hourly nuisance alerts ($\mathcal{A}_h = 3.6 \times f_{\mathrm{FPS}} \times p_{\mathrm{neg}} \times \mathrm{FP/1k}$), demonstrating that data-level negative curation governs edge operational reliability independently of architectural inference latency.

### Experimental Matrix

| Axis | Levels |
|------|--------|
| Architecture | YOLO11n, YOLO26n, D-FINE-N |
| Negative ratio (random) | 0%, 20%, 40%, 60%, 80% (arithmetic 20% stepping) |
| Negative type (at each arch's best ratio only) | Random subsample vs. curated hard negatives |
| Core metrics | Precision, Recall, mAP@50, mAP@50:95, FP rate /1,000 frames |
| Efficiency context (report once per architecture) | Params/FLOPs/latency — used only to frame C3's cost discussion |

| Split Name | Ratio | Positive frames | Negative frames | Total training set |
|---|---|---|---|---|
| `train_00_pos_only` | 0% | 2,401 | 0 | 2,401 |
| `train_20_low_neg` | 20% | 2,401 | 600 | 3,001 |
| `train_40_mod_neg` | 40% | 2,401 | 1,600 | 4,001 |
| `train_60_high_neg` | 60% | 2,401 | 3,602 | 6,003 |
| `train_80_max_neg` | 80% | 2,401 | 9,604 | 12,005 |

> **Held-Out Benchmarks (Fixed Natural Distribution):**
> * **Validation:** 1,572 frames (300 positive, 1,272 negative · 80.9% neg)
> * **Test:** 1,572 frames (300 positive, 1,272 negative · 80.9% neg)


## Current Benchmark Status

- [x] **Dataset Curation & Splits:** Unified 80/10/10 stratified partition generated (`seed=42`).
- [x] **Negative Ratio Configurations:** 5 nested training configurations prepared (0%, 20%, 40%, 60%, 80%).
- [x] **Manifests & Configs:** YOLO text manifests, COCO JSONs, Ultralytics dataset YAMLs, and D-FINE YAMLs verified.
- [ ] **Detector Training:** YOLO11n, YOLO26n, and D-FINE-N ratio sweep runs.
- [ ] **Hard-Negative Mining (RQ2):** Best-ratio curation benchmark.
- [ ] **Inference & Profiling:** FP/1k frames and edge deployment cost analysis.

## Quick Reproduction

### 1. Generate Dataset Splits
Ensure dataset annotations (`data/annotations/RGB/annotations.json`) and images are present under `data/`. Split generation requires only standard Python libraries:
```bash
python src/data/create_splits.py
```
This deterministically generates all 5 experimental training configurations (`0%`, `20%`, `40%`, `60%`, `80%`) and held-out validation/test manifests (`SEED=42`).

### 2. Verify Split Integrity
Validate zero data leakage ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$), path integrity, and dataset configuration files:
```bash
python src/data/verify_splits.py
```

### 3. Train Detectors
All models are trained according to the frozen protocol documented in [docs/internal/experimental_protocol.md](docs/internal/experimental_protocol.md) (RTX 4060 8GB, single fixed `seed=42`).

#### Option A: Automated PowerShell Sequence Runner (Recommended)
Run dry-run verification or full multi-detector sweeps directly in PowerShell:
```powershell
# Dry-run validation (checks environment, CUDA, splits, manifests, and configs)
.\scripts\run_training_sequence.ps1 -Detector yolo11n -DryRun

# Full 5-ratio training sweep for YOLO11n (0%, 20%, 40%, 60%, 80%)
.\scripts\run_training_sequence.ps1 -Detector yolo11n

# Full 5-ratio training sweep for YOLO26n
.\scripts\run_training_sequence.ps1 -Detector yolo26n

# Full 5-ratio training sweep for D-FINE-N
.\scripts\run_training_sequence.ps1 -Detector dfine

# Smoke test on a single split (1 epoch)
.\scripts\run_training_sequence.ps1 -Detector yolo11n -Splits "00" -Epochs 1
```

#### Option B: Direct Python Ratio Sweep Scripts
```powershell
# YOLO11n ratio sweep
python src\training\train_yolo_sweep.py --model yolo11n.pt

# YOLO26n ratio sweep
python src\training\train_yolo_sweep.py --model yolo26n.pt

# D-FINE-N ratio sweep
python src\training\train_dfine_sweep.py --dfine-dir DFINE
```

#### Option C: Individual Split CLI Commands
```powershell
# Example: YOLO11n on the 20% negative split
yolo detect train data=configs/yolo/yolo_20_low_neg.yaml model=yolo11n.pt epochs=100 batch=16 imgsz=640 seed=42 close_mosaic=10 optimizer=auto amp=False
```

### 4. Hard-Negative Mining (RQ2 Curation)
Once baseline detector (`train_00_pos_only`) training finishes, curate hard negatives from the 10,178 background candidate pool at matched ratio count:
```powershell
python src\data\mine_hard_negatives.py `
  --weights runs\yolo11n_ratio_sweep\train_00_pos_only\weights\best.pt `
  --target-count 600 `
  --tau 0.25 `
  --tag yolo11n_best_curated
```


## Repository Organization

```text
├── configs/
│   ├── dfine/             # D-FINE dataset and model configs for each ratio split
│   └── yolo/              # Ultralytics dataset configs for each ratio split
├── data/
│   ├── annotations/       # Master annotations (15,723 frames)
│   └── processed/RGB/     # Images, labels, YOLO manifests, and COCO JSONs
├── docs/
│   ├── internal/          # Research questions, methodology, and notes
│   └── manuscript/        # IEEE AIoT conference manuscript
└── src/
    ├── data/              # Split generation and verification scripts
    └── training/          # Automated ratio sweep runners (YOLO & D-FINE)
```


## Authors & Citation

*Author identities, institutional affiliations, and contact information omitted for double-blind peer review.*

For the conference manuscript itself, use:

```bibtex
@unpublished{anonymous2026ieee-aiot,
  title     = {Negative Frames Are Not Architecture-Agnostic: A Cross-Detector Study of Negative-Frame Configuration for Edge AIoT Driver Monitoring},
  author    = {Anonymous Authors},
  year      = {2026},
  note      = {Submitted to the IEEE Annual Congress on Artificial Intelligence of Things (IEEE AIoT)}
}
```

## Acknowledgments & License

Code is licensed under [Apache License 2.0](LICENSE); third-party datasets and dependencies retain their own licenses.
