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

### Research Question

Is the optimal negative-frame configuration for training lightweight object detectors architecture-invariant, or does it depend on detector design — and what does that mean for false-positive-driven operational cost in AIoT deployment?

RQ1 (Sensitivity & Transferability):
> How do precision, recall, mAP, and false-positive rate respond to negative-frame ratio across YOLO11n, YOLO26n, and D-FINE-N — and is the performance-maximizing ratio the same across all three?

RQ2 (Curation Quality):
> At each architecture's empirically best ratio, does replacing random negatives with curated hard negatives (mined from false-positive-prone frames) produce further gains — and is that gain architecture-dependent?

### Abstract

Edge AIoT object detectors, such as in-cabin driver monitoring systems, continuously process video streams where background-only (negative) frames dominate natural operational distributions (~81% negative prevalence). While incorporating negative frames during training mitigates false positives, prior studies tune negative-to-positive proportions in isolation for individual detector architectures. This paper presents the first controlled cross-architecture empirical investigation of negative-frame ratio sensitivity and curation quality across three lightweight edge detectors spanning distinct paradigms: YOLO11n (CNN), YOLO26n (next-gen CNN), and D-FINE-N (real-time DETR transformer). Evaluating five nested arithmetic ratio configurations (0% to 80% negative frames) and false-positive hard-negative curation at matched dataset volumes on a 15,723-frame driver monitoring corpus, we investigate whether optimal negative configurations generalize across detector lineages. Finally, we translate false-positive rates per 1,000 frames into estimated operational nuisance alerts per hour, demonstrating that training data curation directly governs edge deployment usability independent of model inference latency.

### Contributions

1. First cross-architecture study of negative-frame ratio sensitivity on a real, safety-critical AIoT dataset (in-cabin driver monitoring) rather than a generic benchmark — testing a CNN-based detector (YOLO11n), its next generation (YOLO26n), and a transformer-based real-time detector (D-FINE-N), showing whether one "optimal ratio" generalizes or not.
2. A hard-negative curation protocol built from our own baseline's false-positive-prone frames, benchmarked against random subsampling at matched dataset size — quantifying whether curation effort is worth it, and for which architectures.
3. A deployment-cost reframing: translating negative-frame-driven FP-rate changes into estimated nuisance-alert/wasted-transmission cost per hour of operation — explicitly separated from inference latency, which is architecture-bound, not training-data-bound. Metric to track: false positives per 1,000 frames

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

Run the automated ratio sweep (trains all 5 ratio splits sequentially and aggregates test metrics):
```bash
# YOLO11n ratio sweep (0%, 20%, 40%, 60%, 80%)
python src/training/train_yolo_sweep.py --model yolo11n.pt

# YOLO26n ratio sweep
python src/training/train_yolo_sweep.py --model yolo26n.pt

# D-FINE-N ratio sweep
python src/training/train_dfine_sweep.py --dfine-dir DFINE
```

Alternatively, train individual splits directly with the CLI:
```bash
# Example: YOLO11n on the 20% negative split
yolo detect train data=configs/yolo/yolo_20_low_neg.yaml model=yolo11n.pt epochs=100 batch=16 imgsz=640 seed=42 close_mosaic=10 optimizer=auto amp=True
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


- **Oumar Mamoun Ibrahim** — Department of Computer Engineering, University of Sharjah<br>
  [U22200741@sharjah.ac.ae](mailto:U22200741@sharjah.ac.ae) · [ORCID 0009-0008-0312-1605](https://orcid.org/0009-0008-0312-1605)
- **Dr. Mohamad Khairi bin Ishak** — Department of Computer Engineering, University of Sharjah<br>
  [mishak@sharjah.ac.ae](mailto:mishak@sharjah.ac.ae) · [ORCID 0000-0002-3554-0061](https://orcid.org/0000-0002-3554-0061)

  
For the conference manuscript itself, use:

```bibtex
@unpublished{ibrahim2026ieee-aiot,
  title     = {Negative Frames Aren't Architecture-Agnostic: A Cross-Detector Study for Edge Driver Monitoring},
  author    = {Ibrahim, Oumar Mamoun and bin Ishak, Mohamad Khairi},
  year      = {2026},
  note      = {Conference paper to be submitted to the IEEE Annual Congress on Artificial Intelligence of Things (IEEE AIoT)}
}
```

## Acknowledgments & License

Code is licensed under [Apache License 2.0](LICENSE); third-party datasets and dependencies retain their own licenses.
