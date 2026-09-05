<h1 align="center">Negative Frames Aren't Architecture-Agnostic: A Cross-Detector Study for Edge Driver Monitoring</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Input-640%C3%97640-555?style=flat" alt="Input: 640×640">

</p>

<p align="center">
  <a href="https://raw.githubusercontent.com/IamOumarIbrahim/ieee-aiot/main/docs/manuscript/main.pdf" download="main.pdf"><img src="https://img.shields.io/badge/📄_Manuscript-Download_PDF-e02424?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Download the manuscript PDF"></a>
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


### Contributions

1. First cross-architecture study of negative-frame ratio sensitivity on a real, safety-critical AIoT dataset (in-cabin driver monitoring) rather than a generic benchmark — testing a CNN-based detector (YOLO11n), its next generation (YOLO26n), and a transformer-based real-time detector (D-FINE-N), showing whether one "optimal ratio" generalizes or not.
2. A hard-negative curation protocol built from our own baseline's false-positive-prone frames, benchmarked against random subsampling at matched dataset size — quantifying whether curation effort is worth it, and for which architectures.
3. A deployment-cost reframing: translating negative-frame-driven FP-rate changes into estimated nuisance-alert/wasted-transmission cost per hour of operation — explicitly separated from inference latency, which is architecture-bound, not training-data-bound. Metric to track: false positives per 1,000 frames

### Experimental Matrix

| Axis | Levels |
|------|--------|
| Architecture | YOLO11n, YOLO26n, D-FINE-N |
| Negative ratio (random) | 0%, 20%, 40%, 60%, ~81% (existing full dataset — zero extra labeling) |
| Negative type (at each arch's best ratio only) | Random subsample vs. curated hard negatives |
| Core metrics | Precision, Recall, mAP@50, mAP@50:95, FP rate /1,000 frames |
| Efficiency context (report once per architecture) | Params/FLOPs/latency — used only to frame C3's cost discussion |

| Split Name | Ratio | Positive frames | Negative frames | Total training set |
|---|---|---|---|---|
| `train_00_pos_only` | 0% | 2,401 | 0 | 2,401 |
| `train_20_low_neg` | 20% | 2,401 | 600 | 3,001 |
| `train_40_mod_neg` | 40% | 2,401 | 1,600 | 4,001 |
| `train_60_high_neg` | 60% | 2,401 | 3,602 | 6,003 |
| `train_81_nat_full` | ~81% (natural pool) | 2,401 | 10,178 | 12,579 |

> **Held-Out Benchmarks (Fixed Natural Distribution):**
> * **Validation:** 1,572 frames (300 positive, 1,272 negative · 80.9% neg)
> * **Test:** 1,572 frames (300 positive, 1,272 negative · 80.9% neg)


## Current Benchmark Status

## Quick Reproduction

## Repository Organization

## Authors & Citation


- **Oumar Mamoun Ibrahim** — Department of Computer Engineering, University of Sharjah<br>
  [U22200741@sharjah.ac.ae](mailto:U22200741@sharjah.ac.ae) · [ORCID 0009-0008-0312-1605](https://orcid.org/0009-0008-0312-1605)
- **Dr. Mohamad Khairi bin Ishak** — Department of Computer Engineering, University of Sharjah<br>
  [mishak@sharjah.ac.ae](mailto:mishak@sharjah.ac.ae) · [ORCID 0000-0002-3554-0061](https://orcid.org/0000-0002-3554-0061)

  
For the conference manuscript itself, use:

```bibtex
@unpublished{ibrahim2026ieee-aiot,
  title     = {},
  author    = {Ibrahim, Oumar Mamoun and bin Ishak, Mohamad Khairi},
  year      = {2026},
  note      = {Conference paper to be submitted to the IEEE Annual Congress on Artificial Intelligence of Things (IEEE AIoT)}
}
```

## Acknowledgments & License

This work builds on . Code is licensed under [Apache License 2.0](LICENSE); third-party datasets and dependencies retain their own licenses.
