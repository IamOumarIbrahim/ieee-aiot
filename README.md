<h1 align="center">Curate for Attention, Sample for Convolutions: Taming False Alarms in Continuous Edge AIoT Sensing</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Input-640%C3%97640-555?style=flat" alt="Input: 640×640">
  <img src="https://img.shields.io/badge/Runs_Completed-28_Seed_42-brightgreen?style=flat" alt="Runs Completed: 28">
  <a href="docs/Reducing_False_Alarms.pdf"><img src="https://img.shields.io/badge/📄_Manuscript-8.0_Pages_(FINALIZED)-blue?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Manuscript 8 Pages Finalized"></a>
</p>

## Research Questions

- **RQ1 (Ratio Sensitivity):** Does the negative-frame ratio that maximizes mAP and minimizes false-positive frequency differ across edge detector architectures, or does an architecture-invariant optimum emerge?
- **RQ2 (Curation Quality):** At a compute-efficient reference ratio ($r_{\mathrm{ref}} = 40\%$, $N = 4{,}001$), does substituting random negatives with hard-mined negatives yield significant performance gains at matched dataset cardinality, and does this benefit vary across structural paradigms?

## Abstract

Deep learning object detectors deployed in continuous edge AIoT sensing systems—such as driver monitoring systems (DMS)—operate under extreme temporal background dominance, where background-only frames constitute $80.9\%$ of natural operational feeds. This paper presents the first systematic, cross-architecture empirical investigation of negative-frame ratio sensitivity and curation quality across four lightweight edge detectors (**YOLO11n**, **YOLO26n**, **YOLO12n**, **YOLOv10n**), strictly matched in parameter capacity ($2.3$–$2.6\text{ M}$). Evaluating five nested ratio configurations ($0\%$–$80\%$) across 20 ratio runs and 8 hard-negative curation runs on a 15,723-frame driver-monitoring corpus, we determine the architectural sensitivity and curation limits governing edge reliability.

---

## Architectural Matrix & Latency

All models benchmarked on NVIDIA GeForce RTX 4060 Laptop GPU (batch 1, FP32, $640 \times 640$, 500 timed runs):

| Model | Architectural Paradigm | Parameters | FLOPs | Latency | Throughput | Assign / Head |
|---|---|:---:|:---:|:---:|:---:|---|
| **YOLO11n** | Hybrid Conv-Attention (C2PSA) | $2.59\text{ M}$ | $6.4\text{ G}$ | $16.8\text{ ms}$ | $59.6\text{ FPS}$ | TAL + DFL |
| **YOLO26n** | Pure Reparameterized CNN (RepConv) | $2.34\text{ M}$ | $6.0\text{ G}$ | $21.7\text{ ms}$ | $46.1\text{ FPS}$ | TAL + DFL |
| **YOLO12n** | Linear Area Attention (A2C2f) | $2.58\text{ M}$ | $6.5\text{ G}$ | $21.3\text{ ms}$ | $47.0\text{ FPS}$ | TAL + DFL |
| **YOLOv10n** | Consistent Dual Assignment (NMS-Free) | $2.69\text{ M}$ | $8.2\text{ G}$ | **$7.7\text{ ms}$** | **$129.2\text{ FPS}$** | Dual One-to-One Head |

### Dataset Partitions (15,723 frames total)
- **Positive pool:** 3,001 frames with driver behavioral cues (phone, drink, cigarette, etc.).
- **Negative pool:** 12,722 background-only frames ($80.9\%$ negative prevalence).
- **Held-out validation/test:** 1,572 frames each (300 positive, 1,272 negative · 80.9% negative).
- **Training subsets (nested):** `train_00` 0% neg ($N=2{,}401$) → `train_20` 20% ($N=3{,}001$) → `train_40` 40% ($N=4{,}001$) → `train_60` 60% ($N=6{,}003$) → `train_80` 80% ($N=12{,}005$).

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
1. **High-Ratio Convergence:** At $r = 80\%$, all four architectures converge to $4$–$9$ raw FP ($3.14$–$7.08\,\mathrm{FP/1k}$), confirming high negative prevalence is an architecture-invariant regularizer.
2. **Intermediate Trajectory Variance:** YOLO26n drops $96.6\%$ at $r=20\%$, while hybrid and attention architectures benefit from progressive scaling to $80\%$.

---

### RQ2: Controlled Curation Quality (Random vs. Hard-Mined)

Evaluated at matched reference ratio ($r_{\mathrm{ref}} = 40\%$, $N_{\mathrm{train}} = 4{,}001$; 2,401 positive + 1,600 negative frames):

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

## Quick Reproduction

```powershell
# 1. Generate splits and verify zero leakage
python src/data/create_splits.py && python src/data/verify_splits.py

# 2. RQ1 ratio sweep (repeat for yolo26n.pt, yolo12n.pt, yolov10n.pt)
python src/training/train_yolo_sweep.py --model yolo11n.pt --seed 42

# 3. RQ2 hard-negative mining + curation run
python src/data/mine_hard_negatives.py --weights runs/yolo11n_ratio_sweep/train_00_pos_only/weights/best.pt --target-count 1600 --tau 0.25 --tag yolo11n_rq2_curated
python src/training/train_curated_run.py --model yolo11n.pt --curated-yaml configs/yolo/curated_40_yolo11n.yaml

# 4. Compile manuscript
cd docs/manuscript && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
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

## License

Code is licensed under the [Apache License 2.0](LICENSE); third-party datasets and dependencies retain their respective licenses.
