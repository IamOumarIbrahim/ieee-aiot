# Development Log

## [2026-09-05] - Experimental Protocol & Training Configuration Freeze (RTX 4060 8GB)

### Overview
Formalized and froze the detector training protocol across YOLO11n, YOLO26n, and D-FINE-N tailored for an NVIDIA RTX 4060 (8 GB VRAM) training environment. Established hardware-adapted batch execution parameters, native optimization budgets, official learning rates, and explicit methodological guardrails.

---

### Key Changes & Implementations

#### 1. Hardware-Adapted Configuration Freeze
* Established unified 640 × 640 resolution and mixed precision (AMP) across all three detector families.
* Configured physical batch size 16 for YOLO11n and YOLO26n, and physical batch size 4 with 8-step gradient accumulation for D-FINE-N (effective batch size 32), documented explicitly as a hardware adaptation rather than an official native batch configuration.
* Locked training budgets and augmentation cooldowns: 100 epochs with `close_mosaic=10` for YOLO models; 160 epochs with `stop_epoch=148` for D-FINE-N.
* Retained official unscaled learning rates for D-FINE-N (0.0004 backbone, 0.0008 head/transformer, 0.0001 weight decay, 0.9999 EMA restart decay). Retained Ultralytics optimizer `auto` selection with 0.0005 weight decay.
* Generated complete suite of D-FINE-N configurations in `configs/dfine/` (`dfine_hgnetv2_n_coco.yml` and ratio-specific configs `dfine_00_pos_only.yml` through `dfine_80_max_neg.yml`), validated paths to `data/processed/RGB/coco/dfine/`, and integrated automated D-FINE YAML parsing into `src/data/verify_splits.py`.

#### 2. Protocol Guardrails & Methodological Alignment
* **Single fixed random seed:** Confirmed single-seed evaluation (`seed=42`) without multi-seed averaging (no 13/37/73 seeds) to respect compute budgets while ensuring zero cross-ratio confounding.
* **No subject-disjoint split:** Kept strictly to the verified 80/10/10 stratified random partition across 15,723 frames; rejected subject-disjoint 8/3/3 splits from other protocols.
* Created `docs/internal/experimental_protocol.md` with complete tabular specifications, key methodological statement, and execution commands.
* Synchronized `docs/internal/methodology.md` (Section III-A/C frame counts and Section III-D Training Protocol) and `README.md`.
* Resolved internal documentation inconsistencies: fixed filename typo `introduction.md`, timeline date typo, restored Section III-E Evaluation Metrics, aligned table enumeration (Tables I–V), and harmonized Section IV-C/V operational cost cross-references across all drafts.
* Established `docs/internal/results.md` blueprint for Section IV (IV-A ratio sweep, IV-B curation quality, IV-C deployment cost translation) embedding explicit single-seed claims boundaries.
* Enhanced `src/data/verify_splits.py` to assert disk existence of train, val, and test dataloaders across all D-FINE configs, formalized bibliography in `docs/internal/related_work.md`, and drafted the official Abstract in `README.md`.
* Formalized RQ2 hard-negative mining threshold ($\tau = 0.25$) and edge deployment platform (NVIDIA Jetson Orin Nano 8 GB) in `docs/internal/methodology.md` and `docs/internal/experimental_protocol.md`.
* Implemented automated multi-split sweep runners (`src/training/train_yolo_sweep.py` and `src/training/train_dfine_sweep.py`) with automatic held-out test evaluation, FP/1k computation, and JSON summary logging for unattended execution.

---

## [2026-09-05] - Experimental Framework & Dataset Split Configuration

### Overview
Prepared the repository for multi-detector negative-frame ratio benchmarking (YOLO11n, YOLO26n, D-FINE-N). Implemented stratified dataset splitting, generated the 5 arithmetic negative-to-positive ratio configurations (0%, 20%, 40%, 60%, 80%), created detector configuration files, and established an automated verification pipeline.

---

### Key Changes & Implementations

#### 1. Experimental Framing & Semantic Split Naming
* Formulated core research questions (RQ1: ratio sensitivity and architecture invariance; RQ2: random vs. hard-negative curation; C3: false-positive operational cost translation).
* Defined semantic naming conventions for experimental splits:
  * `train_00_pos_only` (0% Negative / Zero-Negative Baseline)
  * `train_20_low_neg` (20% Negative / Low-Negative)
  * `train_40_mod_neg` (40% Negative / Moderate-Negative)
  * `train_60_high_neg` (60% Negative / High-Negative)
  * `train_80_max_neg` (80% Negative / Dominant-Negative Cap)

#### 2. Dataset Partitioning (Stratified 80/10/10 Random Split)
* Adopted an 80/10/10 frame-level random partition across the entire 15,723-frame driver monitoring corpus (3,001 positive, 12,722 negative):
  * **Held-Out Test Benchmark:** 1,572 frames (300 positive, 1,272 negative · 80.9% negative prevalence).
  * **Held-Out Validation Benchmark:** 1,572 frames (300 positive, 1,272 negative · 80.9% negative prevalence).
  * **Training Configurations (2,401 positive frames held fixed across all splits):**
    * **0% (`train_00_pos_only`):** 2,401 pos, 0 neg (Total: 2,401 frames)
    * **20% (`train_20_low_neg`):** 2,401 pos, 600 neg (Total: 3,001 frames)
    * **40% (`train_40_mod_neg`):** 2,401 pos, 1,600 neg (Total: 4,001 frames)
    * **60% (`train_60_high_neg`):** 2,401 pos, 3,602 neg (Total: 6,003 frames)
    * **80% (`train_80_max_neg`):** 2,401 pos, 9,604 neg (Total: 12,005 frames; 574 excess pool negatives discarded for exact arithmetic 20% stepping)
* Positives stratified across the 4 driver-cue categories (`phone_use`, `drinking`, `yawning`, `hand_over_mouth`).
* Negatives sampled in strictly nested subsets ($\text{Neg}_{0\%} \subset \text{Neg}_{20\%} \subset \text{Neg}_{40\%} \subset \text{Neg}_{60\%} \subset \text{Neg}_{80\%}$) using deterministic seed (`SEED = 42`).

#### 3. Pipelines & Scripts
* `src/data/create_splits.py`: Deterministic split generation producing:
  * YOLO manifest files in `data/processed/RGB/yolo/`
  * COCO JSON annotations in `data/processed/RGB/coco/`
  * Mirror copies for D-FINE in `data/processed/RGB/coco/dfine/` and evaluation in `data/processed/RGB/coco/evaluation/`
  * Split metadata summary in `data/processed/RGB/split_stats.json`
* `src/data/verify_splits.py`: Automated verification suite testing:
  * 100% file existence on disk for all manifest image paths
  * Strict zero data leakage ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$)
  * Negative subset nesting integrity across all 5 training levels
  * COCO JSON schema and annotation counts
  * Ultralytics YAML parsing and dataset compatibility

#### 4. Model Configurations (`configs/yolo/`)
* Created dataset configuration files for Ultralytics YOLO:
  * `configs/yolo/yolo_00_pos_only.yaml`
  * `configs/yolo/yolo_20_low_neg.yaml`
  * `configs/yolo/yolo_40_mod_neg.yaml`
  * `configs/yolo/yolo_60_high_neg.yaml`
  * `configs/yolo/yolo_80_max_neg.yaml`

#### 5. Documentation & Repository Integrity
* Updated `README.md` with:
  * Experimental matrix (exact 5-level split counts, semantic names, held-out benchmarks).
  * Quick Reproduction workflow (deterministic generation, split verification, training example).
  * Repository organization directory tree.
* Fixed `.gitignore` from `data/` to `/data/` so source scripts under `src/data/` remain tracked.

