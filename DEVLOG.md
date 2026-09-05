# Development Log

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

