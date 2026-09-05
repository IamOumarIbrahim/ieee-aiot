# Development Log

## [2026-09-05] - IEEE Manuscript Terminology, Mathematical Problem Formulation, and Notation Standardization

### Overview
Executed a comprehensive academic overhaul of the conference manuscript (`docs/manuscript/main.tex`) and all internal documentation (`docs/internal/` and `README.md`) to elevate the writing style, theoretical depth, and mathematical notation to the standards of top-tier IEEE publications (e.g., IEEE Transactions on Intelligent Transportation Systems, IEEE Internet of Things Journal, IEEE/CVF CVPR).

---

### Key Changes & Implementations

#### 1. Mathematical Problem Formulation & Unified Notation
* Formally defined continuous video streams $\mathcal{S} = \{X_t\}_{t=1}^T$, target bounding annotations $Y_t = \{(c_k, b_k)\}_{k=1}^{K_t}$, disjoint foreground/positive $\mathcal{D}_{\mathrm{pos}}$ and background/negative $\mathcal{D}_{\mathrm{neg}}$ partitions.
* Parameterized the negative ratio $r = \frac{|\mathcal{D}_{\mathrm{neg}}|}{|\mathcal{D}_{\mathrm{pos}}| + |\mathcal{D}_{\mathrm{neg}}|}$ governing strictly nested subsets $\mathcal{D}_{\mathrm{neg}}^{(0\%)} \subset \dots \subset \mathcal{D}_{\mathrm{neg}}^{(80\%)}$.
* Defined filtered detector predictions $\hat{\mathcal{Y}}_\tau(X)$ at threshold $\tau \in (0, 1]$.
* Formulated false positives per 1,000 background frames ($\mathrm{FP/1k}$) and operational hourly nuisance alerts ($\mathcal{A}_h = 3.6 \times f_{\mathrm{FPS}} \times p_{\mathrm{neg}} \times \mathrm{FP/1k}$) under natural background prevalence $p_{\mathrm{neg}} \approx 0.809$.

#### 2. Architectural Loss Dynamics on Negative Frames
* **Anchor-Free Decoupled CNNs (YOLO11n / YOLO26n):** Formulated how for unannotated background frames ($Y = \emptyset$), regression ($\mathcal{L}_{\mathrm{box}}$) and DFL ($\mathcal{L}_{\mathrm{dfl}}$) losses evaluate to zero, collapsing the multi-task loss strictly to Binary Cross-Entropy ($\mathrm{BCE}$) over all spatial anchor cells across pyramid scales $s \in \{P_3, P_4, P_5\}$, penalizing local false activations.
* **Query-Based Real-Time DETRs (D-FINE-N):** Formulated how Hungarian bipartite matching unconditionally assigns all $N_q$ learned queries to the null object $\varnothing$, zeroing out bounding box regression and fine-grained distribution refinement ($\mathcal{L}_{\mathrm{FDR}}$) losses, and supervising queries exclusively through classification Focal Loss.

#### 3. Formalized Hard-Negative Curation Protocol (RQ2)
* Formalized the mining condition $\mathcal{H}_{\mathrm{mined}}$ at $\tau = 0.25$, score-based ranking $s_{\max}(X)$, deterministic backfill (`seed=42`), and exact cardinality matching ($|\mathcal{D}_{\mathrm{train, hard}}^{(r^*)}| = |\mathcal{D}_{\mathrm{train, rand}}^{(r^*)}|$), isolating sample information entropy from volume.

#### 4. IEEE Academic Tone & Standardized Typography
* Replaced colloquial phrasing with formal academic prose.
* Standardized cross-referencing (`\eqref{...}`, `Table~\ref{...}`, `Section~\ref{...}`) and metrics ($\mathrm{mAP}_{50}$, $\mathrm{mAP}_{50:95}$, $\mathrm{P}$, $\mathrm{R}$, $\mathrm{FP/1k}$, $\mathcal{A}_h$).
* Expanded bibliography to 21 authoritative IEEE citations with standard abbreviations.

#### 5. Full Documentation Synchronization & Verification
* Updated `docs/internal/introduction.md`, `docs/internal/related_work.md`, `docs/internal/methodology.md`, `docs/internal/results.md`, `docs/internal/discussion.md`, and `README.md`.
* Automated validation with `python src/data/verify_splits.py` passed with 100% integrity across all manifests, splits, JSONs, and configuration YAMLs.

---

## [2026-09-05] - Training Sequence Diagnostics, Windows cuBLAS Fixes, and RQ2 Pipeline Completion

### Overview
Conducted comprehensive end-to-end debugging and validation of the training and evaluation execution pipelines on Windows PowerShell with an NVIDIA GeForce RTX 4060 Laptop GPU. Identified and resolved all architectural, numerical, and runtime bugs blocking PowerShell execution of the paper's benchmark protocol.

---

### Key Changes & Bug Fixes

#### 1. Recursive YAML Resolution in `src/data/verify_splits.py`
* **Issue:** `verify_splits.py` crashed with `KeyError: 'num_classes'` because D-FINE configuration YAMLs use hierarchical inheritance (`__include__`) where parameters like `num_classes` reside in base configs.
* **Resolution:** Implemented recursive `load_yaml_with_includes` and `deep_merge` in `src/data/verify_splits.py` and `src/training/train_dfine_sweep.py`. Verified all 6 D-FINE configurations, COCO JSONs, and YOLO YAMLs pass with 100% integrity.

#### 2. Batched FP/1k Inference & VRAM OOM Prevention
* **Issue:** `calculate_fp_per_1k` passed all 1,272 test images to `model.predict()` simultaneously, causing CUDA Out-of-Memory crashes on 8 GB VRAM.
* **Resolution:** Batched inference in chunks of 32 images with explicit `del results`, `del model`, `gc.collect()`, and `torch.cuda.empty_cache()`. Added automated export of clean Markdown summary tables (`{model}_sweep_summary.md`) alongside JSON logs.

#### 3. Windows cuBLAS `CUBLAS_STATUS_INTERNAL_ERROR` Resolution
* **Issue:** On Windows with PyTorch 2.6.0+cu124 on Ada Lovelace GPUs (RTX 4060), `cublasGemmStridedBatchedEx` in FP16/BF16 raises `RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR`. Ultralytics `check_amp()` triggers this error on attention blocks and aborts training because it only catches `AssertionError`.
* **Resolution:** Added `--amp` flag (defaulting to `False`) to `src/training/train_yolo_sweep.py` and `-Amp` switch to `scripts/run_training_sequence.ps1`. In full precision FP32 (`amp=False`), training runs flawlessly via `cublasSgemm`, utilizing only ~1.5 GB of the 8 GB VRAM budget at batch 16.

#### 4. Missing RQ2 Hard-Negative Mining Pipeline (`src/data/mine_hard_negatives.py`)
* **Issue:** The repository lacked an executable implementation of the RQ2 curation protocol specified in the paper.
* **Resolution:** Implemented `src/data/mine_hard_negatives.py`, reconstructing the 10,178 training candidate pool, evaluating false positives at $\tau = 0.25$, ranking candidate frames, deterministically backfilling (`seed=42`) to match the target sample count, and generating all YOLO text manifests, COCO JSONs, D-FINE configs, and curation audit logs.

#### 5. PowerShell Native Training Runner (`scripts/run_training_sequence.ps1`)
* **Issue:** Prior documentation contained bash-specific line-continuation backslashes (`\`) and lacked native PowerShell orchestration, while `$ErrorActionPreference = "Stop"` aborted execution on PyTorch library warnings emitted to stderr.
* **Resolution:** Created `scripts/run_training_sequence.ps1` and `scripts/check_env.py` with `PYTHONWARNINGS="ignore"` and robust exit-code propagation. Updated `README.md` and `docs/internal/experimental_protocol.md` with tested PowerShell commands. Validated end-to-end with a 1-epoch training test on the RTX 4060 GPU.

---

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

---

### [2026-09-05] IEEE Manuscript & Documentation Scholarly Transformation

#### 1. IEEE Conference Manuscript (`docs/manuscript/main.tex`)
* **Format & Template:** Migrated fully to standard `IEEEtran` conference format (`\documentclass[conference]{IEEEtran}`, `\IEEEoverridecommandlockouts`, `flushend`). Double-blind review metadata applied (`Anonymous Authors`, `Institution redacted`).
* **Section I (Introduction):** Formalized the AIoT edge operational problem: background dominance ($80.9\%$ negative prevalence), alert fatigue, edge compute duty cycles, and cellular uplink costs. Formulated research questions **RQ1** (Ratio Sensitivity and Paradigm Invariance) and **RQ2** (Curation Quality vs. Sample Volume at Matched Cardinality).
* **Section II (Related Work):** Restructured into four scholarly subsections: Class Imbalance & Negative Sampling, Lightweight Edge Detectors (CNNs vs. Real-Time DETRs), Negative-Frame Training in AIoT, and Identified Research Gap. Integrated 21 IEEE-standard citations.
* **Section III (System Modeling & Methodology):**
  * Formal mathematical problem formulation: $\mathcal{S} = \{X_t\}_{t=1}^T$, $Y_t$, $\mathcal{D}_{\mathrm{pos}}$, $\mathcal{D}_{\mathrm{neg}}$, confidence threshold filtering $\hat{\mathcal{Y}}_\tau$, $\mathrm{FP/1k}$, and hourly nuisance alerts $\mathcal{A}_h = 3.6 \times f_{\mathrm{FPS}} \times p_{\mathrm{neg}} \times \mathrm{FP/1k}$.
  * Mathematical loss dynamics: Anchor-Free CNN decoupled BCE loss ($\mathcal{L}_{\mathrm{YOLO}}^{\mathrm{neg}}$) vs. Real-Time DETR Hungarian matching with Focal Loss ($\mathcal{L}_{\mathrm{DFINE}}^{\mathrm{neg}}$) under null object target ($\varnothing$).
  * Architectural overview (Table I) and frozen training protocol (Table II) explicitly documenting 8 GB VRAM hardware adaptations (D-FINE-N batch 4 $\times$ 8 accumulation; YOLO FP32 `amp=False` for Windows cuBLAS stability).
  * Formal 4-step RQ2 hard-negative mining protocol isolating sample entropy from cardinality via exact sample size matching.
* **Section IV (Results) & Section V (Discussion):**
  * Structured placeholder tables with rigorous column notation: Table III (RQ1 sweep), Table IV (RQ2 curation benchmark), and Table V (operational alert frequency translation across 5, 15, and 30 FPS).
  * Analytical framing decoupling architecture-bound inference latency ($t_{\mathrm{inf}}$) from curation-bound operational reliability ($\mathcal{A}_h$).
  * Explicit claims boundaries eschewing inferential hypothesis testing ($p$-values/ANOVA) on deterministic point estimates (`seed=42`).
* **Typesetting & Compilation Integrity:**
  * Successfully compiled via MiKTeX `pdflatex` to an exact 8-page PDF (`docs/manuscript/main.pdf`).
  * Eliminated all 4 overfull `\hbox` warnings by splitting long equations and compacting table cell widths.
  * Added `flushend` package to automatically equalize column heights on the final page (References).
  * Verified 0 undefined references, 0 missing citations, and 0 label mismatches.

#### 2. Internal Documentation Alignment
* Synchronized `docs/internal/introduction.md`, `related_work.md`, `methodology.md`, `results.md`, `discussion.md`, and `README.md` to reflect unified IEEE terminology, formal mathematical notation, and consistent table numbering (Tables I–V).

