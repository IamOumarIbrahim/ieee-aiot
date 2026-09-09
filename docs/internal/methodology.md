# Section III: System Modeling and Experimental Methodology

## A. Problem Formulation and Mathematical Modeling
Let $\mathcal{S} = \{X_t\}_{t=1}^T$ denote a continuous streaming video feed captured by an edge camera sensor, where $X_t \in \mathcal{X} \subset \mathbb{R}^{H \times W \times 3}$ represents the image frame at time step $t$. Each frame $X_t$ is associated with a ground-truth annotation set $Y_t = \{(c_k, b_k)\}_{k=1}^{K_t}$, where $c_k \in \mathcal{C} = \{1, \dots, C\}$ is the driver behavior class index, $b_k = (x_k, y_k, w_k, h_k) \in [0, 1]^4$ denotes normalized bounding-box coordinates, and $K_t \geq 0$ denotes the number of target instances in frame $t$.

We partition the universe of frames into two disjoint categories:
$$\mathcal{D}_{\mathrm{pos}} = \{(X_t, Y_t) \mid K_t \geq 1\}$$
$$\mathcal{D}_{\mathrm{neg}} = \{(X_t, \emptyset) \mid K_t = 0\}$$

A frame is defined as a *positive frame* ($X_t \in \mathcal{D}_{\mathrm{pos}}$) if it contains at least one driver-cue object, and as a *negative frame* ($X_t \in \mathcal{D}_{\mathrm{neg}}$) if it contains zero annotated target objects.

Given an object detector parameterized by weights $\theta$, inference on an input frame $X$ yields predicted bounding tuples $f_\theta(X) = \{(\hat{c}_j, \hat{b}_j, s_j)\}_{j=1}^M$, where $\hat{c}_j \in \mathcal{C}$, $\hat{b}_j \in [0, 1]^4$, and $s_j \in [0, 1]$ represents the predicted class confidence score. At a deployed confidence operating threshold $\tau \in (0, 1]$, the filtered prediction set is:
$$\hat{\mathcal{Y}}_\tau(X) = \left\{ (\hat{c}_j, \hat{b}_j, s_j) \in f_\theta(X) \;\middle|\; s_j \geq \tau \right\}$$

For any background-only negative frame $X \in \mathcal{D}_{\mathrm{neg}}$, the ground-truth instance set is empty ($Y = \emptyset$). Consequently, any detection belonging to $\hat{\mathcal{Y}}_\tau(X)$ constitutes an operational False Positive ($\mathrm{FP}$). The deployment cost metric, false positives per 1,000 background frames ($\mathrm{FP/1k}$), evaluated on the held-out negative test set $\mathcal{D}_{\mathrm{test}}^{\mathrm{neg}}$, is formalized as:
$$\mathrm{FP/1k} = \frac{1000}{\left|\mathcal{D}_{\mathrm{test}}^{\mathrm{neg}}\right|} \sum_{X \in \mathcal{D}_{\mathrm{test}}^{\mathrm{neg}}} \left| \hat{\mathcal{Y}}_\tau(X) \right|$$

In an operational edge deployment operating at sensor frame rate $f_{\mathrm{FPS}}$ (frames per second) over a duration of one hour ($T_{\mathrm{hr}} = 3600\,\text{s}$), let $p_{\mathrm{neg}} \in (0, 1]$ represent the operational background frame prevalence. The expected hourly nuisance alert rate $\mathcal{A}_h$ is modeled as:
$$\mathcal{A}_h = 3600 \times f_{\mathrm{FPS}} \times p_{\mathrm{neg}} \times \frac{\mathrm{FP/1k}}{1000} = 3.6 \times f_{\mathrm{FPS}} \times p_{\mathrm{neg}} \times \mathrm{FP/1k}$$
Under unconditioned driving streams ($p_{\mathrm{neg}} \approx 0.809$), this formulation computes the real-world nuisance alert burden experienced by the driver and edge communication stack.

---

## B. Architectural Loss Dynamics on Negative Frames

### 1. Anchor-Free Decoupled CNNs (YOLO11n, YOLO26n, YOLO12n)
The multi-task loss function for anchor-free YOLO architectures is:
$$\mathcal{L}_{\mathrm{YOLO}} = \lambda_{\mathrm{cls}} \mathcal{L}_{\mathrm{cls}} + \lambda_{\mathrm{box}} \mathcal{L}_{\mathrm{box}} + \lambda_{\mathrm{dfl}} \mathcal{L}_{\mathrm{dfl}}$$
where $\mathcal{L}_{\mathrm{box}}$ is Complete IoU (CIoU) loss, $\mathcal{L}_{\mathrm{dfl}}$ is Distribution Focal Loss, and $\mathcal{L}_{\mathrm{cls}}$ is Binary Cross-Entropy ($\mathrm{BCE}$).
When training on a negative frame $X \in \mathcal{D}_{\mathrm{neg}}$, no ground-truth bounding boxes exist ($Y = \emptyset$). The Task-Aligned Assigner assigns zero positive anchors across all feature pyramid scales $s \in \{P_3, P_4, P_5\}$. As a result, the regression terms vanish ($\mathcal{L}_{\mathrm{box}} = 0, \mathcal{L}_{\mathrm{dfl}} = 0$), and the loss collapses strictly to the classification objective across all spatial anchor locations:
$$\mathcal{L}_{\mathrm{YOLO}}^{\mathrm{neg}} = -\lambda_{\mathrm{cls}} \sum_{s=1}^S \sum_{i=1}^{H_s W_s} \sum_{c=1}^C \log\left(1 - \hat{p}_{s, i, c}\right)$$
where $\hat{p}_{s, i, c} = \sigma(z_{s, i, c})$ is the predicted probability for class $c$ at grid cell $i$ of pyramid scale $s$. In the absence of negative frames ($r = 0\%$), classification gradients on background regions arise exclusively from non-target areas of positive frames. Adding negative frames exposes spatial filters and attention projections to diverse cabin environments devoid of foreground cues, penalizing spurious activations.

### 2. Dual-Label Assignment (YOLOv10n)
YOLOv10n introduces consistent dual assignments during training: a one-to-many branch (providing rich supervisory gradients) and a one-to-one branch (harmonized with inference for NMS-free prediction). On unannotated negative frames ($Y = \emptyset$), both branches assign zero positive targets, zeroing out bounding-box regression and supervising dual heads purely through classification binary cross-entropy, regularizing end-to-end background predictions without post-processing heuristics.

### 3. Feature Aggregation Mechanisms Across Lineages
* **YOLO26n (Pure Reparameterized CNN):** Negative gradients update local convolutional receptive fields (RepConv), attenuating spatial filters on high-frequency edge/texture noise (e.g., seatbelts, steering wheel rims).
* **YOLO11n (Hybrid CNN-Attention):** Negative gradients modulate both convolutional stages and neck-level self-attention (C2PSA), allowing global feature suppression across prominent spatial locations.
* **YOLO12n (Linear Area Attention):** Negative gradients flow through Area Attention (A2C2f) modules partitioned horizontally and vertically, enabling holistic contextual suppression of extended cabin features (e.g., headrests, window glare).
* **YOLOv10n (NMS-Free Dual Head):** Negative gradients supervise both one-to-many and one-to-one branches simultaneously, enforcing discriminative background silence at ultra-low inference latency ($7.7$\,ms).

---

## C. Dataset and Stratified Partitioning
Experiments use an in-cabin driver-monitoring (DMS) dataset comprising 15,723 total frames, of which 3,001 contain target driver-cue annotations (`phone_use`: 2,437 [81.2%], `drinking`: 264 [8.8%], `yawning`: 159 [5.3%], `hand_over_mouth`: 141 [4.7%]) and 12,722 are background-only (80.9% negative prevalence). All frames are partitioned using a stratified 80/10/10 random split (`seed=42`):
* **Held-Out Validation Benchmark ($\mathcal{D}_{\mathrm{val}}$):** 1,572 frames (300 positive, 1,272 negative; 80.9% negative prevalence)
* **Held-Out Test Benchmark ($\mathcal{D}_{\mathrm{test}}$):** 1,572 frames (300 positive, 1,272 negative; 80.9% negative prevalence)
* **Training Universe ($\mathcal{D}_{\mathrm{train}}^{\mathrm{univ}}$):** 2,401 positive frames ($\mathcal{D}_{\mathrm{pos}}$) held fixed across all training splits, and 10,178 candidate negative frames ($\mathcal{U}_{\mathrm{neg}}$).

---

## D. Detector Architectures and On-Device Latency Benchmarks
Four lightweight edge detectors spanning distinct architectural paradigms are benchmarked at matched capacity:

| Detector | Params | FLOPs | Latency (ms)† | Throughput (FPS)† | Paradigm | Key Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11n** | 2.6 M | 6.5 G | 16.8 ms | 59.6 FPS | Hybrid | C3k2 + C2PSA Attention |
| **YOLO26n** | 2.6 M | 6.2 G | 21.7 ms | 46.1 FPS | Pure Conv | RepConv Reparameterization |
| **YOLO12n** | 2.6 M | 7.6 G | 21.3 ms | 47.0 FPS | Area Attn | A2C2f Area Attention |
| **YOLOv10n**| 2.3 M | 6.6 G | 7.7 ms  | 129.2 FPS | NMS-Free | Dual-Label Head |

† *Latency and throughput measured on NVIDIA RTX 4060 Laptop GPU (PyTorch FP32, batch=1, 640×640, averaged over 200 iterations after 50 warmups).*

---

## E. Negative-Frame Experimental Configurations

### Axis 1 (RQ1) --- Ratio Sweep
For each architecture, five training sets are constructed by holding the full 2,401-frame positive core fixed and sampling strictly nested negative subsets at uniform arithmetic 20%-point steps:
* 0%: 0 neg / 2,401 total (`train_00_pos_only`)
* 20%: 600 neg / 3,001 total (`train_20_low_neg`)
* 40%: 1,600 neg / 4,001 total (`train_40_mod_neg`)
* 60%: 3,602 neg / 6,003 total (`train_60_high_neg`)
* 80%: 9,604 neg / 12,005 total (`train_80_max_neg`)

Nested subsets satisfy $\mathcal{D}_{\mathrm{neg}}^{(0\%)} \subset \mathcal{D}_{\mathrm{neg}}^{(20\%)} \subset \dots \subset \mathcal{D}_{\mathrm{neg}}^{(80\%)}$ with fixed random seed (`seed=42`), yielding $4 \times 5 = 20$ training runs completed for Seed 42.

### Axis 2 (RQ2) --- Hard-Negative Curation
At the compute-efficient reference ratio $r_{\mathrm{ref}}=40\%$ ($N_{\mathrm{target}}=1,600$ negatives, $N=4,001$ frames), a controlled comparison benchmarks uniform random negative sampling against curated hard-negative mining:
1. Train baseline detector $f_{\theta_0}$ on `train_00_pos_only`.
2. Infer over candidate pool $\mathcal{U}_{\mathrm{neg}}$ (10,178 background frames).
3. Identify hard negatives: $\mathcal{H}_{\mathrm{mined}} = \{X \in \mathcal{U}_{\mathrm{neg}} \mid \exists (\hat{c}, \hat{b}, s) \in f_{\theta_0}(X) \text{ s.t. } s \geq 0.25\}$.
4. Rank by maximum score $s_{\max}(X)$, select top $N_{\mathrm{target}}$, and backfill deterministically if needed (`seed=42`):
   $$|\mathcal{D}_{\mathrm{train, hard}}^{(40\%)}| = |\mathcal{D}_{\mathrm{train, rand}}^{(40\%)}| = 2{,}401 + 1{,}600 = 4{,}001\,\text{frames}$$

This yields $4 \times 2 = 8$ curation runs, establishing **28 completed benchmark runs** under Seed 42.

---

## F. Frozen Training Protocol (RTX 4060 8GB)

| Parameter | YOLO11n | YOLO26n | YOLO12n | YOLOv10n |
| :--- | :---: | :---: | :---: | :---: |
| **Input Resolution** | 640 × 640 | 640 × 640 | 640 × 640 | 640 × 640 |
| **Physical / Effective Batch** | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 |
| **Epochs** | 100 | 100 | 100 | 100 |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW |
| **Initial Learning Rate ($\mathrm{lr}_0$)** | 0.00125 | 0.00125 | 0.00125 | 0.00125 |
| **Weight Decay** | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
| **Augmentation Cooldown** | `close_mosaic=10` | `close_mosaic=10` | `close_mosaic=10` | `close_mosaic=10` |
| **Precision** | FP32 (`amp=False`)† | FP32 (`amp=False`)† | FP32 (`amp=False`)† | FP32 (`amp=False`)† |
| **Deterministic Seed** | 42 | 42 | 42 | 42 |

† *Windows cuBLAS Precision Note:* PyTorch 2.6.0 on Windows with Ada Lovelace GPUs triggers `CUBLAS_STATUS_INTERNAL_ERROR` in FP16 batched GEMM; FP32 training is numerically stable and uses ~1.5 GB of 8 GB VRAM at batch 16.

---

## G. Evaluation Metrics, Operating Points & Multi-Seed Extension
* **Detection Accuracy:** $\mathrm{mAP}_{50}$, $\mathrm{mAP}_{50:95}$, Precision ($\mathrm{P}$), Recall ($\mathrm{R}$) evaluated over the test set, with $\mathrm{P}$ and $\mathrm{R}$ reported at maximum-$F_1$.
* **Operational Deployment Cost:** False positives per 1,000 background frames ($\mathrm{FP/1k}$) evaluated on the 1,272 test negatives at operational threshold $\tau = 0.25$, $\mathrm{IoU} = 0.70$, and projected hourly nuisance alerts $\mathcal{A}_h$.
* **Phase 4 Multi-Seed Robustness Validation:** To address single-seed limitations, a multi-seed replication sweep (`seed=43`) across all 4 architectures and 5 splits (20 runs total) is actively running (`src/training/run_phase4_second_seed.py`) to quantify empirical variance bounds (mean $\pm$ std).
