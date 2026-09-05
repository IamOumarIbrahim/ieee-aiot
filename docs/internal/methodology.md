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

### 1. Anchor-Free Decoupled CNNs (YOLO11n / YOLO26n)
The multi-task loss function for modern anchor-free YOLO architectures is:
$$\mathcal{L}_{\mathrm{YOLO}} = \lambda_{\mathrm{cls}} \mathcal{L}_{\mathrm{cls}} + \lambda_{\mathrm{box}} \mathcal{L}_{\mathrm{box}} + \lambda_{\mathrm{dfl}} \mathcal{L}_{\mathrm{dfl}}$$
where $\mathcal{L}_{\mathrm{box}}$ is Complete IoU (CIoU) loss, $\mathcal{L}_{\mathrm{dfl}}$ is Distribution Focal Loss, and $\mathcal{L}_{\mathrm{cls}}$ is Binary Cross-Entropy ($\mathrm{BCE}$).
When training on a negative frame $X \in \mathcal{D}_{\mathrm{neg}}$, no ground-truth bounding boxes exist ($Y = \emptyset$). The Task-Aligned Assigner assigns zero positive anchors across all feature pyramid scales $s \in \{P_3, P_4, P_5\}$. As a result, the regression terms vanish ($\mathcal{L}_{\mathrm{box}} = 0, \mathcal{L}_{\mathrm{dfl}} = 0$), and the loss collapses strictly to the classification objective across all spatial anchor locations:
$$\mathcal{L}_{\mathrm{YOLO}}^{\mathrm{neg}} = -\lambda_{\mathrm{cls}} \sum_{s=1}^S \sum_{i=1}^{H_s W_s} \sum_{c=1}^C \log\left(1 - \hat{p}_{s, i, c}\right)$$
where $\hat{p}_{s, i, c} = \sigma(z_{s, i, c})$ is the predicted probability for class $c$ at grid cell $i$ of pyramid scale $s$. In the absence of negative frames ($r = 0\%$), classification gradients on background regions arise exclusively from non-target areas of positive frames. Adding negative frames exposes the local convolutional filters to diverse cabin environments devoid of foreground cues, penalizing spurious local activations.

### 2. Query-Based Real-Time DETRs (D-FINE-N)
In D-FINE-N, detection is formulated via a fixed set of $N_q$ learned object queries ($N_q = 300$). Training relies on optimal bipartite matching via the Hungarian algorithm:
$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_{N_q}} \sum_{i=1}^{N_q} \mathcal{L}_{\mathrm{match}}\left(y_i, \hat{y}_{\sigma(i)}\right)$$
where $\mathfrak{S}_{N_q}$ is the permutation group of $N_q$ elements. On a negative frame ($Y = \emptyset$), the ground-truth set contains only null targets ($\varnothing$). Every query $i \in \{1, \dots, N_q\}$ is unconditionally matched to the background class $\varnothing$. Consequently, the bounding-box L1 loss, Generalized IoU (GIoU) loss, and Fine-grained Distribution Refinement ($\mathcal{L}_{\mathrm{FDR}}$) loss are zero. The training loss reduces entirely to the classification objective:
$$\mathcal{L}_{\mathrm{DFINE}}^{\mathrm{neg}} = \sum_{j=1}^{N_q} \mathrm{FL}\left(\hat{p}_j, 0\right) = -\sum_{j=1}^{N_q} \alpha \hat{p}_j^\gamma \log\left(1 - \hat{p}_j\right)$$
where $\mathrm{FL}$ denotes Focal Loss with focusing parameter $\gamma$ and balance factor $\alpha$. Unlike CNNs where gradients penalize tens of thousands of fixed spatial grid points locally, DETRs supervise a compact set of global queries interacting across the entire image via self-attention and cross-attention.

---

## C. Dataset and Stratified Partitioning
Experiments use an in-cabin driver-monitoring (DMS) dataset comprising 15,723 total frames, of which 3,001 contain target driver-cue annotations (`phone_use`, `drinking`, `yawning`, `hand_over_mouth`) and 12,722 are background-only (80.9% negative prevalence). All frames are partitioned using a stratified 80/10/10 random split (`seed=42`):
* **Held-Out Validation Benchmark ($\mathcal{D}_{\mathrm{val}}$):** 1,572 frames (300 positive, 1,272 negative; 80.9% negative prevalence)
* **Held-Out Test Benchmark ($\mathcal{D}_{\mathrm{test}}$):** 1,572 frames (300 positive, 1,272 negative; 80.9% negative prevalence)
* **Training Universe ($\mathcal{D}_{\mathrm{train}}^{\mathrm{univ}}$):** 2,401 positive frames ($\mathcal{D}_{\mathrm{pos}}$) held fixed across all training splits, and 10,178 candidate negative frames ($\mathcal{U}_{\mathrm{neg}}$).

Evaluating on a fixed, natural-distribution test set preserves the false-positive behavior that negative-frame configuration is intended to address.

---

## D. Detector Architectures
Three lightweight detectors spanning distinct design paradigms:
1. **YOLO11n:** Mainstream anchor-free CNN (2.6M params, 6.5 GFLOPs).
2. **YOLO26n:** Next-generation CNN with reparameterized convolution blocks (2.5M params, 6.3 GFLOPs).
3. **D-FINE-N:** Query-based real-time DETR transformer with fine-grained distribution refinement (4.3M params, 25.0 GFLOPs).

Target deployment platform: NVIDIA Jetson Orin Nano (8 GB VRAM). Baseline latency is architecture-bound, not training-data-bound.

---

## E. Negative-Frame Experimental Configurations

### Axis 1 (RQ1) --- Ratio Sweep
For each architecture, five training sets are constructed by holding the full 2,401-frame positive core fixed and sampling strictly nested negative subsets at uniform arithmetic 20%-point steps:
* 0%: 0 neg / 2,401 total (`train_00_pos_only`)
* 20%: 600 neg / 3,001 total (`train_20_low_neg`)
* 40%: 1,600 neg / 4,001 total (`train_40_mod_neg`)
* 60%: 3,602 neg / 6,003 total (`train_60_high_neg`)
* 80%: 9,604 neg / 12,005 total (`train_80_max_neg`)

Nested subsets satisfy $\mathcal{D}_{\mathrm{neg}}^{(0\%)} \subset \mathcal{D}_{\mathrm{neg}}^{(20\%)} \subset \dots \subset \mathcal{D}_{\mathrm{neg}}^{(80\%)} \subset \mathcal{U}_{\mathrm{neg}}$ with fixed random seed (`seed=42`), yielding $3 \times 5 = 15$ training runs.

### Axis 2 (RQ2) --- Hard-Negative Curation
At each architecture's best-performing ratio $r^*$, a controlled comparison is conducted between random subsampling and hard-negative mining at strictly matched dataset cardinality:
1. Train baseline detector $f_{\theta_0}$ on `train_00_pos_only`.
2. Infer over candidate pool $\mathcal{U}_{\mathrm{neg}}$ (10,178 background frames).
3. Identify hard negatives: $\mathcal{H}_{\mathrm{mined}} = \{X \in \mathcal{U}_{\mathrm{neg}} \mid \exists (\hat{c}, \hat{b}, s) \in f_{\theta_0}(X) \text{ s.t. } s \geq 0.25\}$.
4. Rank by maximum score $s_{\max}(X)$, select top $N_{\mathrm{target}}^{(r^*)}$, and backfill deterministically if needed to match cardinality:
   $$|\mathcal{D}_{\mathrm{train, hard}}^{(r^*)}| = |\mathcal{D}_{\mathrm{train, rand}}^{(r^*)}| = 2{,}401 + N_{\mathrm{target}}^{(r^*)}$$

This yields $3 \times 2 = 6$ additional runs, establishing a total of **21 benchmark runs**.

---

## F. Frozen Training Protocol (RTX 4060 8GB)

| Parameter | YOLO11n | YOLO26n | D-FINE-N |
| :--- | :---: | :---: | :---: |
| **Input Resolution** | 640 × 640 | 640 × 640 | 640 × 640 |
| **Physical Batch Size** | 16 | 16 | 4 |
| **Gradient Accumulation** | 1 | 1 | 8 |
| **Effective Batch Size** | 16 | 16 | 32 |
| **Epochs** | 100 | 100 | 160 |
| **Optimizer** | `auto` | `auto` | AdamW |
| **Backbone LR** | — | — | 0.0004 |
| **Head / Transformer LR** | — | — | 0.0008 |
| **Weight Decay** | 0.0005 | 0.0005 | 0.0001 |
| **EMA Restart Decay** | — | — | 0.9999 |
| **Augmentation Cooldown** | `close_mosaic=10` | `close_mosaic=10` | `stop_epoch=148` |
| **Precision** | FP32 (`amp=False`)† | FP32 (`amp=False`)† | Mixed Precision (AMP) |
| **Deterministic Seed** | 42 | 42 | 42 |

† *Windows cuBLAS Precision Note:* PyTorch 2.6.0 on Windows with Ada Lovelace GPUs triggers `CUBLAS_STATUS_INTERNAL_ERROR` in FP16 batched GEMM; FP32 training is numerically stable and uses ~1.5 GB of 8 GB VRAM at batch 16.

---

## G. Evaluation Metrics & Claims Boundaries
* **Detection Accuracy:** Precision ($\mathrm{P}$), Recall ($\mathrm{R}$), $\mathrm{mAP}_{50}$, $\mathrm{mAP}_{50:95}$ under standard COCO evaluation.
* **Operational Deployment Metric:** False positives per 1,000 frames ($\mathrm{FP/1k}$) on negative benchmark test frames, and projected hourly nuisance alerts $\mathcal{A}_h$.
* **Claims Boundary:** Single-seed point estimates report practical effect magnitude, trajectory curvature (inverted-U vs. monotonic), and cross-architecture consistency without inferential hypothesis testing ($p$-values/ANOVA).
