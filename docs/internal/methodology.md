III. Methodology
A. Dataset

Experiments use an in-cabin driver-monitoring (DMS) dataset comprising 15,723 total frames, of which 3,001 contain target driver-cue annotations (`phone_use`, `drinking`, `yawning`, `hand_over_mouth`) (positive frames) and 12,722 are background-only (negative frames), reflecting a natural negative prevalence of approximately 81%. All frames are partitioned using a stratified 80/10/10 random split (`seed=42`) into training, validation, and test partitions:
* **Validation Benchmark:** 1,572 frames (300 positive, 1,272 negative · 80.9% negative prevalence)
* **Test Benchmark:** 1,572 frames (300 positive, 1,272 negative · 80.9% negative prevalence)
* **Training Pool:** 2,401 positive frames (held fixed across all training splits) and 10,178 negative candidate frames

Critically, the test set composition is held fixed across all experimental conditions and reflects the dataset's natural negative-heavy distribution (~81% negative). Only the training set's negative-frame configuration is varied (Section III-C). This design choice is deliberate: evaluating on an artificially rebalanced test set would mask the false-positive behavior that negative-frame configuration is intended to address, and would not reflect the negative-dominant conditions object detectors encounter in continuous AIoT video streams.

B. Detector Architectures

Three lightweight detectors, selected to span distinct design paradigms relevant to edge deployment, are trained under identical conditions:

YOLO11n: a one-stage, anchor-free CNN detector representing the current mainstream YOLO lineage.
YOLO26n: the succeeding YOLO generation, included to test whether architectural refinements change sensitivity to negative-frame configuration relative to YOLO11n.
D-FINE-N: a transformer-based, query-driven real-time detector, included to test whether a fundamentally different detection paradigm (set-based prediction vs. dense anchor-free prediction) responds differently to negative-frame manipulation than the CNN-based YOLO models.

All three are evaluated at their smallest ("nano") variant to reflect realistic AIoT compute budgets. Table [X] reports each model's parameter count, FLOPs, and baseline inference latency measured on the target deployment platform; since latency is a function of model structure and input resolution, not training-data composition, it is reported as fixed context rather than as a dependent variable in the negative-frame experiments.

C. Negative-Frame Configurations

Two experimental axes isolate the effect of negative-frame training data, corresponding to RQ1 and RQ2.

1) Ratio sweep (RQ1). For each architecture, five training sets are constructed by holding the full 2,401 positive-frame core fixed and sampling nested negative-frame subsets to reach exact arithmetic 20% stepping: 0% (0 neg / 2,401 total), 20% (600 neg / 3,001 total), 40% (1,600 neg / 4,001 total), 60% (3,602 neg / 6,003 total), and 80% (9,604 neg / 12,005 total). This yields 3 architectures × 5 ratios = 15 training runs. All non-negative-frame factors (positive frames, augmentation schedules, hyperparameters, epochs) are held constant within an architecture across ratios, isolating negative-frame ratio as the sole independent variable.

2) Hard-negative curation (RQ2). For each architecture, the ratio identified as best-performing in the RQ1 sweep is used as the anchor point for a controlled comparison: negatives sampled randomly at that ratio versus negatives sampled via a hard-negative mining protocol, at matched dataset size (so any difference reflects composition, not volume). The mining protocol is:

Train a baseline model (the architecture's 0%-negative or lowest-ratio checkpoint) on the full negative-frame pool at inference time only.
Retain negative frames on which the baseline produces a false-positive prediction above a fixed confidence threshold [τ].
If retained frames are fewer than the target ratio requires, backfill with randomly sampled negatives to match the matched dataset size exactly.

This yields 3 architectures × 2 sampling strategies (random vs. hard, at the matched best ratio) = 6 additional runs, for 21 total training runs.

D. Training Protocol

### Corrected & Frozen Configuration — RTX 4060 8GB

| Parameter                 |           YOLO11n |           YOLO26n |         D-FINE-N |
| ------------------------- | ----------------: | ----------------: | ---------------: |
| **Input Resolution**      |         640 × 640 |         640 × 640 |        640 × 640 |
| **Physical Batch Size**   |                16 |                16 |                4 |
| **Gradient Accumulation** |                 1 |                 1 |                8 |
| **Effective Batch Size**  |                16 |                16 |               32 |
| **Epochs**                |               100 |               100 |              160 |
| **Optimizer**             |            `auto` |            `auto` |            AdamW |
| **AMP**                   |                 ✓ |                 ✓ |                ✓ |
| **Seed**                  |  **1 fixed seed** |  **1 fixed seed** | **1 fixed seed** |
| **Native Augmentation**   |                 ✓ |                 ✓ |                ✓ |
| **Augmentation Stop**     | `close_mosaic=10` | `close_mosaic=10` | `stop_epoch=148` |
| **Backbone LR**           |                 — |                 — |       **0.0004** |
| **Head / Transformer LR** |                 — |                 — |       **0.0008** |
| **Weight Decay**          |            0.0005 |            0.0005 |           0.0001 |
| **EMA Restart Decay**     |                 — |                 — |           0.9999 |

### Key Methodological Statement

> The models were trained using their respective native optimization and augmentation frameworks. Hardware-constrained batch sizes were adjusted for the RTX 4060 (8 GB), with gradient accumulation applied only to D-FINE-N. Official learning rates, weight decay, augmentation schedules, and training budgets were retained where applicable. The same fixed random seed and training configuration were used across all negative-frame ratio configurations, with the negative-frame ratio as the controlled experimental variable.

### Protocol Details & Hardware Adaptation

To ensure reproducible, hardware-isolated comparisons on an NVIDIA RTX 4060 (8 GB VRAM), models are trained from their official pretrained COCO checkpoints at a uniform 640 × 640 resolution under mixed precision (AMP):
1. **Batch Size & Gradient Accumulation:** YOLO11n and YOLO26n operate at a physical and effective batch size of 16 without gradient accumulation. For D-FINE-N, VRAM constraints at 8 GB require a physical batch size of 4; gradient accumulation across 8 steps yields an effective batch size of 32. This represents a pragmatic hardware adaptation, rather than an official native batch configuration; we explicitly do not claim effective batch 32 is numerically equivalent to the official batch 128.
2. **Optimization & Learning Rates:** Ultralytics models utilize `optimizer: auto` (allowing the framework to configure SGD/AdamW defaults with official weight decay 0.0005). D-FINE-N utilizes AdamW with official, unscaled learning rates (0.0004 backbone, 0.0008 head/transformer, 0.0001 weight decay, and EMA restart decay of 0.9999).
3. **Training Budgets & Augmentation Schedules:** YOLO variants are trained for 100 epochs with mosaic augmentation disabled for the final 10 epochs (`close_mosaic=10`). D-FINE-N is trained for 160 epochs with augmentation disabled at epoch 148 (`stop_epoch=148`).
4. **Deterministic Single-Seed Evaluation:** A single deterministic seed (`seed=42`) is held constant across all models and ratio splits, strictly avoiding repeated multi-seed variance to respect compute budgets while maintaining zero confounding across negative-ratio treatments.

Table I: run summary (21 runs)
Figure 1: ratio sweep across architectures (main finding)
Table II: best ratio per architecture + endpoints
Table III: random vs. hard-mined negatives at matched ratio
Table IV: FP-per-1k → estimated nuisance-alert rate per hour
