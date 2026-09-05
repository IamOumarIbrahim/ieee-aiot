# IV. Experimental Results (Draft Blueprint)

> **Author Note & Claims Boundary:**
> All models are evaluated via single point estimates (`seed=42`) without multi-seed averaging.
> * **Safe to claim:** The shape of ratio–performance curves (inverted-U vs. monotonic); best-performing ratio *among tested levels (0%, 20%, 40%, 60%, 80%)*; paired comparisons at matched dataset size (RQ2); analytical nuisance-alert estimates.
> * **Not safe to claim:** Inferential statistical significance (no p-values, ANOVA, or "significantly"); claiming an unsearched global optimum; generalizability to non-nano scales or unmeasured hardware latency shifts.

---

## A. Negative-Frame Ratio Sensitivity Sweep (RQ1)

* **Objective:** Determine whether the performance-maximizing negative-frame ratio differs across YOLO11n, YOLO26n, and D-FINE-N, or whether an architecture-invariant optimal ratio exists.
* **Associated Artifacts:**
  * **Table II:** Experimental Run Summary (15 ratio sweep runs across 5 ratio levels).
  * **Table III:** Best-performing ratio per architecture among tested levels, alongside boundary endpoints (0% positive-only baseline and 80% dominant-negative cap).
  * **Figure 1:** Cross-architecture performance trajectories across the 5 ratio levels, plotting mAP@50, Precision, Recall, and False Positives per 1,000 frames (FP/1k).

### Key Result Observations (To fill upon run completion):
* **YOLO11n Operating Point:** [Best ratio observed among tested levels: _% · mAP@50 delta from 0% baseline: +_._ · FP/1k reduction: _%]
* **YOLO26n Operating Point:** [Best ratio observed among tested levels: _% · mAP@50 delta from 0% baseline: +_._ · FP/1k reduction: _%]
* **D-FINE-N Operating Point:** [Best ratio observed among tested levels: _% · mAP@50 delta from 0% baseline: +_._ · FP/1k reduction: _%]
* **Cross-Architecture Invariance Verdict:** [Results indicate that the best-performing ratio (is identical at _% across all three / differs meaningfully between CNN and transformer architectures)].

---

## B. Random Subsampling vs. Hard-Negative Curation (RQ2)

* **Objective:** Benchmark curated hard negatives (mined from baseline false positives) against random negative subsampling at matched dataset size to quantify whether active curation yields architecture-dependent benefits.
* **Associated Artifacts:**
  * **Table IV:** Paired comparison of detection accuracy (mAP@50, mAP@50:95) and FP/1k between random subsampling and hard-negative mining at each architecture's best-performing ratio.

### Key Result Observations (To fill upon run completion):
* **Curation Delta (YOLO11n):** [Hard-negative mining produced a (positive / neutral / negative) delta of _._ mAP@50 and _._ FP/1k relative to matched random negatives].
* **Curation Delta (YOLO26n):** [Hard-negative mining produced a delta of _._ mAP@50 and _._ FP/1k].
* **Curation Delta (D-FINE-N):** [Hard-negative mining produced a delta of _._ mAP@50 and _._ FP/1k].
* **Takeaway:** [Curating hard negatives (justifies / does not justify) additional data-engineering overhead relative to random collection].

---

## C. False-Positive to Operational Deployment Cost Translation (Contribution 3)

* **Objective:** Reframe detector evaluation for continuous edge AIoT deployment by translating measured FP/1k rates into estimated operational costs (nuisance alerts per hour and unnecessary edge compute wakeups).
* **Associated Artifacts:**
  * **Table I:** Model Architecture & Baseline Efficiency (Parameters, FLOPs, Edge Latency — reported once as fixed context).
  * **Table V:** False Positives per 1,000 frames mapped to estimated nuisance alerts per hour across streaming operational rates (e.g., 5 FPS, 15 FPS, 30 FPS).

### Key Result Observations (To fill upon run completion):
* **Operating Point Spread:** The transition from 0% negatives to the best-performing ratio reduced the test-set FP/1k rate from [_._] to [_._], translating to an estimated [_._]-fold reduction in nuisance alerts per operating hour at 30 FPS.
* **Architecture Latency Decoupling:** While inference latency is governed strictly by model architecture (Table I), nuisance alert frequency is governed by negative-frame training data composition, confirming that data curation directly dictates real-world deployment usability.
