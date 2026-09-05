# Section IV: Experimental Results and Empirical Evaluation (Draft Blueprint)

> **Author Note & Claims Boundary:**
> All models are evaluated via single point estimates (`seed=42`) without multi-seed averaging.
> * **Safe to claim:** The shape of ratio–performance curves (inverted-U vs. monotonic); best-performing ratio *among tested levels (0%, 20%, 40%, 60%, 80%)*; paired comparisons at matched dataset cardinality (RQ2); analytical nuisance-alert estimates ($\mathcal{A}_h$).
> * **Not safe to claim:** Inferential statistical significance (no p-values, ANOVA, or claiming significance without empirical distributions); claiming an unsearched global optimum; generalizability to non-nano scales or unmeasured hardware latency shifts.

---

## A. Negative-Frame Ratio Sensitivity Sweep (RQ1)

* **Objective:** Determine whether the performance-maximizing negative-frame ratio $r^*$ differs across YOLO11n, YOLO26n, and D-FINE-N, or whether an architecture-invariant optimal ratio exists.
* **Associated Manuscript Table:** **Table III** (Ratio Sweep: Detection Accuracy and Deployment Cost across 15 runs).
* **Associated Metrics:** Mean Average Precision ($\mathrm{mAP}_{50}$, $\mathrm{mAP}_{50:95}$), Precision ($\mathrm{P}$), Recall ($\mathrm{R}$), and False Positives per 1,000 frames ($\mathrm{FP/1k}$).

### Key Result Observations (To fill upon run completion):
* **YOLO11n Operating Point:** [Best ratio observed among tested levels: _% · $\mathrm{mAP}_{50}$ delta from 0% baseline: +_._ · $\mathrm{FP/1k}$ reduction: _%]
* **YOLO26n Operating Point:** [Best ratio observed among tested levels: _% · $\mathrm{mAP}_{50}$ delta from 0% baseline: +_._ · $\mathrm{FP/1k}$ reduction: _%]
* **D-FINE-N Operating Point:** [Best ratio observed among tested levels: _% · $\mathrm{mAP}_{50}$ delta from 0% baseline: +_._ · $\mathrm{FP/1k}$ reduction: _%]
* **Cross-Architecture Invariance Verdict:** [Results indicate that the best-performing ratio (is identical at _% across all three / differs meaningfully between CNN and transformer architectures)].

---

## B. Hard-Negative Curation vs. Uniform Random Sampling (RQ2)

* **Objective:** Benchmark curated hard negatives (mined from baseline false positives at $\tau=0.25$) against uniform random negative subsampling at matched dataset cardinality ($|\mathcal{D}_{\mathrm{train, hard}}^{(r^*)}| = |\mathcal{D}_{\mathrm{train, rand}}^{(r^*)}|$) to isolate sample information entropy from volume.
* **Associated Manuscript Table:** **Table IV** (Random Subsampling vs. Hard-Negative Mining at Best Ratio).
* **Associated Metrics:** $\mathrm{mAP}_{50}$, $\mathrm{mAP}_{50:95}$, $\mathrm{FP/1k}$, and $\Delta\mathrm{FP/1k}$.

### Key Result Observations (To fill upon run completion):
* **Curation Delta (YOLO11n):** [Hard-negative mining produced a (positive / neutral / negative) delta of _._ $\mathrm{mAP}_{50}$ and _._ $\mathrm{FP/1k}$ relative to matched random negatives].
* **Curation Delta (YOLO26n):** [Hard-negative mining produced a delta of _._ $\mathrm{mAP}_{50}$ and _._ $\mathrm{FP/1k}$].
* **Curation Delta (D-FINE-N):** [Hard-negative mining produced a delta of _._ $\mathrm{mAP}_{50}$ and _._ $\mathrm{FP/1k}$].
* **Takeaway:** [Curating hard negatives (justifies / does not justify) additional data-engineering overhead relative to random collection].

---

## C. Operational Deployment Cost Translation (Contribution 3)

* **Objective:** Reframe detector evaluation for continuous edge AIoT deployment by translating measured $\mathrm{FP/1k}$ rates into estimated operational nuisance alerts per hour ($\mathcal{A}_h$) via:
  $$\mathcal{A}_h = 3600 \times f_{\mathrm{FPS}} \times p_{\mathrm{neg}} \times \frac{\mathrm{FP/1k}}{1000} = 3.6 \times f_{\mathrm{FPS}} \times p_{\mathrm{neg}} \times \mathrm{FP/1k}$$
* **Associated Manuscript Table:** **Table V** (Projected Operational Nuisance Alerts per Hour: 0% Baseline vs. Optimal Ratio across 5 FPS, 15 FPS, 30 FPS under natural background prevalence $p_{\mathrm{neg}} = 0.809$).
* **Architecture Latency Decoupling:** While inference latency ($t_{\mathrm{inf}}$) is governed strictly by model architecture (Table I), nuisance alert frequency ($\mathcal{A}_h$) is governed by negative-frame training data composition, confirming that data curation directly dictates real-world deployment usability.

### Key Result Observations (To fill upon run completion):
* **Operating Point Spread:** The transition from 0% negatives to the best-performing ratio reduced the test-set $\mathrm{FP/1k}$ rate from [_._] to [_._], translating to an estimated [_._]-fold reduction in nuisance alerts per operating hour at 30 FPS.
* **Alert Suppression Summary:** At 30 FPS ($p_{\mathrm{neg}} = 0.809$), baseline models trigger [_._] alerts/hour vs. [_._] alerts/hour at the optimal ratio, directly alleviating driver alert fatigue and cellular uplink transmission waste.
