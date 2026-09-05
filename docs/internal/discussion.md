# Section V: Discussion and Edge Deployment Implications

## A. Practical Guidelines for AIoT Edge Data Curation
The central research question motivating this work---whether negative-frame configuration can be tuned once and applied across detector architectures, or whether it must be re-tuned per architecture---has direct implications for how AIoT engineering teams allocate a fixed data-curation budget.

* **If empirical results confirm architecture-dependent optima ($r^*$ diverges):** Edge practitioners cannot treat dataset curation as an independent upstream step isolated from model selection. Swapping detector architectures (e.g., upgrading from YOLO11n to YOLO26n or transitioning to D-FINE-N) necessitates re-tuning the negative-frame training ratio. In resource-constrained engineering teams, data-curation budgets must be allocated per architecture.
* **If empirical results demonstrate architecture-invariant optima ($r^*$ is identical):** The optimal negative-frame ratio can be characterized as an intrinsic property of the application domain and background visual distribution rather than the detector backbone. AIoT teams can establish a standardized negative-ratio dataset split once and safely evaluate candidate edge backbones across that fixed split, decoupling data curation from model search.
* **Regarding hard-negative mining utility (RQ2):** If hard-negative mining yields marginal $\mathrm{mAP}$ improvements over matched random sampling, engineering teams can bypass multi-stage mining pipelines in favor of automated random background ingestion, substantially reducing data-pipeline complexity. Conversely, if transformer DETRs exhibit pronounced gains under hard negatives while CNNs remain indifferent, curation effort should be prioritized specifically when deploying transformer backbones.

---

## B. Decoupling Model Latency from Curation-Governed Reliability
The deployment-cost framework in Section IV-C formalizes the decoupling between model execution speed and system operational reliability:
1. **Inference Latency is Architecture-Bound:** Latency ($t_{\mathrm{inf}}$) is governed by parameter volume, layer depth, and FLOPs. It cannot be reduced by training-data configuration.
2. **Operational False Positives are Curation-Bound:** Nuisance alert frequency ($\mathcal{A}_h$) is directly governed by negative-frame prevalence and curation quality, and cannot be mitigated by hardware acceleration (e.g., TensorRT FP16 quantization) without retraining.

In continuous vehicular surveillance and automotive monitoring, false-positive alerts directly incur severe penalties:
* **Bandwidth & Cellular Uplink Costs:** In edge-cloud topologies, spurious detections trigger cellular video snippet uploads over LTE/5G, inflating operational data bills.
* **Edge Compute Duty Cycles:** Spurious detections prevent IoT SoCs from entering low-power sleep states, accelerating battery depletion.
* **Human-Machine Interaction & Alert Fatigue:** Repeated false alerts erode operator trust, leading drivers to disable vital active safety warnings.

---

## C. Limitations and Threats to Validity
1. **Domain and Dataset Scope:** Experiments are conducted on an in-cabin vehicular monitoring dataset. While its 80.9% negative prevalence mirrors continuous AIoT streaming conditions, findings may not fully extrapolate to domains with distinct background textures (e.g., aerial remote sensing, maritime surveillance).
2. **Model Scale:** We evaluate lightweight "nano" detector variants to match micro-edge compute constraints. Larger variants (e.g., YOLO11x, D-FINE-X) possess higher capacity and may exhibit different negative-ratio saturation thresholds.
3. **Deterministic Point Estimates:** Each experimental configuration utilizes a single deterministic training run (`seed=42`) to respect computational budgets on single-GPU hardware, precluding multi-seed inferential hypothesis testing ($p$-values, ANOVA).
4. **Heuristic Mining Threshold:** Hard-negative mining evaluates candidate frames at a fixed operational threshold ($\tau = 0.25$) using a positive-only baseline checkpoint. Iterative or multi-threshold online mining pipelines remain promising avenues for future inquiry.
5. **Analytical Alert Modeling:** Nuisance alert rates $\mathcal{A}_h$ are modeled analytically from test-set $\mathrm{FP/1k}$ values rather than measured via in-vehicle physical telemetry over months of road testing.

---

# Section VI: Conclusion

This paper presents the first systematic, cross-architecture empirical investigation of negative-frame training configurations for lightweight edge object detectors in a safety-critical AIoT application. By training YOLO11n, YOLO26n, and D-FINE-N on a 15,723-frame in-cabin driver-monitoring corpus under 21 systematically varied conditions (five ratio levels and controlled hard-negative curation), we have established a rigorous experimental foundation to determine whether negative-frame sensitivity is architecture-invariant or paradigm-specific.

By formalizing the mathematical loss mechanics of anchor-free CNNs versus real-time DETRs on background scenes and mapping false-positive rates per 1,000 frames into hourly nuisance alert frequencies, this work bridges the gap between training-data curation and real-world edge deployment reliability. Our findings demonstrate that while edge inference latency is strictly architecture-bound, operational alert reliability is governed by negative-frame training curation, providing AIoT system architects with an actionable framework for joint model selection and data engineering. Future work will extend this benchmark across multi-domain AIoT sensor feeds, evaluate larger detector variants, and validate nuisance alert telemetry on physical in-vehicle hardware testbeds.