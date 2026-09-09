# Section V: Discussion and Edge Deployment Implications

## A. Practical Guidelines for AIoT Edge Data Curation
The central research questions motivating this work---whether negative-frame configuration can be tuned once and applied across detector architectures, and whether hard-negative mining justifies its curation overhead---yield clear, actionable guidelines for AIoT practitioners:

1. **Architecture-Invariant Convergence at High Negative Ratios:**
   At high negative ratios ($r = 80\%$), all four edge architectures (YOLO11n, YOLO26n, YOLO12n, YOLOv10n) converge to a remarkably narrow false-positive band ($4$--$9$ raw false detections across $1,272$ negative test frames, or $3.14$--$7.08\,\mathrm{FP/1k}$). However, detector design dictates the optimal operating point: while YOLO11n, YOLO26n, and YOLO12n achieve their lowest false alarms at $80\%$ without sacrificing recall, YOLOv10n achieves optimal balance at $r = 60\%$ (matching $80\%$ FP suppression at $4.72\,\mathrm{FP/1k}$ while retaining $7.3$ percentage points higher recall: $0.9758$ vs. $0.9024$, and superior $\mathrm{mAP}_{50}$: $0.9749$ vs. $0.9524$).

2. **Divergent Intermediate Suppression Trajectories:**
   While endpoints converge, intermediate trajectories vary by up to $31.5\times$:
   * **Pure Reparameterized CNN (YOLO26n):** Achieves near-complete false-positive suppression immediately at $r = 20\%$ (plunging from $206$ to $7$ raw false positives, a $96.6\%$ reduction). For compute-constrained edge teams with limited training epochs or storage, pure convolutions provide rapid background regularization at minimal negative data overhead.
   * **Hybrid and Attention Models (YOLO11n, YOLO12n, YOLOv10n):** Require higher negative volumes ($r = 60\%\text{--}80\%$) to reach optimal false-positive suppression (e.g., YOLO11n false positives decline progressively: $126 \to 54 \to 31 \to 12 \to 4$).

3. **Paradigm-Dependent Curation Utility (RQ2):**
   At the compute-efficient reference ratio ($r_{\mathrm{ref}} = 40\%$, $N = 4,001$ images), hard-negative mining produces marked paradigm divergence:
   * **Attention and Dual-Assignment Detectors Benefit Huggingly:** YOLO11n achieves a $-71.0\%$ reduction in false positives ($31 \to 9$), YOLO12n achieves $-65.2\%$ ($23 \to 8$), and YOLOv10n achieves $-55.6\%$ ($18 \to 8$) when switching from random to mined negatives at identical dataset size. The high informational entropy of mined frames provides essential discriminative gradients for global attention mechanisms and dual-assignment heads.
   * **Pure Convolutions Hit Curation Saturation:** For YOLO26n, hard-negative mining and random sampling tie exactly ($7$ raw false positives in both regimes, $0.0\%$ reduction). Reparameterized convolutional kernels saturate early on localized cabin features, rendering complex offline mining pipelines unnecessary for pure CNN architectures.

---

## B. Decoupling Model Latency from Curation-Governed Reliability
The empirical results formalize a vital principle in edge AIoT systems engineering: **inference latency and operational reliability are completely decoupled.**

1. **Inference Latency is Strictly Architecture-Bound:**
   On an NVIDIA RTX 4060 Laptop GPU (batch size 1, FP32, $640 \times 640$), YOLOv10n achieves the fastest inference at $7.7\,\mathrm{ms}$ ($129.2\,\mathrm{FPS}$) due to its NMS-free dual-head design. YOLO11n operates at $16.8\,\mathrm{ms}$ ($59.6\,\mathrm{FPS}$), YOLO12n at $21.3\,\mathrm{ms}$ ($47.0\,\mathrm{FPS}$), and YOLO26n at $21.7\,\mathrm{ms}$ ($46.1\,\mathrm{FPS}$). These speeds are governed by kernel execution profiles and post-processing, unaffected by training data composition.

2. **Operational Reliability is Strictly Curation-Bound:**
   At $30\,\mathrm{FPS}$ continuous streaming, a baseline detector trained without negative frames ($r = 0\%$) triggers between $382$ and $6,993$ nuisance alerts per hour ($\mathcal{A}_h$). Elevating negative prevalence to $r = 80\%$ slashes alert rates to $339$--$765\,\mathcal{A}_h/\mathrm{hr}$ across all models---a dramatic reduction achieved without changing a single line of model architecture or hardware code.

In deployed vehicular and industrial IoT setups, uncurated models cause:
* **Severe Cellular Uplink Congestion:** Thousands of false alerts per hour trigger unnecessary cloud telematics video uploads.
* **Depleted Thermal/Battery Budgets:** Constant post-detection processing keeps edge processors from entering low-power sleep states.
* **Driver Alert Fatigue:** Continuous spurious auditory or visual warnings induce operators to disable safety monitoring systems entirely.

---

## C. Limitations and Ongoing Multi-Seed Validation
1. **Domain and Dataset Scope:** Benchmarks are conducted on an in-cabin vehicular driver-monitoring dataset ($15,723$ frames, $80.9\%$ natural negative prevalence). While highly representative of continuous edge perception, extreme outdoor environments (e.g., adverse weather aerial surveillance) may present different background feature distributions.
2. **Model Scale:** We intentionally evaluated nano-scale models ($2.3$--$2.6\text{M}$ parameters) to strictly isolate architectural paradigm differences under edge constraints. Larger backbones (e.g., YOLO11x, YOLO12x) possess greater capacity and may exhibit different saturation limits.
3. **Multi-Seed Statistical Replication (Actively Running):** The initial 28-run benchmark established deterministic empirical point estimates under `seed=42`. To rigorously quantify stochastic run-to-run variance, Phase 4 multi-seed replication (`seed=43`) across all 4 architectures and 5 nested ratios is actively executing in the background, which will yield mean $\pm$ standard deviation confidence intervals across all primary metrics.
4. **Offline Mining Threshold:** Hard-negative mining in RQ2 was conducted via an offline forward pass at $\tau = 0.25$ using a positive-only baseline. Dynamic, online multi-epoch mining represents an exciting avenue for continuous lifelong edge learning.

---

# Section VI: Conclusion

This paper presents the first comprehensive cross-architecture empirical study investigating negative-frame training configurations across four lightweight edge object detection paradigms: hybrid convolutional-attention (YOLO11n), pure reparameterized convolution (YOLO26n), linear area-attention (YOLO12n), and consistent dual-assignment NMS-free inference (YOLOv10n).

Through 24 systematically controlled training runs and 4 mining inference passes across a 15,723-frame driver-monitoring corpus, we demonstrate that:
1. Negative-frame scaling is the single most powerful lever for false-positive suppression in edge AIoT, driving all architectures to a sub-$10\,\mathrm{FP/1k}$ convergence zone at $80\%$ negative prevalence despite wide variation in intermediate suppression trajectories.
2. Hard-negative curation provides dramatic false-positive reductions ($-55.6\%$ to $-71.0\%$) for attention-based and dual-label architectures at matched dataset cardinality, while pure reparameterized CNNs experience curation saturation.
3. Inference latency ($7.7$--$21.7\,\mathrm{ms}$) is strictly architecture-dependent, whereas operational nuisance alert rates are curation-governed.

These findings equip AIoT system designers with a rigorous, empirically validated framework for co-optimizing data engineering and edge model selection. Future work will extend this framework to multi-camera edge nodes and continuous on-device continual negative mining.