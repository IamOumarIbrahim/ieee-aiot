# Section II: Related Work

## A. Class Imbalance and Negative Sampling in Object Detection
Foreground-background class imbalance represents a canonical challenge in object detection [1]. Dense one-stage detectors evaluate tens of thousands of spatial candidate locations per frame, of which only a minuscule fraction overlap foreground objects. At the loss formulation level, Focal Loss [1] dynamically scales the standard cross-entropy loss by $(1 - p_t)^\gamma$ to suppress gradient contributions from easy, well-classified negative instances. Online Hard Example Mining (OHEM) [14] constructs training batches exclusively from high-loss regions, ensuring non-trivial gradient updates. Zhang *et al.* [3] demonstrated through Adaptive Training Sample Selection (ATSS) that the mechanism of assigning positive and negative samples exerts a more decisive influence on detector performance than whether the underlying regression head is anchor-based or anchor-free.

Beyond regional assignment, Chen *et al.* [2] showed that negative sample representations are non-static; they undergo representation drift during iterative backpropagation, which can confound decision boundaries unless consistently mined. Similarly, SNIPER [4] confirmed that extracting context chips from false-positive background regions significantly elevates detection precision without incurring the computational overhead of multi-scale image pyramids.

## B. Lightweight Edge Detectors: Hybrid Attention, Reparameterization, Area Attention, and NMS-Free Assignments
Edge AIoT sensing mandates compact detector architectures capable of executing under stringent latency, memory, and thermal constraints [15]. To establish rigorous comparative validity, we evaluate four lightweight edge detector paradigms with strictly matched parameter capacity ($2.3$--$2.6\text{M}$ parameters):

1. **Hybrid CNN-Attention (YOLO11n):** Couples standard convolutional backbones and neck-level feature pyramids with partial self-attention modules (C2PSA) [9]. It employs decoupled heads, Task-Aligned Assigners (TAL) for joint score-overlap optimization, and Distribution Focal Loss (DFL) [16].
2. **Pure Reparameterized CNN (YOLO26n):** Introduces structural reparameterization blocks (RepConv) [10] to enhance feature reuse and maximize feedforward kernel execution throughput without non-convolutional attention branches.
3. **Linear Area Attention (YOLO12n):** Introduces an attention-centric architecture via Area Attention modules (A2C2f) [11], partitioning feature maps into horizontal and vertical visual areas to achieve linear complexity while retaining global receptive fields.
4. **Consistent Dual-Assignment NMS-Free (YOLOv10n):** Eliminates post-processing latency bottlenecks by deploying consistent dual assignments [20]---employing a one-to-many head during training to provide rich supervisory signals while executing inference via an efficient one-to-one head that requires no Non-Maximum Suppression (NMS).

*(Note: Real-time DETRs such as D-FINE-N [11 in earlier exploration] were initially considered, but modern real-time models like YOLO12n and YOLOv10n provide closer parameter parity ($2.3$--$2.6\text{M}$) and direct on-device edge deployment compatibility within the unified AIoT benchmark).*

## C. Negative-Frame Training and Operational Reliability in AIoT
While region-level negative sampling within object-containing images is well-studied, the deliberate inclusion of entire unannotated background frames (negative frames) has received comparatively little formal attention. Practitioner guidelines for YOLO models recommend incorporating 0–10% background images as a heuristic against false positives [8]. However, empirical investigations in specialized domains show marked divergence from this guideline. In floating-debris detection, Deng *et al.* [5] identified an inverted-U performance trajectory across negative proportions from 0% to 40%, with $\mathrm{mAP}$ peaking at approximately 20% before excessive background representation degraded recall. Conversely, in autonomous driving roadwork monitoring, Rezaei *et al.* [6] demonstrated that scaling background images from 10% to over 60% reduced false-positive detections by more than 80% while simultaneously cutting false negatives by 25%. In human-object interaction (HOI), Gupta *et al.* [7] reported that scaling negative-to-positive pair ratios from 10:1 to 1000:1 was essential to regularize complex interaction spaces.

In edge driver-monitoring systems (DMS), visual feeds are captured under dynamic cabin illumination, occlusions, and diverse driver postures [12, 19]. While standard DMS datasets (e.g., State Farm Distracted Driver [18], DMD [19]) emphasize positive cue diversity, deployed embedded nodes encounter vast stretches of routine driving devoid of target cues. False detections directly degrade user trust and induce alert fatigue [6, 13].

## D. Identified Research Gap
Existing studies on negative-frame training suffer from a fundamental limitation: prior investigations tune the negative-to-positive ratio for a single, isolated detector architecture. No prior work investigates whether the performance-maximizing negative ratio or the efficacy of hard-negative mining is architecture-invariant or paradigm-dependent across modern edge detection paradigms (hybrid attention, pure convolutions, area attention, and dual-assignment NMS-free models). This paper addresses this gap through a controlled cross-architecture study across four matched-capacity edge detectors on a standardized AIoT driver-monitoring benchmark.

---

### Reference Mapping

* [1] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE/CVF Int. Conf. Comput. Vision (ICCV)*, Oct. 2017, pp. 2980–2988.
* [2] H. Chen, Y. Wang, G. Wang, and Y. Qiao, "Improving object detection with consistent negative sample mining," in *Proc. Int. Conf. Image Graphics (ICIG)*, Springer, 2019, pp. 667–678.
* [3] S. Zhang, C. Chi, Y. Yao, Z. Lei, and S. Li, "Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2020, pp. 9759–9768.
* [4] B. Singh, M. Najibi, and L. S. Davis, "SNIPER: Efficient multi-scale training," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 31, Dec. 2018, pp. 9333–9343.
* [5] Z. Deng, H. Zhang, L. Zhou, and X. Yang, "Floating-debris detection with empirical negative sampling," *arXiv:2510.23798*, Oct. 2025.
* [6] T. Rezaei, J. Kolb, and W. Stork, "Deep neural network based roadwork detection for autonomous driving," *arXiv:2604.02282*, Apr. 2026.
* [7] T. Gupta, A. Schwing, and D. Hoiem, "No-frills human-object interaction detection," in *Proc. IEEE/CVF Int. Conf. Comput. Vision (ICCV)*, Oct. 2019, pp. 1960–1969.
* [8] Ultralytics, "Tips for best training results: Background images," *Ultralytics Documentation*, 2024.
* [9] G. Jocher and J. Qiu, "Ultralytics YOLO11," *GitHub repository*, 2024.
* [10] Ultralytics, "Ultralytics YOLO26 architecture and models," *Ultralytics Documentation*, 2026.
* [11] Y. Tian, Q. Ye, and M. Doermann, "YOLOv12: Attention-centric real-time object detectors," *arXiv:2502.12524*, Feb. 2025.
* [12] A. Sengupta, V. Sharma, and R. Mall, "Computer vision for in-cabin driver monitoring: A comprehensive review," *IEEE Trans. Intell. Transp. Syst.*, vol. 24, no. 9, pp. 8945–8965, Sep. 2023.
* [13] J. Chen, K. Li, Z. Rong, and K. Bilgic, "False alert suppression and energy-efficient edge inferencing in smart surveillance IoT," *IEEE Internet Things J.*, vol. 10, no. 14, pp. 12340–12352, Jul. 2023.
* [14] A. Shrivastava, A. Gupta, and R. Girshick, "Training region-based object detectors with online hard example mining," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2016, pp. 761–769.
* [15] E. Wang, M. Davis, S. Xu, and T. Huang, "Deep learning on micro-edge devices: A survey on architectural efficiency and inference constraints," *IEEE Access*, vol. 11, pp. 45120–45138, May 2023.
* [16] X. Li et al., "Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, Dec. 2020, pp. 18312–18323.
* [17] L. Fridman, P. Toyman, and B. Seaman, "Cognitive load and distraction detection via multi-sensor driver monitoring in automated vehicles," *IEEE Trans. Intell. Veh.*, vol. 4, no. 2, pp. 270–281, Jun. 2019.
* [18] State Farm, "State Farm Distracted Driver Detection," *Kaggle Competition Dataset*, 2016.
* [19] I. Ortega et al., "DMD: A large-scale multi-modal driver monitoring dataset for attention and distraction analysis," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, Jun. 2020, pp. 498–507.
* [20] A. Wang et al., "YOLOv10: Real-time end-to-end object detection," *arXiv:2405.14458*, May 2024.