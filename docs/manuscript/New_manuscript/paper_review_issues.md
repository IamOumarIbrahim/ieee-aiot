# Reviewer Notes: What Is Stopping This Paper From Being Submission Ready

Reviewed as an IEEE conference submission. Items are grouped by severity. Section 1 alone should block submission until resolved.

---

## 1. CRITICAL: Reference List Integrity

I spot checked nine references against their real sources. Two matched exactly. Five did not.

1. **`lambert2026chinese`**: the bibliography title reads "Reported methods of circumvention in automotive cabin monitoring systems." The actual Electrek article at that URL is titled "Tesla's self-driving safeguards fooled by $30 doll heads" (June 15, 2026). A bibliography entry must use the real title of the source, not a paraphrase.
2. **`yolodrive2025`**: listed as "IEEE Trans. Intell. Veh., early access, 2025." A paper with this exact title was located in *Computers, Materials & Continua* (Tech Science Press), vol. 87, no. 2, 2026, not IEEE Transactions on Intelligent Vehicles. The listed authors (H. Zhang, W. Wang, K. Li) could not be confirmed against that publication.
3. **`pyolov82024`**: cited as "M. A. Hossain, R. A. Khan, and S. Sajid, 'P-YOLOv8: An efficient TinyML object detection framework for embedded driver distraction systems,' arXiv:2410.15602, 2024." The real paper at arXiv:2410.15602 is titled "P-YOLOv8: Efficient and Accurate Real-Time Detection of Distracted Driving," and its authors include Mohamed R. Elshamy and Abdel-Hameed A. Badawy. Neither the title, the authors, nor the "structural channel pruning" description used in the Related Work text matches the real paper (the real P-YOLOv8 uses a pretrained backbone plus classification head, not channel pruning).
4. **`yolodrowsiness2025`**: cited as "K. Sharma and R. Patel, 'Comprehensive evaluation of real-time computer vision YOLO models for embedded driver drowsiness detection,' arXiv:2509.17498, 2025." The real paper at that arXiv ID is "Vision-Based Driver Drowsiness Monitoring: Comparative Analysis of YOLOv5-v11 Models" by Dilshara Herath, Chinthaka Abeyrathne, and Prabhani Jayaweera.
5. **`yolov12dms2025`**: cited as "A. Al-Mansoor, M. S. Rahman, and T. Hussain, ... Int. J. Saf. Secur. Eng., vol. 15, no. 9, pp. 115-126, 2025." A paper matching that exact volume and issue was located, titled "YOLOv12-Based Driver Monitoring System: Real-Time Applications in ADAS," pp. 1921-1932, with a different author affiliation (Ibn Tofail University). Title, authors, and page range do not match.

**Verified correct:** `jocher2026yolo26` (arXiv:2606.03748, title and authors match) and `ec2026safercars` (the 7 July 2024 / 7 July 2026 ADDW dates are accurate per EUR-Lex and Regulation (EU) 2019/2144).

This pattern (real identifier, fabricated title/authors/venue) appeared in five of seven non-classic references checked. Every remaining reference must be manually verified against its actual source (title, full author list, venue, volume, issue, and page numbers) before submission. Submitting a reference list with fabricated metadata is a research-integrity problem, not a formatting nitpick, and reviewers or an editorial integrity check are likely to catch it.

---

## 2. Numerical Inconsistency in the Text

6. In Section V-C ("Nuisance Alert Projection"), the text states hard-mined 40% models produce "550-619 alerts/hr at 30 FPS, an 84% reduction from baseline." Using the paper's own Table VI values, the actual per-model reductions from the 0% baseline to the 40% hard-mined condition at 30 FPS are:
   - YOLO11n: 5,873 to 619 = 89.5% reduction
   - YOLO26n: 7,041 to 550 = 92.2% reduction
   - YOLO12n: 3,949 to 550 = 86.1% reduction
   - YOLOv10n: 5,014 to 619 = 87.7% reduction

   None of these equal 84%, and 84% is below even the smallest true reduction (86.1%). The true range is roughly 86% to 92%. This number needs to be recalculated and corrected, or the authors need to clarify what "84%" was actually meant to represent.

---

## 3. Validity Threats

7. **Frame leakage risk between splits.** Frames were extracted at 1 FPS from continuous video, and the split is stratified by cue instance rather than by driver or session. Consecutive or visually near-duplicate background frames from the same driver and recording could land in both a training negative pool and the held-out validation or test negative pool. If so, the reported false-alarm suppression (the paper's central result) may be partly an artifact of the model memorizing driver-specific background texture rather than learning generalizable suppression. This is currently mentioned only as a passing limitation ("Driver Partition Overlap"); given that it could inflate the paper's headline numbers, it deserves a prominent discussion, and ideally a supplementary subject-disjoint replication.
8. **Threshold ambiguity.** FP/1k is computed at a fixed confidence threshold (tau = 0.25), but precision and recall in the same table are reported "at maximum-F1," which is a different, per-configuration threshold. It is not stated whether that threshold was chosen on the validation split (acceptable) or the test split itself (data leakage/optimistic bias). As written, a reader cannot reconstruct a single confusion matrix that produces all four reported numbers (FP/1k, P, R, F1) for a given row.
9. **Single-seed conclusions for the headline configurations.** The 60% and 80% negative-ratio conditions, which anchor the paper's main practical recommendation ("an 80% ratio achieves the lowest false-alarm floor"), were trained with only one seed. No variance estimate exists for the configuration the paper recommends most strongly.
10. **n=2 seed claims stated with more confidence than the sample size supports.** For example, "YOLO26n displays seed sensitivity, with suppression ranging from 0.0% under Seed 42 to 47.0% under Seed 43" is a conclusion about architectural behavior drawn from two data points.
11. **Single label per frame.** The annotation protocol enforces exactly one bounding box per positive frame and a mutual-exclusion rule between overlapping cues (e.g., yawning under a covering hand is always labeled hand-over-mouth, never both). This simplifies the detection problem relative to naturalistic driving, where simultaneous cues are common (the paper's own related-work discussion of YOLO-Drive notes this explicitly), and could understate real-world false-negative risk.
12. **Pool reuse between RQ1 and RQ2.** It is not stated whether the negative pool used for the RQ1 ratio sweep and the pool mined for RQ2 hard negatives are drawn from disjoint subsets, or whether "random negative" and "hard-mined negative" configurations at 40% could share frames beyond the mining process itself.

---

## 4. Statistical Reporting Problems

13. **Pseudoreplication in the Spearman correlations.** The reported correlations (rho = -0.937, p < 1e-4; rho = +0.546, p < 0.05; rho = +0.012, p > 0.50) are computed across 20 points formed by 4 models times 5 ratios. Because four of the "independent" points share the same model, this is a repeated-measures structure, and the standard Spearman p-value assumes independent observations. The correlation coefficients themselves are fine as descriptive statistics, but the p-values as reported are not statistically defensible without a correction for the model-level clustering (e.g., a mixed-effects or cluster-robust approach).
14. **No confidence intervals** anywhere on the headline claims (6.4x-25.7x reduction; 33.3%-67.8% reduction).
15. **"Stable" mAP50:95** (0.7086-0.7766, roughly a 9% relative spread) is asserted without a formal test of stability.
16. **No significance test** for the RQ2 random-vs-hard-mined comparison; the claim rests on two-seed means and +/-1 SD error bars, several of which are large relative to the mean (e.g., YOLO26n random-40% FP/1k = 9.43 +/- 5.56).

---

## 5. Additional Citation and Factual Accuracy Checks

17. Table I lists YOLO12n at 7.6 GFLOPs (2.6M params). The officially published Ultralytics specification for YOLO12n is 6.5 GFLOPs at 640x640 (docs.ultralytics.com/models/yolo12), a roughly 17% discrepancy. Either the measurement methodology differs from Ultralytics' own benchmark (in which case say so) or the value needs correcting.
18. The Electrek source (item 1 above) is also a non-peer-reviewed news/opinion blog being used to support an academic claim about automation "disuse" alongside Parasuraman and Riley (1997). Even once the title is fixed, consider pairing it with a more rigorous primary source (an NHTSA filing, or a peer-reviewed automation-disuse study) so the claim does not rest solely on a blog post.
19. The introduction and the nuisance-alert discussion both use plural framing ("naturalistic on-road studies indicate...") while citing a single 2005 source for the 14.5% figure. Either add a corroborating, more recent citation, or rephrase to singular ("one widely cited study").

---

## 6. Ethics, Licensing, and Data Governance

20. The ethics footnote self-declares an exemption under 45 CFR 46.104(d)(4) rather than stating that an Institutional Review Board made this determination. Most venues require the exemption to be certified by the institution's IRB or ethics committee, not self-assessed by the authors. Clarify whether an IRB actually issued this determination.
21. Figure 1(a) reproduces DMD face-crop images directly in the paper. Confirm that DMD's non-commercial research license permits publishing identifiable subject imagery in a third-party paper's figures, as opposed to only permitting research use of the dataset itself. Some datasets require blurring, synthetic substitution, or explicit written permission for this.
22. Given the EU regulatory framing, a short data-governance statement (e.g., GDPR basis for processing facial data) would strengthen the ethics section.

---

## 7. Reproducibility

23. No code, trained weights, or exact dataset split files/IDs are released or linked (no repository, no archive DOI). As written, the 20-configuration sweep and the mining protocol cannot be independently reproduced.
24. Hyperparameter reporting is incomplete: no weight decay, no warm-up length, no early-stopping criterion, no statement about deterministic training (cuDNN determinism, dataloader worker seeding), despite the paper's own conclusions resting partly on seed-to-seed variance.
25. Figure 1 depends on five external image files (`figures/cue_yawn.jpg`, `cue_hand_mouth.jpg`, `cue_drink.jpg`, `cue_phone.jpg`, `cue_negative.jpg`) that are not included with the source. The document will not compile with Figure 1 rendered correctly until these are supplied.

---

## 8. Figures, Tables, Presentation

26. In Fig. 6 ("results_combined"), the four 40%-hard-mined star markers are manually offset to x = 39.0, 39.7, 40.3, 41.0 purely for visual separation, even though all four represent the same 40% ratio. The caption only partially clarifies this; state explicitly that the x-offset is jitter for legibility, not four different ratios.
27. No per-class (yawn / hand-over-mouth / drink / phone) precision, recall, or AP breakdown is reported anywhere, despite the mutual-exclusion rule creating an inherently ambiguous boundary between yawn and hand-over-mouth. A per-class table or confusion matrix would substantiate the claim that recall is unaffected by negative curation.
28. Fig. 3 (RQ1 sweep) reports point estimates only with no error bars, while Fig. 5 (RQ2 mining) does report +/-1 SD bars. The inconsistent level of rigor between the two central results figures is easy for a reviewer to notice.
29. Several caption sentences are interpretive rather than descriptive (e.g., "confirming that architectural complexity does not violate frame-rate requirements"). IEEE style favors moving interpretation into body text and keeping captions descriptive.

---

## 9. Venue and Compliance Checks

30. The paper contains 6 tables and 6 figures (one full width). Confirm this fits the target venue's page limit; this is dense for a typical IEEE conference page budget.
31. Confirm the venue's double-blind policy on ethics/acknowledgment footnotes. The current footnote does not name an institution, so it is likely fine, but some venues ask that IRB/consent language be deferred to the camera-ready version out of caution.
32. The third listed contribution ("Execution Latency and Alert Projection") is closer to a standard benchmarking exercise than a novel methodological contribution. Consider reframing it as part of the evaluation rather than a headline contribution, or sharpen what is actually novel about the alert-projection formulation.

---

## Suggested Order of Fixes

1. Audit and correct every bibliography entry (Section 1).
2. Correct or explain the 84% figure (Section 2).
3. Resolve the threshold ambiguity and add the leakage caveat (Section 3, items 7-8).
4. Add the statistical caveats or rerun with proper clustering-aware tests (Section 4).
5. Everything else can be addressed in a final polish pass.
