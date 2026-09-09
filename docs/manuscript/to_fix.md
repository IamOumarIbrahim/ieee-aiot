# `main.tex` — Issues to Fix or Review

> **Source file:** [main.tex](file:///C:/Dev/repos/Public%20repos/ieee-aiot/docs/manuscript/main.tex)

---

## 1. Internal Consistency & Numerical Discrepancies

### 1.1 — Abstract vs. Body: Hard-negative reduction ranges disagree
- **Abstract (L39):** States hard-mined negatives cut false alarms by **"55.6%–71.0%"** for attention-based and dual-assignment models.
- **Table 2 / Body (L376–L389):** Reports reductions of **52.6%–67.8%** (YOLO11n −67.8%, YOLO12n −66.7%, YOLOv10n −52.6%).
- **Body text (L408–L409):** Also says **"52.6%–67.8%"**.
- The abstract's "55.6%–71.0%" matches none of the reported numbers. Fix the abstract to match the table values.

### 1.2 — Abstract vs. Body: Spearman rho for recall
- **Abstract (L39):** Reports Spearman rho = **+0.227** (not significant) for recall.
- **Table 1 footnote block (L256):** Reports Spearman rho = **+0.012** (n.s., p > 0.50).
- These are different values. Determine which is correct and reconcile.

### 1.3 — Body text: Spearman rho for FP/1k disagrees with table
- **Body (L266):** States pooled Spearman rho = **-0.921** (p < 10^-8).
- **Table 1 (L256):** Reports Spearman rho = **-0.937*** (p < 10^-4).
- Different rho values and different p-value thresholds. Reconcile.

### 1.4 — Body text: Pearson r for precision disagrees with table
- **Body (L266):** States Pearson correlation **r = +0.593** (p = 0.006).
- **Table 1 (L257):** Reports Pearson r = **+0.628*** (p < 0.05).
- Different values. One of them is wrong; fix whichever is incorrect.

### 1.5 — Suppression factor discrepancies
- **Body (L463):** Claims YOLO26n suppresses false alarms by **31.5x** — but 80.59 / 3.14 = 25.7x, not 31.5x.
- **Body (L463):** Claims YOLOv10n by **14.5x** — but 57.39 / 4.72 = 12.2x, not 14.5x.
- **Body (L463):** Claims YOLO11n by **12.7x** — but 67.22 / 4.72 = 14.2x, not 12.7x.
- **Body (L463):** Claims YOLO12n by **7.3x** — but 45.20 / 7.08 = 6.4x, not 7.3x.
- None of the suppression factors match the Table 1 data. Recalculate all four from FP/1k at r=0% divided by FP/1k at r=80%.

### 1.6 — Discussion: mAP 50:95 for YOLO12n at 40%
- **Body (L461):** Claims mAP_50:95 = **0.7772** for YOLO12n at 40% ratio.
- **Table 1 (L243):** Reports mAP_50:95 = **0.7659** for YOLO12n at 40%.
- **Table 2 (L385):** Reports mAP_50:95 = **0.7839** for YOLO12n hard-mined at 40%.
- 0.7772 appears nowhere. It may be a dual-seed mean; if so, clarify or use the table value.

### 1.7 — "Multi-seed" description inconsistency
- **Table 1 footnote (L261):** Says "Multi-seed means (seed=42,43 for r <= 40%; single-seed seed=42 for remaining)".
- **Table 2 footnote (L393):** Says "Values report dual-seed sample means (seed=42,43)."
- **Limitations (L472):** Says "Dual-seed replication (seed=42, 43) across all four architectures".
- Table 1 implies only some splits used dual seeds, while limitations text implies all did. Clarify the actual seeding strategy.

### 1.8 — Nuisance alert computation spot-check
- Formula: A_h = 3.6 x f_FPS x p_neg x FP/1k
- YOLO26n 0% baseline at 5 FPS: 3.6 x 5 x 0.809 x 80.59 = ~1,173, but Table 3 (L433) says **1,443**.
- YOLO11n 0% baseline at 5 FPS: 3.6 x 5 x 0.809 x 67.22 = ~979, but Table 3 (L429) says **870**.
- The A_h values in Table 3 do not match the formula with the reported FP/1k and p_neg. Re-derive all entries or check if a different p_neg or rounding was used.

---

## 2. Missing or Unused References

### 2.1 — bibitem b16 (Generalized Focal Loss) is never cited
- **L545–L548:** Reference [b16] (Li et al., "Generalized focal loss") is defined but **never cited** anywhere in the body text.
- Either cite it where relevant (e.g., in the loss-dynamics section discussing DFL) or remove it to save space.

---

## 3. Terminology & Clarity

### 3.1 — "area-attention" vs. "Area Attention" vs. "attention-centric" for YOLO12n
- **Abstract (L39):** YOLO12n is called "attention-centric".
- **Body (L51):** YOLO12n uses "area-attention".
- **Table 1 (L155):** Paradigm listed as "Attn", mechanism as "A2C2f Area".
- Not strictly wrong, but standardize the short label across all mentions.

### 3.2 — First use of acronyms not expanded
- **DMS** first appears (L45) without a formal `(DMS)` parenthetical.
- **NMS** first appears in the abstract (L39) without expansion. Expand to "non-maximum suppression (NMS)" at first use.
- **FPS** is defined at L110 but used earlier in Table I header (L151).
- **BCE** (L120) — confirm this is expanded before first use.
- **SoC** (L181–L182) — "SoCs" is never expanded. Expand to "System on Chip (SoC)".
- **FP32** (L174, L193) — never expanded. Consider expanding once.

### 3.3 — "A2C2f" module name not defined
- The module name "A2C2f" is domain-specific jargon. Consider defining it once (e.g., "Area-Attention C2f block (A2C2f)").

### 3.4 — "C2PSA" and "C3k2" not defined
- **L77, L128, L153:** These module names are never expanded. Define once (e.g., "Cross Stage Partial Self-Attention (C2PSA)").

---

## 4. Writing & Style

### 4.1 — Abstract length and density
- The abstract is a single ~218-word paragraph. Review whether it can be tightened to <= 200 words, or verify the IEEE AIoT 2026 abstract word limit.

### 4.2 — Very long sentence in abstract
- The sentence beginning "Sweeping five negative-frame ratios..." (L39) is ~58 words. Consider splitting into two sentences.

### 4.3 — Inconsistent math formatting for "M parameters"
- **Body (L83):** Uses `$2.3$--$2.6\text{M}$` (roman M inside math).
- **Abstract (L39):** Uses `$2.3$--$2.6$\,M` (M outside math, with thin space).
- Standardize across all occurrences.

### 4.4 — "31.5x" notation
- **Body (L463):** Uses `$\times$` notation for suppression factors. Verify IEEE style consistency (some prefer "fold" notation).

---

## 5. Formatting & LaTeX

### 5.1 — Missing figure file: `fig_negative_taxonomy.png`
- **L400:** `\includegraphics[width=\columnwidth]{fig_negative_taxonomy.png}`
- Verify this file exists in the manuscript directory. If missing, the paper will not compile.

### 5.2 — `\flushend` package potential issues
- **L25:** This package can cause compilation issues in some IEEE templates. Verify it works correctly with the final compiled output.

### 5.3 — `\IEEEoverridecommandlockouts` potentially deprecated
- **L13:** This command is deprecated in newer IEEEtran versions. Verify it's needed for the June 2024 revision template.

### 5.4 — Table footnotes as plain `\footnotesize` blocks
- **Tables 1–4:** Footnotes are implemented as `{\footnotesize ...}` blocks after `\end{tabular}`. Review whether this satisfies the IEEE AIoT template requirements.

### 5.5 — Manual `\vspace{1pt}` after tables
- **L159, L202, L260, L392, L445:** This is fragile and may behave unexpectedly across different builds.

### 5.6 — Two blank lines between `\maketitle` and `\begin{abstract}` (L36–37)
- Cosmetically harmless but unnecessary.

---

## 6. Page Limit & Space Concerns

### 6.1 — Hard 6-page limit
- **Header comment (L7–L9):** States a hard 6-page limit including references.
- Compile and verify the paper fits within 6 pages. With 20 references, 4 tables, 2 figures, this is tight.

### 6.2 — Bibliography compactness hack
- **L483:** `\setlength{\itemsep}{0.6pt}` is used to compress bibliography. Verify this doesn't violate IEEE formatting guidelines.

---

## 7. Scientific / Methodological Review Points

### 7.1 — No confidence intervals or error bars
- Tables 1 and 2 report single-point values (or dual-seed means) but no standard deviations, confidence intervals, or error bars. Consider adding +/- values for FP/1k at minimum.

### 7.2 — Statistical test details incomplete
- **L256–L257:** Correlation significance uses threshold notation in the table but exact p-values in the body. Standardize reporting.

### 7.3 — Operational threshold tau = 0.25 never justified
- The confidence threshold tau = 0.25 is used throughout but never justified. Why 0.25 and not 0.5 or the max-F1 threshold?

### 7.4 — No per-class breakdown
- The limitations section (L471) acknowledges phone_use dominance (81.2%), but no per-class mAP or recall breakdown is provided. Reviewers may ask for this.

### 7.5 — Training step confound not controlled
- **Limitations (L468):** Acknowledged — 100 epochs over variable dataset sizes means more gradient steps at 80%. Consider adding a sentence about why epoch-based training is standard YOLO practice.

### 7.6 — No cross-validation or subject-level splits
- **Limitations (L470):** Frame-level random partition, not subject-level. Reviewers may flag data leakage risk from consecutive frames.

### 7.7 — Ethics / IRB detail is thin
- **L136:** "institutional consent and de-identification" is the only mention. For human-subject DMS data, consider adding "IRB details provided upon acceptance" if double-blind prevents disclosure.

---

## 8. References — Accuracy Concerns

### 8.1 — [b6] Felzenszwalb et al. 2010 — potentially misattributed
- **L89:** Cited as evidence that omitting background images degrades open-world resilience in autonomous driving.
- **Actual paper:** This is the DPM (Deformable Part Models) paper, a foundational detection work — NOT an autonomous driving study about negative-frame omission.
- Verify the claim is supported by this reference, or replace with a more appropriate one.

### 8.2 — [b13] Chen et al. 2023 — verify existence
- "False alert suppression and energy-efficient edge inferencing in smart surveillance IoT" — Verify this is a real published paper with these exact authors, title, and venue.

### 8.3 — [b15] Wang et al. 2023 — verify existence
- "Deep learning on micro-edge devices: A survey..." — Verify this reference exists in IEEE Access with these details.

### 8.4 — [b17] Fridman et al. 2019 — verify author list
- Co-authors "P. Toyman and B. Seaman" are unusual for Lex Fridman's known publications. Verify this is a real paper with these exact co-authors.

### 8.5 — [b10] YOLO26 reference date "2026"
- **L524:** Year listed as 2026. Verify the URL is accessible and the reference is stable.

### 8.6 — Reference formatting inconsistency
- Some references use abbreviated conference names (e.g., "ICCV", "CVPR") while others use longer forms. IEEE style prefers consistent abbreviation. Standardize.

### 8.7 — [b11] Author name: "M. Doermann"
- **L526:** The well-known researcher is David Doermann, not "M. Doermann". Verify actual YOLOv12 paper authors from arXiv:2502.12524.

---

## 9. Double-Blind Compliance

### 9.1 — Verify no identifying information leaks
- Author block (L33–34) and acknowledgment (L480) are correctly anonymized.
- The custom dataset ("15,723-frame in-cabin DMS corpus") could potentially de-anonymize the authors if it is publicly associated with them. Verify.

---

## Summary

| Category | Count |
|---|---|
| Internal consistency / numerical | 8 |
| Missing/unused references | 1 |
| Terminology & clarity | 4 (+sub-items) |
| Writing & style | 4 |
| Formatting & LaTeX | 6 |
| Page limit & space | 2 |
| Scientific / methodological | 7 |
| References accuracy | 7 |
| Double-blind compliance | 1 |
| **Total** | **~40 items** |
