# Source of Truth: Training, Validation, and Test Evidence

**Audit snapshot:** 2026-09-09. This file is based on executable source, run
artifacts, dataset manifests, and logs in this repository. It deliberately does
not use `README`, `DEVLOG`, or any other documentation as evidence.

## How to read this file

The evidence hierarchy is intentional:

1. A run's `args.yaml` is the authority for its saved training options.
2. Its `results.csv` is the authority for epoch-by-epoch training/validation
   output. The tables below use its final recorded epoch.
3. The sweep and RQ2 JSON files are the authority for the explicit post-training
   evaluation of `best.pt` on validation and held-out test data.
4. Configuration and source code establish intended behavior only where there is
   no completed output artifact. Such rows are labelled **configured** or
   **not recorded**, never treated as measured results.

`N/A` means that the output does not record the quantity. It is not zero. The
dash in YOLO26n's DFL columns means the CSV has no DFL column for that model.

## Audit coverage and evidence limits

| Scope inspected (documentation excluded) | Files / artifacts | Result |
|---|---:|---|
| Project Python source under `src/` | 8 | Read and parsed successfully |
| Automation Python under `scripts/` | 9 | Read and parsed successfully |
| PowerShell automation under `scripts/` | 2 | Read |
| Untracked analysis/visualization Python under `scratch/` | 37 | Read and parsed successfully; none is the training authority |
| Vendored D-FINE Python source | 90 | Read and parsed successfully |
| Vendored D-FINE non-Python source/configuration (YAML, C++, shell, build files, Dockerfile) | 53 | Read; all YAML parsed successfully |
| YAML dataset/model/runtime configurations | 24 | Read; D-FINE include chain also resolved by executable verifier |
| YOLO run options | 42 `args.yaml` | Read and grouped below |
| YOLO learning curves | 41 `results.csv` | Final rows read below |
| Persisted YOLO evaluation summaries | 8 sweep JSON + 2 RQ2 JSON + 2 multi-seed JSON | Read below |
| Hard-negative curation audits | 4 JSON | Read below |
| D-FINE training logs | 2 `log.txt` | Parsed as JSON-lines below |
| Checkpoints and qualitative images | 94 checkpoints, 856 image artifacts | File presence checked; they are binary evidence, not the numeric source used here |

The embedded D-FINE checkout is at commit `956d1709314c2c6a4df6f34de232054578a7449f`
and is locally modified: it imports/applies `safe_linear`, normalizes numeric
CUDA device strings, and has an untracked data junction. Those changes are part
of the executable environment.

## Dataset ground truth from generated artifacts

The canonical manifests and COCO files agree on the following counts. All four
classes are present (`yawning`, `hand_over_mouth`, `drinking`, `phone_use`).

| Split / configuration | Total images | Positive images / annotations | Background-only images | Source artifacts |
|---|---:|---:|---:|---|
| Validation | 1,572 | 300 / 300 | 1,272 | `yolo/val.txt`, `coco/instances_val.json` |
| Held-out test | 1,572 | 300 / 300 | 1,272 | `yolo/test.txt`, `coco/instances_test.json` |
| 00% positive-only | 2,401 | 2,401 / 2,401 | 0 | `train_00_pos_only` |
| 20% low-negative | 3,001 | 2,401 / 2,401 | 600 | `train_20_low_neg` |
| 40% moderate-negative | 4,001 | 2,401 / 2,401 | 1,600 | `train_40_mod_neg` |
| 60% high-negative | 6,003 | 2,401 / 2,401 | 3,602 | `train_60_high_neg` |
| 80% maximum-negative | 12,005 | 2,401 / 2,401 | 9,604 | `train_80_max_neg` |
| Each curated configuration | 4,001 | 2,401 / 2,401 | 1,600 | `train_curated_<model>_best_curated` |

Every arithmetic training manifest has zero overlap with either evaluation
manifest; validation and test also have zero overlap with each other. The four
curated sets are model-specific replacements of the random 40% set:

| Curated set | Frames shared with random-40% | Replaced random negatives | New negatives |
|---|---:|---:|---:|
| YOLO11n-curated | 2,662 | 1,339 | 1,339 |
| YOLO12n-curated | 2,650 | 1,351 | 1,351 |
| YOLO26n-curated | 2,645 | 1,356 | 1,356 |
| YOLOv10n-curated | 2,624 | 1,377 | 1,377 |

## What the metric-producing code actually does

| Stage | Executable implementation | Reported quantities / exact rule |
|---|---|---|
| YOLO training | [`train_yolo_sweep.py`](src/training/train_yolo_sweep.py) calls `model.train(...)` | Ultralytics CSV records `train/box_loss`, `train/cls_loss`, model-dependent `train/dfl_loss`, validation losses, P, R, mAP50, and mAP50-95 each epoch. |
| YOLO post-training validation and test | The same file calls `model.val(..., split="val")` and separately `model.val(..., split="test")` on `best.pt`. | P = `metrics.box.mp`, R = `metrics.box.mr`, mAP50 = `metrics.box.map50`, mAP50-95 = `metrics.box.map`. These are the `Val` and `Test` values in the sweep/RQ2 summaries. |
| YOLO background false positives | `calculate_fp_per_1k()` predicts each background-only frame at `conf=0.25`. | `FP/1k = raw predicted boxes / 1,272 × 1,000`. Every displayed test denominator is 1,272. |
| RQ2 curation | [`mine_hard_negatives.py`](src/data/mine_hard_negatives.py) ranks negative-pool frames by maximum detection confidence at `tau=0.25`; it backfills with seed 42 if fewer than 1,600 are mined. | Four audit JSON files record pool size, mined count, backfill, and final count. |
| D-FINE training evaluation | [`DFINE/src/solver/det_solver.py`](DFINE/src/solver/det_solver.py) calls `evaluate(..., self.val_dataloader, ...)` each epoch. | Its log key is named `test_coco_eval_bbox`, but it is **validation** evaluation because the actual loader is `val_dataloader`. COCO stats index 0 = mAP50-95 and index 1 = mAP50. |
| D-FINE held-out-test helper | [`dfine_utils.py`](src/training/dfine_utils.py) can override the validation loader with `instances_test.json`, then compute precision/recall and FP/1k. | This is source capability only: no D-FINE sweep summary or held-out-test/FP output exists in the repository. |

The final CSV row is not necessarily the same as a summary's `Val` row: CSV is
the final training epoch, while summary validation/test is an extra evaluation
of the selected `best.pt` checkpoint.

## YOLO options actually saved with the runs

All saved YOLO runs are 100 epochs at 640 pixels, batch 16, `amp: false`,
`deterministic: true`, `pretrained: true`, `val: true`, `save: true`, and
`resume: false`. They record mosaic `1.0`, `close_mosaic: 10`, horizontal flip
`0.5`, HSV augmentation `(0.015, 0.7, 0.4)`, `auto_augment: randaugment`, and
erasing `0.4`. They record `mixup`, `cutmix`, `copy_paste`, `flipud`, rotation,
translation aside from the default `0.1`, shear, perspective, and BGR as off or
zero; `cos_lr` and `save_json` are false. These facts come from each run's
`args.yaml`, not from current script defaults.

| Saved-option group | Runs | Seed | Optimizer saved in artifact | `lr0` saved in artifact | Important consequence |
|---|---:|---:|---|---:|---|
| A | 8 | 42 | `auto` | 0.01 | YOLO11n and YOLO26n, ratios 0/20/40/60. The effective optimizer chosen by `auto` was not persisted, so it must not be called AdamW from the artifact. |
| B | 16 | 42 | `AdamW` | 0.00125 | YOLO11n and YOLO26n 80% + curated; all YOLO12n and YOLOv10n arithmetic + curated runs. |
| C | 18 | 43 | `AdamW` | 0.00125 | All seed-43 run directories, including the incomplete YOLO11n 80% directory. |

This is a real protocol split: the current sweep source defaults to AdamW and
0.00125, but group A's completed artifacts say `auto` and 0.01. Do not combine
group A with groups B/C as one identical hyperparameter condition.

## YOLO artifact completeness

| Model | Seed 42 | Seed 43 | Evidence-backed conclusion |
|---|---|---|---|
| YOLO11n | 0/20/40/60/80 + curated all complete | 0/20/40/60 + curated complete; 80% has only `args.yaml` | Seed-43 80% has no CSV, checkpoint, or evaluation summary. |
| YOLO12n | 0/20/40/60/80 + curated all complete | 0/20/40 + curated complete | No 60% or 80% seed-43 run directory/artifact exists. |
| YOLO26n | 0/20/40/60/80 + curated all complete | 0/20/40 + curated complete | No 60% or 80% seed-43 run directory/artifact exists. |
| YOLOv10n | 0/20/40/60/80 + curated all complete | 0/20/40 + curated complete | No 60% or 80% seed-43 run directory/artifact exists. |
| YOLOv8n | No saved configuration or run artifact | No saved configuration or run artifact | A local `yolo8n.pt` exists, but none of the benchmark drivers or outputs uses it; there are no numbers. |

## Training endpoints: final row of every available YOLO `results.csv`

`T` columns are training losses; `V-loss` and `V-metric` columns are the
last-epoch validation entries recorded during training. They are not the
post-training best-checkpoint results in the next sections.

| Model | Seed | Config | T box | T cls | T DFL | V box | V cls | V DFL | V P | V R | V mAP50 | V mAP50-95 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo11n | 42 | 00_pos_only | 0.68041 | 0.26786 | 0.98794 | 0.85145 | 0.82078 | 1.05994 | 0.86426 | 0.96260 | 0.94172 | 0.70900 |
| yolo11n | 42 | 20_low_neg | 0.69166 | 0.27456 | 0.99130 | 0.85822 | 0.58381 | 1.05195 | 0.90586 | 0.97140 | 0.96480 | 0.72475 |
| yolo11n | 42 | 40_mod_neg | 0.69601 | 0.27997 | 0.98900 | 0.85029 | 0.51817 | 1.06424 | 0.89871 | 0.97906 | 0.94574 | 0.71146 |
| yolo11n | 42 | 60_high_neg | 0.71477 | 0.28666 | 1.00540 | 0.83811 | 0.49923 | 1.07373 | 0.94399 | 0.96195 | 0.96564 | 0.72497 |
| yolo11n | 42 | 80_max_neg | 0.74837 | 0.32450 | 1.01102 | 0.88165 | 0.50863 | 1.07254 | 0.91706 | 0.96424 | 0.96321 | 0.71079 |
| yolo11n | 42 | curated | 0.72873 | 0.30736 | 1.01218 | 0.84306 | 0.34833 | 1.05039 | 0.95343 | 0.96998 | 0.98183 | 0.74536 |
| yolo11n | 43 | 00_pos_only | 0.69001 | 0.27486 | 0.99456 | 0.87934 | 0.90366 | 1.10529 | 0.82349 | 0.96370 | 0.95659 | 0.70768 |
| yolo11n | 43 | 20_low_neg | 0.70585 | 0.27987 | 0.99989 | 0.87169 | 0.51543 | 1.07111 | 0.88488 | 0.97164 | 0.93492 | 0.69517 |
| yolo11n | 43 | 40_mod_neg | 0.70279 | 0.28365 | 0.99340 | 0.84767 | 0.41738 | 1.06860 | 0.92152 | 0.97994 | 0.95911 | 0.71265 |
| yolo11n | 43 | 60_high_neg | 0.72596 | 0.29439 | 1.01106 | 0.85588 | 0.50138 | 1.05401 | 0.92093 | 0.96685 | 0.95340 | 0.71595 |
| yolo11n | 43 | curated | 0.72128 | 0.29775 | 1.00649 | 0.86028 | 0.49818 | 1.07789 | 0.93153 | 0.94716 | 0.95345 | 0.68934 |
| yolo12n | 42 | 00_pos_only | 0.69916 | 0.27965 | 1.03553 | 0.86727 | 0.93644 | 1.06928 | 0.88480 | 0.96640 | 0.92938 | 0.69247 |
| yolo12n | 42 | 20_low_neg | 0.69943 | 0.28346 | 1.02886 | 0.88714 | 0.54431 | 1.11320 | 0.89539 | 0.96824 | 0.92475 | 0.67301 |
| yolo12n | 42 | 40_mod_neg | 0.71438 | 0.28799 | 1.01378 | 0.87740 | 0.56652 | 1.10130 | 0.91205 | 0.97006 | 0.95848 | 0.70291 |
| yolo12n | 42 | 60_high_neg | 0.72678 | 0.30263 | 1.07958 | 0.89714 | 0.49973 | 1.13671 | 0.93427 | 0.96132 | 0.94302 | 0.68154 |
| yolo12n | 42 | 80_max_neg | 0.75892 | 0.32878 | 1.05660 | 0.87450 | 0.47855 | 1.09470 | 0.91166 | 0.97203 | 0.96409 | 0.71354 |
| yolo12n | 42 | curated | 0.71231 | 0.29416 | 1.04697 | 0.87520 | 0.45674 | 1.08958 | 0.93638 | 0.96459 | 0.96747 | 0.70893 |
| yolo12n | 43 | 00_pos_only | 0.69371 | 0.27735 | 1.05182 | 0.87420 | 0.80721 | 1.11583 | 0.86213 | 0.95644 | 0.93482 | 0.69081 |
| yolo12n | 43 | 20_low_neg | 0.70278 | 0.28390 | 1.05825 | 0.86952 | 0.58741 | 1.10179 | 0.89801 | 0.96287 | 0.93577 | 0.69399 |
| yolo12n | 43 | 40_mod_neg | 0.70761 | 0.28650 | 1.05983 | 0.87262 | 0.54682 | 1.11664 | 0.90423 | 0.97434 | 0.95603 | 0.69212 |
| yolo12n | 43 | curated | 0.71942 | 0.29932 | 1.08868 | 0.87603 | 0.50526 | 1.14350 | 0.94019 | 0.97593 | 0.97544 | 0.71530 |
| yolo26n | 42 | 00_pos_only | 0.67987 | 0.19352 | N/A | 0.81615 | 0.80352 | N/A | 0.83812 | 0.90444 | 0.88658 | 0.66567 |
| yolo26n | 42 | 20_low_neg | 0.68200 | 0.19208 | N/A | 0.84306 | 0.31502 | N/A | 0.87368 | 0.96670 | 0.93138 | 0.71637 |
| yolo26n | 42 | 40_mod_neg | 0.69105 | 0.19257 | N/A | 0.86389 | 0.28798 | N/A | 0.88724 | 0.97271 | 0.95014 | 0.71884 |
| yolo26n | 42 | 60_high_neg | 0.71021 | 0.21074 | N/A | 0.84741 | 0.29773 | N/A | 0.92305 | 0.93106 | 0.95982 | 0.72482 |
| yolo26n | 42 | 80_max_neg | 0.76044 | 0.23040 | N/A | 0.87412 | 0.28714 | N/A | 0.92889 | 0.89832 | 0.95828 | 0.71167 |
| yolo26n | 42 | curated | 0.71792 | 0.21095 | N/A | 0.86127 | 0.28184 | N/A | 0.90708 | 0.97433 | 0.94851 | 0.71079 |
| yolo26n | 43 | 00_pos_only | 0.69857 | 0.20752 | N/A | 0.87323 | 0.85870 | N/A | 0.82524 | 0.94162 | 0.90357 | 0.66723 |
| yolo26n | 43 | 20_low_neg | 0.70126 | 0.21664 | N/A | 0.82825 | 0.34942 | N/A | 0.84743 | 0.93025 | 0.88810 | 0.67010 |
| yolo26n | 43 | 40_mod_neg | 0.70950 | 0.20913 | N/A | 0.85812 | 0.33565 | N/A | 0.86559 | 0.97417 | 0.93102 | 0.68869 |
| yolo26n | 43 | curated | 0.74195 | 0.23940 | N/A | 0.86508 | 0.31302 | N/A | 0.90541 | 0.93347 | 0.95052 | 0.72370 |
| yolov10n | 42 | 00_pos_only | 0.69139 | 0.21444 | 1.01228 | 0.89450 | 0.58522 | 1.09410 | 0.82371 | 0.88079 | 0.89756 | 0.64944 |
| yolov10n | 42 | 20_low_neg | 0.70567 | 0.20719 | 0.97972 | 0.83322 | 0.35149 | 0.99228 | 0.85672 | 0.91720 | 0.91860 | 0.67562 |
| yolov10n | 42 | 40_mod_neg | 0.70578 | 0.20309 | 0.98830 | 0.89115 | 0.32582 | 1.08391 | 0.89823 | 0.94436 | 0.94284 | 0.69072 |
| yolov10n | 42 | 60_high_neg | 0.72317 | 0.22017 | 1.00610 | 0.85810 | 0.29291 | 1.04771 | 0.91292 | 0.95510 | 0.93069 | 0.69283 |
| yolov10n | 42 | 80_max_neg | 0.72784 | 0.23565 | 0.98723 | 0.89947 | 0.33400 | 1.07505 | 0.90581 | 0.92961 | 0.92807 | 0.67577 |
| yolov10n | 42 | curated | 0.70288 | 0.21672 | 0.99071 | 0.88453 | 0.32346 | 1.07445 | 0.92713 | 0.93324 | 0.95104 | 0.70175 |
| yolov10n | 43 | 00_pos_only | 0.69478 | 0.20079 | 0.97330 | 0.82430 | 0.72746 | 0.97896 | 0.83139 | 0.87618 | 0.85853 | 0.65359 |
| yolov10n | 43 | 20_low_neg | 0.69484 | 0.21172 | 0.99009 | 0.87560 | 0.35961 | 1.07953 | 0.87309 | 0.98335 | 0.94474 | 0.70395 |
| yolov10n | 43 | 40_mod_neg | 0.70717 | 0.20745 | 0.98884 | 0.92024 | 0.30071 | 1.08538 | 0.90081 | 0.94772 | 0.93718 | 0.68241 |
| yolov10n | 43 | curated | 0.70356 | 0.21833 | 0.99143 | 0.87927 | 0.29488 | 1.06478 | 0.92732 | 0.93761 | 0.95065 | 0.71580 |

## RQ1 post-training validation and held-out-test results

These are the explicit evaluations of `best.pt`, not the CSV endpoint. `FP`
uses the background-only subset of each evaluation split at confidence 0.25.

| Model | Seed | Ratio | Val P | Val R | Val mAP50 | Val mAP50-95 | Val FP/1k | Test P | Test R | Test mAP50 | Test mAP50-95 | Test FP/1k | Test raw FP/1,272 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo11n | 42 | 0% | 0.8813 | 0.9762 | 0.9520 | 0.7246 | 69.97 | 0.8806 | 0.9741 | 0.9769 | 0.7538 | 59.75 | 76/1272 |
| yolo11n | 42 | 20% | 0.9033 | 0.9816 | 0.9673 | 0.7362 | 24.37 | 0.9202 | 0.9728 | 0.9878 | 0.7518 | 31.45 | 40/1272 |
| yolo11n | 42 | 40% | 0.9099 | 0.9627 | 0.9597 | 0.7253 | 18.87 | 0.9376 | 0.9839 | 0.9932 | 0.7699 | 24.37 | 31/1272 |
| yolo11n | 42 | 60% | 0.9192 | 0.9697 | 0.9690 | 0.7267 | 15.72 | 0.9248 | 0.9920 | 0.9881 | 0.7747 | 12.58 | 16/1272 |
| yolo11n | 42 | 80% | 0.9163 | 0.9604 | 0.9629 | 0.7155 | 14.15 | 0.9644 | 0.9766 | 0.9933 | 0.7606 | 4.72 | 6/1272 |
| yolo11n | 43 | 0% | 0.8350 | 0.9444 | 0.9488 | 0.7206 | 94.34 | 0.8345 | 0.9832 | 0.9878 | 0.7602 | 74.69 | 95/1272 |
| yolo11n | 43 | 20% | 0.8914 | 0.9615 | 0.9489 | 0.7033 | 25.94 | 0.9364 | 0.9900 | 0.9797 | 0.7530 | 30.66 | 39/1272 |
| yolo11n | 43 | 40% | 0.9034 | 0.9793 | 0.9657 | 0.7207 | 18.08 | 0.9240 | 0.9823 | 0.9905 | 0.7723 | 19.65 | 25/1272 |
| yolo11n | 43 | 60% | 0.9280 | 0.9786 | 0.9703 | 0.7301 | 9.43 | 0.9689 | 0.9691 | 0.9916 | 0.7699 | 7.08 | 9/1272 |
| yolo12n | 42 | 0% | 0.8971 | 0.9746 | 0.9472 | 0.7033 | 70.75 | 0.9096 | 0.9840 | 0.9899 | 0.7653 | 51.89 | 66/1272 |
| yolo12n | 42 | 20% | 0.9023 | 0.9650 | 0.9447 | 0.6945 | 34.59 | 0.9147 | 0.9837 | 0.9830 | 0.7481 | 35.38 | 45/1272 |
| yolo12n | 42 | 40% | 0.9186 | 0.9675 | 0.9626 | 0.7095 | 18.08 | 0.9150 | 0.9959 | 0.9882 | 0.7772 | 18.08 | 23/1272 |
| yolo12n | 42 | 60% | 0.9300 | 0.9644 | 0.9578 | 0.7085 | 11.79 | 0.9348 | 0.9960 | 0.9888 | 0.7766 | 9.43 | 12/1272 |
| yolo12n | 42 | 80% | 0.8915 | 0.9509 | 0.9637 | 0.7164 | 14.15 | 0.9621 | 0.9767 | 0.9925 | 0.7529 | 7.08 | 9/1272 |
| yolo12n | 43 | 0% | 0.8798 | 0.9472 | 0.9594 | 0.7076 | 47.17 | 0.9360 | 0.9805 | 0.9922 | 0.7339 | 38.52 | 49/1272 |
| yolo12n | 43 | 20% | 0.8860 | 0.9687 | 0.9569 | 0.7094 | 33.02 | 0.8762 | 0.9889 | 0.9684 | 0.7416 | 36.16 | 46/1272 |
| yolo12n | 43 | 40% | 0.8899 | 0.9754 | 0.9678 | 0.7128 | 16.51 | 0.9248 | 0.9938 | 0.9850 | 0.7546 | 19.65 | 25/1272 |
| yolo26n | 42 | 0% | 0.8523 | 0.9276 | 0.9161 | 0.6855 | 109.28 | 0.8649 | 0.8988 | 0.9176 | 0.7095 | 99.06 | 126/1272 |
| yolo26n | 42 | 20% | 0.8872 | 0.9663 | 0.9417 | 0.7160 | 14.94 | 0.9390 | 0.9785 | 0.9744 | 0.7632 | 15.72 | 20/1272 |
| yolo26n | 42 | 40% | 0.9270 | 0.9253 | 0.9619 | 0.7345 | 7.08 | 0.9411 | 0.9653 | 0.9830 | 0.7555 | 5.50 | 7/1272 |
| yolo26n | 42 | 60% | 0.9230 | 0.9787 | 0.9736 | 0.7375 | 9.43 | 0.9587 | 0.9086 | 0.9660 | 0.7594 | 10.22 | 13/1272 |
| yolo26n | 42 | 80% | 0.8944 | 0.9651 | 0.9617 | 0.7155 | 6.29 | 0.9217 | 0.9794 | 0.9736 | 0.7671 | 3.14 | 4/1272 |
| yolo26n | 43 | 0% | 0.8800 | 0.8710 | 0.9283 | 0.6830 | 87.26 | 0.8465 | 0.9223 | 0.9164 | 0.7076 | 62.11 | 79/1272 |
| yolo26n | 43 | 20% | 0.8852 | 0.9723 | 0.9321 | 0.6945 | 15.72 | 0.9223 | 0.9741 | 0.9715 | 0.7460 | 18.08 | 23/1272 |
| yolo26n | 43 | 40% | 0.9017 | 0.9391 | 0.9590 | 0.7119 | 17.30 | 0.9038 | 0.9385 | 0.9621 | 0.7220 | 13.36 | 17/1272 |
| yolov10n | 42 | 0% | 0.8956 | 0.8699 | 0.9429 | 0.6842 | 70.75 | 0.9016 | 0.8361 | 0.9196 | 0.7124 | 68.40 | 87/1272 |
| yolov10n | 42 | 20% | 0.9019 | 0.9540 | 0.9550 | 0.7113 | 20.44 | 0.9264 | 0.9083 | 0.9459 | 0.7253 | 23.58 | 30/1272 |
| yolov10n | 42 | 40% | 0.8934 | 0.9583 | 0.9558 | 0.7074 | 16.51 | 0.8918 | 0.9765 | 0.9615 | 0.7428 | 14.15 | 18/1272 |
| yolov10n | 42 | 60% | 0.9150 | 0.9478 | 0.9345 | 0.6991 | 10.22 | 0.9221 | 0.9758 | 0.9749 | 0.7525 | 4.72 | 6/1272 |
| yolov10n | 42 | 80% | 0.8722 | 0.9221 | 0.9370 | 0.6836 | 7.08 | 0.9010 | 0.9024 | 0.9524 | 0.7384 | 4.72 | 6/1272 |
| yolov10n | 43 | 0% | 0.8389 | 0.9667 | 0.9171 | 0.6828 | 37.74 | 0.9115 | 0.9300 | 0.9650 | 0.7449 | 46.38 | 59/1272 |
| yolov10n | 43 | 20% | 0.9023 | 0.9483 | 0.9539 | 0.7083 | 11.01 | 0.8402 | 0.9717 | 0.9386 | 0.7288 | 18.87 | 24/1272 |
| yolov10n | 43 | 40% | 0.8537 | 0.9382 | 0.9462 | 0.7034 | 13.36 | 0.8708 | 0.9810 | 0.9668 | 0.7707 | 15.72 | 20/1272 |

## RQ2: random 40% versus model-specific curated 40%

The `random` row is the RQ1 40% result; the `curated` row is the corresponding
model-specific 4,001-frame curated dataset. Both are post-training evaluations
of `best.pt`.

| Model | Seed | Condition | Train min | Val P/R | Val mAP50 / mAP50-95 | Val FP/1k | Test P/R | Test mAP50 / mAP50-95 | Test FP/1k | Raw test FP/1,272 |
|---|---:|---|---:|---|---|---:|---|---|---:|---:|
| yolo11n | 42 | random 40% | 84.02 | 0.9099 / 0.9627 | 0.9597 / 0.7253 | 18.87 | 0.9376 / 0.9839 | 0.9932 / 0.7699 | 24.37 | 31 |
| yolo11n | 42 | curated | 87.81 | 0.9536 / 0.9694 | 0.9818 / 0.7448 | 6.29 | 0.9615 / 0.9934 | 0.9946 / 0.7786 | 7.08 | 9 |
| yolo11n | 43 | random 40% | 88.00 | 0.9034 / 0.9793 | 0.9657 / 0.7207 | 18.08 | 0.9240 / 0.9823 | 0.9905 / 0.7723 | 19.65 | 25 |
| yolo11n | 43 | curated | 84.47 | 0.9196 / 0.9673 | 0.9600 / 0.7145 | 6.29 | 0.9491 / 0.9935 | 0.9918 / 0.7691 | 7.08 | 9 |
| yolo26n | 42 | random 40% | 97.47 | 0.9270 / 0.9253 | 0.9619 / 0.7345 | 7.08 | 0.9411 / 0.9653 | 0.9830 / 0.7555 | 5.50 | 7 |
| yolo26n | 42 | curated | 98.77 | 0.9232 / 0.9387 | 0.9632 / 0.7215 | 7.86 | 0.9365 / 0.9584 | 0.9845 / 0.7585 | 5.50 | 7 |
| yolo26n | 43 | random 40% | 104.24 | 0.9017 / 0.9391 | 0.9590 / 0.7119 | 17.30 | 0.9038 / 0.9385 | 0.9621 / 0.7220 | 13.36 | 17 |
| yolo26n | 43 | curated | 98.48 | 0.8712 / 0.9672 | 0.9544 / 0.7259 | 8.65 | 0.8728 / 0.9676 | 0.9673 / 0.7452 | 7.08 | 9 |
| yolo12n | 42 | random 40% | 115.25 | 0.9186 / 0.9675 | 0.9626 / 0.7095 | 18.08 | 0.9150 / 0.9959 | 0.9882 / 0.7772 | 18.08 | 23 |
| yolo12n | 42 | curated | 118.16 | 0.9513 / 0.9636 | 0.9743 / 0.7165 | 7.08 | 0.9744 / 0.9764 | 0.9933 / 0.7838 | 6.29 | 8 |
| yolo12n | 43 | random 40% | 117.51 | 0.8899 / 0.9754 | 0.9678 / 0.7128 | 16.51 | 0.9248 / 0.9938 | 0.9850 / 0.7546 | 19.65 | 25 |
| yolo12n | 43 | curated | 115.85 | 0.9307 / 0.9966 | 0.9720 / 0.7278 | 9.43 | 0.9734 / 0.9777 | 0.9923 / 0.7839 | 6.29 | 8 |
| yolov10n | 42 | random 40% | 106.97 | 0.8934 / 0.9583 | 0.9558 / 0.7074 | 16.51 | 0.8918 / 0.9765 | 0.9615 / 0.7428 | 14.15 | 18 |
| yolov10n | 42 | curated | 106.90 | 0.9348 / 0.9383 | 0.9501 / 0.7009 | 7.08 | 0.9328 / 0.9653 | 0.9836 / 0.7697 | 6.29 | 8 |
| yolov10n | 43 | random 40% | 108.96 | 0.8537 / 0.9382 | 0.9462 / 0.7034 | 13.36 | 0.8708 / 0.9810 | 0.9668 / 0.7707 | 15.72 | 20 |
| yolov10n | 43 | curated | 107.81 | 0.8881 / 0.9688 | 0.9539 / 0.7181 | 10.22 | 0.9313 / 0.9320 | 0.9741 / 0.7650 | 7.86 | 10 |

### Curation inputs actually recorded

| Curated model | Candidate negative pool | Confidence threshold | Hard-mined | Random backfill | Final negatives / total training frames |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 10,178 | 0.25 | 596 | 1,004 | 1,600 / 4,001 |
| YOLO12n | 10,178 | 0.25 | 548 | 1,052 | 1,600 / 4,001 |
| YOLO26n | 10,178 | 0.25 | 948 | 652 | 1,600 / 4,001 |
| YOLOv10n | 10,178 | 0.25 | 602 | 998 | 1,600 / 4,001 |

## D-FINE-N: configured settings versus actual output

The D-FINE include chain was resolved by
[`verify_dfine_config_merge.py`](scripts/verify_dfine_config_merge.py), not by
comments in YAML. Each of the five arithmetic configurations resolves as follows:

| Configuration(s) | Epochs | Train batch | Classes | AdamW base / backbone LR | Weight decay | Stop epoch | AMP / scaler | EMA | Train / validation / test annotation |
|---|---:|---:|---:|---|---:|---:|---|---|---|
| 00%, 20%, 40%, 60%, 80% | 160 | 8 | 4 | 0.0008 / 0.0004 | 0.0001 | 148 | disabled; launcher injects `use_amp=False, scaler=None` | enabled, decay 0.9999 | ratio-specific train / `instances_val.json` / `instances_test.json` |

The verifier itself contains a stale check expecting batch 4; its actual merge
output is batch **8**, and therefore the verifier ends with a warning. The
D-FINE launcher describes `accum_steps=4` / effective batch 32, but no
accumulation setting is present in the resolved configuration or the vendored
training engine; only batch 8 is source-verified. The D-FINE source defines
`test_dataloader`, but the ordinary training loop evaluates only
`val_dataloader`.

| D-FINE artifact | Training evidence | Validation evidence (the log's `test_coco_eval_bbox`) | Held-out test / FP evidence | Trust status |
|---|---|---|---|---|
| `runs/dfine_ratio_sweep/dfine_00_pos_only` | 24 logged epochs, numbered 0–23; final loss 17.55084 (VFL 0.39265, bbox 0.27019, GIoU 0.30088, FGL 1.15292); 3,724,463 parameters | Best logged at epoch 12: mAP50-95 0.61793, mAP50 0.85463. Final epoch 23: mAP50-95 0.59894, mAP50 0.80810. | No sweep summary; no test precision/recall, test mAP, or FP/1k persisted. | **Incomplete; not comparable to completed YOLO test rows.** |
| `runs/test_dfine_2_epochs_archive` | 2 logged epochs (0–1); final loss 27.83978 | Final validation mAP50-95 0.16124, mAP50 0.27214 | None | Archived smoke/partial run, not a benchmark result. |
| 20%, 40%, 60%, 80%, and all D-FINE curated configs | Configuration exists | No D-FINE run log/CSV/summary | None | **Configured only; no numbers.** |

## Multi-seed aggregate caveat

`multi_seed_rq1_summary.json` and `multi_seed_rq2_summary.json` combine only
available seed-42/43 pairs. For two values the code uses
`abs(seed42 - seed43) / sqrt(2)` as the standard deviation. These aggregates do
not create missing seed-43 60%/80% measurements and do not make the D-FINE run
complete. Use the per-seed tables above as the primary record.

## Bottom line

The repository contains complete, test-backed YOLO evidence for the rows shown
above, but it does **not** contain a complete uniform all-model/all-seed sweep.
In particular, the early seed-42 YOLO11n/YOLO26n runs have a different saved
optimizer/LR condition, several seed-43 ratio runs were never completed, and
D-FINE has no held-out-test result. Those facts should accompany any comparison
or claim derived from these numbers.
