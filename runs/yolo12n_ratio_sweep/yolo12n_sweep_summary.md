# Evaluation Summary: C:\Dev\repos\Public repos\ieee-aiot\yolo12n.pt

- **Timestamp:** 2026-09-07 05:22:57
- **Protocol:** epochs=100, batch=16, imgsz=640, seed=42

### Validation & Test Split Performance

| Split | Ratio | Val mAP50 | Val mAP50:95 | Val FP/1k | Test mAP50 | Test mAP50:95 | Test FP/1k | Train Time (min) |
|---|---|---|---|---|---|---|---|---|
| `train_00_pos_only` | 0% | 0.9472 | 0.7033 | 70.75 | 0.9899 | 0.7653 | 51.89 | 76.7 |
| `train_20_low_neg` | 20% | 0.9447 | 0.6945 | 34.59 | 0.9830 | 0.7481 | 35.38 | 91.1 |
| `train_40_mod_neg` | 40% | 0.9626 | 0.7095 | 18.08 | 0.9882 | 0.7772 | 18.08 | 115.2 |
| `train_60_high_neg` | 60% | 0.9578 | 0.7085 | 11.79 | 0.9888 | 0.7766 | 9.43 | 164.7 |
| `train_80_max_neg` | 80% | 0.9637 | 0.7164 | 14.15 | 0.9925 | 0.7529 | 7.08 | 319.5 |
