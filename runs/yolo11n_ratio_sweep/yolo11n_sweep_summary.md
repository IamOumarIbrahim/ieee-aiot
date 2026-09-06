# Evaluation Summary: yolo11n.pt

- **Timestamp:** 2026-09-06 17:25:39
- **Protocol:** epochs=100, batch=16, imgsz=640, seed=42

### Validation & Test Split Performance

| Split | Ratio | Val mAP50 | Val mAP50:95 | Val FP/1k | Test mAP50 | Test mAP50:95 | Test FP/1k | Train Time (min) |
|---|---|---|---|---|---|---|---|---|
| `train_00_pos_only` | 0% | 0.9520 | 0.7246 | 69.97 | 0.9769 | 0.7538 | 59.75 | 58.5 |
| `train_20_low_neg` | 20% | 0.9673 | 0.7362 | 24.37 | 0.9878 | 0.7518 | 31.45 | 66.6 |
| `train_40_mod_neg` | 40% | 0.9597 | 0.7253 | 18.87 | 0.9932 | 0.7699 | 24.37 | 84.0 |
| `train_60_high_neg` | 60% | 0.9690 | 0.7267 | 15.72 | 0.9881 | 0.7747 | 12.58 | 122.0 |
