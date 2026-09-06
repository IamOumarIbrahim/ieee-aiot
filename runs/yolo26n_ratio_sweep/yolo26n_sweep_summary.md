# Evaluation Summary: yolo26n.pt

- **Timestamp:** 2026-09-06 17:30:48
- **Protocol:** epochs=100, batch=16, imgsz=640, seed=42

### Validation & Test Split Performance

| Split | Ratio | Val mAP50 | Val mAP50:95 | Val FP/1k | Test mAP50 | Test mAP50:95 | Test FP/1k | Train Time (min) |
|---|---|---|---|---|---|---|---|---|
| `train_00_pos_only` | 0% | 0.9161 | 0.6855 | 109.28 | 0.9176 | 0.7095 | 99.06 | 65.5 |
| `train_20_low_neg` | 20% | 0.9417 | 0.7160 | 14.94 | 0.9744 | 0.7632 | 15.72 | 76.9 |
| `train_40_mod_neg` | 40% | 0.9619 | 0.7345 | 7.08 | 0.9830 | 0.7555 | 5.50 | 97.5 |
| `train_60_high_neg` | 60% | 0.9736 | 0.7375 | 9.43 | 0.9660 | 0.7594 | 10.22 | 139.1 |
