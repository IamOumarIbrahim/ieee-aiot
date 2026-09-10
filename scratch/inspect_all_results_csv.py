import os
import glob
import pandas as pd
from pathlib import Path

repo_root = Path(r"C:\Dev\repos\Public repos\ieee-aiot")
results_files = sorted(glob.glob(str(repo_root / "runs" / "*" / "*" / "results.csv")))

print(f"Found {len(results_files)} results.csv files:")
summary_rows = []

for f in results_files:
    rel = str(Path(f).relative_to(repo_root))
    df = pd.read_csv(f)
    # clean column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    num_epochs = len(df)
    last_row = df.iloc[-1]
    last_epoch = int(last_row.get("epoch", num_epochs))
    
    # best map50 epoch
    best_map50_idx = df["metrics/mAP50(B)"].idxmax() if "metrics/mAP50(B)" in df.columns else None
    best_row = df.iloc[best_map50_idx] if best_map50_idx is not None else None
    
    summary_rows.append({
        "run": rel.replace("\\results.csv", ""),
        "epochs_logged": num_epochs,
        "last_epoch": last_epoch,
        "final_train_box_loss": round(float(last_row.get("train/box_loss", -1)), 4),
        "final_train_cls_loss": round(float(last_row.get("train/cls_loss", -1)), 4),
        "final_train_dfl_loss": round(float(last_row.get("train/dfl_loss", -1)), 4),
        "final_val_box_loss": round(float(last_row.get("val/box_loss", -1)), 4),
        "final_val_cls_loss": round(float(last_row.get("val/cls_loss", -1)), 4),
        "final_val_dfl_loss": round(float(last_row.get("val/dfl_loss", -1)), 4),
        "final_val_p": round(float(last_row.get("metrics/precision(B)", -1)), 4),
        "final_val_r": round(float(last_row.get("metrics/recall(B)", -1)), 4),
        "final_val_map50": round(float(last_row.get("metrics/mAP50(B)", -1)), 4),
        "final_val_map50_95": round(float(last_row.get("metrics/mAP50-95(B)", -1)), 4),
        "best_val_epoch": int(best_row["epoch"]) if best_row is not None else None,
        "best_val_map50": round(float(best_row["metrics/mAP50(B)"]), 4) if best_row is not None else None,
        "best_val_map50_95": round(float(best_row["metrics/mAP50-95(B)"]), 4) if best_row is not None else None,
    })

res_df = pd.DataFrame(summary_rows)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
print(res_df[["run", "epochs_logged", "best_val_epoch", "best_val_map50", "final_val_map50", "final_train_cls_loss", "final_val_cls_loss"]])
