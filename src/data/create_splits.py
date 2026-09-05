"""
create_splits.py - Generate 80/10/10 random stratified splits and negative ratio configurations.

Ratios produced:
- 0%  (Positive-Only): 2,401 pos, 0 neg (Total: 2,401)
- 20% (Low-Negative): 2,401 pos, 600 neg (Total: 3,001)
- 40% (Moderate-Negative): 2,401 pos, 1,600 neg (Total: 4,001)
- ~81% (Natural Full Pool): 2,401 pos, 10,178 neg (Total: 12,579)

Evaluation sets:
- Test: 300 pos, 1,272 neg (Total: 1,572, 80.9% negative)
- Val:  300 pos, 1,272 neg (Total: 1,572, 80.9% negative)
"""

import os
import json
import random
import shutil
from pathlib import Path

SEED = 42

def load_master_coco(path):
    with open(path, "r") as f:
        return json.load(f)

def save_coco(coco_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(coco_dict, f, indent=2)

def save_yolo_txt(image_paths, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for p in image_paths:
            f.write(f"{p}\n")

def make_coco_subset(master_coco, image_ids):
    image_ids_set = set(image_ids)
    sub_images = [img for img in master_coco["images"] if img["id"] in image_ids_set]
    sub_annotations = [ann for ann in master_coco["annotations"] if ann["image_id"] in image_ids_set]
    return {
        "info": master_coco.get("info", {}),
        "licenses": master_coco.get("licenses", []),
        "categories": master_coco.get("categories", []),
        "images": sub_images,
        "annotations": sub_annotations
    }

def main():
    repo_root = Path(__file__).resolve().parents[2]
    annotations_path = repo_root / "data" / "annotations" / "RGB" / "annotations.json"
    coco_out_dir = repo_root / "data" / "processed" / "RGB" / "coco"
    dfine_out_dir = coco_out_dir / "dfine"
    eval_out_dir = coco_out_dir / "evaluation"
    yolo_out_dir = repo_root / "data" / "processed" / "RGB" / "yolo"

    print(f"Loading master COCO: {annotations_path}")
    master_coco = load_master_coco(annotations_path)

    images = master_coco["images"]
    annotations = master_coco["annotations"]

    print(f"Total images: {len(images)}, Total annotations: {len(annotations)}")

    # Map image_id to annotations
    img_to_anns = {img["id"]: [] for img in images}
    for ann in annotations:
        img_to_anns[ann["image_id"]].append(ann)

    # Separate positive and negative images
    pos_images = [img for img in images if len(img_to_anns[img["id"]]) > 0]
    neg_images = [img for img in images if len(img_to_anns[img["id"]]) == 0]

    print(f"Positive frames: {len(pos_images)}, Negative frames: {len(neg_images)}")

    # Group positive images by category for stratified sampling
    cat_to_pos_imgs = {}
    for img in pos_images:
        cat_id = img_to_anns[img["id"]][0]["category_id"]
        cat_to_pos_imgs.setdefault(cat_id, []).append(img)

    rng = random.Random(SEED)

    train_pos = []
    val_pos = []
    test_pos = []

    for cat_id, cat_imgs in sorted(cat_to_pos_imgs.items()):
        rng.shuffle(cat_imgs)
        n_test = round(len(cat_imgs) * 0.10)
        n_val = round(len(cat_imgs) * 0.10)
        
        test_chunk = cat_imgs[:n_test]
        val_chunk = cat_imgs[n_test:n_test + n_val]
        train_chunk = cat_imgs[n_test + n_val:]
        
        test_pos.extend(test_chunk)
        val_pos.extend(val_chunk)
        train_pos.extend(train_chunk)
        print(f"Category {cat_id} ({len(cat_imgs)}): train={len(train_chunk)}, val={len(test_chunk)}, test={len(val_chunk)}")

    print(f"Total stratified positives -> Train: {len(train_pos)}, Val: {len(val_pos)}, Test: {len(test_pos)}")
    assert len(train_pos) == 2401, f"Expected 2401 train pos, got {len(train_pos)}"
    assert len(val_pos) == 300, f"Expected 300 val pos, got {len(val_pos)}"
    assert len(test_pos) == 300, f"Expected 300 test pos, got {len(test_pos)}"

    # Partition negative images
    rng.shuffle(neg_images)
    n_neg_test = 1272
    n_neg_val = 1272
    test_neg = neg_images[:n_neg_test]
    val_neg = neg_images[n_neg_test:n_neg_test + n_neg_val]
    train_neg_pool = neg_images[n_neg_test + n_neg_val:]

    print(f"Negative partitioning -> Train pool: {len(train_neg_pool)}, Val: {len(val_neg)}, Test: {len(test_neg)}")
    assert len(test_neg) == 1272
    assert len(val_neg) == 1272
    assert len(train_neg_pool) == 10178

    # Build test and val full sets
    test_set = test_pos + test_neg
    val_set = val_pos + val_neg
    rng.shuffle(test_set)
    rng.shuffle(val_set)

    print(f"Final Test set: {len(test_set)} ({len(test_pos)} pos, {len(test_neg)} neg, {len(test_neg)/len(test_set)*100:.1f}%)")
    print(f"Final Val set: {len(val_set)} ({len(val_pos)} pos, {len(val_neg)} neg, {len(val_neg)/len(val_set)*100:.1f}%)")

    # Build the 4 training configurations
    # Subsample negative frames with fixed seed for nested consistency
    rng_train = random.Random(SEED)
    shuffled_train_neg = list(train_neg_pool)
    rng_train.shuffle(shuffled_train_neg)

    train_configs = {
        "train_00_pos_only": {
            "name": "train_00_pos_only",
            "pos": train_pos,
            "neg": [],
            "target_ratio": "0%"
        },
        "train_20_low_neg": {
            "name": "train_20_low_neg",
            "pos": train_pos,
            "neg": shuffled_train_neg[:600],
            "target_ratio": "20%"
        },
        "train_40_mod_neg": {
            "name": "train_40_mod_neg",
            "pos": train_pos,
            "neg": shuffled_train_neg[:1600],
            "target_ratio": "40%"
        },
        "train_81_nat_full": {
            "name": "train_81_nat_full",
            "pos": train_pos,
            "neg": shuffled_train_neg,
            "target_ratio": "~81%"
        }
    }

    # Helper to convert image objects to YOLO relative path format: ./../images/{file_name}
    def to_yolo_paths(img_list):
        paths = []
        for img in img_list:
            fn = img["file_name"]
            if fn.startswith("images/"):
                fn = fn[len("images/"):]
            paths.append(f"./../images/{fn}")
        return paths

    # 1. Save Test and Val YOLO manifests and COCO JSONs
    save_yolo_txt(to_yolo_paths(test_set), yolo_out_dir / "test.txt")
    save_yolo_txt(to_yolo_paths(val_set), yolo_out_dir / "val.txt")

    test_coco = make_coco_subset(master_coco, [img["id"] for img in test_set])
    val_coco = make_coco_subset(master_coco, [img["id"] for img in val_set])

    save_coco(test_coco, coco_out_dir / "instances_test.json")
    save_coco(val_coco, coco_out_dir / "instances_val.json")

    save_coco(test_coco, dfine_out_dir / "instances_test.json")
    save_coco(val_coco, dfine_out_dir / "instances_val.json")

    save_coco(test_coco, eval_out_dir / "instances_test.json")
    save_coco(val_coco, eval_out_dir / "instances_val.json")

    # 2. Save each training configuration
    stats = {}
    for cfg_key, cfg in train_configs.items():
        combined = cfg["pos"] + cfg["neg"]
        rng.shuffle(combined)
        
        n_pos = len(cfg["pos"])
        n_neg = len(cfg["neg"])
        total = len(combined)
        ratio_pct = (n_neg / total * 100) if total > 0 else 0.0

        print(f"Generated {cfg_key}: {total} frames ({n_pos} pos, {n_neg} neg, {ratio_pct:.2f}% neg)")
        stats[cfg_key] = {
            "total": total,
            "pos": n_pos,
            "neg": n_neg,
            "neg_ratio": f"{ratio_pct:.2f}%"
        }

        # Save YOLO txt
        save_yolo_txt(to_yolo_paths(combined), yolo_out_dir / f"{cfg_key}.txt")

        # Save COCO json
        cfg_coco = make_coco_subset(master_coco, [img["id"] for img in combined])
        save_coco(cfg_coco, coco_out_dir / f"instances_{cfg_key}.json")
        save_coco(cfg_coco, dfine_out_dir / f"instances_{cfg_key}.json")
        save_coco(cfg_coco, eval_out_dir / f"instances_{cfg_key}.json")

    # Also keep train.txt and instances_train.json pointing to natural full for default compatibility
    save_yolo_txt(to_yolo_paths(train_configs["train_81_nat_full"]["pos"] + train_configs["train_81_nat_full"]["neg"]), yolo_out_dir / "train.txt")
    save_coco(make_coco_subset(master_coco, [img["id"] for img in train_configs["train_81_nat_full"]["pos"] + train_configs["train_81_nat_full"]["neg"]]), coco_out_dir / "instances_train.json")
    save_coco(make_coco_subset(master_coco, [img["id"] for img in train_configs["train_81_nat_full"]["pos"] + train_configs["train_81_nat_full"]["neg"]]), dfine_out_dir / "instances_train.json")
    save_coco(make_coco_subset(master_coco, [img["id"] for img in train_configs["train_81_nat_full"]["pos"] + train_configs["train_81_nat_full"]["neg"]]), eval_out_dir / "instances_train.json")

    # Save summary stats
    summary_path = repo_root / "data" / "processed" / "RGB" / "split_stats.json"
    with open(summary_path, "w") as f:
        json.dump({
            "seed": SEED,
            "test": {"total": len(test_set), "pos": len(test_pos), "neg": len(test_neg), "neg_ratio": f"{len(test_neg)/len(test_set)*100:.2f}%"},
            "val": {"total": len(val_set), "pos": len(val_pos), "neg": len(val_neg), "neg_ratio": f"{len(val_neg)/len(val_set)*100:.2f}%"},
            "training_configs": stats
        }, f, indent=2)

    print(f"\nAll splits generated successfully. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
