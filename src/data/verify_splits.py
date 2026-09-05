"""
verify_splits.py - Automated verification for generated dataset splits and configurations.
"""

import os
import json
import yaml
from pathlib import Path
from ultralytics.data.utils import check_det_dataset

def main():
    repo_root = Path(__file__).resolve().parents[2]
    yolo_dir = repo_root / "data" / "processed" / "RGB" / "yolo"
    coco_dir = repo_root / "data" / "processed" / "RGB" / "coco"
    configs_dir = repo_root / "configs" / "yolo"

    print("=== 1. VERIFYING YOLO MANIFESTS ===")
    split_files = {
        "val": (yolo_dir / "val.txt", 1572),
        "test": (yolo_dir / "test.txt", 1572),
        "train_00_pos_only": (yolo_dir / "train_00_pos_only.txt", 2401),
        "train_20_low_neg": (yolo_dir / "train_20_low_neg.txt", 3001),
        "train_40_mod_neg": (yolo_dir / "train_40_mod_neg.txt", 4001),
        "train_60_high_neg": (yolo_dir / "train_60_high_neg.txt", 6003),
        "train_80_max_neg": (yolo_dir / "train_80_max_neg.txt", 12005),
    }

    manifest_paths = {}
    for name, (path, expected_count) in split_files.items():
        assert path.exists(), f"Missing manifest: {path}"
        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == expected_count, f"{name}: expected {expected_count}, got {len(lines)}"
        
        # Verify that all image paths in the manifest exist on disk
        for l in lines:
            # l is like ./../images/...
            resolved = (yolo_dir / l).resolve()
            assert resolved.exists(), f"Missing file on disk: {resolved}"
            
        manifest_paths[name] = set(lines)
        print(f"  [OK] {name}: {len(lines)} paths (all exist on disk)")

    print("\n=== 2. ZERO DATA LEAKAGE CHECKS ===")
    train_full = manifest_paths["train_80_max_neg"]
    val_set = manifest_paths["val"]
    test_set = manifest_paths["test"]

    assert len(train_full.intersection(val_set)) == 0, "DATA LEAKAGE: Train & Val overlap!"
    assert len(train_full.intersection(test_set)) == 0, "DATA LEAKAGE: Train & Test overlap!"
    assert len(val_set.intersection(test_set)) == 0, "DATA LEAKAGE: Val & Test overlap!"
    print("  [OK] Train ∩ Val = ∅")
    print("  [OK] Train ∩ Test = ∅")
    print("  [OK] Val ∩ Test = ∅")

    print("\n=== 3. NESTED NEGATIVE SUBSET INTEGRITY ===")
    # Positives must be identical across all training splits
    t0 = manifest_paths["train_00_pos_only"]
    t20 = manifest_paths["train_20_low_neg"]
    t40 = manifest_paths["train_40_mod_neg"]
    t60 = manifest_paths["train_60_high_neg"]
    t80 = manifest_paths["train_80_max_neg"]

    assert t0.issubset(t20), "Train 0% is not a subset of Train 20%!"
    assert t20.issubset(t40), "Train 20% is not a subset of Train 40%!"
    assert t40.issubset(t60), "Train 40% is not a subset of Train 60%!"
    assert t60.issubset(t80), "Train 60% is not a subset of Train 80%!"
    print("  [OK] Train(0%) ⊂ Train(20%) ⊂ Train(40%) ⊂ Train(60%) ⊂ Train(80%)")

    print("\n=== 4. VERIFYING COCO JSONS ===")
    coco_files = {
        "val": (coco_dir / "instances_val.json", 1572, 300),
        "test": (coco_dir / "instances_test.json", 1572, 300),
        "train_00": (coco_dir / "instances_train_00_pos_only.json", 2401, 2401),
        "train_20": (coco_dir / "instances_train_20_low_neg.json", 3001, 2401),
        "train_40": (coco_dir / "instances_train_40_mod_neg.json", 4001, 2401),
        "train_60": (coco_dir / "instances_train_60_high_neg.json", 6003, 2401),
        "train_80": (coco_dir / "instances_train_80_max_neg.json", 12005, 2401),
    }

    for name, (path, exp_imgs, exp_anns) in coco_files.items():
        assert path.exists(), f"Missing COCO file: {path}"
        with open(path, "r") as f:
            d = json.load(f)
        imgs = len(d["images"])
        anns = len(d["annotations"])
        assert imgs == exp_imgs, f"{name}: expected {exp_imgs} imgs, got {imgs}"
        assert anns == exp_anns, f"{name}: expected {exp_anns} anns, got {anns}"
        print(f"  [OK] {name}: {imgs} images, {anns} annotations")

    print("\n=== 5. ULTRALYTICS YAML CONFIG VERIFICATION ===")
    for yml in ["yolo_00_pos_only.yaml", "yolo_20_low_neg.yaml", "yolo_40_mod_neg.yaml", "yolo_60_high_neg.yaml", "yolo_80_max_neg.yaml"]:
        yml_path = configs_dir / yml
        assert yml_path.exists(), f"Missing config: {yml_path}"
        # Validate that ultralytics can parse and find datasets
        data_dict = check_det_dataset(str(yml_path))
        assert "train" in data_dict and "val" in data_dict and "test" in data_dict
        print(f"  [OK] {yml} successfully parsed by Ultralytics")

    print("\n=== 6. D-FINE YAML CONFIG VERIFICATION ===")
    def deep_merge(base, update):
        merged = dict(base)
        for k, v in update.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = deep_merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    def load_yaml_with_includes(file_path):
        file_path = Path(file_path).resolve()
        with open(file_path, "r") as f:
            data = yaml.safe_load(f) or {}
        merged = {}
        if "__include__" in data:
            includes = data["__include__"]
            if isinstance(includes, str):
                includes = [includes]
            for inc in includes:
                inc_path = (file_path.parent / inc).resolve()
                base_data = load_yaml_with_includes(inc_path)
                merged = deep_merge(merged, base_data)
        merged = deep_merge(merged, data)
        return merged

    dfine_configs_dir = repo_root / "configs" / "dfine"
    dfine_ymls = [
        "dfine_hgnetv2_n_coco.yml",
        "dfine_00_pos_only.yml",
        "dfine_20_low_neg.yml",
        "dfine_40_mod_neg.yml",
        "dfine_60_high_neg.yml",
        "dfine_80_max_neg.yml"
    ]
    for yml in dfine_ymls:
        yml_path = dfine_configs_dir / yml
        assert yml_path.exists(), f"Missing D-FINE config: {yml_path}"
        d = load_yaml_with_includes(yml_path)
        assert d.get("num_classes") == 4, f"{yml}: expected 4 classes, got {d.get('num_classes')}"
        for loader_key in ["train_dataloader", "val_dataloader", "test_dataloader"]:
            assert loader_key in d, f"{yml}: missing {loader_key}"
            assert "dataset" in d[loader_key], f"{yml}: missing dataset in {loader_key}"
            ann_rel = d[loader_key]["dataset"]["ann_file"]
            img_rel = d[loader_key]["dataset"]["img_folder"]
            ann_path = repo_root / ann_rel
            img_path = repo_root / img_rel
            assert ann_path.exists(), f"{yml} ({loader_key}): missing annotation file: {ann_path}"
            assert img_path.exists(), f"{yml} ({loader_key}): missing image folder: {img_path}"
    print("\n=== 7. CANONICAL TRAIN NEGATIVE POOL VERIFICATION (RQ2) ===")
    pool_file = repo_root / "data" / "processed" / "RGB" / "train_neg_pool_ids.json"
    assert pool_file.exists(), f"Missing train_neg_pool_ids.json: {pool_file}"
    with open(pool_file, "r") as f:
        pool_data = json.load(f)
    pool_ids = set(pool_data["train_neg_pool_ids"])
    assert len(pool_ids) == 10178, f"Expected 10178 IDs, got {len(pool_ids)}"
    
    with open(coco_dir / "instances_val.json", "r") as f:
        val_img_ids = {img["id"] for img in json.load(f)["images"]}
    with open(coco_dir / "instances_test.json", "r") as f:
        test_img_ids = {img["id"] for img in json.load(f)["images"]}
    
    assert len(pool_ids & val_img_ids) == 0, f"DATA LEAKAGE: train_neg_pool intersects with val set ({len(pool_ids & val_img_ids)} frames)!"
    assert len(pool_ids & test_img_ids) == 0, f"DATA LEAKAGE: train_neg_pool intersects with test set ({len(pool_ids & test_img_ids)} frames)!"
    print(f"  [OK] train_neg_pool_ids.json verified: {len(pool_ids)} candidates, 0 val/test leakage")

    print("\nALL VERIFICATIONS PASSED! Directory and configurations are 100% ready for training.")

if __name__ == "__main__":
    main()
