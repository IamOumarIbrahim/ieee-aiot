"""
mine_hard_negatives.py - Hard-Negative Mining Pipeline for RQ2 (IEEE AIoT Benchmark).

Implements the frozen RQ2 experimental protocol:
1. Reconstructs the exact 10,178 training negative candidate pool (seed=42).
2. Runs batched inference using the 0% negative baseline detector checkpoint.
3. Filters frames yielding false-positive detections >= tau (default 0.25).
4. Matches the exact sample count of the best RQ1 ratio (with deterministic random backfill if needed).
5. Generates paired COCO JSONs, YOLO manifests, and dataset configuration YAMLs.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
import torch

SEED = 42

def parse_args():
    parser = argparse.ArgumentParser(description="RQ2 Hard-Negative Mining Pipeline")
    parser.add_argument("--weights", type=str, required=True, help="Path to baseline detector checkpoint (trained on 0% negative baseline)")
    parser.add_argument("--target-count", type=int, default=1600, help="Target number of negative frames (matches best RQ1 ratio, e.g. 1600 for 40%)")
    parser.add_argument("--tau", type=float, default=0.25, help="Confidence threshold for false-positive detection (paper default: 0.25)")
    parser.add_argument("--tag", type=str, default="yolo11n_best_curated", help="Output tag for generated dataset and configs")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--dry-run", action="store_true", help="Inspect candidates and simulate curation without running full inference")
    return parser.parse_args()

def load_master_negatives(repo_root):
    annotations_path = repo_root / "data" / "annotations" / "RGB" / "annotations.json"
    with open(annotations_path, "r") as f:
        master_coco = json.load(f)

    images = master_coco["images"]
    annotations = master_coco["annotations"]

    img_to_anns = {img["id"]: [] for img in images}
    for ann in annotations:
        img_to_anns[ann["image_id"]].append(ann)

    pos_images = [img for img in images if len(img_to_anns[img["id"]]) > 0]
    neg_images = [img for img in images if len(img_to_anns[img["id"]]) == 0]

    # Replicate create_splits.py deterministic partitioning
    rng = random.Random(SEED)
    rng.shuffle(neg_images)
    n_neg_test = 1272
    n_neg_val = 1272
    train_neg_pool = neg_images[n_neg_test + n_neg_val:]
    assert len(train_neg_pool) == 10178, f"Expected 10178 train negs, got {len(train_neg_pool)}"

    # Also extract train_pos (2,401 frames)
    cat_to_pos_imgs = {}
    for img in pos_images:
        cat_id = img_to_anns[img["id"]][0]["category_id"]
        cat_to_pos_imgs.setdefault(cat_id, []).append(img)

    rng_pos = random.Random(SEED)
    train_pos = []
    for cat_id, cat_imgs in sorted(cat_to_pos_imgs.items()):
        rng_pos.shuffle(cat_imgs)
        n_test = round(len(cat_imgs) * 0.10)
        n_val = round(len(cat_imgs) * 0.10)
        train_chunk = cat_imgs[n_test + n_val:]
        train_pos.extend(train_chunk)

    assert len(train_pos) == 2401, f"Expected 2401 train pos, got {len(train_pos)}"
    return master_coco, train_pos, train_neg_pool

def to_yolo_paths(img_list):
    paths = []
    for img in img_list:
        fn = img["file_name"]
        if fn.startswith("images/"):
            fn = fn[len("images/"):]
        paths.append(f"./../images/{fn}")
    return paths

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
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    print("============================================================")
    print("  RQ2 Hard-Negative Mining Pipeline")
    print(f"  Detector Checkpoint: {args.weights}")
    print(f"  Confidence Threshold (tau): {args.tau}")
    print(f"  Target Negative Count: {args.target_count}")
    print(f"  Output Tag: {args.tag}")
    print("============================================================\n")

    master_coco, train_pos, train_neg_pool = load_master_negatives(repo_root)
    print(f"Reconstructed Candidate Pool: {len(train_neg_pool)} background frames (fixed positive core: {len(train_pos)} frames)")

    if args.dry_run:
        print("[DRY-RUN] Simulating mining with deterministic pseudo-curation...")
        rng_sim = random.Random(SEED)
        simulated_mined = train_neg_pool[:args.target_count]
        curated_negatives = simulated_mined
        hard_mined_count = len(simulated_mined)
        backfill_count = 0
    else:
        from ultralytics import YOLO
        print(f"Loading detector weights: {args.weights}...")
        model = YOLO(args.weights)

        img_dir = repo_root / "data" / "processed" / "RGB"
        candidate_paths = [(img_dir / img["file_name"]).resolve() for img in train_neg_pool]

        print(f"Scoring {len(candidate_paths)} candidate negatives (batch_size={args.batch_size}, conf={args.tau})...")
        mined_scores = []
        for i in range(0, len(candidate_paths), args.batch_size):
            batch_paths = [str(p) for p in candidate_paths[i:i + args.batch_size]]
            results = model.predict(batch_paths, conf=args.tau, device=args.device, verbose=False)
            for idx_in_batch, r in enumerate(results):
                global_idx = i + idx_in_batch
                if len(r.boxes) > 0:
                    max_conf = float(r.boxes.conf.max().item())
                    mined_scores.append((global_idx, max_conf, len(r.boxes)))
            del results
            if (i // args.batch_size) % 50 == 0 and i > 0:
                print(f"  Processed {min(len(candidate_paths), i + args.batch_size)}/{len(candidate_paths)} images... (Found {len(mined_scores)} hard negatives so far)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Rank by maximum false-positive confidence
        mined_scores.sort(key=lambda x: x[1], reverse=True)
        hard_mined_indices = [x[0] for x in mined_scores]
        hard_mined_count = len(hard_mined_indices)
        print(f"\nMining completed! Total frames with false positives >= {args.tau}: {hard_mined_count}")

        mined_neg_imgs = [train_neg_pool[idx] for idx in hard_mined_indices]

        # Match exact target sample count
        if hard_mined_count >= args.target_count:
            curated_negatives = mined_neg_imgs[:args.target_count]
            backfill_count = 0
            print(f"Selected top {args.target_count} highest-confidence hard negatives (no backfill required).")
        else:
            curated_negatives = list(mined_neg_imgs)
            mined_set = set(hard_mined_indices)
            remaining_negatives = [img for idx, img in enumerate(train_neg_pool) if idx not in mined_set]
            rng_backfill = random.Random(SEED)
            rng_backfill.shuffle(remaining_negatives)
            needed = args.target_count - hard_mined_count
            backfill_imgs = remaining_negatives[:needed]
            curated_negatives.extend(backfill_imgs)
            backfill_count = len(backfill_imgs)
            print(f"Retained all {hard_mined_count} mined negatives + backfilled {backfill_count} random negatives to reach {args.target_count} total.")

    # Build curated training dataset
    curated_train_imgs = train_pos + curated_negatives
    rng_curated = random.Random(SEED)
    rng_curated.shuffle(curated_train_imgs)

    print(f"\nFinal Curated Training Set: {len(curated_train_imgs)} frames ({len(train_pos)} pos, {len(curated_negatives)} neg)")

    # 1. Save YOLO Manifest
    yolo_manifest_dir = repo_root / "data" / "processed" / "RGB" / "yolo"
    yolo_manifest_path = yolo_manifest_dir / f"train_curated_{args.tag}.txt"
    yolo_paths = to_yolo_paths(curated_train_imgs)
    with open(yolo_manifest_path, "w") as f:
        for p in yolo_paths:
            f.write(f"{p}\n")
    print(f"  [OK] YOLO manifest written: {yolo_manifest_path}")

    # 2. Save COCO JSON
    coco_dir = repo_root / "data" / "processed" / "RGB" / "coco"
    coco_curated_path = coco_dir / f"instances_train_curated_{args.tag}.json"
    curated_coco = make_coco_subset(master_coco, [img["id"] for img in curated_train_imgs])
    with open(coco_curated_path, "w") as f:
        json.dump(curated_coco, f, indent=2)
    print(f"  [OK] COCO JSON written: {coco_curated_path}")

    # Also save in dfine coco directory
    dfine_coco_path = coco_dir / "dfine" / f"instances_train_curated_{args.tag}.json"
    with open(dfine_coco_path, "w") as f:
        json.dump(curated_coco, f, indent=2)
    print(f"  [OK] D-FINE COCO JSON written: {dfine_coco_path}")

    # 3. Save Ultralytics YAML Config
    yolo_cfg_path = repo_root / "configs" / "yolo" / f"yolo_curated_{args.tag}.yaml"
    yolo_cfg_content = f"""# Ultralytics YOLO RQ2 Curated Hard-Negative Configuration
path: data/processed/RGB
train: yolo/train_curated_{args.tag}.txt
val: yolo/val.txt
test: yolo/test.txt

names:
  0: yawning
  1: hand_over_mouth
  2: drinking
  3: phone_use
"""
    with open(yolo_cfg_path, "w") as f:
        f.write(yolo_cfg_content)
    print(f"  [OK] YOLO dataset config written: {yolo_cfg_path}")

    # 4. Save D-FINE YAML Config
    dfine_cfg_path = repo_root / "configs" / "dfine" / f"dfine_curated_{args.tag}.yml"
    dfine_cfg_content = f"""# D-FINE-N RQ2 Curated Hard-Negative Configuration
__include__: [
  './dfine_hgnetv2_n_coco.yml'
]

output_dir: ./output/dfine_curated_{args.tag}

train_dataloader:
  dataset:
    ann_file: data/processed/RGB/coco/dfine/instances_train_curated_{args.tag}.json
"""
    with open(dfine_cfg_path, "w") as f:
        f.write(dfine_cfg_content)
    print(f"  [OK] D-FINE dataset config written: {dfine_cfg_path}")

    # 5. Persist Curation Audit Statistics
    stats_dir = repo_root / "runs" / "curation_stats"
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = stats_dir / f"curation_{args.tag}.json"
    curation_stats = {
        "tag": args.tag,
        "detector_weights": args.weights,
        "tau": args.tau,
        "candidate_pool_size": len(train_neg_pool),
        "hard_mined_detected": hard_mined_count,
        "target_count": args.target_count,
        "backfill_count": backfill_count,
        "total_curated_negatives": len(curated_negatives),
        "total_train_frames": len(curated_train_imgs),
        "yolo_config": str(yolo_cfg_path.relative_to(repo_root)),
        "dfine_config": str(dfine_cfg_path.relative_to(repo_root)),
    }
    with open(stats_file, "w") as f:
        json.dump(curation_stats, f, indent=2)
    print(f"  [OK] Curation statistics written: {stats_file}")
    print("\nHard-negative dataset creation completed successfully.")

if __name__ == "__main__":
    main()
