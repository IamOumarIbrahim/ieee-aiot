"""
dfine_utils.py - Shared D-FINE Native Utilities for IEEE AIoT Benchmark.

Provides unified, native helpers for:
1. Locating D-FINE checkpoints (best_stg2.pth, best_stg1.pth, last.pth).
2. Constructing D-FINE training commands with unified AMP toggle.
3. Loading native D-FINE models in deployment or evaluation mode.
4. Evaluating checkpoints on COCO test splits (mAP@50, mAP@50:95, precision, recall).
5. Calculating false-positive rate per 1,000 negative test frames (exact 1,272 denominator).
6. Scoring candidate negative frames for hard-negative mining (RQ2).

Safety Note:
All heavy/torch/CUDA imports are deferred inside functions to ensure importing
this module remains strictly GPU-free and never initializes CUDA.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from safe_linear import apply_safe_linear_patch
    apply_safe_linear_patch()
except ImportError:
    pass

DEFAULT_DFINE_AMP = False


def get_repo_root() -> Path:
    """Returns absolute path to the repository root."""
    return Path(__file__).resolve().parents[2]


def get_dfine_root(dfine_dir: Optional[str] = None) -> Path:
    """Resolves and returns the path to the vendored DFINE directory."""
    repo_root = get_repo_root()
    if dfine_dir:
        p = Path(dfine_dir)
        if not p.is_absolute():
            p = repo_root / p
        return p.resolve()
    return (repo_root / "DFINE").resolve()


def patch_dfine_if_needed(dfine_dir: Optional[str] = None) -> Path:
    """Applies Windows compatibility patches to upstream DFINE if needed."""
    dfine_path = get_dfine_root(dfine_dir)
    repo_root = get_repo_root()
    if not (dfine_path / "train.py").exists():
        return dfine_path

    # 1. Install SafeLinear patch
    core_dir = dfine_path / "src" / "core"
    safe_linear_src = repo_root / "src" / "training" / "safe_linear.py"
    safe_linear_dst = core_dir / "safe_linear.py"
    if safe_linear_src.exists():
        if not safe_linear_dst.exists() or safe_linear_dst.read_text(encoding="utf-8") != safe_linear_src.read_text(encoding="utf-8"):
            import shutil
            shutil.copy2(safe_linear_src, safe_linear_dst)

    init_file = core_dir / "__init__.py"
    if init_file.exists():
        content = init_file.read_text(encoding="utf-8")
        if "apply_safe_linear_patch" not in content:
            with open(init_file, "a", encoding="utf-8") as f:
                f.write("\nfrom .safe_linear import apply_safe_linear_patch\napply_safe_linear_patch()\n")

    # 2. Patch device string normalization in _solver.py
    solver_file = dfine_path / "src" / "solver" / "_solver.py"
    if solver_file.exists():
        s_content = solver_file.read_text(encoding="utf-8")
        old_snippet = "device = torch.device(cfg.device)"
        new_snippet = 'dev_str = str(cfg.device).strip()\n            if dev_str.isdigit():\n                dev_str = f"cuda:{dev_str}"\n            device = torch.device(dev_str)'
        if old_snippet in s_content and new_snippet not in s_content:
            s_content = s_content.replace(old_snippet, new_snippet, 1)
            solver_file.write_text(s_content, encoding="utf-8")

    return dfine_path


def ensure_dfine_sys_path(dfine_dir: Optional[str] = None) -> Path:
    """Adds DFINE directory to sys.path if not already present, ensuring patches are applied."""
    dfine_path = patch_dfine_if_needed(dfine_dir)
    dfine_str = str(dfine_path)
    if dfine_str not in sys.path:
        sys.path.insert(0, dfine_str)
    return dfine_path


def find_dfine_checkpoint(
    search_dir: Path | str,
    preferred_names: Tuple[str, ...] = ("best_stg2.pth", "best_stg1.pth", "last.pth", "best.pt")
) -> Optional[Path]:
    """
    Finds the latest/best D-FINE checkpoint within search_dir or known fallback directories.
    """
    repo_root = get_repo_root()
    search_path = Path(search_dir)
    if not search_path.is_absolute():
        search_path = repo_root / search_path

    # Candidate directories to search
    dirs_to_check = [
        search_path,
        search_path / "weights",
        repo_root / "DFINE" / "output" / search_path.name,
        repo_root / "output" / search_path.name,
        repo_root / "runs" / "dfine_ratio_sweep" / search_path.name,
    ]

    for d in dirs_to_check:
        if d.exists() and d.is_dir():
            for name in preferred_names:
                ckpt = d / name
                if ckpt.exists() and ckpt.is_file():
                    return ckpt.resolve()

    return None


def build_dfine_train_cmd(
    dfine_train_py: Path | str,
    config_path: Path | str,
    use_amp: bool = False,
    seed: int = 42,
    output_dir: Optional[Path | str] = None,
    device: Optional[str] = None
) -> List[str]:
    """
    Constructs the canonical D-FINE training command adhering to the frozen protocol.
    Shared by both RQ1 (sweep) and RQ2 (curated) training invocations to prevent config drift.
    """
    cmd = [
        sys.executable,
        str(dfine_train_py),
        "-c", str(config_path),
        "--seed", str(seed)
    ]
    if use_amp:
        cmd.append("--use-amp")
    if output_dir is not None:
        cmd.extend(["--output-dir", str(output_dir)])
    if device is not None:
        dev_str = str(device).strip()
        if dev_str.isdigit():
            dev_str = f"cuda:{dev_str}"
        cmd.extend(["-d", dev_str])
    return cmd


def load_dfine_model(
    config_path: Path | str,
    checkpoint_path: Optional[Path | str] = None,
    device: str = "0",
    deploy: bool = True,
    dfine_dir: Optional[str] = None
):
    """
    Loads a native D-FINE model from config and optional checkpoint.
    Returns:
        (model, cfg) if deploy=True, or (model, postprocessor, cfg) if deploy=False.
    """
    ensure_dfine_sys_path(dfine_dir)

    import torch
    import torch.nn as nn
    from src.core import YAMLConfig

    config_path = str(Path(config_path).resolve())
    cfg = YAMLConfig(config_path)

    # Disable pretrained download when evaluating or tuning custom checkpoints
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"D-FINE checkpoint not found at: {ckpt_path}")

        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(checkpoint, dict):
            if "ema" in checkpoint and "module" in checkpoint["ema"]:
                state = checkpoint["ema"]["module"]
            elif "model" in checkpoint:
                state = checkpoint["model"]
            else:
                state = checkpoint
        else:
            state = checkpoint

        # Clean prefix if saved under DDP/DataParallel
        clean_state = {}
        for k, v in state.items():
            clean_k = k[7:] if k.startswith("module.") else k
            clean_state[clean_k] = v

        cfg.model.load_state_dict(clean_state, strict=False)

    torch_device = torch.device(f"cuda:{device}" if (torch.cuda.is_available() and device != "cpu") else "cpu")

    if deploy:
        class DFINEInferenceWrapper(nn.Module):
            def __init__(self, core_cfg):
                super().__init__()
                self.model = core_cfg.model.deploy()
                self.postprocessor = core_cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                labels, boxes, scores = self.postprocessor(outputs, orig_target_sizes)
                return labels, boxes, scores

        deploy_model = DFINEInferenceWrapper(cfg).to(torch_device)
        deploy_model.eval()
        return deploy_model, cfg
    else:
        cfg.model.to(torch_device).eval()
        cfg.postprocessor.to(torch_device).eval()
        return cfg.model, cfg.postprocessor, cfg


def evaluate_dfine_coco(
    config_path: Path | str,
    checkpoint_path: Path | str,
    test_ann_file: Optional[Path | str] = None,
    device: str = "0",
    batch_size: int = 8,
    dfine_dir: Optional[str] = None
) -> Tuple[float, float, float, float]:
    """
    Evaluates a D-FINE checkpoint on the held-out COCO test benchmark.
    Returns:
        (mAP@50, mAP@50:95, precision, recall) as floats.
    """
    repo_root = get_repo_root()
    ensure_dfine_sys_path(dfine_dir)

    import torch
    from src.core import YAMLConfig
    from src.data import get_coco_api_from_dataset
    from src.solver.validator import Validator, scale_boxes

    if test_ann_file is None:
        test_ann_file = repo_root / "data" / "processed" / "RGB" / "coco" / "dfine" / "instances_test.json"
    else:
        test_ann_file = Path(test_ann_file).resolve()

    if not test_ann_file.exists():
        raise FileNotFoundError(f"Test COCO annotation file not found: {test_ann_file}")

    # Override val_dataloader to evaluate against test set
    update_dict = {
        "val_dataloader": {
            "dataset": {
                "ann_file": str(test_ann_file)
            },
            "batch_size": batch_size,
            "total_batch_size": batch_size
        }
    }

    cfg = YAMLConfig(str(Path(config_path).resolve()), **update_dict)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    # Load checkpoint state
    ckpt_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        if "ema" in checkpoint and "module" in checkpoint["ema"]:
            state = checkpoint["ema"]["module"]
        elif "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    clean_state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    cfg.model.load_state_dict(clean_state, strict=False)

    torch_device = torch.device(f"cuda:{device}" if (torch.cuda.is_available() and device != "cpu") else "cpu")
    model = cfg.model.to(torch_device).eval()
    postprocessor = cfg.postprocessor.to(torch_device).eval()

    val_loader = cfg.val_dataloader
    base_ds = get_coco_api_from_dataset(val_loader.dataset)
    evaluator = cfg.evaluator

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []

    print(f"  [D-FINE EVAL] Evaluating {ckpt_path.name} on {test_ann_file.name} ({len(val_loader.dataset)} samples)...")

    with torch.no_grad():
        for samples, targets in val_loader:
            samples = samples.to(torch_device)
            targets = [{k: v.to(torch_device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            outputs = model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)

            res = {target["image_id"].item(): output for target, output in zip(targets, results)}
            evaluator.update(res)

            for idx, (target, result) in enumerate(zip(targets, results)):
                gt.append({
                    "boxes": scale_boxes(
                        target["boxes"],
                        (target["orig_size"][1], target["orig_size"][0]),
                        (samples[idx].shape[-1], samples[idx].shape[-2]),
                    ),
                    "labels": target["labels"],
                })
                labels = result["labels"]
                preds.append({"boxes": result["boxes"], "labels": labels, "scores": result["scores"]})

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    coco_stats = evaluator.coco_eval["bbox"].stats.tolist()
    map50_95 = float(coco_stats[0])
    map50 = float(coco_stats[1])

    metrics = Validator(gt, preds).compute_metrics()
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))

    print(f"  [D-FINE EVAL] Completed: mAP@50: {map50:.4f}, mAP@50:95: {map50_95:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    return map50, map50_95, precision, recall


def calculate_dfine_fp_per_1k(
    model_or_wrapper,
    test_manifest_path: Path | str,
    conf_thresh: float = 0.25,
    device: str = "0",
    batch_size: int = 32
) -> Tuple[float, int, int]:
    """
    Evaluates D-FINE false-positive predictions on background-only negative test frames.
    Uses the EXACT matching denominator convention as train_yolo_sweep.py:
        fp_per_1k = (fp_count / total_neg_frames) * 1000.0, where total_neg_frames == 1272.
    Returns:
        (fp_per_1k, total_fp, total_neg_frames)
    """
    import torch
    import torchvision.transforms as T
    from PIL import Image

    manifest_file = Path(test_manifest_path).resolve()
    manifest_dir = manifest_file.parent

    with open(manifest_file, "r") as f:
        all_test_paths = [l.strip() for l in f if l.strip()]

    # Load test COCO annotations to identify negative frames
    val_test_dir = manifest_dir.parent / "coco"
    with open(val_test_dir / "instances_test.json", "r") as f:
        test_coco = json.load(f)

    pos_img_ids = {ann["image_id"] for ann in test_coco["annotations"]}
    neg_img_filenames = {img["file_name"] for img in test_coco["images"] if img["id"] not in pos_img_ids}

    neg_test_paths = []
    for p in all_test_paths:
        fn = p.replace("./../", "")
        if fn in neg_img_filenames or fn.replace("images/", "") in {f.replace("images/", "") for f in neg_img_filenames}:
            resolved_p = (manifest_dir / p).resolve()
            neg_test_paths.append(str(resolved_p))

    total_neg_frames = len(neg_test_paths)
    assert total_neg_frames == 1272, f"Expected 1272 negative test frames, got {total_neg_frames}"

    if total_neg_frames == 0:
        return 0.0, 0, 0

    torch_device = torch.device(f"cuda:{device}" if (torch.cuda.is_available() and device != "cpu") else "cpu")
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    fp_count = 0
    with torch.no_grad():
        for i in range(0, total_neg_frames, batch_size):
            batch_fps = neg_test_paths[i : i + batch_size]
            pil_imgs = [Image.open(p).convert("RGB") for p in batch_fps]
            orig_sizes = torch.tensor([[im.width, im.height] for im in pil_imgs], dtype=torch.float32).to(torch_device)
            tensors = torch.stack([transforms(im) for im in pil_imgs], dim=0).to(torch_device)

            labels, boxes, scores = model_or_wrapper(tensors, orig_sizes)
            mask = scores >= conf_thresh
            fp_count += int(mask.sum().item())

            del pil_imgs, orig_sizes, tensors, labels, boxes, scores

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fp_per_1k = (fp_count / total_neg_frames) * 1000.0
    return fp_per_1k, fp_count, total_neg_frames


def score_candidates_dfine(
    model_or_wrapper,
    candidate_paths: List[Path],
    tau: float = 0.25,
    batch_size: int = 32,
    device: str = "0"
) -> List[Tuple[int, float, int]]:
    """
    Scores candidate training negative frames for false-positive detections >= tau.
    Returns:
        List of (global_idx, max_conf, num_boxes) sorted by confidence.
    """
    import torch
    import torchvision.transforms as T
    from PIL import Image

    torch_device = torch.device(f"cuda:{device}" if (torch.cuda.is_available() and device != "cpu") else "cpu")
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    mined_scores: List[Tuple[int, float, int]] = []
    total_candidates = len(candidate_paths)

    with torch.no_grad():
        for i in range(0, total_candidates, batch_size):
            batch_paths = candidate_paths[i : i + batch_size]
            pil_imgs = [Image.open(str(p)).convert("RGB") for p in batch_paths]
            orig_sizes = torch.tensor([[im.width, im.height] for im in pil_imgs], dtype=torch.float32).to(torch_device)
            tensors = torch.stack([transforms(im) for im in pil_imgs], dim=0).to(torch_device)

            labels, boxes, scores = model_or_wrapper(tensors, orig_sizes)

            for b_idx in range(len(batch_paths)):
                img_scores = scores[b_idx]
                valid_mask = img_scores >= tau
                num_boxes = int(valid_mask.sum().item())
                if num_boxes > 0:
                    max_conf = float(img_scores[valid_mask].max().item())
                    global_idx = i + b_idx
                    mined_scores.append((global_idx, max_conf, num_boxes))

            del pil_imgs, orig_sizes, tensors, labels, boxes, scores

            if (i // batch_size) % 50 == 0 and i > 0:
                print(f"  Processed {min(total_candidates, i + batch_size)}/{total_candidates} candidates... (Found {len(mined_scores)} hard negatives so far)")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mined_scores.sort(key=lambda x: x[1], reverse=True)
    return mined_scores
