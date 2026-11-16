"""
Evaluation Pipeline for 3D Brain Tumor Segmentation.
Performs full-volume sliding window inference and computes per-class Dice scores (TC, WT, ET).
"""

import os
import json
import argparse
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.model import build_swinunetr_model
from src.dataset import (
    build_case_dicts,
    BratsNiftiDataset,
    get_val_transforms,
)
from src.losses import build_loss_and_metrics, post_process_logits
from src.utils import (
    set_determinism,
    run_sliding_window_inference,
    reconstruct_integer_label_map,
    plot_slice_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3D SwinUNETR on BraTS Cohort")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        help="Path to folder containing BraTS patient cases"
    )
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Directory to save evaluation artifacts")
    parser.add_argument("--val_cases", type=int, default=20, help="Number of held-out validation cases")
    parser.add_argument("--feature_size", type=int, default=48, help="SwinUNETR feature dimension")
    parser.add_argument("--sw_batch_size", type=int, default=4, help="Sliding window inference batch size")
    parser.add_argument("--seed", type=int, default=101, help="Random determinism seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_determinism(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("  EVALUATION ENGINE: 3D SwinUNETR Full-Volume BraTS Evaluation")
    print("=" * 75)
    print(f"Device        : {device}")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Dataset Path  : {args.data_root}")
    print("=" * 75)

    # Ingest cases
    all_cases = build_case_dicts(args.data_root)
    if not all_cases:
        raise FileNotFoundError(f"No valid cases found in {args.data_root}")

    val_cases = all_cases[-args.val_cases:]
    print(f"Loaded {len(val_cases)} held-out patient cases for evaluation.")

    val_ds = BratsNiftiDataset(val_cases, transform=get_val_transforms())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    # Initialize model
    model = build_swinunetr_model(
        in_channels=4,
        out_channels=3,
        feature_size=args.feature_size,
        use_checkpoint=False,
        device=device,
        multi_gpu=False
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    # Handle DataParallel prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    print("Weights loaded successfully.")

    _, dice_metric, dice_metric_batch = build_loss_and_metrics()
    roi_size = (96, 96, 96)

    print("\nRunning sliding-window inference across evaluation cohort ...")
    dice_metric.reset()
    dice_metric_batch.reset()

    case_metrics = []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(val_loader, desc="Evaluating")):
            x, y = x.to(device), y.to(device)
            logits = run_sliding_window_inference(
                model=model,
                inputs=x,
                roi_size=roi_size,
                sw_batch_size=args.sw_batch_size,
                overlap=0.5,
                mode="gaussian"
            )
            preds = post_process_logits(logits)

            dice_metric(y_pred=preds, y=y)
            dice_metric_batch(y_pred=preds, y=y)

            # Generate visualization for first 3 cases
            if i < 3:
                img_np = x[0, 0].cpu().numpy()  # FLAIR channel
                gt_binary = y[0].cpu().numpy()   # (3, D, H, W)
                pred_binary = preds[0].cpu().numpy()

                gt_integer = reconstruct_integer_label_map(gt_binary)
                pred_integer = reconstruct_integer_label_map(pred_binary)

                mid_slice = img_np.shape[0] // 2
                viz_path = os.path.join(args.output_dir, f"case_{i+1}_slice_{mid_slice}.png")
                plot_slice_comparison(
                    image_slice=img_np[mid_slice],
                    ground_truth_slice=gt_integer[mid_slice],
                    predicted_slice=pred_integer[mid_slice],
                    save_path=viz_path,
                    slice_idx=mid_slice,
                    title_suffix=f"(Case {i+1})"
                )

    mean_dice = float(dice_metric.aggregate().item())
    per_class = dice_metric_batch.aggregate().cpu().numpy()

    results = {
        "mean_dice": mean_dice,
        "tumor_core_dice_tc": float(per_class[0]),
        "whole_tumor_dice_wt": float(per_class[1]),
        "enhancing_tumor_dice_et": float(per_class[2]),
        "num_evaluated_cases": len(val_cases),
    }

    print("\n" + "=" * 65)
    print("  FINAL EVALUATION PERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"  Mean Dice Score            : {mean_dice:.4f} ({mean_dice*100:.2f}%)")
    print(f"  Whole Tumor (WT) Dice      : {per_class[1]:.4f} ({per_class[1]*100:.2f}%)")
    print(f"  Enhancing Tumor (ET) Dice  : {per_class[2]:.4f} ({per_class[2]*100:.2f}%)")
    print(f"  Tumor Core (TC) Dice       : {per_class[0]:.4f} ({per_class[0]*100:.2f}%)")
    print("=" * 65)

    res_json_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Results exported to -> {res_json_path}")


if __name__ == "__main__":
    main()
