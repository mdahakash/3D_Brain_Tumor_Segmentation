"""
Training Pipeline for 3D SwinUNETR Multimodal Brain Tumor Segmentation on BraTS.
Features:
  - MONAI SwinUNETR Architecture with Patch-based Random Spatial Cropping (96^3)
  - Full-Volume Sliding Window Inference Validation
  - Multi-Label Sub-Region Dice Tracking (TC, WT, ET)
  - Automatic Checkpoint Selection Monitored on Mean Validation Dice
  - Training History and Metric Visualization Export
"""

import os
import json
import time
import argparse
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.model import build_swinunetr_model, get_model_parameters
from src.dataset import (
    build_case_dicts,
    BratsNiftiDataset,
    get_train_transforms,
    get_val_transforms,
)
from src.losses import build_loss_and_metrics, post_process_logits
from src.utils import (
    set_determinism,
    run_sliding_window_inference,
    plot_training_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D SwinUNETR on BraTS Dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        help="Path to folder containing BraTS training patient cases"
    )
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save checkpoints and logs")
    parser.add_argument("--config", type=str, default="configs/default_config.json", help="Path to config JSON")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Initial AdamW learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--val_interval", type=int, default=None, help="Validation frequency (in epochs)")
    parser.add_argument("--val_cases", type=int, default=None, help="Number of cases to hold out for validation")
    parser.add_argument("--seed", type=int, default=101, help="Random determinism seed")
    parser.add_argument("--feature_size", type=int, default=48, help="SwinUNETR feature dimension")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker threads")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def validate_full_volume(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dice_metric,
    dice_metric_batch,
    roi_size=(96, 96, 96),
    sw_batch_size: int = 4
) -> Tuple[float, np.ndarray]:
    """Runs sliding-window inference over full 3D validation volumes."""
    model.eval()
    dice_metric.reset()
    dice_metric_batch.reset()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = run_sliding_window_inference(
                model=model,
                inputs=x,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=0.5,
                mode="gaussian"
            )
            preds = post_process_logits(logits)
            dice_metric(y_pred=preds, y=y)
            dice_metric_batch(y_pred=preds, y=y)

    mean_dice = float(dice_metric.aggregate().item())
    per_class_dice = dice_metric_batch.aggregate().cpu().numpy()
    return mean_dice, per_class_dice


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = args.seed or cfg.get("seed", 101)
    set_determinism(seed)

    epochs = args.epochs or cfg.get("epochs", 30)
    lr = args.lr or cfg.get("learning_rate", 1e-4)
    weight_decay = cfg.get("weight_decay", 1e-5)
    val_interval = args.val_interval or cfg.get("val_interval", 5)
    val_holdout = args.val_cases or cfg.get("val_holdout_cases", 20)
    feature_size = args.feature_size or cfg.get("feature_size", 48)
    roi_size = tuple(cfg.get("roi_size", [96, 96, 96]))

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    train_batch_size = args.batch_size or (1 * num_devices)
    sw_batch_size = cfg.get("sw_batch_size", 4) * num_devices

    print("=" * 80)
    print("  TRAINING PIPELINE: 3D SwinUNETR Multi-Modal Brain Tumor Segmentation")
    print("=" * 80)
    print(f"Device               : {device} (Total GPUs: {num_devices})")
    print(f"Dataset Path         : {args.data_root}")
    print(f"Output Directory     : {args.output_dir}")
    print(f"Input / Target Chans : 4 (FLAIR, T1, T1ce, T2) -> 3 (TC, WT, ET)")
    print(f"Patch ROI Size       : {roi_size}")
    print(f"Batch Size (Train/SW): {train_batch_size} / {sw_batch_size}")
    print(f"Optimization         : AdamW (lr={lr}, weight_decay={weight_decay})")
    print(f"Total Epochs         : {epochs} (Val Interval: every {val_interval} eps)")
    print("=" * 80)

    # Ingest cases
    all_cases = build_case_dicts(args.data_root)
    if not all_cases:
        raise FileNotFoundError(f"No valid BraTS cases found in {args.data_root}. Verify dataset directory.")

    train_cases = all_cases[:-val_holdout]
    val_cases = all_cases[-val_holdout:]
    print(f"Dataset Partitioning -> Train Cases: {len(train_cases)} | Validation Cases: {len(val_cases)}")

    train_ds = BratsNiftiDataset(train_cases, transform=get_train_transforms(roi_size))
    val_ds = BratsNiftiDataset(val_cases, transform=get_val_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=torch.cuda.is_available()
    )

    # Initialize SwinUNETR
    model = build_swinunetr_model(
        in_channels=4,
        out_channels=3,
        feature_size=feature_size,
        use_checkpoint=True,
        device=device,
        multi_gpu=True
    )

    tot_params, train_params = get_model_parameters(model)
    print(f"SwinUNETR Initialized -> Total Parameters: {tot_params:,} ({tot_params/1e6:.1f}M)")

    # Loss and Metrics
    loss_fn, dice_metric, dice_metric_batch = build_loss_and_metrics()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_dice = -1.0
    best_epoch = 0
    best_path = os.path.join(args.output_dir, cfg.get("checkpoint_name", "brats_swinunetr_best.pt"))
    history = []

    print("-" * 80)
    print("Beginning Training Loop ...")
    print("-" * 80)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for step, (x, y) in enumerate(pbar, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{epoch_loss / step:.4f}")

        epoch_loss /= max(1, len(train_loader))
        row = {"epoch": epoch, "loss": float(epoch_loss)}

        if epoch % val_interval == 0 or epoch == epochs:
            mean_dice, per_class = validate_full_volume(
                model=model,
                loader=val_loader,
                device=device,
                dice_metric=dice_metric,
                dice_metric_batch=dice_metric_batch,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size
            )
            row.update({
                "val_dice": float(mean_dice),
                "val_dice_tc": float(per_class[0]),
                "val_dice_wt": float(per_class[1]),
                "val_dice_et": float(per_class[2]),
            })

            is_best = mean_dice > best_dice
            if is_best:
                best_dice = mean_dice
                best_epoch = epoch
                state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": state,
                    "mean_dice": best_dice,
                    "per_class_dice": per_class.tolist(),
                    "feature_size": feature_size,
                    "roi_size": roi_size,
                }, best_path)

            mark = " *BEST*" if is_best else ""
            print(
                f"Epoch {epoch:3d}/{epochs:3d} | Train Loss: {epoch_loss:.4f} | "
                f"Val Mean Dice: {mean_dice:.4f} [TC: {per_class[0]:.4f}, WT: {per_class[1]:.4f}, ET: {per_class[2]:.4f}]{mark}"
            )
        else:
            print(f"Epoch {epoch:3d}/{epochs:3d} | Train Loss: {epoch_loss:.4f}")

        history.append(row)

    total_duration = (time.time() - start_time) / 3600.0
    print("=" * 80)
    print(f"Training completed in {total_duration:.2f} hours.")
    print(f"Best Validation Mean Dice: {best_dice:.4f} achieved at Epoch {best_epoch}")
    print(f"Best Checkpoint saved to : {best_path}")

    # Export history and plots
    history_json_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    curves_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_history(history, curves_path)
    print(f"Convergence curves plotted to -> {curves_path}")


if __name__ == "__main__":
    main()
