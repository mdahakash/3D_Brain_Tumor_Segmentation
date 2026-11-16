"""
Single-Patient 3D Brain Tumor Segmentation Inference CLI.
Takes multimodal MRI scans (FLAIR, T1, T1ce, T2), runs sliding-window inference with SwinUNETR,
and exports the reconstructed 3D segmentation mask (NIfTI) and visualization slices.
"""

import os
import glob
import argparse
from typing import Dict, Optional

import numpy as np
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ConcatItemsd,
    NormalizeIntensityd,
    EnsureTyped,
)

from src.model import build_swinunetr_model
from src.losses import post_process_logits
from src.utils import (
    run_sliding_window_inference,
    reconstruct_integer_label_map,
    plot_slice_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment 3D Brain Tumor Sub-Regions for a Single Patient")
    parser.add_argument("--case_dir", type=str, default=None, help="Directory containing patient's 4 NIfTI files")
    parser.add_argument("--flair", type=str, default=None, help="Path to FLAIR NIfTI file")
    parser.add_argument("--t1", type=str, default=None, help="Path to T1 NIfTI file")
    parser.add_argument("--t1ce", type=str, default=None, help="Path to T1ce NIfTI file")
    parser.add_argument("--t2", type=str, default=None, help="Path to T2 NIfTI file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained SwinUNETR checkpoint (.pt)")
    parser.add_argument("--output_nifti", type=str, default=None, help="Path to export predicted 3D segmentation NIfTI (.nii.gz)")
    parser.add_argument("--output_png", type=str, default="segmentation_preview.png", help="Path to export visualization PNG")
    parser.add_argument("--feature_size", type=int, default=48, help="SwinUNETR feature dimension")
    parser.add_argument("--slice_idx", type=int, default=None, help="Axial slice index for visualization preview")
    return parser.parse_args()


def resolve_modalities(args: argparse.Namespace) -> Dict[str, str]:
    """Finds or validates the 4 modality file paths."""
    if args.case_dir:
        case_id = os.path.basename(args.case_dir.rstrip("/\\"))
        def find_file(suffix: str) -> str:
            candidates = [
                os.path.join(args.case_dir, f"{case_id}_{suffix}.nii"),
                os.path.join(args.case_dir, f"{case_id}_{suffix}.nii.gz"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    return c
            matches = glob.glob(os.path.join(args.case_dir, f"*{suffix}.nii*"))
            if matches:
                return matches[0]
            raise FileNotFoundError(f"Missing {suffix} scan in {args.case_dir}")

        return {
            "flair": find_file("flair"),
            "t1": find_file("t1"),
            "t1ce": find_file("t1ce"),
            "t2": find_file("t2"),
        }

    missing = [m for m, p in [("flair", args.flair), ("t1", args.t1), ("t1ce", args.t1ce), ("t2", args.t2)] if not p]
    if missing:
        raise ValueError(f"Missing modality paths: {missing}. Provide --case_dir or all four modality flags.")

    return {"flair": args.flair, "t1": args.t1, "t1ce": args.t1ce, "t2": args.t2}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = resolve_modalities(args)

    print("=" * 70)
    print("  3D SWINUNETR PATIENT INFERENCE ENGINE")
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Checkpoint    : {args.checkpoint}")
    for k, v in modalities.items():
        print(f"  {k.upper():<6} Scan  : {v}")
    print("=" * 70)

    # Preprocessing transform
    modality_keys = ["flair", "t1", "t1ce", "t2"]
    infer_transform = Compose([
        LoadImaged(keys=modality_keys),
        EnsureChannelFirstd(keys=modality_keys),
        ConcatItemsd(keys=modality_keys, name="image", dim=0),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])

    data_dict = infer_transform(modalities)
    image_tensor = data_dict["image"].unsqueeze(0).to(device)  # (1, 4, D, H, W)
    print(f"Preprocessed input tensor shape: {tuple(image_tensor.shape)}")

    # Load model
    model = build_swinunetr_model(
        in_channels=4,
        out_channels=3,
        feature_size=args.feature_size,
        use_checkpoint=False,
        device=device,
        multi_gpu=False
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Sliding-window inference
    roi_size = (96, 96, 96)
    with torch.no_grad():
        logits = run_sliding_window_inference(
            model=model,
            inputs=image_tensor,
            roi_size=roi_size,
            sw_batch_size=4,
            overlap=0.5,
            mode="gaussian"
        )
        binary_preds = post_process_logits(logits).squeeze(0).cpu().numpy()  # (3, D, H, W)

    # Reconstruct integer segmentation mask
    integer_mask = reconstruct_integer_label_map(binary_preds)  # (D, H, W)
    print(f"Reconstructed 3D Mask Shape: {integer_mask.shape} | Unique labels: {np.unique(integer_mask)}")

    # Volume Quantification
    voxel_counts = {
        "Necrotic/Core (Label 1)": int((integer_mask == 1).sum()),
        "Edema (Label 2)": int((integer_mask == 2).sum()),
        "Enhancing (Label 4)": int((integer_mask == 4).sum()),
    }
    total_tumor_voxels = sum(voxel_counts.values())
    print("\nTumor Sub-Region Quantification:")
    for region, count in voxel_counts.items():
        pct = (count / max(1, total_tumor_voxels)) * 100.0
        print(f"  - {region:<26}: {count:8,d} voxels ({pct:5.1f}%)")
    print(f"  - Total Tumor Volume Voxels : {total_tumor_voxels:8,d}")

    # Export NIfTI if requested
    if args.output_nifti:
        try:
            import nibabel as nib
            ref_nii = nib.load(modalities["flair"])
            affine = ref_nii.affine
            header = ref_nii.header
            out_nii = nib.Nifti1Image(integer_mask.astype(np.int16), affine, header)
            os.makedirs(os.path.dirname(os.path.abspath(args.output_nifti)), exist_ok=True)
            nib.save(out_nii, args.output_nifti)
            print(f"\nSaved 3D Segmentation NIfTI volume to: {args.output_nifti}")
        except ImportError:
            print("\nWarning: nibabel not installed. Could not save .nii.gz directly.")

    # Export visualization preview
    flair_volume = image_tensor[0, 0].cpu().numpy()
    slice_idx = args.slice_idx if args.slice_idx is not None else (flair_volume.shape[0] // 2)
    plot_slice_comparison(
        image_slice=flair_volume[slice_idx],
        ground_truth_slice=None,
        predicted_slice=integer_mask[slice_idx],
        save_path=args.output_png,
        slice_idx=slice_idx,
        title_suffix="(Predicted Mask)"
    )
    print(f"Diagnostic slice visualization saved to: {args.output_png}")


if __name__ == "__main__":
    main()
