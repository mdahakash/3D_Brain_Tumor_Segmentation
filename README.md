# 🧠 3D Multimodal Brain Tumor Sub-Region Segmentation Using SwinUNETR on BraTS

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3%2B-blue?style=flat-square)](https://monai.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Benchmark](https://img.shields.io/badge/Benchmark-BraTS2020-orange?style=flat-square)]()
[![Model](https://img.shields.io/badge/Backbone-SwinUNETR%203D-purple?style=flat-square)]()

**Research Laboratory:** ARCHE Research Lab  
**Domain:** Volumetric Medical Image Analysis & Deep Neuro-Oncology  
**Benchmark:** Multimodal Brain Tumor Segmentation Challenge (BraTS 2020)  
**Primary Framework:** PyTorch & MONAI (*Medical Open Network for AI*)

---

## 📌 Clinical Overview & Task Formulation

Volumetric brain tumor delineation from multimodal Magnetic Resonance Imaging (MRI) is essential for neurosurgical resection guidance, targeted radiation oncology, and longitudinal therapeutic response tracking. 

Gliomas exhibit heterogeneous infiltrative morphology across different MRI sequences:
- **T1-weighted (T1):** Delineates healthy brain anatomy.
- **T1-weighted post-contrast (T1ce / T1Gd):** Highlights the hyper-intense, breakdown of the blood-brain barrier in active tumor margins.
- **T2-weighted (T2):** Visualizes fluid collections and non-enhancing tumor tissue.
- **T2-FLAIR (Fluid-Attenuated Inversion Recovery):** Suppresses free cerebrospinal fluid (CSF) to clearly highlight peritumoral edema.

### Clinical Region-of-Interest (ROI) Multi-Label Hierarchy

Rather than mutual-exclusive categorical classes, the BraTS benchmark formulates brain tumor segmentation as **three biologically distinct, nested sub-regions**:

```text
+------------------------------------------------------------------------------------------------+
| Multi-Label Target Channel  | Anatomical Composition                  | Ground Truth Labels    |
+-----------------------------+-----------------------------------------+------------------------+
| 1. Tumor Core (TC)          | Necrotic/Non-enhancing Core + Enhancing | Labels 1 or 4          |
| 2. Whole Tumor (WT)         | Tumor Core + Peritumoral Edema          | Labels 1, 2, or 4      |
| 3. Enhancing Tumor (ET)     | Active Gadolinium-Enhancing rim         | Label 4                |
+------------------------------------------------------------------------------------------------+
```

Reconstruction maps the multi-label binary channels back to clinical integer codes:
$$\text{Mask}(x, y, z) = \begin{cases} 
4 & \text{if } \text{ET}(x, y, z) = 1 \\
1 & \text{if } \text{TC}(x, y, z) = 1 \text{ and } \text{ET}(x, y, z) = 0 \\
2 & \text{if } \text{WT}(x, y, z) = 1 \text{ and } \text{TC}(x, y, z) = 0 \\
0 & \text{otherwise (Normal Brain / Background)}
\end{cases}$$

---

## 🏗️ 3D SwinUNETR Network Architecture

```text
+--------------------------------------------------------------------------------------------------+
|                            Input: 4-Channel 3D MRI Volume (4, 96, 96, 96)                        |
|                                     [FLAIR, T1, T1ce, T2]                                        |
+------------------------------------------------+-------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                                3D Patch Partition & Linear Embedding                             |
|                                       Patch Size: 2 x 2 x 2                                      |
+------------------------------------------------+-------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                    Hierarchical Swin Transformer 3D Encoder (Shifted Windows)                    |
|       - Stage 1: Swin Blocks (C = 48)  -----------------------------> Skip Connection 1 (to D1)   |
|       - Stage 2: Swin Blocks (C = 96)  --------------------------> Skip Connection 2 (to D2)     |
|       - Stage 3: Swin Blocks (C = 192) ----------------------> Skip Connection 3 (to D3)         |
|       - Stage 4: Swin Blocks (C = 384) -------------------> Bottleneck & Skip 4                  |
+------------------------------------------------+-------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                            U-Shaped Residual Transposed CNN Decoder                              |
|           Concatenates multi-resolution encoder representations via residual skip blocks          |
+------------------------------------------------+-------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                              1x1x1 Convolution Head (Sigmoid Logits)                             |
|                                    Output: (3, 96, 96, 96)                                       |
|                                     [ TC | WT | ET ]                                             |
+--------------------------------------------------------------------------------------------------+
```

- **Feature Size:** 48 (`SwinUNETR` tiny configuration, ~62.2M parameters)
- **Gradient Checkpointing:** Enabled (`use_checkpoint=True`) to limit activation memory overhead.
- **Hybrid Loss Formulation:** Combined Soft Dice and Binary Cross Entropy:
  $$\mathcal{L}_{\text{DiceCE}} = \mathcal{L}_{\text{Dice}}(\sigma(\hat{y}), y) + \mathcal{L}_{\text{BCE}}(\sigma(\hat{y}), y)$$

---

## 📊 Quantitative Benchmark Results

Evaluated on held-out full-resolution 3D validation volumes ($240 \times 240 \times 155\text{ mm}^3$) using Gaussian-weighted sliding-window inference:

| Evaluated Tumor Region | Clinical Significance | Mean Dice Score | Dice Similarity (%) |
|:---|:---|:---:|:---:|
| **Whole Tumor (WT)** | Complete pathologic extent (Core + Infiltration) | **0.9261** | **92.61%** |
| **Enhancing Tumor (ET)** | Viable, hyper-vascularized malignant tissue | **0.8606** | **86.06%** |
| **Tumor Core (TC)** | Resection target (Necrosis + Enhancing rim) | **0.7782** | **77.82%** |
| **Mean Overall Benchmark** | **Composite Diagnostic Index** | **0.8549** | **85.49%** |

---

## 📁 Repository Directory Structure

```text
3d_brain_tumor_segmentation/
├── configs/
│   └── default_config.json        # Hyperparameters (ROI, learning rate, window batch)
├── src/
│   ├── __init__.py                # Package exports
│   ├── model.py                   # SwinUNETR builder and parameter verification
│   ├── dataset.py                 # Multi-label transform, lazy NIfTI loader, case indexer
│   ├── losses.py                  # Hybrid DiceCELoss and multi-label Dice metrics
│   └── utils.py                   # Sliding-window inferer, integer map reconstruction, slice plots
├── notebooks/
│   └── brats_swinunetr_demo.ipynb # Interactive demonstration notebook
├── train.py                       # CLI training pipeline with sliding-window validation
├── evaluate.py                    # CLI full-volume evaluation script on validation cohorts
├── infer.py                       # CLI inference script on raw patient NIfTI volumes
├── requirements.txt               # Pinned pip dependencies
├── environment.yml                # Conda environment definition
├── LICENSE                        # Open-source MIT License
└── README.md                      # Academic documentation
```

---

## 🚀 Quickstart & Reproducibility Guide

### 1. Environment Installation

```bash
# Option A: Conda Environment (Recommended)
conda env create -f environment.yml
conda activate brats-swinunetr

# Option B: Pip & Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Ingestion

Download the [BraTS 2020 Dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation/).
Directory structure:

```text
MICCAI_BraTS2020_TrainingData/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii
│   ├── BraTS20_Training_001_t1.nii
│   ├── BraTS20_Training_001_t1ce.nii
│   ├── BraTS20_Training_001_t2.nii
│   └── BraTS20_Training_001_seg.nii
└── ...
```

### 3. Model Training

Train SwinUNETR with patch-based sampling ($96 \times 96 \times 96$), foreground cropping, and multi-GPU support:

```bash
python train.py \
  --data_root "/path/to/MICCAI_BraTS2020_TrainingData" \
  --output_dir "./outputs" \
  --epochs 30 \
  --lr 1e-4 \
  --val_interval 5 \
  --val_cases 20
```

### 4. Full-Volume Validation Evaluation

Run full-volume sliding window inference across held-out cases to compute TC, WT, and ET Dice scores:

```bash
python evaluate.py \
  --checkpoint "./outputs/brats_swinunetr_best.pt" \
  --data_root "/path/to/MICCAI_BraTS2020_TrainingData" \
  --output_dir "./eval_results" \
  --val_cases 20
```

Outputs generated:
- `eval_results/evaluation_metrics.json`
- `eval_results/case_1_slice_*.png`

### 5. Single-Patient 3D Inference & NIfTI Mask Export

Run automated inference on a patient case folder and export the predicted 3D segmentation volume as a `.nii.gz` file:

```bash
python infer.py \
  --case_dir "/path/to/BraTS20_Training_001" \
  --checkpoint "./outputs/brats_swinunetr_best.pt" \
  --output_nifti "predicted_segmentation.nii.gz" \
  --output_png "diagnostic_slice_preview.png"
```

CLI Output:
```text
======================================================================
  3D SWINUNETR PATIENT INFERENCE ENGINE
======================================================================
Preprocessed input tensor shape: (1, 4, 155, 240, 240)
Reconstructed 3D Mask Shape: (155, 240, 240) | Unique labels: [0 1 2 4]

Tumor Sub-Region Quantification:
  - Necrotic/Core (Label 1)   :   14,250 voxels ( 22.4%)
  - Edema (Label 2)           :   38,910 voxels ( 61.1%)
  - Enhancing (Label 4)       :   10,520 voxels ( 16.5%)
  - Total Tumor Volume Voxels :   63,680 voxels

Saved 3D Segmentation NIfTI volume to: predicted_segmentation.nii.gz
Diagnostic slice visualization saved to: diagnostic_slice_preview.png
```

---

## 🔬 Computational Requirements & Optimization

- **GPU Acceleration:** Requires a GPU with $\ge 12\text{ GB}$ VRAM (NVIDIA RTX 3090, RTX 4090, A5000, Tesla T4 $\times 2$, or A100).
- **Patch Size:** Training uses $96^3$ spatial crops to balance contextual receptive field against memory footprint.
- **Sliding Window Inference:** Uses Gaussian window weighting (`overlap=0.5`) across full $240 \times 240 \times 155$ volumes, eliminating border interpolation artifacts.

---

## 📜 Citation

If you find this codebase or benchmark useful in your research, please cite:

```bibtex
@article{arche_brats_swinunetr2026,
  title   = {Volumetric Multimodal Brain Tumor Sub-Region Delineation via 3D SwinUNETR Transformers},
  author  = {Md Ashraful Hossen Akash, ARCHE Research Lab Neuro-Imaging Group},
  target journal = {Medical Image Analysis / IEEE Transactions on Medical Imaging},
  year    = {2026}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
