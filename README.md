# Multi-Task SegFormer for Knee Ultrasound: Effusion ROI Segmentation and Patient-Level Classification of Inflammatory Arthritis

This repository contains the complete implementation, evaluation notebooks, and visualization scripts for the manuscript:

> *"Multi-Task SegFormer for Knee Ultrasound Effusion ROI Segmentation and Patient-Level Classification of Inflammatory Arthritis"*

All notebooks are provided with their **original cell execution outputs preserved**, so all tables and figures reported in the manuscript can be directly verified without retraining the models.

---

## Repository Structure

### `1_Training/` — Model Training Pipelines

| Notebook | Description | Manuscript Reference |
|---|---|---|
| `SegFormer_MTL_Training.ipynb` | Multi-Task SegFormer (MIT-B2) training with Optuna HPO. Includes nested CV setup, AdamW + cosine scheduler, joint BCE+Dice+Focal seg loss. | Table 1 (SegFormer rows) |
| `YOLOv8_Training.ipynb` | YOLOv8-Seg single-stage training pipeline with Optuna HPO. | Table 1 (YOLOv8-Seg rows) |
| `MaskRCNN_Training.ipynb` | Mask R-CNN (ResNet50-FPN) two-stage training pipeline with Optuna HPO. COCO-format annotation conversion included. | Table 1 (Mask R-CNN rows), Figure 3B |

---

### `2_Evaluation/` — Test Set Evaluation (Per Fold)

| Notebook | Description | Manuscript Reference |
|---|---|---|
| `SegFormer_Evaluation.ipynb` | Patient-level test evaluation of the proposed SegFormer model. Loads fold-wise prediction CSVs, computes Dice, Accuracy, Sensitivity, Specificity, F1 per fold, and generates the aggregated confusion matrix. | Table 2, Table 3 (SegFormer rows), Figure 6A |
| `YOLOv8_Evaluation.ipynb` | Patient-level test evaluation of YOLOv8-Seg. Includes per-fold confusion matrix generation. | Table 2, Table 3 (YOLOv8-Seg rows), Figure 6C |
| `MaskRCNN_Evaluation.ipynb` | Patient-level test evaluation of Mask R-CNN. Includes segmentation table, fold-wise Dice boxplots, and per-fold confusion matrices. | Table 2, Table 3 (Mask R-CNN rows), Figure 6B |

---

### `3_Visualization_and_Analysis/` — Figures and Tables from the Manuscript

| Notebook | Description | Manuscript Reference |
|---|---|---|
| `Dice_Boxplots_Figure4.ipynb` | Fold-wise Dice score boxplots across SegFormer, Mask R-CNN, and YOLOv8-Seg. | **Figure 4** |
| `Prediction_vs_GT_Overlay_Figure5.ipynb` | Qualitative segmentation overlays: GT (white contours) vs. Prediction (red contours) for all three models. | **Figure 5** |
| `Confusion_Matrices_Figure6.ipynb` | Patient-level confusion matrices for SegFormer (A), Mask R-CNN (B), and YOLOv8-Seg (C). Includes both per-fold breakdown and aggregated combined matrices for all three models. | **Figure 6A, 6B, 6C** |
| `Results_Tables_Table2_Table3.ipynb` | Computes and formats Table 2 (Dice, IoU) and Table 3 (Accuracy, Sensitivity, Specificity, F1). | **Table 2, Table 3** |
| `ROC_Calibration_Analysis.ipynb` | ROC curve with 95% bootstrap CI, Precision-Recall curve, Calibration plot (Brier Score), threshold sensitivity analysis. | Supplementary / Results section |
| `Epoch_Timing_Analysis.ipynb` | Per-epoch and per-fold training time analysis across all models. | Discussion (computational complexity) |

---

### `Outputs/` — Pre-Generated Results

#### `Outputs/Figures/`
High-resolution figures directly corresponding to the manuscript:

| File | Manuscript Figure |
|---|---|
| `confusion_matrix.png` | Figure 6A — SegFormer aggregated confusion matrix |
| `confusion_matrix_segformer_perfold.png` | Figure 6A — SegFormer per-fold confusion matrices (5 folds + combined) |
| `confusion_matrix_maskrcnn.png` | Figure 6B — Mask R-CNN per-fold confusion matrices (5 folds + combined) |
| `confusion_matrix_yolo.png` | Figure 6C — YOLOv8-Seg per-fold confusion matrices (5 folds + combined) |
| `per_fold_boxplots.png` | Figure 4 — Fold-wise Dice score boxplots |
| `per_fold_boxplots_fixed.png` | Figure 4 — Fold-wise Dice score boxplots (alternative layout) |
| `Figure_Prediction_vs_GroundTruth_ROI_BlueRed_Pairs.png` | Figure 5 — SegFormer qualitative segmentation overlays |
| `maskrcnn_overlay_grid.png` | Figure 5 — Mask R-CNN qualitative segmentation overlay grid |
| `image_005_R_P_001_pred_arthrit_with_GT_ROI.png` | Figure 5 — SegFormer, arthritis patient (005_R_P) |
| `image_008_L_C_001_pred_control_with_GT_ROI.png` | Figure 5 — SegFormer, control patient (008_L_C) |
| `image_016_R_C_001_pred_control_with_GT_ROI.png` | Figure 5 — SegFormer, control patient (016_R_C) |
| `image_017_L_P_001_pred_arthrit_with_GT_ROI.png` | Figure 5 — SegFormer, arthritis patient (017_L_P) |
| `roc_curve_with_ci.png` | ROC Curve with 95% bootstrap CI |
| `pr_curve.png` | Precision-Recall Curve |
| `bland_altman.png` | Bland-Altman Agreement Plot |
| `learning_curves.png` | Learning dynamics (validation metrics per epoch) |
| `calibration_plot.png` | Calibration / Reliability diagram |
| `epoch_timing_analysis.png` | Per-epoch training time comparison across models |

#### `Outputs/Tables/`
CSV and LaTeX tables for direct paper verification:

| File | Manuscript Reference |
|---|---|
| `main_results.csv` / `main_results.tex` | Table 2 & Table 3 (aggregate) |
| `per_fold_statistics.csv` | Per-fold breakdown |
| `per_fold_test_metrics_by_run.csv` | Detailed per-fold test metrics |
| `best_hyperparameters.csv` / `best_hyperparameters.tex` | Table 1 (optimal HPO results) |
| `statistical_tests.csv` | Statistical significance tests |
| `calibration_metrics.csv` | Brier Score |
| `time_complexity.csv` | Inference/training time |
| `error_analysis.csv` | Misclassification analysis |
| `overall_epoch_time_stats.csv` | Overall per-epoch training time stats |
| `per_epoch_time_stats.csv` | Detailed per-epoch time records |

---

## Experimental Setup

- **Dataset**: 65 individuals (26 inflammatory arthritis, 39 non-IA controls); 10 B-mode slices per patient (650 total images)
- **Cross-validation**: 5-fold patient-level (no patient data leakage between splits)
- **HPO**: Nested CV — Optuna with 30 trials per outer fold
- **Hardware**: Single GPU training

## Key Results (from Table 2 & Table 3)

| Model | Dice (%) | IoU (%) | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1 (%) |
|---|---|---|---|---|---|---|
| **SegFormer (proposed)** | **91.63 ± 1.25** | **85.57 ± 1.88** | **90.77 ± 6.44** | **97.14 ± 6.39** | 86.79 ± 13.46 | **90.45 ± 6.48** |
| Mask R-CNN | 75.42 ± 2.99 | 65.09 ± 2.65 | 87.69 ± 8.77 | 89.81 ± 9.52 | **86.79 ± 13.46** | 86.51 ± 9.49 |
| YOLOv8-Seg | 79.20 ± 3.91 | 68.87 ± 4.13 | 84.62 ± 7.69 | 100.00 ± 0.00 | 74.10 ± 13.43 | 83.23 ± 8.35 |

## Requirements

```
Python >= 3.8
torch >= 1.12
torchvision
transformers
ultralytics
optuna
scikit-learn
pandas, numpy
matplotlib, seaborn
opencv-python
```
