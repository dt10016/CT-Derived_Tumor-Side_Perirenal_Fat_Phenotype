# Preliminary Segmentation Performance Results

## 1. Kidney/Tumor Boundary Segmentation

Validation dataset: KIPA cases with manual kidney/tumor labels.

Model: KiTS/organ SegResNet.

Evaluation space: preprocessed/transformed space.

### KiTS organ SegResNet full_z075

| Case | Kidney Dice | Kidney + Tumor Dice |
|---|---:|---:|
| 0 | 0.9151 | 0.9195 |
| 11 | 0.9194 | 0.8746 |
| 20 | 0.7618 | 0.9379 |
| Mean | 0.8654 | 0.9107 |
| Median | 0.9151 | 0.9195 |
| Range | 0.7618-0.9194 | 0.8746-0.9379 |

### KiTS organ SegResNet lowmem

| Case | Kidney Dice | Kidney + Tumor Dice |
|---|---:|---:|
| 0 | 0.9260 | 0.9353 |
| 11 | 0.9034 | 0.8854 |
| 20 | 0.7375 | 0.8989 |
| Mean | 0.8556 | 0.9065 |
| Median | 0.9034 | 0.8989 |
| Range | 0.7375-0.9260 | 0.8854-0.9353 |

Interpretation: the kidney + tumor union is more stable than kidney-only segmentation and is more appropriate as the renal outer boundary for PRF extraction.

## 2. Renal Vein Segmentation and Plane Localization

Validation dataset: 10 KIPA cases with manual renal vein labels.

Model: KIPA vessel-finetune SegResNet.

### Renal vein Dice

| Metric | Value |
|---|---:|
| N | 10 |
| Mean Dice | 0.7048 |
| Median Dice | 0.6964 |
| Range | 0.6020-0.8286 |

### Per-case vein Dice

| Case | Tumor Side | Vein Dice |
|---|---|---:|
| 0 | Left | 0.7550 |
| 1 | Left | 0.7992 |
| 3 | Left | 0.7346 |
| 5 | Left | 0.8286 |
| 7 | Left | 0.7080 |
| 11 | Left | 0.6519 |
| 13 | Right | 0.6109 |
| 20 | Left | 0.6848 |
| 24 | Left | 0.6730 |
| 30 | Right | 0.6020 |

### Renal vein plane localization error

| Plane Selection Method | Mean Absolute Error, slices | Median Absolute Error, slices | Range, slices |
|---|---:|---:|---:|
| All-vein selected plane | 1.5 | 0.0 | 0-14 |
| Raw tumor-side selected plane | 16.8 | 11.5 | 0-66 |
| Tumor-side selected plane constrained within all-vein +/- 5 slices | 2.1 | 0.0 | 0-19 |

Interpretation: all-vein selection is stable for the general renal vein level. Tumor-side plane selection should be constrained around the all-vein plane to avoid remote false-positive vessel regions.

## 3. VAT Segmentation

Model: AATTCT fat TransUNet.

Available result: qualitative QC on TCGA/KiTS examples showed plausible VAT/SAT separation in abdominal CT images. However, no voxel-level manual VAT labels have been evaluated locally yet, so Dice should not be reported for VAT unless a manual VAT reference set is added.

Suggested wording: VAT segmentation quality was assessed visually in representative cases, and voxel-wise Dice evaluation was available for kidney/tumor and renal vein segmentation.

## 4. Suggested Manuscript-Friendly Summary

In preliminary validation, the kidney/tumor boundary segmentation achieved a mean Dice of 0.9107 for the kidney + tumor union in three manually labelled KIPA cases. Renal vein segmentation achieved a mean Dice of 0.7048 in ten KIPA cases. The all-vein selected renal vein plane showed a mean absolute localization error of 1.5 slices, while tumor-side plane selection constrained within +/- 5 slices of the all-vein plane reduced the tumor-side plane localization error to 2.1 slices.

These preliminary results support the use of AI-assisted segmentation and anatomical plane localization for downstream PRF extraction, with radiologist quality control for cases with uncertain organ boundary or vessel plane localization.
