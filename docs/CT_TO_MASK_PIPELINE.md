# CT-to-Mask Pipeline

This document describes the upstream image-processing layer used before AI-PRF feature extraction.

## Required Inputs

For each case, the downstream PRF measurement scripts require:

| Input | Purpose |
|---|---|
| CT volume | Anatomical image and voxel spacing reference |
| Kidney/tumor mask | Defines the renal outer boundary |
| Renal vein mask | Selects the renal venous measurement plane |
| Fat mask | Defines VAT; PRAT/PRF is extracted as kidney shell intersected with VAT |

All masks should be in the same space as the CT image.

## Recommended Segmentation Strategy

```text
CT
  -> kidney/tumor segmentation
  -> renal vein segmentation
  -> VAT segmentation
  -> radiologist QC
  -> PRF extraction and measurement
```

The code does not require a specific model architecture. Any model can be used if its output labels match `configs/label_conventions.md`.

## Anatomical Rules

### Kidney/Tumor Boundary

Use the union of kidney and tumor labels as the renal outer boundary. This is more robust than kidney-only segmentation when the tumor distorts the renal contour.

### Tumor Side and Opposite Side

The tumor-bearing side is determined from the tumor mask or provided explicitly by the user. The opposite side is the contralateral kidney side.

### Renal Vein Plane

The general renal venous level is selected as the axial slice with the largest renal vein segmentation area.

For tumor-side analysis, select the tumor-side renal vein plane within a small window around the all-vein selected plane. In our experiments, a `+/-5` slice window reduced remote false-positive vessel selection.

### VAT and PRF Extraction

VAT is intersected with a dilated shell around the renal outer boundary:

```text
PRF candidate region = dilated(kidney + tumor boundary) - kidney/tumor boundary
AI-derived PRF = PRF candidate region intersected with VAT
```

This produces tumor-side, opposite-side, and bilateral PRF regions.

## Example Command

```bash
python scripts/preview_prf_pipeline_plane.py \
  --case CASE_001,/path/to/ct.nii.gz,/path/to/organ_mask.nii.gz,/path/to/vein_mask.nii.gz,/path/to/fat_mask.nii.gz \
  --output-dir outputs/qc \
  --kidney-label 1 \
  --tumor-label 2 \
  --vein-label 3 \
  --vat-label 2 \
  --tumor-side-window 5
```

## Suggested Methods Wording

Preoperative CT images were processed using an AI-assisted workflow. Kidney/tumor segmentation was used to define the renal outer boundary and to identify the tumor-bearing and contralateral sides. Renal vein segmentation was used to localize the renal venous reference plane. VAT segmentation was then intersected with a perirenal shell around the kidney/tumor boundary to obtain side-specific perirenal fat regions. Quantitative features from these regions were used to construct the AI-T-PRF index and Low/High groups for downstream clinical analysis.
