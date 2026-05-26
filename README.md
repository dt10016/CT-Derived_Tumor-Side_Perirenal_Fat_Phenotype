# AI-PRF Assessment

End-to-end code for AI-assisted perirenal fat assessment on abdominal CT.

The workflow converts CT images and segmentation masks into side-specific perirenal fat measurements and an interpretable AI-T-PRF index for clinical analysis.

## Workflow

```text
CT image
  -> kidney/tumor boundary segmentation
  -> renal vein segmentation and renal vein plane localization
  -> visceral adipose tissue segmentation
  -> perirenal shell extraction: kidney/tumor shell intersected with VAT
  -> tumor-side, opposite-side, and bilateral PRF measurements
  -> AI-T/O/B-PRF index and Low/High groups
```

This repository contains the downstream extraction, QC, index generation, and validation utilities. Model weights and clinical imaging data are not included.

## Repository Layout

```text
scripts/
  infer_aattct_fat_nifti.py          # optional AATTCT/TransUNet fat inference wrapper
  extract_prat_from_vat.py           # extract PRAT/PRF masks from organ and VAT masks
  measure_prft_from_masks.py         # measure directional PRFT features
  generate_ai_t_prf_index_table.py   # generate AI-T/O/B-PRF index tables
  run_ai_t_prf_analysis.py           # distribution, agreement, and QC figures
  preview_*.py                       # visual QC utilities
docs/
  CT_TO_MASK_PIPELINE.md
  MODEL_PIPELINE.md
  PRELIMINARY_SEGMENTATION_DICE_RESULTS.md
configs/
  example_cases.csv
  label_conventions.md
examples/
  ai_t_prf_input_template.csv
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Minimal Usage

### 1. Extract PRAT masks from organ and VAT masks

```bash
python scripts/extract_prat_from_vat.py \
  --patient-id CASE_001 \
  --ct /path/to/CASE_001_ct.nii.gz \
  --organ-mask /path/to/CASE_001_organ_mask.nii.gz \
  --vat-mask /path/to/CASE_001_fat_mask.nii.gz \
  --tumor-side left \
  --output-dir outputs/CASE_001/prat
```

### 2. Measure directional PRFT from masks

```bash
python scripts/measure_prft_from_masks.py \
  --patient-id CASE_001 \
  --ct /path/to/CASE_001_ct.nii.gz \
  --kidney-mask /path/to/CASE_001_organ_mask.nii.gz \
  --tumor-mask /path/to/CASE_001_organ_mask.nii.gz \
  --vein-mask /path/to/CASE_001_vessel_mask.nii.gz \
  --fat-mask /path/to/CASE_001_fat_mask.nii.gz \
  --output outputs/ai_prf_features.csv
```

### 3. Generate AI-PRF index and groups

```bash
python scripts/generate_ai_t_prf_index_table.py \
  --input outputs/ai_prf_features.csv \
  --output outputs/cohort_ai_t_prf_index_distribution.csv \
  --save-model outputs/ai_t_prf_index_model.json \
  --summary outputs/ai_t_prf_index_summary.csv
```

### 4. Analyze distribution and manual agreement

```bash
python scripts/run_ai_t_prf_analysis.py \
  --input outputs/cohort_ai_t_prf_index_distribution.csv \
  --output-dir outputs/analysis
```

## Segmentation Models

This code can consume masks from any segmentation model if the labels are mapped correctly. In our development workflow:

- kidney/tumor boundary: KiTS-style organ SegResNet or manually QC'ed organ masks
- renal vein: vessel SegResNet
- VAT: AATTCT/TransUNet fat segmentation

Weights are intentionally excluded from the repository. Pass model checkpoints through command-line arguments or run the downstream scripts on already generated masks.

## Preliminary Validation

See `docs/PRELIMINARY_SEGMENTATION_DICE_RESULTS.md`.

Short version:

- kidney + tumor union Dice on KIPA validation examples: mean 0.9107
- renal vein Dice on KIPA validation examples: mean 0.7048
- all-vein renal plane localization error: mean 1.5 slices
- constrained tumor-side renal plane localization error: mean 2.1 slices

## Data and Privacy

Do not commit patient imaging data, masks, trained weights, or clinical tables. The `.gitignore` excludes common medical image and model weight formats.

## Citation

If you use this repository, cite the associated manuscript once available.
