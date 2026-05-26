# AI-PRF Index Model Pipeline

This document describes the downstream tabular pipeline after PRF measurements have been extracted from image masks.

## Input Feature Table

The index generator expects one row per case:

```text
patient_id
ai_t_anterior_prft_mm
ai_t_lateral_prft_mm
ai_t_posterior_prft_mm
ai_t_mean_prft_mm
ai_o_anterior_prft_mm
ai_o_lateral_prft_mm
ai_o_posterior_prft_mm
ai_o_mean_prft_mm
ai_b_anterior_prft_mm
ai_b_lateral_prft_mm
ai_b_posterior_prft_mm
ai_b_mean_prft_mm
manual_mean_t_prft_mm        optional
manual_mean_o_prft_mm        optional
manual_mean_b_prft_mm        optional
```

See `examples/ai_t_prf_input_template.csv`.

## Step 1: Generate AI-PRF Index

```bash
python scripts/generate_ai_t_prf_index_table.py \
  --input outputs/ai_prf_features.csv \
  --output outputs/cohort_ai_t_prf_index_distribution.csv \
  --save-model outputs/ai_t_prf_index_model.json \
  --summary outputs/ai_t_prf_index_summary.csv
```

The script creates continuous AI-T/O/B-PRF indices and Low/High groups.

## Step 2: Analyze Distribution and Manual Agreement

```bash
python scripts/run_ai_t_prf_analysis.py \
  --input outputs/cohort_ai_t_prf_index_distribution.csv \
  --output-dir outputs/analysis \
  --threshold 0.43
```

Outputs include:

| Output | Purpose |
|---|---|
| `ai_t_prf_index_summary.csv` | Cohort counts and validation metrics |
| `cohort_ai_t_prf_index_distribution.csv` | Per-case index and Low/High group |
| `figure_ai_t_prf_index_distribution.svg` | Distribution plot |
| `figure_ai_index_manual_prft_correlation.svg` | Correlation with manual PRFT |
| `figure_ai_manual_prft_bland_altman.svg` | Agreement plot |

## Interpretation

The AI-T-PRF index is intended as an interpretable continuous biomarker derived from side-specific perirenal fat measurements. It can be dichotomized into Low/High groups using a prespecified or cohort-derived threshold, then associated with clinicopathologic variables and survival outcomes.
