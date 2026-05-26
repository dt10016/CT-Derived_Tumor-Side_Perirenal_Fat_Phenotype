"""
Generate cohort_ai_t_prf_index_distribution.csv from AI-extracted PRFT features.

This is the lightweight "AI-T-PRF index model" used to bridge the image AI
pipeline and the clinical manuscript analyses.

Model idea
----------
The image AI module should produce tumor-side, opposite-side, and bilateral
perirenal fat measurements for each patient, preferably at the renal venous
plane:

    ai_t_anterior_prft_mm / ai_o_anterior_prft_mm / ai_b_anterior_prft_mm
    ai_t_lateral_prft_mm  / ai_o_lateral_prft_mm  / ai_b_lateral_prft_mm
    ai_t_posterior_prft_mm / ai_o_posterior_prft_mm / ai_b_posterior_prft_mm

This script converts those measurements into continuous indices:

    AI-T-PRF index: tumor-side PRF
    AI-O-PRF index: opposite-side PRF
    AI-B-PRF index: bilateral PRF

    1. log-transform positive thickness features
    2. z-score features using the cohort or a saved model config
    3. combine features using predefined or fitted weights
    4. min-max normalize the raw score to a 0-1 index
    5. assign AI-T-PRF-Low/High by threshold

The default model is intentionally interpretable and conservative:

    AI-T-PRF raw score = mean(z_anterior, z_lateral, z_posterior)

If a saved model JSON is supplied, the same feature means, standard deviations,
weights, min/max normalization, and threshold are reused, supporting external
validation.

Input CSV
---------
Preferred required columns:

    patient_id
    ai_t_anterior_prft_mm
    ai_t_lateral_prft_mm
    ai_t_posterior_prft_mm
    ai_o_anterior_prft_mm
    ai_o_lateral_prft_mm
    ai_o_posterior_prft_mm
    ai_b_anterior_prft_mm
    ai_b_lateral_prft_mm
    ai_b_posterior_prft_mm

Backward-compatible tumor-side aliases are also accepted:

    ai_anterior_t_prft_mm
    ai_lateral_t_prft_mm
    ai_posterior_t_prft_mm

Optional but recommended for validation:

    manual_mean_t_prft_mm
    ai_mean_t_prft_mm

Output CSV
----------
    cohort_ai_t_prf_index_distribution.csv

Columns:

    patient_id
    ai_t_prf_index / ai_t_prf_group
    ai_o_prf_index / ai_o_prf_group
    ai_b_prf_index / ai_b_prf_group
    manual_mean_t_prft_mm
    ai_mean_t_prft_mm
    ai_anterior_t_prft_mm
    ai_lateral_t_prft_mm
    ai_posterior_t_prft_mm

Example
-------
python generate_ai_t_prf_index_table.py ^
  --input ai_t_prf_input_template.csv ^
  --output cohort_ai_t_prf_index_distribution.csv ^
  --save-model ai_t_prf_index_model.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SIDE_FEATURES = {
    "t": [
        "ai_t_anterior_prft_mm",
        "ai_t_lateral_prft_mm",
        "ai_t_posterior_prft_mm",
    ],
    "o": [
        "ai_o_anterior_prft_mm",
        "ai_o_lateral_prft_mm",
        "ai_o_posterior_prft_mm",
    ],
    "b": [
        "ai_b_anterior_prft_mm",
        "ai_b_lateral_prft_mm",
        "ai_b_posterior_prft_mm",
    ],
}
BACKWARD_COMPAT_FEATURES = {
    "ai_anterior_t_prft_mm": "ai_t_anterior_prft_mm",
    "ai_lateral_t_prft_mm": "ai_t_lateral_prft_mm",
    "ai_posterior_t_prft_mm": "ai_t_posterior_prft_mm",
    "ai_mean_t_prft_mm": "ai_t_mean_prft_mm",
}
FEATURES = SIDE_FEATURES["t"]
LOW_GROUP = "AI-T-PRF-Low"
HIGH_GROUP = "AI-T-PRF-High"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required numeric value for column {key!r}, patient={row.get('patient_id', '')}")
    return float(value)


def normalize_input_columns(rows: list[dict]) -> None:
    """Accept older tumor-side column names and derive bilateral values if possible."""
    for row in rows:
        for old, new in BACKWARD_COMPAT_FEATURES.items():
            if new not in row and old in row:
                row[new] = row[old]
        # If bilateral directional values are absent but T/O are present, use their mean.
        for idx, direction in enumerate(("anterior", "lateral", "posterior")):
            b_key = f"ai_b_{direction}_prft_mm"
            t_key = f"ai_t_{direction}_prft_mm"
            o_key = f"ai_o_{direction}_prft_mm"
            if (b_key not in row or str(row.get(b_key, "")).strip() == "") and t_key in row and o_key in row:
                try:
                    row[b_key] = f"{(as_float(row, t_key) + as_float(row, o_key)) / 2:.2f}"
                except ValueError:
                    pass
        if "ai_t_mean_prft_mm" not in row or str(row.get("ai_t_mean_prft_mm", "")).strip() == "":
            if all(key in row and str(row.get(key, "")).strip() for key in SIDE_FEATURES["t"]):
                row["ai_t_mean_prft_mm"] = f"{mean([as_float(row, key) for key in SIDE_FEATURES['t']]):.2f}"
        if "ai_o_mean_prft_mm" not in row or str(row.get("ai_o_mean_prft_mm", "")).strip() == "":
            if all(key in row and str(row.get(key, "")).strip() for key in SIDE_FEATURES["o"]):
                row["ai_o_mean_prft_mm"] = f"{mean([as_float(row, key) for key in SIDE_FEATURES['o']]):.2f}"
        if "ai_b_mean_prft_mm" not in row or str(row.get("ai_b_mean_prft_mm", "")).strip() == "":
            if all(key in row and str(row.get(key, "")).strip() for key in SIDE_FEATURES["b"]):
                row["ai_b_mean_prft_mm"] = f"{mean([as_float(row, key) for key in SIDE_FEATURES['b']]):.2f}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sd(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    m = mean(values)
    value = math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))
    return value if value > 1e-12 else 1.0


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def transform_feature(value: float) -> float:
    """Stable transform for positive thickness measurements."""
    if value < 0:
        raise ValueError(f"PRFT measurements must be non-negative, got {value}")
    return math.log1p(value)


def build_default_model(rows: list[dict], threshold: float | None, low_fraction: float | None) -> dict:
    """Calibrate the shared PRF scoring model on tumor-side features."""
    transformed = {
        feature: [transform_feature(as_float(row, feature)) for row in rows]
        for feature in FEATURES
    }
    means = {feature: mean(values) for feature, values in transformed.items()}
    sds = {feature: sd(values) for feature, values in transformed.items()}
    weights = {feature: 1.0 / len(FEATURES) for feature in FEATURES}
    raw_scores = [
        sum(
            ((transform_feature(as_float(row, feature)) - means[feature]) / sds[feature]) * weights[feature]
            for feature in FEATURES
        )
        for row in rows
    ]
    raw_min = min(raw_scores)
    raw_max = max(raw_scores)
    if raw_max <= raw_min:
        raw_max = raw_min + 1.0
    indexes = [(score - raw_min) / (raw_max - raw_min) for score in raw_scores]
    if threshold is None:
        if low_fraction is None:
            threshold = percentile(indexes, 0.40)
        else:
            threshold = percentile(indexes, low_fraction)
    return {
        "model_name": "AI-T-PRF index model",
        "version": "0.1",
        "features": FEATURES,
        "transform": "log1p_then_zscore",
        "means": means,
        "sds": sds,
        "weights": weights,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "threshold": threshold,
        "group_rule": "AI-T-PRF-Low if index < threshold else AI-T-PRF-High",
    }


def load_model(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        model = json.load(fp)
    missing = [key for key in ["features", "means", "sds", "weights", "raw_min", "raw_max", "threshold"] if key not in model]
    if missing:
        raise ValueError(f"Model JSON is missing required keys: {missing}")
    return model


def save_model(path: Path, model: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(model, fp, indent=2)


def score_row(row: dict, model: dict) -> tuple[float, float]:
    raw = 0.0
    for feature in model["features"]:
        value = transform_feature(as_float(row, feature))
        raw += ((value - float(model["means"][feature])) / float(model["sds"][feature])) * float(model["weights"][feature])
    raw_min = float(model["raw_min"])
    raw_max = float(model["raw_max"])
    index = (raw - raw_min) / (raw_max - raw_min)
    index = max(0.0, min(1.0, index))
    return raw, index


def score_side(row: dict, model: dict, side: str) -> tuple[float, float]:
    side_features = SIDE_FEATURES[side]
    tumor_features = SIDE_FEATURES["t"]
    raw = 0.0
    for source_feature, model_feature in zip(side_features, tumor_features):
        value = transform_feature(as_float(row, source_feature))
        raw += (
            (value - float(model["means"][model_feature]))
            / float(model["sds"][model_feature])
        ) * float(model["weights"][model_feature])
    raw_min = float(model["raw_min"])
    raw_max = float(model["raw_max"])
    index = (raw - raw_min) / (raw_max - raw_min)
    index = max(0.0, min(1.0, index))
    return raw, index


def optional_value(row: dict, key: str, fallback: str = "") -> str:
    value = row.get(key, fallback)
    return fallback if value is None else str(value)


def generate_table(rows: list[dict], model: dict) -> list[dict]:
    output = []
    for row in rows:
        _, t_index = score_side(row, model, "t")
        _, o_index = score_side(row, model, "o") if all(key in row and str(row.get(key, "")).strip() for key in SIDE_FEATURES["o"]) else (float("nan"), float("nan"))
        _, b_index = score_side(row, model, "b") if all(key in row and str(row.get(key, "")).strip() for key in SIDE_FEATURES["b"]) else (float("nan"), float("nan"))
        threshold = float(model["threshold"])
        t_group = LOW_GROUP if t_index < threshold else HIGH_GROUP
        o_group = LOW_GROUP.replace("T-", "O-") if o_index < threshold else HIGH_GROUP.replace("T-", "O-")
        b_group = LOW_GROUP.replace("T-", "B-") if b_index < threshold else HIGH_GROUP.replace("T-", "B-")
        output.append(
            {
                "patient_id": row["patient_id"],
                "ai_t_prf_index": f"{t_index:.4f}",
                "ai_t_prf_group": t_group,
                "ai_o_prf_index": "" if math.isnan(o_index) else f"{o_index:.4f}",
                "ai_o_prf_group": "" if math.isnan(o_index) else o_group,
                "ai_b_prf_index": "" if math.isnan(b_index) else f"{b_index:.4f}",
                "ai_b_prf_group": "" if math.isnan(b_index) else b_group,
                "manual_mean_t_prft_mm": optional_value(row, "manual_mean_t_prft_mm"),
                "manual_mean_o_prft_mm": optional_value(row, "manual_mean_o_prft_mm"),
                "manual_mean_b_prft_mm": optional_value(row, "manual_mean_b_prft_mm"),
                "ai_t_mean_prft_mm": optional_value(row, "ai_t_mean_prft_mm"),
                "ai_o_mean_prft_mm": optional_value(row, "ai_o_mean_prft_mm"),
                "ai_b_mean_prft_mm": optional_value(row, "ai_b_mean_prft_mm"),
                "ai_t_anterior_prft_mm": f"{as_float(row, 'ai_t_anterior_prft_mm'):.2f}",
                "ai_t_lateral_prft_mm": f"{as_float(row, 'ai_t_lateral_prft_mm'):.2f}",
                "ai_t_posterior_prft_mm": f"{as_float(row, 'ai_t_posterior_prft_mm'):.2f}",
                "ai_o_anterior_prft_mm": optional_value(row, "ai_o_anterior_prft_mm"),
                "ai_o_lateral_prft_mm": optional_value(row, "ai_o_lateral_prft_mm"),
                "ai_o_posterior_prft_mm": optional_value(row, "ai_o_posterior_prft_mm"),
                "ai_b_anterior_prft_mm": optional_value(row, "ai_b_anterior_prft_mm"),
                "ai_b_lateral_prft_mm": optional_value(row, "ai_b_lateral_prft_mm"),
                "ai_b_posterior_prft_mm": optional_value(row, "ai_b_posterior_prft_mm"),
            }
        )
    return output


def write_summary(path: Path, rows: list[dict], model: dict) -> None:
    low = [row for row in rows if row["ai_t_prf_group"] == LOW_GROUP]
    high = [row for row in rows if row["ai_t_prf_group"] == HIGH_GROUP]
    indexes = [float(row["ai_t_prf_index"]) for row in rows]
    summary_rows = [
        {"metric": "total_n", "value": len(rows)},
        {"metric": "threshold", "value": f'{float(model["threshold"]):.4f}'},
        {"metric": "low_n", "value": len(low)},
        {"metric": "high_n", "value": len(high)},
        {"metric": "low_pct", "value": f"{len(low) / len(rows) * 100:.2f}"},
        {"metric": "high_pct", "value": f"{len(high) / len(rows) * 100:.2f}"},
        {"metric": "index_mean", "value": f"{mean(indexes):.4f}"},
        {"metric": "index_min", "value": f"{min(indexes):.4f}"},
        {"metric": "index_max", "value": f"{max(indexes):.4f}"},
    ]
    for prefix, label in (("ai_o", "o"), ("ai_b", "b")):
        index_key = f"{prefix}_prf_index"
        group_key = f"{prefix}_prf_group"
        values = [float(row[index_key]) for row in rows if str(row.get(index_key, "")).strip()]
        if values:
            low_label = f"AI-{label.upper()}-PRF-Low"
            high_label = f"AI-{label.upper()}-PRF-High"
            low_rows = [row for row in rows if row.get(group_key) == low_label]
            high_rows = [row for row in rows if row.get(group_key) == high_label]
            summary_rows.extend(
                [
                    {"metric": f"{label}_low_n", "value": len(low_rows)},
                    {"metric": f"{label}_high_n", "value": len(high_rows)},
                    {"metric": f"{label}_low_pct", "value": f"{len(low_rows) / len(rows) * 100:.2f}"},
                    {"metric": f"{label}_high_pct", "value": f"{len(high_rows) / len(rows) * 100:.2f}"},
                    {"metric": f"{label}_index_mean", "value": f"{mean(values):.4f}"},
                ]
            )
    write_csv(path, summary_rows, ["metric", "value"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AI-T-PRF index cohort table from AI PRFT features.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV with patient_id and AI directional PRFT measurements.",
    )
    parser.add_argument(
        "--output",
        default="cohort_ai_t_prf_index_distribution.csv",
        help="Output cohort_ai_t_prf_index_distribution.csv path.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional saved model JSON. If omitted, a model is fit/calibrated on the input cohort.",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help="Optional path to save calibrated model JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fixed Low/High threshold on normalized AI-T-PRF index.",
    )
    parser.add_argument(
        "--low-fraction",
        type=float,
        default=0.40,
        help="If threshold is absent, choose threshold by this lower quantile. Default: 0.40.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional summary CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.input))
    if not rows:
        raise ValueError("Input CSV is empty")
    normalize_input_columns(rows)
    for feature in ["patient_id"] + FEATURES:
        if feature not in rows[0]:
            raise ValueError(f"Input CSV must contain column {feature!r}")
    if args.model:
        model = load_model(Path(args.model))
    else:
        model = build_default_model(rows, args.threshold, args.low_fraction)
    output_rows = generate_table(rows, model)
    fieldnames = [
        "patient_id",
        "ai_t_prf_index",
        "ai_t_prf_group",
        "ai_o_prf_index",
        "ai_o_prf_group",
        "ai_b_prf_index",
        "ai_b_prf_group",
        "manual_mean_t_prft_mm",
        "manual_mean_o_prft_mm",
        "manual_mean_b_prft_mm",
        "ai_t_mean_prft_mm",
        "ai_o_mean_prft_mm",
        "ai_b_mean_prft_mm",
        "ai_t_anterior_prft_mm",
        "ai_t_lateral_prft_mm",
        "ai_t_posterior_prft_mm",
        "ai_o_anterior_prft_mm",
        "ai_o_lateral_prft_mm",
        "ai_o_posterior_prft_mm",
        "ai_b_anterior_prft_mm",
        "ai_b_lateral_prft_mm",
        "ai_b_posterior_prft_mm",
    ]
    write_csv(Path(args.output), output_rows, fieldnames)
    if args.save_model:
        save_model(Path(args.save_model), model)
    if args.summary:
        write_summary(Path(args.summary), output_rows, model)
    low_n = sum(1 for row in output_rows if row["ai_t_prf_group"] == LOW_GROUP)
    high_n = len(output_rows) - low_n
    print(f"Wrote: {args.output}")
    print(f"Patients: {len(output_rows)}")
    print(f"Threshold: {float(model['threshold']):.4f}")
    print(f"{LOW_GROUP}: {low_n} ({low_n / len(output_rows) * 100:.1f}%)")
    print(f"{HIGH_GROUP}: {high_n} ({high_n / len(output_rows) * 100:.1f}%)")
    if args.save_model:
        print(f"Saved model: {args.save_model}")


if __name__ == "__main__":
    main()
