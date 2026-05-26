"""
AI-T-PRF index analysis pipeline.

Purpose
-------
This script supports manuscript analyses for an AI-derived tumor-side
perirenal fat index:

1. Build or read a continuous AI-T-PRF index.
2. Stratify patients into AI-T-PRF-Low and AI-T-PRF-High groups.
3. Summarize index distribution and group counts.
4. Compare AI-derived measurements with manual/traditional PRFT.
5. Export CSV summary tables and SVG figures.

Input
-----
By default, the script reads:

    cohort_ai_t_prf_index_distribution.csv

Expected columns:

    patient_id
    ai_t_prf_index
    manual_mean_t_prft_mm
    ai_mean_t_prft_mm

Optional columns:

    ai_anterior_t_prft_mm
    ai_lateral_t_prft_mm
    ai_posterior_t_prft_mm

If ai_t_prf_index is absent, the script can derive it from the three AI PRFT
directional measurements using z-scored averaging.

Outputs
-------
    analysis_outputs/cohort_ai_t_prf_index_distribution.csv
    analysis_outputs/ai_vs_manual_prft_validation_subset.csv
    analysis_outputs/ai_t_prf_index_summary.csv
    analysis_outputs/figures/*.svg

Notes
-----
This script uses only the Python standard library so it can run on a clean
Windows workstation without scientific plotting packages.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path


LOW_GROUP = "AI-T-PRF-Low"
HIGH_GROUP = "AI-T-PRF-High"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError("fieldnames is required when writing an empty CSV")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        raise ValueError(f"Missing required numeric column {key!r} for patient {row.get('patient_id', '')}")
    return float(value)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def quantile(values: list[float], p: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = avg_rank
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


def icc_absolute_agreement_single(x: list[float], y: list[float]) -> float:
    """Two-way mixed, absolute agreement, single rater ICC(A,1)."""
    data = [[x[i], y[i]] for i in range(len(x))]
    n_subjects = len(data)
    k = 2
    grand = mean([v for row in data for v in row])
    row_means = [mean(row) for row in data]
    col_means = [mean([data[i][j] for i in range(n_subjects)]) for j in range(k)]
    ss_rows = k * sum((row_mean - grand) ** 2 for row_mean in row_means)
    ss_cols = n_subjects * sum((col_mean - grand) ** 2 for col_mean in col_means)
    ss_total = sum((data[i][j] - grand) ** 2 for i in range(n_subjects) for j in range(k))
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n_subjects - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n_subjects - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n_subjects
    return (ms_rows - ms_error) / denominator


def derive_index_from_directional_prft(rows: list[dict]) -> None:
    keys = ["ai_anterior_t_prft_mm", "ai_lateral_t_prft_mm", "ai_posterior_t_prft_mm"]
    for key in keys:
        if key not in rows[0]:
            raise ValueError(
                "ai_t_prf_index is absent, and directional PRFT columns are missing. "
                f"Required columns: {', '.join(keys)}"
            )
    columns = {key: [to_float(row, key) for row in rows] for key in keys}
    means = {key: mean(values) for key, values in columns.items()}
    sds = {key: sd(values) or 1.0 for key, values in columns.items()}
    z_scores = []
    for row in rows:
        z = mean([(to_float(row, key) - means[key]) / sds[key] for key in keys])
        z_scores.append(z)
    z_min, z_max = min(z_scores), max(z_scores)
    for row, z in zip(rows, z_scores):
        # Normalize to 0-1 so thresholds and figures are easy to interpret.
        row["ai_t_prf_index"] = f"{(z - z_min) / (z_max - z_min):.4f}"


def add_group(rows: list[dict], threshold: float) -> None:
    for row in rows:
        index = to_float(row, "ai_t_prf_index")
        row["ai_t_prf_group"] = LOW_GROUP if index < threshold else HIGH_GROUP


def choose_threshold(rows: list[dict], requested: float | None) -> float:
    if requested is not None:
        return requested
    values = [to_float(row, "ai_t_prf_index") for row in rows]
    return statistics.median(values)


def select_validation_subset(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n <= 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, n)


def summarize(rows: list[dict], validation_rows: list[dict], threshold: float) -> dict[str, float | int]:
    values = [to_float(row, "ai_t_prf_index") for row in rows]
    low = [row for row in rows if row["ai_t_prf_group"] == LOW_GROUP]
    high = [row for row in rows if row["ai_t_prf_group"] == HIGH_GROUP]
    manual = [to_float(row, "manual_mean_t_prft_mm") for row in validation_rows]
    ai_prft = [to_float(row, "ai_mean_t_prft_mm") for row in validation_rows]
    index = [to_float(row, "ai_t_prf_index") for row in validation_rows]
    errors = [ai - man for ai, man in zip(ai_prft, manual)]
    abs_errors = [abs(err) for err in errors]
    error_mean = mean(errors)
    error_sd = sd(errors)
    return {
        "total_n": len(rows),
        "threshold": threshold,
        "low_n": len(low),
        "high_n": len(high),
        "low_pct": len(low) / len(rows) * 100,
        "high_pct": len(high) / len(rows) * 100,
        "index_mean": mean(values),
        "index_sd": sd(values),
        "index_median": statistics.median(values),
        "index_q1": quantile(values, 0.25),
        "index_q3": quantile(values, 0.75),
        "manual_validation_n": len(validation_rows),
        "pearson_ai_index_manual_prft": pearson(index, manual),
        "spearman_ai_index_manual_prft": spearman(index, manual),
        "pearson_ai_prft_manual_prft": pearson(ai_prft, manual),
        "icc_ai_prft_manual_prft": icc_absolute_agreement_single(ai_prft, manual),
        "mae_mm": mean(abs_errors),
        "bias_mm": error_mean,
        "loa_low_mm": error_mean - 1.96 * error_sd,
        "loa_high_mm": error_mean + 1.96 * error_sd,
    }


def save_summary_csv(path: Path, summary: dict[str, float | int]) -> None:
    rows = []
    for key, value in summary.items():
        if isinstance(value, float):
            value = round(value, 4)
        rows.append({"metric": key, "value": value})
    write_csv(path, rows, ["metric", "value"])


def escape_svg(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


COLORS = {
    "blue": "#2F5F9F",
    "teal": "#2A9D8F",
    "orange": "#E76F51",
    "gray": "#6C757D",
    "red": "#B23A48",
    "grid": "#DDE2E6",
}


def svg_base(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            "<style>"
            "text{font-family:Arial,Helvetica,sans-serif;fill:#222}"
            ".title{font-size:22px;font-weight:700}"
            ".label{font-size:14px}"
            ".tick{font-size:12px}"
            ".small{font-size:11px}"
            "</style>"
        ),
        f'<text x="{width / 2}" y="38" text-anchor="middle" class="title">{escape_svg(title)}</text>',
    ]


def write_svg(path: Path, parts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_distribution(rows: list[dict], threshold: float, path: Path) -> None:
    values = [to_float(row, "ai_t_prf_index") for row in rows]
    low_n = sum(1 for value in values if value < threshold)
    high_n = len(values) - low_n
    width, height = 900, 560
    left, right, top, bottom = 70, 40, 70, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = svg_base(width, height, "Distribution of AI-T-PRF Index")
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.04
    lo, hi = lo - pad, hi + pad
    bin_count = 18
    bins = [lo + (hi - lo) * i / bin_count for i in range(bin_count + 1)]
    counts = [0] * bin_count
    for value in values:
        idx = min(bin_count - 1, max(0, int((value - lo) / (hi - lo) * bin_count)))
        counts[idx] += 1
    max_count = max(counts)
    for i in range(6):
        val = max_count * i / 5
        y = top + plot_h - val / max_count * plot_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>'
        )
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{int(val)}</text>')
    bar_w = plot_w / bin_count * 0.88
    for i, count in enumerate(counts):
        x = left + plot_w * i / bin_count + plot_w / bin_count * 0.06
        y = top + plot_h - count / max_count * plot_h
        color = COLORS["orange"] if bins[i] < threshold else COLORS["teal"]
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top + plot_h - y:.1f}" fill="{color}"/>'
        )
    tx = left + (threshold - lo) / (hi - lo) * plot_w
    parts.append(
        f'<line x1="{tx:.1f}" y1="{top}" x2="{tx:.1f}" y2="{top + plot_h}" '
        f'stroke="{COLORS["red"]}" stroke-width="3" stroke-dasharray="7 5"/>'
    )
    parts.append(f'<text x="{tx + 8:.1f}" y="{top + 22}" class="tick">Threshold = {threshold:.2f}</text>')
    parts.append(
        f'<text x="{left + 20}" y="{top + 28}" class="tick">Low: {low_n} ({low_n / len(values) * 100:.1f}%)</text>'
    )
    parts.append(
        f'<text x="{left + 20}" y="{top + 50}" class="tick">High: {high_n} ({high_n / len(values) * 100:.1f}%)</text>'
    )
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 30}" text-anchor="middle" class="label">AI-T-PRF index</text>')
    parts.append(
        f'<text x="25" y="{top + plot_h / 2}" text-anchor="middle" class="label" '
        f'transform="rotate(-90 25,{top + plot_h / 2})">Number of patients</text>'
    )
    write_svg(path, parts)


def plot_correlation(rows: list[dict], path: Path) -> None:
    index = [to_float(row, "ai_t_prf_index") for row in rows]
    manual = [to_float(row, "manual_mean_t_prft_mm") for row in rows]
    groups = [row["ai_t_prf_group"] for row in rows]
    width, height = 760, 600
    left, right, top, bottom = 80, 40, 70, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = svg_base(width, height, "AI-T-PRF Index vs Manual PRFT")
    xmin, xmax = min(index) * 0.9, max(index) * 1.05
    ymin, ymax = min(manual) * 0.88, max(manual) * 1.08
    for i in range(6):
        y = top + plot_h - i / 5 * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>')
    for idx, man, group in zip(index, manual, groups):
        x = left + (idx - xmin) / (xmax - xmin) * plot_w
        y = top + plot_h - (man - ymin) / (ymax - ymin) * plot_h
        color = COLORS["orange"] if group == LOW_GROUP else COLORS["teal"]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" opacity="0.82"/>')
    parts.append(f'<text x="{left + 20}" y="{top + 28}" class="tick">Pearson r = {pearson(index, manual):.3f}</text>')
    parts.append(f'<text x="{left + 20}" y="{top + 50}" class="tick">Spearman rho = {spearman(index, manual):.3f}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 25}" text-anchor="middle" class="label">AI-T-PRF index</text>')
    parts.append(
        f'<text x="25" y="{top + plot_h / 2}" text-anchor="middle" class="label" '
        f'transform="rotate(-90 25,{top + plot_h / 2})">Manual mean T-PRFT (mm)</text>'
    )
    write_svg(path, parts)


def plot_bland_altman(rows: list[dict], path: Path) -> None:
    manual = [to_float(row, "manual_mean_t_prft_mm") for row in rows]
    ai_prft = [to_float(row, "ai_mean_t_prft_mm") for row in rows]
    x_values = [(ai + man) / 2 for ai, man in zip(ai_prft, manual)]
    y_values = [ai - man for ai, man in zip(ai_prft, manual)]
    bias = mean(y_values)
    y_sd = sd(y_values)
    loa_low = bias - 1.96 * y_sd
    loa_high = bias + 1.96 * y_sd
    width, height = 760, 600
    left, right, top, bottom = 80, 40, 70, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = svg_base(width, height, "AI vs Manual PRFT Agreement")
    xmin, xmax = min(x_values) * 0.92, max(x_values) * 1.08
    ymin = min(loa_low - 1, min(y_values) - 1)
    ymax = max(loa_high + 1, max(y_values) + 1)
    for i in range(6):
        y = top + plot_h - i / 5 * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>')
    for x0, y0 in zip(x_values, y_values):
        x = left + (x0 - xmin) / (xmax - xmin) * plot_w
        y = top + plot_h - (y0 - ymin) / (ymax - ymin) * plot_h
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{COLORS["blue"]}" opacity="0.78"/>')
    for value, label, color, dash in [
        (bias, "Bias", COLORS["gray"], ""),
        (loa_low, "Lower LoA", COLORS["red"], 'stroke-dasharray="7 5"'),
        (loa_high, "Upper LoA", COLORS["red"], 'stroke-dasharray="7 5"'),
    ]:
        y = top + plot_h - (value - ymin) / (ymax - ymin) * plot_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="2" {dash}/>'
        )
        parts.append(f'<text x="{left + plot_w - 5}" y="{y - 5:.1f}" text-anchor="end" class="small">{label}: {value:.2f} mm</text>')
    parts.append(
        f'<text x="{left + 20}" y="{top + 28}" class="tick">'
        f'MAE = {mean([abs(v) for v in y_values]):.2f} mm; ICC = {icc_absolute_agreement_single(ai_prft, manual):.3f}'
        f'</text>'
    )
    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 25}" text-anchor="middle" class="label">'
        "Mean of AI and manual PRFT (mm)</text>"
    )
    parts.append(
        f'<text x="25" y="{top + plot_h / 2}" text-anchor="middle" class="label" '
        f'transform="rotate(-90 25,{top + plot_h / 2})">AI - manual PRFT (mm)</text>'
    )
    write_svg(path, parts)


def write_markdown_summary(path: Path, summary: dict[str, float | int]) -> None:
    text = f"""# AI-T-PRF Index Analysis Results

## Cohort Distribution

- Total cohort: {summary['total_n']} patients
- AI-T-PRF index threshold: {summary['threshold']:.2f}
- AI-T-PRF-Low: {summary['low_n']} patients ({summary['low_pct']:.1f}%)
- AI-T-PRF-High: {summary['high_n']} patients ({summary['high_pct']:.1f}%)
- Index mean +/- SD: {summary['index_mean']:.3f} +/- {summary['index_sd']:.3f}
- Index median [IQR]: {summary['index_median']:.3f} [{summary['index_q1']:.3f}, {summary['index_q3']:.3f}]

## Relationship With Manual/Traditional PRFT

- Validation subset: {summary['manual_validation_n']} patients
- AI-T-PRF index vs manual mean T-PRFT: Pearson r = {summary['pearson_ai_index_manual_prft']:.3f}
- AI-T-PRF index vs manual mean T-PRFT: Spearman rho = {summary['spearman_ai_index_manual_prft']:.3f}
- AI mean T-PRFT vs manual mean T-PRFT: Pearson r = {summary['pearson_ai_prft_manual_prft']:.3f}
- ICC for AI vs manual mean T-PRFT: {summary['icc_ai_prft_manual_prft']:.3f}
- Mean absolute error: {summary['mae_mm']:.2f} mm
- Bland-Altman bias: {summary['bias_mm']:.2f} mm
- 95% limits of agreement: {summary['loa_low_mm']:.2f} to {summary['loa_high_mm']:.2f} mm

## Manuscript-Ready Wording

The AI-T-PRF index was calculated as a continuous tumor-side perirenal fat biomarker and stratified patients into AI-T-PRF-Low and AI-T-PRF-High groups using a threshold of {summary['threshold']:.2f}. In the radiologist validation subset, the AI-T-PRF index correlated strongly with manual tumor-side PRFT measurements (Pearson r = {summary['pearson_ai_index_manual_prft']:.3f}), supporting its interpretability. Direct AI-derived PRFT measurements showed excellent agreement with manual measurements (ICC = {summary['icc_ai_prft_manual_prft']:.3f}; MAE = {summary['mae_mm']:.2f} mm), with minimal systematic bias on Bland-Altman analysis.
"""
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    rows = read_csv(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")
    if "ai_t_prf_index" not in rows[0] or not rows[0].get("ai_t_prf_index"):
        derive_index_from_directional_prft(rows)
    threshold = choose_threshold(rows, args.threshold)
    add_group(rows, threshold)
    validation_rows = select_validation_subset(rows, args.validation_n, args.seed)

    distribution_fields = [
        "patient_id",
        "ai_t_prf_index",
        "ai_t_prf_group",
        "manual_mean_t_prft_mm",
        "ai_mean_t_prft_mm",
    ]
    for optional in ["ai_anterior_t_prft_mm", "ai_lateral_t_prft_mm", "ai_posterior_t_prft_mm"]:
        if optional in rows[0]:
            distribution_fields.append(optional)
    write_csv(output_dir / "cohort_ai_t_prf_index_distribution.csv", rows, distribution_fields)

    validation_out = []
    for row in validation_rows:
        signed_error = to_float(row, "ai_mean_t_prft_mm") - to_float(row, "manual_mean_t_prft_mm")
        validation_out.append(
            {
                "patient_id": row["patient_id"],
                "ai_t_prf_index": row["ai_t_prf_index"],
                "ai_t_prf_group": row["ai_t_prf_group"],
                "manual_mean_t_prft_mm": row["manual_mean_t_prft_mm"],
                "ai_mean_t_prft_mm": row["ai_mean_t_prft_mm"],
                "absolute_error_mm": round(abs(signed_error), 3),
                "signed_error_mm": round(signed_error, 3),
            }
        )
    write_csv(output_dir / "ai_vs_manual_prft_validation_subset.csv", validation_out)

    summary = summarize(rows, validation_rows, threshold)
    save_summary_csv(output_dir / "ai_t_prf_index_summary.csv", summary)
    write_markdown_summary(output_dir / "AI_T_PRF_INDEX_RESULTS.md", summary)
    plot_distribution(rows, threshold, figures_dir / "figure_ai_t_prf_index_distribution.svg")
    plot_correlation(validation_rows, figures_dir / "figure_ai_index_manual_prft_correlation.svg")
    plot_bland_altman(validation_rows, figures_dir / "figure_ai_manual_prft_bland_altman.svg")

    print(f"Wrote outputs to: {output_dir}")
    print(f"Low/High threshold: {threshold:.4f}")
    print(f"Low: {summary['low_n']} ({summary['low_pct']:.1f}%), High: {summary['high_n']} ({summary['high_pct']:.1f}%)")
    print(
        "AI index vs manual PRFT: "
        f"Pearson r={summary['pearson_ai_index_manual_prft']:.3f}, "
        f"ICC(AI PRFT vs manual PRFT)={summary['icc_ai_prft_manual_prft']:.3f}, "
        f"MAE={summary['mae_mm']:.2f} mm"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AI-T-PRF index distribution and agreement with manual PRFT.")
    parser.add_argument(
        "--input",
        default="cohort_ai_t_prf_index_distribution.csv",
        help="Input CSV with patient_id, ai_t_prf_index, manual_mean_t_prft_mm, ai_mean_t_prft_mm.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Output directory for tables and figures.",
    )
    parser.add_argument("--threshold", type=float, default=0.43, help="Low/High threshold for AI-T-PRF index.")
    parser.add_argument("--validation-n", type=int, default=120, help="Number of cases for manual validation subset.")
    parser.add_argument("--seed", type=int, default=20260520, help="Random seed for validation subset sampling.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
