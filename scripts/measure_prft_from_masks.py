"""
Measure tumor-side, opposite-side, and bilateral perirenal fat thickness (PRFT)
from AI segmentation masks.

This script is the upstream measurement step for the AI-T-PRF index model.
It converts image-level AI outputs into per-patient directional PRFT features:

    ai_t_anterior_prft_mm / ai_o_anterior_prft_mm / ai_b_anterior_prft_mm
    ai_t_lateral_prft_mm  / ai_o_lateral_prft_mm  / ai_b_lateral_prft_mm
    ai_t_posterior_prft_mm / ai_o_posterior_prft_mm / ai_b_posterior_prft_mm
    ai_t_mean_prft_mm / ai_o_mean_prft_mm / ai_b_mean_prft_mm

Recommended pipeline
--------------------
    CT + kidney/tumor/renal-vein/fat masks
      -> measure_prft_from_masks.py
      -> ai_t_prf_feature_input.csv
      -> generate_ai_t_prf_index_table.py
      -> cohort_ai_t_prf_index_distribution.csv

Key assumptions
---------------
1. The renal vein mask identifies the axial measurement plane.
   By default, the script chooses the axial slice with the largest renal vein
   mask area. If a tumor side is provided, and the mask contains both sides,
   the script can restrict renal-vein voxels to the tumor-side half of the image.

2. The kidney/tumor mask identifies the inner boundary.
   The organ mask is the union of kidney and tumor masks on the selected slice.

3. The fat mask identifies the measurable adipose compartment.
   If a fat mask is not provided, a HU threshold on CT can be used instead.

4. Directional PRFT is computed by casting rays from the organ centroid outward.
   For robustness, each reported direction is the median distance across a
   small angular window rather than a single ray.

Inputs
------
Required:
    --patient-id
    --ct
    --kidney-mask
    --tumor-side left|right

At least one fat source:
    --fat-mask
    or --hu-fat-from-ct

Recommended:
    --vein-mask
    --tumor-mask

Outputs
-------
    CSV with one row:
        patient_id
        selected_slice
        tumor_side
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

Example
-------
python measure_prft_from_masks.py ^
  --patient-id TCGA-B0-4698 ^
  --ct TCGA-B0-4698_hu.nii.gz ^
  --kidney-mask kidney_pred.nii.gz ^
  --tumor-mask tumor_pred.nii.gz ^
  --vein-mask vein_pred.nii.gz ^
  --fat-mask fat_pred.nii.gz ^
  --tumor-side left ^
  --output ai_t_prf_feature_input.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from pathlib import Path

import numpy as np


def require_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "nibabel is required to read NIfTI files. Install it in the active Python environment "
            "or run this script with the project environment that has medical imaging dependencies."
        ) from exc
    return nib


def load_nifti(path: str | None):
    if path is None:
        return None, None
    nib = require_nibabel()
    img = nib.load(path)
    return np.asarray(img.dataobj), img


def spacing_xy_from_img(img) -> tuple[float, float]:
    zooms = img.header.get_zooms()
    if len(zooms) < 2:
        return 1.0, 1.0
    return float(zooms[0]), float(zooms[1])


def ensure_3d(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a 3D volume, got shape {arr.shape}")
    return arr


def as_binary(arr: np.ndarray, label: int | None = None) -> np.ndarray:
    if label is None:
        return arr > 0
    return arr == label


def side_mask(shape_2d: tuple[int, int], side: str) -> np.ndarray:
    h, w = shape_2d
    mask = np.zeros((h, w), dtype=bool)
    if side == "left":
        mask[:, : w // 2] = True
    elif side == "right":
        mask[:, w // 2 :] = True
    else:
        raise ValueError("--tumor-side must be left or right")
    return mask


def opposite_side(side: str) -> str:
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    raise ValueError("--tumor-side must be left or right")


def choose_slice_from_vein(vein: np.ndarray, tumor_side: str | None = None, vein_label: int | None = None) -> int:
    vein = as_binary(vein, vein_label)
    scores = []
    for z in range(vein.shape[2]):
        sl = vein[:, :, z]
        if tumor_side in {"left", "right"}:
            sl = sl & side_mask(sl.shape, tumor_side)
        scores.append(int(sl.sum()))
    if max(scores) == 0:
        raise ValueError("Vein mask is empty; cannot determine renal venous plane")
    return int(np.argmax(scores))


def choose_slice_from_organ(organ: np.ndarray, tumor_side: str) -> int:
    scores = []
    for z in range(organ.shape[2]):
        sl = organ[:, :, z] & side_mask(organ[:, :, z].shape, tumor_side)
        scores.append(int(sl.sum()))
    if max(scores) == 0:
        raise ValueError("Organ mask is empty on tumor side; cannot choose fallback measurement plane")
    return int(np.argmax(scores))


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    visited = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if len(comp) > len(best):
                best = comp
    out = np.zeros(mask.shape, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Cannot compute centroid of empty mask")
    return float(ys.mean()), float(xs.mean())


def ray_distance_mm(
    organ: np.ndarray,
    fat: np.ndarray,
    center_yx: tuple[float, float],
    angle_deg: float,
    spacing_xy: tuple[float, float],
    max_steps: int = 800,
) -> float | None:
    """Distance from organ boundary to the outer fat boundary along one ray."""
    cy, cx = center_yx
    theta = math.radians(angle_deg)
    # Image coordinates: x rightward, y downward.
    dx = math.cos(theta)
    dy = math.sin(theta)
    h, w = organ.shape
    seen_organ = False
    in_fat_after_organ = False
    fat_start_step: int | None = None
    last_fat_step: int | None = None
    for step in range(max_steps):
        y = int(round(cy + dy * step))
        x = int(round(cx + dx * step))
        if y < 0 or y >= h or x < 0 or x >= w:
            break
        if organ[y, x]:
            seen_organ = True
            in_fat_after_organ = False
            fat_start_step = None
            last_fat_step = None
            continue
        if not seen_organ:
            continue
        if fat[y, x]:
            if not in_fat_after_organ:
                fat_start_step = step
                in_fat_after_organ = True
            last_fat_step = step
            continue
        if in_fat_after_organ:
            break
    if fat_start_step is None or last_fat_step is None or last_fat_step <= fat_start_step:
        return None
    sx, sy = spacing_xy
    step_mm = math.sqrt((dx * sx) ** 2 + (dy * sy) ** 2)
    return (last_fat_step - fat_start_step + 1) * step_mm


def median_direction_distance(
    organ: np.ndarray,
    fat: np.ndarray,
    center_yx: tuple[float, float],
    base_angle: float,
    spacing_xy: tuple[float, float],
    angle_window: float,
    angle_step: float,
) -> float | None:
    values = []
    n_steps = int(round((2 * angle_window) / angle_step))
    for idx in range(n_steps + 1):
        angle = base_angle - angle_window + idx * angle_step
        value = ray_distance_mm(organ, fat, center_yx, angle, spacing_xy)
        if value is not None and value > 0:
            values.append(value)
    if not values:
        return None
    return float(np.median(values))


def build_fat_mask(ct: np.ndarray, fat_mask: np.ndarray | None, hu_min: float, hu_max: float) -> np.ndarray:
    if fat_mask is not None:
        return as_binary(fat_mask)
    return (ct >= hu_min) & (ct <= hu_max)


def measure_side_prft(
    organ: np.ndarray,
    fat: np.ndarray,
    selected_slice: int,
    side: str,
    spacing_xy: tuple[float, float],
    args: argparse.Namespace,
) -> dict[str, float | None]:
    organ_slice = organ[:, :, selected_slice] & side_mask(organ[:, :, selected_slice].shape, side)
    organ_slice = largest_component(organ_slice)
    fat_slice = fat[:, :, selected_slice] & side_mask(fat[:, :, selected_slice].shape, side)
    if organ_slice.sum() == 0:
        if args.allow_missing:
            return {"anterior": None, "lateral": None, "posterior": None}
        raise ValueError(f"Empty {side}-side organ mask on selected slice {selected_slice}")
    center = centroid(organ_slice)
    lateral_angle = 180.0 if side == "left" else 0.0
    angles = {
        "anterior": float(args.anterior_angle),
        "lateral": lateral_angle,
        "posterior": float(args.posterior_angle),
    }
    distances: dict[str, float | None] = {}
    for name, angle in angles.items():
        distances[name] = median_direction_distance(
            organ_slice,
            fat_slice,
            center,
            angle,
            spacing_xy,
            args.angle_window,
            args.angle_step,
        )
    missing = [name for name, value in distances.items() if value is None]
    if missing and not args.allow_missing:
        raise ValueError(
            f"Could not measure {side}-side PRFT for directions {missing} on slice {selected_slice}. "
            "Check masks, side orientation, or pass --allow-missing."
        )
    return distances


def mean_available(values: list[float | None]) -> float:
    numeric = [value for value in values if value is not None and not math.isnan(value)]
    return float(np.mean(numeric)) if numeric else float("nan")


def measure_prft(args: argparse.Namespace) -> dict[str, str]:
    ct, ct_img = load_nifti(args.ct)
    kidney, _ = load_nifti(args.kidney_mask)
    tumor, _ = load_nifti(args.tumor_mask)
    vein, _ = load_nifti(args.vein_mask)
    fat_mask, _ = load_nifti(args.fat_mask)
    ct = ensure_3d(ct, "ct")
    kidney = ensure_3d(kidney, "kidney_mask")
    tumor_bin = np.zeros(kidney.shape, dtype=bool) if tumor is None else as_binary(ensure_3d(tumor, "tumor_mask"), args.tumor_label)
    kidney_bin = as_binary(kidney, args.kidney_label)
    organ = kidney_bin | tumor_bin
    fat = build_fat_mask(ct, None if fat_mask is None else ensure_3d(fat_mask, "fat_mask"), args.hu_min, args.hu_max)
    if vein is not None:
        selected_slice = choose_slice_from_vein(ensure_3d(vein, "vein_mask"), args.tumor_side, args.vein_label)
    elif args.slice_index is not None:
        selected_slice = args.slice_index
    else:
        selected_slice = choose_slice_from_organ(organ, args.tumor_side)
    spacing_xy = spacing_xy_from_img(ct_img)
    # Angles in image coordinates. Assumption: y increases anterior-to-posterior is scanner-dependent.
    # These labels are radiology-facing and should be confirmed during validation.
    tumor_dist = measure_side_prft(organ, fat, selected_slice, args.tumor_side, spacing_xy, args)
    opp_side = opposite_side(args.tumor_side)
    opp_dist = measure_side_prft(organ, fat, selected_slice, opp_side, spacing_xy, args)
    bilateral_dist = {
        key: mean_available([tumor_dist[key], opp_dist[key]])
        for key in ("anterior", "lateral", "posterior")
    }
    tumor_mean = mean_available(list(tumor_dist.values()))
    opp_mean = mean_available(list(opp_dist.values()))
    bilateral_mean = mean_available([tumor_mean, opp_mean])
    def fmt(value: float | None) -> str:
        return "" if value is None or math.isnan(value) else f"{value:.2f}"
    return {
        "patient_id": args.patient_id,
        "selected_slice": str(selected_slice),
        "tumor_side": args.tumor_side,
        "opposite_side": opp_side,
        "ai_t_anterior_prft_mm": fmt(tumor_dist["anterior"]),
        "ai_t_lateral_prft_mm": fmt(tumor_dist["lateral"]),
        "ai_t_posterior_prft_mm": fmt(tumor_dist["posterior"]),
        "ai_t_mean_prft_mm": fmt(tumor_mean),
        "ai_o_anterior_prft_mm": fmt(opp_dist["anterior"]),
        "ai_o_lateral_prft_mm": fmt(opp_dist["lateral"]),
        "ai_o_posterior_prft_mm": fmt(opp_dist["posterior"]),
        "ai_o_mean_prft_mm": fmt(opp_mean),
        "ai_b_anterior_prft_mm": fmt(bilateral_dist["anterior"]),
        "ai_b_lateral_prft_mm": fmt(bilateral_dist["lateral"]),
        "ai_b_posterior_prft_mm": fmt(bilateral_dist["posterior"]),
        "ai_b_mean_prft_mm": fmt(bilateral_mean),
    }


def append_or_write_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
        "patient_id",
        "selected_slice",
        "tumor_side",
        "opposite_side",
        "ai_t_anterior_prft_mm",
        "ai_t_lateral_prft_mm",
        "ai_t_posterior_prft_mm",
        "ai_t_mean_prft_mm",
        "ai_o_anterior_prft_mm",
        "ai_o_lateral_prft_mm",
        "ai_o_posterior_prft_mm",
        "ai_o_mean_prft_mm",
        "ai_b_anterior_prft_mm",
        "ai_b_lateral_prft_mm",
        "ai_b_posterior_prft_mm",
        "ai_b_mean_prft_mm",
    ]
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure directional tumor-side PRFT from AI masks.")
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--ct", required=True, help="CT NIfTI path.")
    parser.add_argument("--kidney-mask", required=True, help="Kidney mask NIfTI path.")
    parser.add_argument("--tumor-mask", default=None, help="Optional tumor mask NIfTI path.")
    parser.add_argument("--vein-mask", default=None, help="Optional renal vein mask NIfTI path for plane selection.")
    parser.add_argument("--fat-mask", default=None, help="Optional fat mask NIfTI path. If omitted, HU threshold is used.")
    parser.add_argument("--tumor-side", required=True, choices=["left", "right"])
    parser.add_argument("--output", required=True, help="Output CSV path; rows are appended.")
    parser.add_argument("--slice-index", type=int, default=None, help="Manual fallback slice index.")
    parser.add_argument("--kidney-label", type=int, default=None, help="Optional kidney label value in mask.")
    parser.add_argument("--tumor-label", type=int, default=None, help="Optional tumor label value in mask.")
    parser.add_argument("--vein-label", type=int, default=None, help="Optional renal vein label value in multiclass vein mask.")
    parser.add_argument("--hu-min", type=float, default=-190.0)
    parser.add_argument("--hu-max", type=float, default=-30.0)
    parser.add_argument("--anterior-angle", type=float, default=-90.0)
    parser.add_argument("--posterior-angle", type=float, default=90.0)
    parser.add_argument("--angle-window", type=float, default=20.0)
    parser.add_argument("--angle-step", type=float, default=2.0)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = measure_prft(args)
    append_or_write_csv(Path(args.output), row)
    print(row)


if __name__ == "__main__":
    main()
