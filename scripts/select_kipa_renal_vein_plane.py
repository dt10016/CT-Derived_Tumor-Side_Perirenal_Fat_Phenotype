"""
Select renal vein plane from KIPA manual labels and create QC previews.

KIPA label convention:
    1 artery, 2 kidney, 3 renal vein, 4 tumor

The selected plane is the axial slice with the largest renal-vein label area.
QC contact sheets include neighboring slices so a radiologist can judge whether
the selected plane matches the intended renal venous level.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


COLORS = {
    1: (255, 80, 80),
    2: (80, 220, 120),
    3: (80, 180, 255),
    4: (255, 200, 80),
}

NAMES = {
    1: "artery",
    2: "kidney",
    3: "vein",
    4: "tumor",
}


def load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def orient_axial(arr: np.ndarray) -> np.ndarray:
    # Current KIPA NIfTI examples are LPS. Transpose gives radiology axial view:
    # top anterior, bottom posterior, image left patient right.
    return arr.T


def normalize_ct(slice_2d: np.ndarray) -> np.ndarray:
    arr = slice_2d.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def overlay(gray: np.ndarray, labels: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    for cls, color in COLORS.items():
        region = labels == cls
        if region.any():
            rgb[region] = (1 - alpha) * rgb[region] + alpha * np.array(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def tumor_side(label: np.ndarray) -> str:
    tumor = label == 4
    mid = label.shape[0] // 2
    right_voxels = int(tumor[:mid, :, :].sum())
    left_voxels = int(tumor[mid:, :, :].sum())
    if left_voxels == right_voxels:
        return "unknown"
    return "left" if left_voxels > right_voxels else "right"


def choose_vein_slice(label: np.ndarray, side: str | None) -> tuple[int, int]:
    vein = label == 3
    if side in {"left", "right"}:
        mid = label.shape[0] // 2
        side_mask = np.zeros(label.shape[:2], dtype=bool)
        if side == "left":
            side_mask[mid:, :] = True
        else:
            side_mask[:mid, :] = True
        vein = vein & side_mask[:, :, None]
    areas = np.array([int(vein[:, :, z].sum()) for z in range(label.shape[2])])
    if int(areas.max()) == 0:
        raise ValueError("No renal vein label found")
    z = int(areas.argmax())
    return z, int(areas[z])


def draw_panel(case_id: str, z: int, image_slice: np.ndarray, label_slice: np.ndarray, selected_z: int, scale: int) -> Image.Image:
    gray = normalize_ct(orient_axial(image_slice))
    lab = orient_axial(label_slice)
    panel = Image.fromarray(overlay(gray, lab))
    if scale > 1:
        panel = panel.resize((panel.width * scale, panel.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(panel)
    title_color = (255, 230, 120) if z == selected_z else (255, 255, 255)
    draw.rectangle((0, 0, panel.width, 72), fill=(0, 0, 0))
    draw.text((10, 8), f"{case_id} z={z}" + ("  SELECTED" if z == selected_z else ""), fill=title_color)
    x = 10
    for cls in (1, 2, 3, 4):
        draw.rectangle((x, 42, x + 14, 56), fill=COLORS[cls])
        draw.text((x + 18, 39), NAMES[cls], fill=(255, 255, 255))
        x += 90
    return panel


def save_case(case_id: str, image_path: Path, label_path: Path, output_dir: Path, window: int, scale: int, side_mode: str) -> dict[str, str | int]:
    image = load(image_path)
    label = load(label_path).astype(np.uint8)
    inferred_side = tumor_side(label)
    side = inferred_side if side_mode == "tumor-side" else None
    selected_z, selected_area = choose_vein_slice(label, side)
    z0 = max(0, selected_z - window)
    z1 = min(label.shape[2] - 1, selected_z + window)
    panels = [
        draw_panel(case_id, z, image[:, :, z], label[:, :, z], selected_z, scale)
        for z in range(z0, z1 + 1)
    ]
    w = max(panel.width for panel in panels)
    h = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (w * len(panels), h), (0, 0, 0))
    for i, panel in enumerate(panels):
        sheet.paste(panel, (i * w, 0))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{case_id}_renal_vein_plane_contact.png"
    sheet.save(out_path)
    return {
        "case_id": case_id,
        "tumor_side": inferred_side,
        "selection_mode": side_mode,
        "selected_slice": selected_z,
        "selected_vein_voxels": selected_area,
        "preview": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--side-mode", choices=["all-vein", "tumor-side"], default="all-vein")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = []
    for case_id in args.case_ids:
        rows.append(
            save_case(
                case_id,
                Path(args.image_dir) / f"{case_id}.nii.gz",
                Path(args.label_dir) / f"{case_id}.nii.gz",
                output_dir / case_id,
                args.window,
                args.scale,
                args.side_mode,
            )
        )
        print(rows[-1])

    summary_path = output_dir / f"renal_vein_plane_summary_{args.side_mode}.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
