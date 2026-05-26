"""
Create visual previews for PRAT extraction quality control.

Panels per case:
    1. CT + kidney/tumor mask
    2. CT + VAT mask
    3. CT + tumor-side PRAT
    4. CT + opposite-side PRAT
    5. CT + bilateral PRAT

The script selects representative axial slices based on kidney/tumor and PRAT
area, so QC previews stay near the kidney instead of drifting to unrelated VAT.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


def load_img(path: str):
    return nib.load(path)


def load(path: str) -> np.ndarray:
    return np.asarray(load_img(path).dataobj)


def orient_axial(slice_2d: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return slice_2d
    if mode == "radiology":
        # For common LPS axial NIfTI data: display rows as anterior->posterior
        # and columns as patient right->left.
        return slice_2d.T
    raise ValueError(f"Unknown display orientation: {mode}")


def norm_ct(slice_2d: np.ndarray, hu_min: float = -150.0, hu_max: float = 250.0) -> np.ndarray:
    arr = np.clip(slice_2d.astype(np.float32), hu_min, hu_max)
    arr = (arr - hu_min) / max(hu_max - hu_min, 1e-6)
    return (arr * 255).astype(np.uint8)


def overlay(gray: np.ndarray, masks: list[tuple[np.ndarray, tuple[int, int, int], float]]) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    for mask, color, alpha in masks:
        region = mask.astype(bool)
        if region.any():
            color_arr = np.array(color, dtype=np.float32)
            rgb[region] = (1 - alpha) * rgb[region] + alpha * color_arr
    return np.clip(rgb, 0, 255).astype(np.uint8)


def label_panel(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 28), fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255))
    return img


def choose_slices(organ: np.ndarray, prat: np.ndarray, n: int, mode: str) -> list[int]:
    organ_area = np.array([int(organ[:, :, z].sum()) for z in range(organ.shape[2])], dtype=np.float64)
    prat_area = np.array([int(prat[:, :, z].sum()) for z in range(prat.shape[2])], dtype=np.float64)
    if mode == "organ":
        score_arr = organ_area
    elif mode == "prat":
        score_arr = prat_area
    else:
        organ_norm = organ_area / max(float(organ_area.max()), 1.0)
        prat_norm = prat_area / max(float(prat_area.max()), 1.0)
        score_arr = 0.7 * organ_norm + 0.3 * prat_norm
        score_arr[organ_area == 0] = 0
    scores = [(float(score_arr[z]), z) for z in range(prat.shape[2])]
    scores = [(score, z) for score, z in scores if score > 0]
    scores.sort(reverse=True)
    return sorted(z for _, z in scores[:n])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create PRAT overlay previews.")
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--ct", required=True)
    parser.add_argument("--organ-mask", required=True, help="Multiclass organ/vessel prediction.")
    parser.add_argument("--vat-mask", required=True)
    parser.add_argument("--t-prat", required=True)
    parser.add_argument("--o-prat", required=True)
    parser.add_argument("--b-prat", required=True)
    parser.add_argument("--kidney-label", type=int, default=2)
    parser.add_argument("--tumor-label", type=int, default=4)
    parser.add_argument("--vat-label", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-slices", type=int, default=4)
    parser.add_argument("--slice-mode", choices=["combined", "organ", "prat"], default="combined")
    parser.add_argument("--display-orientation", choices=["radiology", "raw"], default="radiology")
    args = parser.parse_args()

    ct = load(args.ct)
    organ = load(args.organ_mask)
    vat = load(args.vat_mask) == args.vat_label
    t_prat = load(args.t_prat) > 0
    o_prat = load(args.o_prat) > 0
    b_prat = load(args.b_prat) > 0
    kidney = organ == args.kidney_label
    tumor = organ == args.tumor_label
    organ_mask = kidney | tumor

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slices = choose_slices(organ_mask, b_prat, args.max_slices, args.slice_mode)
    if not slices:
        raise ValueError("No PRAT voxels found for preview")

    summary = ["patient_id,slice,preview_path"]
    for z in slices:
        gray = norm_ct(orient_axial(ct[:, :, z], args.display_orientation))
        panels = [
            (
                "Kidney/Tumor",
                overlay(
                    gray,
                    [
                        (orient_axial(kidney[:, :, z], args.display_orientation), (80, 220, 120), 0.45),
                        (orient_axial(tumor[:, :, z], args.display_orientation), (255, 200, 80), 0.55),
                    ],
                ),
            ),
            ("VAT", overlay(gray, [(orient_axial(vat[:, :, z], args.display_orientation), (80, 180, 255), 0.42)])),
            (
                "Tumor-side PRAT",
                overlay(gray, [(orient_axial(t_prat[:, :, z], args.display_orientation), (231, 111, 81), 0.65)]),
            ),
            (
                "Opposite-side PRAT",
                overlay(gray, [(orient_axial(o_prat[:, :, z], args.display_orientation), (42, 157, 143), 0.65)]),
            ),
            (
                "Bilateral PRAT",
                overlay(gray, [(orient_axial(b_prat[:, :, z], args.display_orientation), (233, 196, 106), 0.65)]),
            ),
        ]
        pil_panels = [label_panel(Image.fromarray(panel), f"{args.patient_id} z={z}  {name}") for name, panel in panels]
        w, h = pil_panels[0].size
        canvas = Image.new("RGB", (w * len(pil_panels), h), (255, 255, 255))
        for i, panel in enumerate(pil_panels):
            canvas.paste(panel, (i * w, 0))
        out_path = out_dir / f"{args.patient_id}_z{z:03d}_prat_preview.png"
        canvas.save(out_path)
        summary.append(f"{args.patient_id},{z},{out_path}")

    (out_dir / f"{args.patient_id}_preview_summary.csv").write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved {len(slices)} previews to {out_dir}")


if __name__ == "__main__":
    main()
