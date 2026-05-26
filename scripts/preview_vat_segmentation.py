"""
Create QC previews for fat segmentation.

Expected labels:
    1 subcutaneous fat (SAT)
    2 visceral adipose tissue (VAT)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


def load(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj)


def orient_axial(arr: np.ndarray) -> np.ndarray:
    # Most local NIfTI examples are LPS/RAS-like axial arrays; transpose gives
    # the familiar axial display with anterior at top.
    return arr.T


def normalize_ct(slice_2d: np.ndarray, hu_min: float = -200.0, hu_max: float = 250.0) -> np.ndarray:
    arr = np.clip(slice_2d.astype(np.float32), hu_min, hu_max)
    arr = (arr - hu_min) / max(hu_max - hu_min, 1e-6)
    return (arr * 255).astype(np.uint8)


def overlay(gray: np.ndarray, sat: np.ndarray, vat: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    if sat.any():
        rgb[sat] = 0.55 * rgb[sat] + 0.45 * np.array([80, 180, 255], dtype=np.float32)
    if vat.any():
        rgb[vat] = 0.50 * rgb[vat] + 0.50 * np.array([233, 196, 106], dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def choose_slices(pred: np.ndarray, n: int) -> list[int]:
    scores = []
    for z in range(pred.shape[2]):
        vat = int((pred[:, :, z] == 2).sum())
        sat = int((pred[:, :, z] == 1).sum())
        score = 3 * vat + sat
        if score > 0:
            scores.append((score, z))
    scores.sort(reverse=True)
    return sorted(z for _, z in scores[:n])


def panel(case_id: str, z: int, ct: np.ndarray, pred: np.ndarray, scale: int) -> Image.Image:
    gray = normalize_ct(orient_axial(ct[:, :, z]))
    lab = orient_axial(pred[:, :, z])
    img = Image.fromarray(overlay(gray, lab == 1, lab == 2))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 62), fill=(0, 0, 0))
    draw.text((10, 8), f"{case_id} z={z}", fill=(255, 255, 255))
    draw.rectangle((10, 36, 26, 52), fill=(80, 180, 255))
    draw.text((32, 34), "SAT", fill=(255, 255, 255))
    draw.rectangle((92, 36, 108, 52), fill=(233, 196, 106))
    draw.text((114, 34), "VAT", fill=(255, 255, 255))
    return img


def save_case(case_id: str, ct_path: str, pred_path: str, out_dir: Path, max_slices: int, scale: int) -> list[dict[str, str]]:
    ct = load(ct_path)
    pred = load(pred_path).astype(np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    panels = []
    for z in choose_slices(pred, max_slices):
        img = panel(case_id, z, ct, pred, scale)
        out_path = out_dir / f"{case_id}_z{z:03d}_vat_qc.png"
        img.save(out_path)
        panels.append(img)
        rows.append(
            {
                "case_id": case_id,
                "slice": str(z),
                "sat_voxels": str(int((pred[:, :, z] == 1).sum())),
                "vat_voxels": str(int((pred[:, :, z] == 2).sum())),
                "preview_path": str(out_path),
            }
        )
    if panels:
        w = max(img.width for img in panels)
        h = max(img.height for img in panels)
        sheet = Image.new("RGB", (w * len(panels), h), (0, 0, 0))
        for i, img in enumerate(panels):
            sheet.paste(img, (i * w, 0))
        sheet_path = out_dir / f"{case_id}_vat_contact_sheet.png"
        sheet.save(sheet_path)
        for row in rows:
            row["contact_sheet"] = str(sheet_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-csv", required=True, help="CSV with patient_id,hu_path,pred_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-slices", type=int, default=5)
    parser.add_argument("--scale", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    rows = []
    with open(args.cases_csv, "r", encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            rows.extend(
                save_case(
                    row["patient_id"],
                    row["hu_path"],
                    row["pred_path"],
                    out_dir / row["patient_id"],
                    args.max_slices,
                    args.scale,
                )
            )
    summary_path = out_dir / "vat_preview_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} previews to {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
