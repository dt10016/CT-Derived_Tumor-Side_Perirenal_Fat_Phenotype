"""
Create radiology-oriented QC previews for KIPA kidney/vessel/tumor masks.

Label convention:
    1 artery, 2 kidney, 3 renal vein, 4 tumor

Display convention:
    axial radiology view for LPS NIfTI data: top anterior, bottom posterior,
    image left patient right, image right patient left.
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
    return arr.T


def normalize_ct(slice_2d: np.ndarray) -> np.ndarray:
    arr = slice_2d.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def overlay(gray: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    for cls, color in COLORS.items():
        region = labels == cls
        if region.any():
            rgb[region] = (1 - alpha) * rgb[region] + alpha * np.array(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def choose_slices(mask: np.ndarray, max_slices: int) -> list[int]:
    scores = []
    for z in range(mask.shape[2]):
        sl = mask[:, :, z]
        tumor = int((sl == 4).sum())
        vein = int((sl == 3).sum())
        kidney = int((sl == 2).sum())
        artery = int((sl == 1).sum())
        score = 8 * tumor + 5 * vein + 2 * kidney + artery
        if score > 0:
            scores.append((score, z))
    scores.sort(reverse=True)
    return sorted(z for _, z in scores[:max_slices])


def label_panel(img: Image.Image, title: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 44), fill=(0, 0, 0))
    draw.text((8, 7), title, fill=(255, 255, 255))
    x = 8
    for cls in (1, 2, 3, 4):
        draw.rectangle((x, 26, x + 12, 38), fill=COLORS[cls])
        draw.text((x + 16, 24), NAMES[cls], fill=(255, 255, 255))
        x += 92
    return img


def save_case(
    case_id: str,
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    max_slices: int,
    scale: int,
) -> list[dict[str, str]]:
    image = load(image_path)
    mask = load(mask_path).astype(np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for z in choose_slices(mask, max_slices):
        gray = normalize_ct(orient_axial(image[:, :, z]))
        label = orient_axial(mask[:, :, z])
        panel = label_panel(Image.fromarray(overlay(gray, label)), f"{case_id} z={z}")
        if scale > 1:
            panel = panel.resize((panel.width * scale, panel.height * scale), Image.Resampling.NEAREST)
        out_path = output_dir / f"{case_id}_z{z:03d}_kipa_mask_qc.png"
        panel.save(out_path)
        rows.append(
            {
                "case_id": case_id,
                "slice": str(z),
                "preview_path": str(out_path),
                "artery_voxels": str(int((mask[:, :, z] == 1).sum())),
                "kidney_voxels": str(int((mask[:, :, z] == 2).sum())),
                "vein_voxels": str(int((mask[:, :, z] == 3).sum())),
                "tumor_voxels": str(int((mask[:, :, z] == 4).sum())),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview KIPA masks in radiology orientation.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--mask-suffix", default=".nii.gz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-slices", type=int, default=4)
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    rows = []
    for case_id in args.case_ids:
        image_path = image_dir / f"{case_id}.nii.gz"
        mask_path = mask_dir / f"{case_id}{args.mask_suffix}"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        rows.extend(save_case(case_id, image_path, mask_path, output_dir / case_id, args.max_slices, args.scale))

    summary_path = output_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} previews to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
