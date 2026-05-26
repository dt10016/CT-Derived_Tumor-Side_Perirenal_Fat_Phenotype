"""
Extract perirenal adipose tissue (PRAT) from VAT using kidney/tumor masks.

This script implements the segmentation post-processing idea commonly used in
clinical radiomics papers:

    CT -> AI segmentation of tumor, kidney, and VAT
       -> identify affected/tumor-side kidney
       -> create a shell around kidney/tumor by dilation/erosion
       -> intersect shell with VAT
       -> PRAT mask

The PRAT mask can then be used for:

1. Tumor-side, opposite-side, and bilateral PRF volume quantification.
2. PRFT measurement on the renal venous plane.
3. AI-T/O/B-PRF index construction.

The script uses only numpy + nibabel + scipy.ndimage.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def require_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except ImportError as exc:
        raise SystemExit("nibabel is required to read/write NIfTI files.") from exc
    return nib


def load_nifti(path: str):
    nib = require_nibabel()
    img = nib.load(path)
    return np.asarray(img.dataobj), img


def save_nifti(path: Path, mask: np.ndarray, ref_img) -> None:
    nib = require_nibabel()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(out, str(path))


def binary(arr: np.ndarray, label: int | None = None) -> np.ndarray:
    if label is None:
        return arr > 0
    return arr == label


def side_mask(shape: tuple[int, int, int], side: str) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mid = shape[0] // 2
    # The TCGA/KiTS-style NIfTI examples used here are LPS oriented: axis 0
    # increases from patient right to patient left.
    if side == "left":
        mask[mid:, :, :] = True
    elif side == "right":
        mask[:mid, :, :] = True
    else:
        raise ValueError("--tumor-side must be left or right")
    return mask


def opposite_side(side: str) -> str:
    return "right" if side == "left" else "left"


def ball_structure(radius_voxels: int) -> np.ndarray:
    r = int(radius_voxels)
    grid = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
    dist2 = sum(axis * axis for axis in grid)
    return dist2 <= r * r


def shell_around(mask: np.ndarray, outer_voxels: int, inner_voxels: int = 0) -> np.ndarray:
    outer = ndimage.binary_dilation(mask, structure=ball_structure(outer_voxels))
    if inner_voxels > 0:
        inner = ndimage.binary_erosion(mask, structure=ball_structure(inner_voxels), border_value=0)
    else:
        inner = mask
    return outer & ~inner


def keep_largest_component(mask: np.ndarray, min_voxels: int = 0) -> np.ndarray:
    labels, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = int(counts.argmax())
    if min_voxels > 0 and counts[largest] < min_voxels:
        return np.zeros_like(mask, dtype=bool)
    return labels == largest


def z_window(mask: np.ndarray, margin: int) -> np.ndarray:
    keep = np.zeros(mask.shape, dtype=bool)
    z_idx = np.flatnonzero(mask.any(axis=(0, 1)))
    if z_idx.size == 0:
        return keep
    z0 = max(int(z_idx.min()) - margin, 0)
    z1 = min(int(z_idx.max()) + margin, mask.shape[2] - 1)
    keep[:, :, z0 : z1 + 1] = True
    return keep


def volume_ml(mask: np.ndarray, img) -> float:
    zooms = img.header.get_zooms()[:3]
    voxel_ml = float(zooms[0] * zooms[1] * zooms[2]) / 1000.0
    return float(mask.sum()) * voxel_ml


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PRAT mask from VAT and kidney/tumor masks.")
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--ct", required=True, help="Reference CT NIfTI.")
    parser.add_argument("--kidney-mask", required=True)
    parser.add_argument("--vat-mask", required=True, help="VAT mask or multiclass fat mask.")
    parser.add_argument("--tumor-mask", default=None)
    parser.add_argument("--kidney-label", type=int, default=None)
    parser.add_argument("--tumor-label", type=int, default=None)
    parser.add_argument("--vat-label", type=int, default=None, help="If fat mask is multiclass, use this label as VAT.")
    parser.add_argument("--tumor-side", required=True, choices=["left", "right"])
    parser.add_argument("--outer-voxels", type=int, default=6, help="Expansion radius around kidney/tumor.")
    parser.add_argument("--inner-voxels", type=int, default=0, help="Optional erosion radius for shell inner boundary.")
    parser.add_argument(
        "--no-clean-organ-components",
        action="store_true",
        help="Disable largest connected component cleanup for each side.",
    )
    parser.add_argument(
        "--min-organ-voxels",
        type=int,
        default=1000,
        help="Minimum voxel count for a side organ component after cleanup.",
    )
    parser.add_argument(
        "--z-margin",
        type=int,
        default=2,
        help="Restrict PRAT to organ-containing axial slices plus this margin.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    _, ref_img = load_nifti(args.ct)
    kidney_arr, _ = load_nifti(args.kidney_mask)
    vat_arr, _ = load_nifti(args.vat_mask)
    tumor_arr = None
    if args.tumor_mask:
        tumor_arr, _ = load_nifti(args.tumor_mask)

    kidney = binary(kidney_arr, args.kidney_label)
    tumor = np.zeros_like(kidney, dtype=bool) if tumor_arr is None else binary(tumor_arr, args.tumor_label)
    vat = binary(vat_arr, args.vat_label)
    organ = kidney | tumor

    t_side = side_mask(organ.shape, args.tumor_side)
    o_side = side_mask(organ.shape, opposite_side(args.tumor_side))

    organ_t = organ & t_side
    organ_o = organ & o_side
    raw_t_voxels = int(organ_t.sum())
    raw_o_voxels = int(organ_o.sum())

    if not args.no_clean_organ_components:
        organ_t = keep_largest_component(organ_t, args.min_organ_voxels)
        organ_o = keep_largest_component(organ_o, args.min_organ_voxels)

    shell_t = shell_around(organ_t, args.outer_voxels, args.inner_voxels)
    shell_o = shell_around(organ_o, args.outer_voxels, args.inner_voxels)
    if args.z_margin >= 0:
        shell_t &= z_window(organ_t, args.z_margin)
        shell_o &= z_window(organ_o, args.z_margin)
    prat_t = shell_t & vat
    prat_o = shell_o & vat
    prat_b = prat_t | prat_o

    output_dir = Path(args.output_dir)
    save_nifti(output_dir / f"{args.patient_id}_t_clean_organ_mask.nii.gz", organ_t, ref_img)
    save_nifti(output_dir / f"{args.patient_id}_o_clean_organ_mask.nii.gz", organ_o, ref_img)
    save_nifti(output_dir / f"{args.patient_id}_b_clean_organ_mask.nii.gz", organ_t | organ_o, ref_img)
    save_nifti(output_dir / f"{args.patient_id}_t_prat_mask.nii.gz", prat_t, ref_img)
    save_nifti(output_dir / f"{args.patient_id}_o_prat_mask.nii.gz", prat_o, ref_img)
    save_nifti(output_dir / f"{args.patient_id}_b_prat_mask.nii.gz", prat_b, ref_img)
    save_nifti(output_dir / f"{args.patient_id}_t_shell_mask.nii.gz", shell_t, ref_img)
    save_nifti(output_dir / f"{args.patient_id}_o_shell_mask.nii.gz", shell_o, ref_img)

    summary = {
        "patient_id": args.patient_id,
        "tumor_side": args.tumor_side,
        "outer_voxels": args.outer_voxels,
        "inner_voxels": args.inner_voxels,
        "clean_organ_components": not args.no_clean_organ_components,
        "min_organs_voxels": args.min_organ_voxels,
        "z_margin": args.z_margin,
        "raw_t_organ_voxels": raw_t_voxels,
        "raw_o_organ_voxels": raw_o_voxels,
        "clean_t_organ_voxels": int(organ_t.sum()),
        "clean_o_organ_voxels": int(organ_o.sum()),
        "t_prat_voxels": int(prat_t.sum()),
        "o_prat_voxels": int(prat_o.sum()),
        "b_prat_voxels": int(prat_b.sum()),
        "t_prat_volume_ml": round(volume_ml(prat_t, ref_img), 3),
        "o_prat_volume_ml": round(volume_ml(prat_o, ref_img), 3),
        "b_prat_volume_ml": round(volume_ml(prat_b, ref_img), 3),
    }
    (output_dir / f"{args.patient_id}_prat_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_summary(output_dir / f"{args.patient_id}_prat_summary.csv", [summary])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
