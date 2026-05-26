from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from preview_prf_pipeline_plane import axial_axis, build_shell, norm_ct, orient_for_display, take_axial


def load(path: str):
    img = nib.load(path)
    return np.asarray(img.dataobj), img


def make_panel(title: str, ct_sl: np.ndarray, overlays: list[tuple[np.ndarray, tuple[int, int, int], float]]) -> Image.Image:
    gray = norm_ct(ct_sl)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    for mask, color, alpha in overlays:
        region = mask.astype(bool)
        if region.any():
            rgb[region] = (1 - alpha) * rgb[region] + alpha * np.array(color, dtype=np.float32)
    img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 34), fill=(0, 0, 0))
    draw.text((8, 9), title, fill=(255, 255, 255))
    return img


def save_case(case_id: str, ct_path: str, seg_path: str, fat_path: str, out_root: Path, shell_voxels: int) -> dict:
    ct, img = load(ct_path)
    seg, _ = load(seg_path)
    fat, _ = load(fat_path)
    organ = np.isin(seg, [1, 2, 3])
    vat = fat == 2
    axis = axial_axis(img)
    areas = [int(np.take(organ, z, axis=axis).sum()) for z in range(organ.shape[axis])]
    z = int(np.argmax(areas))
    shell = ndimage.binary_dilation(organ, iterations=shell_voxels) & ~organ
    prat = shell & vat
    ct_sl = orient_for_display(take_axial(ct, img, z), img)
    kidney_sl = orient_for_display(take_axial(seg == 1, img, z), img)
    tumor_sl = orient_for_display(take_axial(seg == 2, img, z), img)
    cyst_sl = orient_for_display(take_axial(seg == 3, img, z), img)
    vat_sl = orient_for_display(take_axial(vat, img, z), img)
    shell_sl = orient_for_display(take_axial(shell, img, z), img)
    prat_sl = orient_for_display(take_axial(prat, img, z), img)
    panels = [
        make_panel(f"{case_id} z={z} organ", ct_sl, [(kidney_sl, (80, 220, 120), 0.45), (tumor_sl, (255, 200, 80), 0.55), (cyst_sl, (180, 120, 255), 0.45)]),
        make_panel(f"{case_id} z={z} VAT", ct_sl, [(vat_sl, (233, 196, 106), 0.52)]),
        make_panel(f"{case_id} z={z} shell & VAT", ct_sl, [(shell_sl, (255, 80, 80), 0.30), (prat_sl, (42, 157, 143), 0.75)]),
    ]
    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (w * len(panels), h), (0, 0, 0))
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * w, 0))
    out_dir = out_root / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}_kits_organ_vat_shell.png"
    canvas.save(out_path)
    return {
        "case_id": case_id,
        "representative_slice": z,
        "organ_voxels_on_slice": areas[z],
        "vat_voxels_on_slice": int(take_axial(vat, img, z).sum()),
        "prat_voxels_on_slice": int(take_axial(prat, img, z).sum()),
        "preview": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True, help="case_id,ct,seg,fat")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shell-voxels", type=int, default=6)
    args = parser.parse_args()
    rows = []
    for spec in args.case:
        rows.append(save_case(*spec.split(",", 3), Path(args.output_dir), args.shell_voxels))
        print(rows[-1])
    with (Path(args.output_dir) / "kits_organ_vat_shell_summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
