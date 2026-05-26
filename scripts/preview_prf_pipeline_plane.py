from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def load_img(path: str):
    img = nib.load(path)
    return np.asarray(img.dataobj), img


def axis_for(codes: tuple[str, ...], targets: set[str], fallback: int) -> int:
    for i, code in enumerate(codes):
        if code in targets:
            return i
    return fallback


def axial_axis(img) -> int:
    return axis_for(nib.aff2axcodes(img.affine), {"S", "I"}, 2)


def lr_axis(img) -> int:
    return axis_for(nib.aff2axcodes(img.affine), {"L", "R"}, 0)


def patient_side_mask(shape: tuple[int, int, int], img, side: str) -> np.ndarray:
    codes = nib.aff2axcodes(img.affine)
    ax = lr_axis(img)
    mid = shape[ax] // 2
    mask = np.zeros(shape, dtype=bool)
    slicer = [slice(None)] * 3
    code = codes[ax]
    if (side == "left" and code == "L") or (side == "right" and code == "R"):
        slicer[ax] = slice(mid, None)
    else:
        slicer[ax] = slice(None, mid)
    mask[tuple(slicer)] = True
    return mask


def tumor_side(mask: np.ndarray, img, tumor_label: int) -> str:
    tumor = mask == tumor_label
    left = int((tumor & patient_side_mask(mask.shape, img, "left")).sum())
    right = int((tumor & patient_side_mask(mask.shape, img, "right")).sum())
    if left == right:
        return "unknown"
    return "left" if left > right else "right"


def select_slice(mask: np.ndarray, img, vein_label: int, side: str | None = None, center: int | None = None, window: int | None = None) -> tuple[int, int]:
    axis = axial_axis(img)
    vein = mask == vein_label
    if side in {"left", "right"}:
        vein &= patient_side_mask(mask.shape, img, side)
    scores = np.array([int(np.take(vein, z, axis=axis).sum()) for z in range(mask.shape[axis])])
    if center is not None and window is not None and center >= 0:
        keep = np.zeros_like(scores, dtype=bool)
        keep[max(0, center - window) : min(len(scores), center + window + 1)] = True
        scores = np.where(keep, scores, 0)
    if int(scores.max()) == 0:
        return -1, 0
    z = int(scores.argmax())
    return z, int(scores[z])


def ball(radius: int) -> np.ndarray:
    r = int(radius)
    grid = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return sum(a * a for a in grid) <= r * r


def largest(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(counts.argmax())


def build_shell(organ: np.ndarray, side_mask: np.ndarray, radius: int) -> np.ndarray:
    clean = largest(organ & side_mask)
    return ndimage.binary_dilation(clean, structure=ball(radius)) & ~clean


def take_axial(arr: np.ndarray, img, z: int) -> np.ndarray:
    return np.take(arr, z, axis=axial_axis(img))


def orient_for_display(slice_2d: np.ndarray, img) -> np.ndarray:
    codes = nib.aff2axcodes(img.affine)
    axis = axial_axis(img)
    remaining = [i for i in range(3) if i != axis]
    row_world = axis_for(tuple(codes[i] for i in remaining), {"P", "A"}, 0)
    col_world = axis_for(tuple(codes[i] for i in remaining), {"L", "R"}, 1)
    if row_world == col_world:
        row_world, col_world = 0, 1
    out = slice_2d if (row_world, col_world) == (0, 1) else slice_2d.T
    row_code = codes[remaining[row_world]]
    col_code = codes[remaining[col_world]]
    if row_code == "A":
        out = out[::-1, :]
    if col_code == "R":
        out = out[:, ::-1]
    return out


def norm_ct(sl: np.ndarray) -> np.ndarray:
    arr = np.clip(sl.astype(np.float32), -200, 250)
    arr = (arr + 200) / 450.0
    return (arr * 255).clip(0, 255).astype(np.uint8)


def blend(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if mask.any():
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.array(color, dtype=np.float32)


def make_panel(title: str, ct_sl: np.ndarray, overlays: list[tuple[np.ndarray, tuple[int, int, int], float]]) -> Image.Image:
    gray = norm_ct(ct_sl)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    for mask, color, alpha in overlays:
        blend(rgb, mask.astype(bool), color, alpha)
    img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 34), fill=(0, 0, 0))
    draw.text((8, 9), title, fill=(255, 255, 255))
    return img


def save_case(args, case_id: str, ct_path: str, organ_path: str, vein_path: str, fat_path: str, out_root: Path) -> dict:
    ct, ct_img = load_img(ct_path)
    organ_arr, _ = load_img(organ_path)
    vein_arr, _ = load_img(vein_path)
    fat_arr, _ = load_img(fat_path)
    side = args.tumor_side if args.tumor_side else tumor_side(organ_arr, ct_img, args.tumor_label)
    all_z, all_area = select_slice(vein_arr, ct_img, args.vein_label)
    t_z, t_area = select_slice(vein_arr, ct_img, args.vein_label, side, all_z, args.tumor_side_window)
    opposite = "right" if side == "left" else "left"
    organ = np.isin(organ_arr, [args.kidney_label, args.tumor_label])
    vat = fat_arr == args.vat_label
    shell_t = build_shell(organ, patient_side_mask(organ.shape, ct_img, side), args.shell_voxels)
    shell_o = build_shell(organ, patient_side_mask(organ.shape, ct_img, opposite), args.shell_voxels)
    prat_t = shell_t & vat
    prat_o = shell_o & vat
    out_dir = out_root / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    for label, z, side_name, shell, prat in [
        ("Tumor-side", t_z, side, shell_t, prat_t),
        ("Opposite/all-vein", all_z, opposite, shell_o, prat_o),
    ]:
        if z < 0:
            continue
        ct_sl = orient_for_display(take_axial(ct, ct_img, z), ct_img)
        kidney_sl = orient_for_display(take_axial(organ_arr == args.kidney_label, ct_img, z), ct_img)
        tumor_sl = orient_for_display(take_axial(organ_arr == args.tumor_label, ct_img, z), ct_img)
        vein_sl = orient_for_display(take_axial(vein_arr == args.vein_label, ct_img, z), ct_img)
        vat_sl = orient_for_display(take_axial(vat, ct_img, z), ct_img)
        shell_sl = orient_for_display(take_axial(shell, ct_img, z), ct_img)
        prat_sl = orient_for_display(take_axial(prat, ct_img, z), ct_img)
        panels.extend(
            [
                make_panel(f"{label} z={z} organ/vein", ct_sl, [(kidney_sl, (80, 220, 120), 0.45), (tumor_sl, (255, 200, 80), 0.55), (vein_sl, (80, 180, 255), 0.70)]),
                make_panel(f"{label} z={z} VAT", ct_sl, [(vat_sl, (233, 196, 106), 0.52)]),
                make_panel(f"{label} z={z} shell & PRAT", ct_sl, [(shell_sl, (255, 80, 80), 0.32), (prat_sl, (42, 157, 143), 0.75)]),
            ]
        )

    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (w * 3, h * 2), (0, 0, 0))
    for i, p in enumerate(panels):
        canvas.paste(p, ((i % 3) * w, (i // 3) * h))
    out_path = out_dir / f"{case_id}_prf_pipeline_plane.png"
    canvas.save(out_path)
    summary = {
        "case_id": case_id,
        "tumor_side": side,
        "tumor_side_slice": t_z,
        "tumor_side_vein_voxels": t_area,
        "all_vein_slice": all_z,
        "all_vein_voxels": all_area,
        "t_plane_vat_voxels": int(take_axial(vat, ct_img, t_z).sum()) if t_z >= 0 else 0,
        "o_plane_vat_voxels": int(take_axial(vat, ct_img, all_z).sum()) if all_z >= 0 else 0,
        "t_plane_prat_voxels": int(take_axial(prat_t, ct_img, t_z).sum()) if t_z >= 0 else 0,
        "o_plane_prat_voxels": int(take_axial(prat_o, ct_img, all_z).sum()) if all_z >= 0 else 0,
        "preview": str(out_path),
    }
    (out_dir / f"{case_id}_prf_pipeline_plane.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True, help="case_id,ct,organ,vein,fat")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kidney-label", type=int, default=2)
    parser.add_argument("--tumor-label", type=int, default=4)
    parser.add_argument("--vein-label", type=int, default=3)
    parser.add_argument("--vat-label", type=int, default=2)
    parser.add_argument("--tumor-side", default=None)
    parser.add_argument("--tumor-side-window", type=int, default=5)
    parser.add_argument("--shell-voxels", type=int, default=6)
    args = parser.parse_args()

    rows = []
    for spec in args.case:
        case_id, ct, organ, vein, fat = spec.split(",", 4)
        rows.append(save_case(args, case_id, ct, organ, vein, fat, Path(args.output_dir)))
        print(rows[-1])
    summary_path = Path(args.output_dir) / "prf_pipeline_plane_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
