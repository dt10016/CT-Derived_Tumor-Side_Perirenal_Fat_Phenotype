from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)
from PIL import Image, ImageDraw


COLORS = {
    1: (255, 80, 80),
    2: (80, 220, 120),
    3: (80, 180, 255),
    4: (255, 200, 80),
}


def build_transforms(pixdim: tuple[float, float, float]):
    return Compose(
        [
            LoadImaged(keys=["image", "label"], reader="NibabelReader", dtype=np.float32),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / denom)


def normalize(slice_2d: np.ndarray) -> np.ndarray:
    arr = slice_2d.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def tumor_side(label: np.ndarray) -> str:
    tumor = label == 4
    mid = label.shape[0] // 2
    # After Orientationd(axcodes="RAS"), axis 0 increases toward patient right.
    left_voxels = int(tumor[:mid, :, :].sum())
    right_voxels = int(tumor[mid:, :, :].sum())
    if left_voxels == right_voxels:
        return "unknown"
    return "left" if left_voxels > right_voxels else "right"


def side_mask(shape: tuple[int, int, int], side: str | None) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    if side not in {"left", "right"}:
        return mask
    mask[:] = False
    mid = shape[0] // 2
    # After Orientationd(axcodes="RAS"), axis 0 increases toward patient right.
    if side == "left":
        mask[:mid, :, :] = True
    else:
        mask[mid:, :, :] = True
    return mask


def select_vein_slice(
    label: np.ndarray,
    side: str | None,
    z_center: int | None = None,
    z_window: int | None = None,
) -> tuple[int, int]:
    vein = (label == 3) & side_mask(label.shape, side)
    areas = np.array([int(vein[:, :, z].sum()) for z in range(label.shape[2])])
    if z_center is not None and z_window is not None and z_center >= 0:
        keep = np.zeros_like(areas, dtype=bool)
        z0 = max(0, int(z_center) - int(z_window))
        z1 = min(len(areas) - 1, int(z_center) + int(z_window))
        keep[z0 : z1 + 1] = True
        areas = np.where(keep, areas, 0)
    if int(areas.max()) == 0:
        return -1, 0
    z = int(areas.argmax())
    return z, int(areas[z])


def overlay_label(gray: np.ndarray, label: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    for cls, color in COLORS.items():
        region = label == cls
        if region.any():
            rgb[region] = (1 - alpha) * rgb[region] + alpha * np.array(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def overlay_vein_compare(gray: np.ndarray, gt_vein: np.ndarray, pred_vein: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    both = gt_vein & pred_vein
    gt_only = gt_vein & ~pred_vein
    pred_only = pred_vein & ~gt_vein
    rgb[both] = 0.35 * rgb[both] + 0.65 * np.array([80, 220, 120], dtype=np.float32)
    rgb[gt_only] = 0.35 * rgb[gt_only] + 0.65 * np.array([80, 180, 255], dtype=np.float32)
    rgb[pred_only] = 0.35 * rgb[pred_only] + 0.65 * np.array([255, 80, 80], dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def axial(arr: np.ndarray, z: int) -> np.ndarray:
    return arr[:, :, z].T


def make_panel(title: str, image: np.ndarray, label: np.ndarray | None, gt_vein: np.ndarray | None, pred_vein: np.ndarray | None, z: int, scale: int) -> Image.Image:
    gray = normalize(axial(image, z))
    if label is not None:
        rgb = overlay_label(gray, axial(label, z).astype(np.uint8))
    else:
        rgb = overlay_vein_compare(gray, axial(gt_vein, z).astype(bool), axial(pred_vein, z).astype(bool))
    pil = Image.fromarray(rgb)
    if scale > 1:
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, pil.width, 58), fill=(0, 0, 0))
    draw.text((8, 8), title, fill=(255, 255, 255))
    if label is not None:
        draw.text((8, 32), "red=artery green=kidney blue=vein yellow=tumor", fill=(255, 255, 255))
    else:
        draw.text((8, 32), "green=overlap blue=GT-only red=pred-only", fill=(255, 255, 255))
    return pil


def save_case_previews(
    case_id: str,
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    z_pred_t: int,
    z_pred_all: int,
    z_pred_t_constrained: int,
    out_dir: Path,
    window: int,
    scale: int,
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    z_values = []
    for center in (z_pred_t, z_pred_all, z_pred_t_constrained):
        if center >= 0:
            z_values.extend(range(max(0, center - window), min(image.shape[2] - 1, center + window) + 1))
    z_values = sorted(set(z_values))
    panels = []
    gt_vein = gt == 3
    pred_vein = pred == 3
    for z in z_values:
        mark = []
        if z == z_pred_t:
            mark.append("PRED-T")
        if z == z_pred_all:
            mark.append("PRED-ALL")
        if z == z_pred_t_constrained:
            mark.append("PRED-T-CONSTR")
        suffix = " ".join(mark)
        panels.append(make_panel(f"{case_id} z={z} prediction {suffix}", image, pred, None, None, z, scale))
        panels.append(make_panel(f"{case_id} z={z} vein compare {suffix}", image, None, gt_vein, pred_vein, z, scale))
    if not panels:
        return ""
    w = max(panel.width for panel in panels)
    h = max(panel.height for panel in panels)
    cols = 2
    rows = int(np.ceil(len(panels) / cols))
    sheet = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % cols) * w, (i // cols) * h))
    out_path = out_dir / f"{case_id}_vessel_model_renal_vein_plane.png"
    sheet.save(out_path)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--patch-x", type=int, default=96)
    parser.add_argument("--patch-y", type=int, default=96)
    parser.add_argument("--patch-z", type=int, default=96)
    parser.add_argument("--spacing-x", type=float, default=0.8)
    parser.add_argument("--spacing-y", type=float, default=0.8)
    parser.add_argument("--spacing-z", type=float, default=0.8)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--tumor-side-search-window", type=int, default=5)
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()

    torch.cuda.set_device(int(args.gpu_id)) if torch.cuda.is_available() else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.0,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    transforms = build_transforms((args.spacing_x, args.spacing_y, args.spacing_z))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for case_id in args.case_ids:
            item = transforms(
                {
                    "image": str(Path(args.image_dir) / f"{case_id}.nii.gz"),
                    "label": str(Path(args.label_dir) / f"{case_id}.nii.gz"),
                }
            )
            image_t = item["image"].unsqueeze(0).to(device)
            logits = sliding_window_inference(
                image_t,
                roi_size=(args.patch_x, args.patch_y, args.patch_z),
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
                sw_device=device,
                device=torch.device("cpu"),
            )
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            image = item["image"].squeeze(0).cpu().numpy()
            gt = item["label"].squeeze(0).cpu().numpy().astype(np.uint8)
            side = tumor_side(gt)
            gt_t_z, gt_t_area = select_vein_slice(gt, side)
            gt_all_z, gt_all_area = select_vein_slice(gt, None)
            pred_t_z, pred_t_area = select_vein_slice(pred, side)
            pred_all_z, pred_all_area = select_vein_slice(pred, None)
            gt_t_constrained_z, gt_t_constrained_area = select_vein_slice(
                gt, side, gt_all_z, args.tumor_side_search_window
            )
            pred_t_constrained_z, pred_t_constrained_area = select_vein_slice(
                pred, side, pred_all_z, args.tumor_side_search_window
            )
            preview = save_case_previews(
                case_id,
                image,
                gt,
                pred,
                pred_t_z,
                pred_all_z,
                pred_t_constrained_z,
                out_dir / case_id,
                args.window,
                args.scale,
            )
            row = {
                "case_id": case_id,
                "tumor_side": side,
                "vein_dice": f"{dice(pred == 3, gt == 3):.4f}",
                "gt_tumor_side_slice": gt_t_z,
                "pred_tumor_side_slice": pred_t_z,
                "gt_tumor_side_vein_voxels": gt_t_area,
                "pred_tumor_side_vein_voxels": pred_t_area,
                "gt_all_vein_slice": gt_all_z,
                "pred_all_vein_slice": pred_all_z,
                "gt_all_vein_voxels": gt_all_area,
                "pred_all_vein_voxels": pred_all_area,
                "gt_tumor_side_constrained_slice": gt_t_constrained_z,
                "pred_tumor_side_constrained_slice": pred_t_constrained_z,
                "gt_tumor_side_constrained_vein_voxels": gt_t_constrained_area,
                "pred_tumor_side_constrained_vein_voxels": pred_t_constrained_area,
                "tumor_side_search_window": args.tumor_side_search_window,
                "preview": preview,
            }
            rows.append(row)
            print(row)

    summary_path = out_dir / "vessel_model_renal_vein_plane_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
