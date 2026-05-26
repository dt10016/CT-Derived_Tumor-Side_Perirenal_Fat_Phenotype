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


def overlay_compare(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    gt_only = gt & ~pred
    pred_only = pred & ~gt
    both = gt & pred
    rgb[gt_only] = 0.45 * rgb[gt_only] + 0.55 * np.array([80, 180, 255], dtype=np.float32)
    rgb[pred_only] = 0.45 * rgb[pred_only] + 0.55 * np.array([255, 80, 80], dtype=np.float32)
    rgb[both] = 0.45 * rgb[both] + 0.55 * np.array([80, 220, 120], dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def choose_slices(gt: np.ndarray, pred: np.ndarray, n: int) -> list[int]:
    scores = []
    for z in range(gt.shape[2]):
        score = int(gt[:, :, z].sum() + pred[:, :, z].sum())
        if score > 0:
            scores.append((score, z))
    scores.sort(reverse=True)
    return sorted(z for _, z in scores[:n])


def save_previews(case_id: str, image: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_dir: Path, n: int) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for z in choose_slices(gt, pred, n):
        gray = normalize(image[:, :, z]).T
        panel = overlay_compare(gray, gt[:, :, z].T, pred[:, :, z].T)
        pil = Image.fromarray(panel).resize((panel.shape[1] * 3, panel.shape[0] * 3), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(pil)
        draw.rectangle((0, 0, pil.width, 58), fill=(0, 0, 0))
        draw.text((10, 8), f"{case_id} z={z}  green=overlap blue=GT-only red=pred-only", fill=(255, 255, 255))
        out_path = out_dir / f"{case_id}_z{z:03d}_kits_on_kipa_kidney.png"
        pil.save(out_path)
        paths.append(out_path)
    if not paths:
        return ""
    images = [Image.open(path).convert("RGB") for path in paths]
    w = max(img.width for img in images)
    h = max(img.height for img in images)
    sheet = Image.new("RGB", (w * len(images), h), (0, 0, 0))
    for i, img in enumerate(images):
        sheet.paste(img, (i * w, 0))
    sheet_path = out_dir / f"{case_id}_contact_sheet.png"
    sheet.save(sheet_path)
    return str(sheet_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--patch-x", type=int, default=80)
    parser.add_argument("--patch-y", type=int, default=80)
    parser.add_argument("--patch-z", type=int, default=80)
    parser.add_argument("--spacing-x", type=float, default=1.5)
    parser.add_argument("--spacing-y", type=float, default=1.5)
    parser.add_argument("--spacing-z", type=float, default=2.0)
    parser.add_argument("--max-slices", type=int, default=4)
    args = parser.parse_args()

    torch.cuda.set_device(int(args.gpu_id)) if torch.cuda.is_available() else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
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
            label = item["label"].squeeze(0).cpu().numpy().astype(np.uint8)
            gt_kidney = label == 2
            pred_kidney = pred == 1
            gt_kidney_mass = np.isin(label, [2, 4])
            pred_kidney_mass = np.isin(pred, [1, 2, 3])
            preview = save_previews(case_id, image, gt_kidney, pred_kidney, out_dir / case_id, args.max_slices)
            rows.append(
                {
                    "case_id": case_id,
                    "kidney_dice": f"{dice(pred_kidney, gt_kidney):.4f}",
                    "kidney_plus_tumor_dice": f"{dice(pred_kidney_mass, gt_kidney_mass):.4f}",
                    "gt_kidney_voxels": int(gt_kidney.sum()),
                    "pred_kidney_voxels": int(pred_kidney.sum()),
                    "gt_kidney_plus_tumor_voxels": int(gt_kidney_mass.sum()),
                    "pred_kidney_plus_tumor_voxels": int(pred_kidney_mass.sum()),
                    "preview": preview,
                }
            )
            print(rows[-1])

    summary_path = out_dir / "kits_organ_on_kipa_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
