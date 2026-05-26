from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom


def add_transunet_to_path(path: str) -> None:
    sys.path.insert(0, str(Path(path).resolve()))


def normalize_ct(volume_hu: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    volume = np.clip(volume_hu.astype(np.float32), hu_min, hu_max)
    return ((volume - hu_min) / max(hu_max - hu_min, 1e-6)).astype(np.float32)


def build_model(snapshot_path: str, transunet_dir: str, vit_name: str, img_size: int, num_classes: int, n_skip: int, vit_patches_size: int):
    add_transunet_to_path(transunet_dir)
    from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
    from networks.vit_seg_modeling import VisionTransformer as ViT_seg

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = n_skip
    config_vit.patches.size = (vit_patches_size, vit_patches_size)
    if "R50" in vit_name:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
    model.load_state_dict(torch.load(snapshot_path))
    model.eval()
    return model


def axial_axis(img) -> int:
    codes = nib.aff2axcodes(img.affine)
    for idx, code in enumerate(codes):
        if code in {"S", "I"}:
            return idx
    return 2


def infer_nifti(model, ct_xyz: np.ndarray, img_size: int, transpose_for_model: bool, axis: int) -> np.ndarray:
    norm = normalize_ct(ct_xyz, -190.0, -30.0)
    pred_xyz = np.zeros(ct_xyz.shape, dtype=np.uint8)
    n_slices = norm.shape[axis]
    with torch.no_grad():
        for z in range(n_slices):
            sl = np.take(norm, z, axis=axis)
            if transpose_for_model:
                sl = sl.T
            h, w = sl.shape
            resized = zoom(sl, (img_size / h, img_size / w), order=3) if (h != img_size or w != img_size) else sl
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().cuda()
            logits = model(tensor)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()
            if h != img_size or w != img_size:
                pred = zoom(pred, (h / img_size, w / img_size), order=0)
            if transpose_for_model:
                pred = pred.T
            if axis == 0:
                pred_xyz[z, :, :] = pred.astype(np.uint8)
            elif axis == 1:
                pred_xyz[:, z, :] = pred.astype(np.uint8)
            else:
                pred_xyz[:, :, z] = pred.astype(np.uint8)
    return pred_xyz


def save_like(path: Path, data: np.ndarray, ref_img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(data.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(out, str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", required=True, help="Entries like case_id=ct_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument(
        "--transunet-dir",
        required=True,
        help="Path to a local TransUNet checkout that provides networks/vit_seg_modeling.py.",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--vit-name", default="R50-ViT-B_16")
    parser.add_argument("--n-skip", type=int, default=3)
    parser.add_argument("--vit-patches-size", type=int, default=16)
    parser.add_argument("--no-transpose-for-model", action="store_true")
    args = parser.parse_args()

    model = build_model(
        args.snapshot,
        args.transunet_dir,
        args.vit_name,
        args.img_size,
        args.num_classes,
        args.n_skip,
        args.vit_patches_size,
    )
    out_dir = Path(args.output_dir)
    rows = []
    for entry in args.cases:
        case_id, ct_path = entry.split("=", 1)
        img = nib.load(ct_path)
        ct = np.asarray(img.dataobj)
        pred = infer_nifti(model, ct, args.img_size, not args.no_transpose_for_model, axial_axis(img))
        pred_path = out_dir / f"{case_id}_fat_pred.nii.gz"
        save_like(pred_path, pred, img)
        unique, counts = np.unique(pred, return_counts=True)
        row = {"case_id": case_id, "ct_path": ct_path, "pred_path": str(pred_path)}
        for cls, count in zip(unique.tolist(), counts.tolist()):
            row[f"class_{cls}_voxels"] = int(count)
        rows.append(row)
        print(row)

    summary = out_dir / "aattct_nifti_fat_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
