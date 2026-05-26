from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def make_sheet(case_dir: Path, output_path: Path, title: str) -> None:
    paths = sorted(case_dir.glob("*_kipa_mask_qc.png"))
    if not paths:
        raise FileNotFoundError(f"No preview PNGs in {case_dir}")
    images = [Image.open(path).convert("RGB") for path in paths]
    w = max(img.width for img in images)
    h = max(img.height for img in images)
    title_h = 34
    canvas = Image.new("RGB", (w * len(images), h + title_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(255, 255, 255))
    for i, img in enumerate(images):
        canvas.paste(img, (i * w, title_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    make_sheet(Path(args.case_dir), Path(args.output), args.title)
    print(args.output)


if __name__ == "__main__":
    main()
