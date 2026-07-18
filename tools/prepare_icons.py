#!/usr/bin/env python3
"""Prepare content-pack icons: crop to the circular ring, transparent outside.

Detects the large ring in each image (Hough circle detection, with a
gold-pixel fallback), masks everything outside it to transparency with a
soft 2px edge, and exports uniform squares.

Usage:
    python tools/prepare_icons.py raw/*.png -o assets/stats --size 512
    python tools/prepare_icons.py raw/*.png -o out --size 512 --pad 1.05

--pad grows (>1.0) or shrinks (<1.0) the detected radius, if the cut sits
slightly inside or outside the ring.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def detect_ring(img: np.ndarray) -> tuple[float, float, float]:
    """Return (cx, cy, radius) of the dominant ring."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    m = min(h, w)
    circles = cv2.Hough_circles if False else cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=m / 4,
        param1=120, param2=35,
        minRadius=int(0.28 * m), maxRadius=int(0.58 * m),
    )
    if circles is not None:
        # Candidates arrive strongest-first, but inner art can be circular
        # too (rings, eyes, targets). Prefer a candidate that is centered
        # and plausibly outer-ring-sized; fall back to the strongest.
        for cx, cy, r in circles[0]:
            centered = (abs(cx - w / 2) < 0.15 * w
                        and abs(cy - h / 2) < 0.15 * h)
            if centered and 0.32 * m <= r <= 0.55 * m:
                return float(cx), float(cy), float(r)
        cx, cy, r = circles[0][0]
        return float(cx), float(cy), float(r)

    # Fallback: bounding box of gold-ish pixels (the ring is the outermost gold)
    b = img[..., 0].astype(int)
    g = img[..., 1].astype(int)
    r_ = img[..., 2].astype(int)
    gold = (r_ > 120) & (g > 70) & (r_ > b + 40) & (g > b + 20)
    ys, xs = np.nonzero(gold)
    if len(xs) < 100:  # nothing detected; assume centered
        return w / 2, h / 2, 0.45 * m
    cx = (xs.min() + xs.max()) / 2
    cy = (ys.min() + ys.max()) / 2
    r = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2
    return float(cx), float(cy), float(r)


def process(path: Path, out_dir: Path, size: int, pad: float) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return f"SKIP {path.name}: not a readable image"

    cx, cy, r = detect_ring(img)
    r *= pad

    # Pad the canvas so the crop never leaves the image
    margin = int(r) + 4
    img = cv2.copyMakeBorder(img, margin, margin, margin, margin,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cx += margin
    cy += margin

    side = int(round(2 * r)) + 2
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    crop = img[y0:y0 + side, x0:x0 + side]

    # Soft circular alpha: 2px feathered edge at radius r
    yy, xx = np.mgrid[0:side, 0:side]
    dist = np.sqrt((xx - side / 2) ** 2 + (yy - side / 2) ** 2)
    alpha = np.clip(r - dist + 1.0, 0.0, 2.0) / 2.0

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = (alpha * 255).astype(np.uint8)

    out = cv2.resize(rgba, (size, size), interpolation=cv2.INTER_AREA)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (path.stem + ".png")
    cv2.imwrite(str(out_path), out)
    return f"OK   {path.name} -> {out_path} (ring r={int(r)} at {int(cx)},{int(cy)})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--pad", type=float, default=1.06)
    args = ap.parse_args()

    for p in args.images:
        print(process(p, args.out, args.size, args.pad))
    return 0


if __name__ == "__main__":
    sys.exit(main())