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
    gold = (r_ > 120) & (g > 80) & (r_ > b + 40) & (g > b + 20)
    ys, xs = np.nonzero(gold)
    if len(xs) < 100:  # nothing detected; assume centered
        return w / 2, h / 2, 0.45 * m
    cx = (xs.min() + xs.max()) / 2
    cy = (ys.min() + ys.max()) / 2
    r = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2
    return float(cx), float(cy), float(r)


def remove_bg_flood(img: np.ndarray, tol: int) -> np.ndarray:
    """Make edge-connected background transparent by flood-fill from corners.

    Works regardless of ring color (or no ring): samples the four corners,
    flood-fills matching regions from every edge pixel, and sets those to
    transparent. A small close+blur softens the alpha edge.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    ff = img.copy()
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    seed_val = np.mean([img[y, x] for y, x in corners], axis=0)
    lo = (int(tol),) * 3
    hi = (int(tol),) * 3
    # Seed from all four corners so a uniform dark frame is fully caught
    for y, x in corners:
        if mask[y + 1, x + 1] == 0:
            cv2.floodFill(ff, mask, (x, y), (0, 0, 0), lo, hi,
                          cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    bg = mask[1:-1, 1:-1]
    bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    alpha = np.where(bg > 0, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = alpha
    return rgba


def _dominant_color(img: np.ndarray, cx: float, cy: float, r: float) -> tuple:
    """Sample the emblem's dominant color from the inner region."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    inner = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) < r * 0.55)
    px = img[inner]
    # ignore near-black and near-white; take the median of what's left
    lum = px.mean(axis=1)
    keep = px[(lum > 40) & (lum < 220)]
    if len(keep) < 20:
        keep = px
    return tuple(int(v) for v in np.median(keep, axis=0))


def make_badge(img, cx, cy, r, fill, ring_frac, size):
    """Flat `fill` disc inside the ring, transparent outside, ring kept."""
    h, w = img.shape[:2]
    if fill == "auto":
        fill_bgr = _dominant_color(img, cx, cy, r)
    else:  # "#rrggbb"
        hx = fill.lstrip("#")
        fill_bgr = (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))

    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_inner = r * ring_frac  # where the ring's inner edge sits

    out = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Identify the emblem: pixels inside that differ from the (dark) original
    # background. Original inner bg is near-black; emblem is brighter/colored.
    inside = dist < r_inner
    lum = img.mean(axis=2)
    bg_lum = np.median(lum[inside & (lum < 90)]) if (inside & (lum < 90)).any() else 30
    emblem = inside & (lum > bg_lum + 35)
    # thicken emblem slightly so glow edges aren't lost
    emblem = cv2.dilate(emblem.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    # Paint flat fill on the inner disc, then restore the emblem on top
    paint = inside & ~emblem
    out[paint, 0], out[paint, 1], out[paint, 2] = fill_bgr

    # alpha: opaque within r (1.5px feather), transparent beyond
    alpha = np.clip(r - dist + 1.5, 0.0, 3.0) / 3.0
    out[..., 3] = (alpha * 255).astype(np.uint8)

    # crop to the ring square + resize
    m = int(r) + 4
    out = cv2.copyMakeBorder(out, m, m, m, m, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    cxx, cyy = cx + m, cy + m
    side = int(round(2 * r)) + 2
    x0, y0 = int(round(cxx - side / 2)), int(round(cyy - side / 2))
    crop = out[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), fill_bgr


def _tint(color_bgr, amount):
    """Lighten a BGR color toward white by `amount` (0..1)."""
    return tuple(int(c + (255 - c) * amount) for c in color_bgr)


def make_tinted_badge(img, tol, accent, tint_amt, size):
    """Transparent outside the ring; interior filled with a light tint of
    `accent` (or the emblem's own color if accent=='auto'); emblem + ring kept.
    Assumes a light/white exterior (as generated) so flood-fill finds the
    background from the corners.
    """
    h, w = img.shape[:2]

    # 1) exterior = light background reachable from the corners -> transparent
    mask = np.zeros((h + 2, w + 2), np.uint8)
    ff = img.copy()
    for (y, x) in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if mask[y + 1, x + 1] == 0:
            cv2.floodFill(ff, mask, (x, y), (0, 0, 0),
                          (tol,) * 3, (tol,) * 3,
                          cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    exterior = mask[1:-1, 1:-1] > 0

    # 2) accent color
    if accent == "auto":
        lum = img.mean(axis=2)
        colored = (~exterior) & (lum > 40) & (lum < 225)
        px = img[colored]
        # most saturated-ish pixels drive the accent (the red ring/emblem)
        acc = tuple(int(v) for v in np.median(px, axis=0)) if len(px) else (90, 90, 200)
    else:
        hx = accent.lstrip("#")
        acc = (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
    fill = _tint(acc, tint_amt)

    # 3) interior background = light pixels NOT in exterior (inside the ring)
    lum = img.mean(axis=2)
    interior_bg = (~exterior) & (lum > 210)   # the white disc behind the emblem
    out = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    out[interior_bg, 0], out[interior_bg, 1], out[interior_bg, 2] = fill

    # 4) alpha: transparent exterior, opaque elsewhere, feathered 1px
    alpha = np.where(exterior, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    out[..., 3] = alpha

    # 5) trim to content bounding box, square-pad, resize
    ys, xs = np.nonzero(out[..., 3] > 10)
    if len(xs):
        out = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = max(out.shape[0], out.shape[1])
    t, l = (s - out.shape[0]) // 2, (s - out.shape[1]) // 2
    sq = np.zeros((s, s, 4), np.uint8)
    sq[t:t + out.shape[0], l:l + out.shape[1]] = out
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA), fill


def process(path: Path, out_dir: Path, size: int, pad: float,
            overrides: dict[str, float], mode: str, tol: int,
            fill: str, ring_frac: float, accent: str, tint_amt: float) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return f"SKIP {path.name}: not a readable image"

    if mode == "tint":
        out, used = make_tinted_badge(img, tol, accent, tint_amt, size)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (path.stem + ".png")
        cv2.imwrite(str(out_path), out)
        return f"OK   {path.name} -> {out_path} (tint, inner fill BGR={used})"

    if mode == "badge":
        cx, cy, r = detect_ring(img)
        r *= overrides.get(path.stem, pad)
        out, used = make_badge(img, cx, cy, r, fill, ring_frac, size)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (path.stem + ".png")
        cv2.imwrite(str(out_path), out)
        return f"OK   {path.name} -> {out_path} (badge, fill BGR={used})"

    if mode == "corners":
        rgba = remove_bg_flood(img, tol)
        # trim to the non-transparent bounding box, then square-pad
        ys, xs = np.nonzero(rgba[..., 3] > 10)
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            rgba = rgba[y0:y1 + 1, x0:x1 + 1]
        s = max(rgba.shape[0], rgba.shape[1])
        top = (s - rgba.shape[0]) // 2
        left = (s - rgba.shape[1]) // 2
        sq = np.zeros((s, s, 4), np.uint8)
        sq[top:top + rgba.shape[0], left:left + rgba.shape[1]] = rgba
        out = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (path.stem + ".png")
        cv2.imwrite(str(out_path), out)
        return f"OK   {path.name} -> {out_path} (corners, tol={tol})"

    cx, cy, r = detect_ring(img)
    r *= overrides.get(path.stem, pad)

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
    ap.add_argument("--pad", type=float, default=1.0,
                    help="scale detected radius (>1 wider, <1 tighter)")
    ap.add_argument("--override", nargs="*", default=[],
                    metavar="NAME=PAD",
                    help="per-file pad, e.g. --override bond=0.96 steel=1.04")
    ap.add_argument("--mode", choices=["ring", "corners", "badge", "tint"],
                    default="ring",
                    help="ring: crop to gold ring. corners: transparent bg. "
                         "badge: recolor inside a dark-bg icon. tint: light "
                         "exterior -> transparent, interior -> light tint of "
                         "accent (emblem + ring kept).")
    ap.add_argument("--tol", type=int, default=32,
                    help="corners/tint: color tolerance for background match")
    ap.add_argument("--fill", default="auto",
                    help="badge mode: inner fill, 'auto' or #rrggbb")
    ap.add_argument("--ring-frac", type=float, default=0.86,
                    help="badge mode: inner-edge radius as fraction of ring r")
    ap.add_argument("--accent", default="auto",
                    help="tint mode: base color, 'auto' or #rrggbb")
    ap.add_argument("--tint-amt", type=float, default=0.82,
                    help="tint mode: lightening toward white (0..1, higher=paler)")
    args = ap.parse_args()

    overrides = {}
    for item in args.override:
        name, _, val = item.partition("=")
        overrides[name] = float(val)

    for p in args.images:
        print(process(p, args.out, args.size, args.pad, overrides,
                      args.mode, args.tol, args.fill, args.ring_frac,
                      args.accent, args.tint_amt))
    return 0


if __name__ == "__main__":
    sys.exit(main())