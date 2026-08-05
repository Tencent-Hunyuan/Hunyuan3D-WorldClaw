#!/usr/bin/env python3
"""Turns the delivered orbit and walk renders into the stills the Results grid shows.

    in   <SRC>/<Orbit|Walk>/<internal-name>/<1|2|3>.png
    out  public/assets/views/<scene-id>/<orbit|walk>-<1..3>.webp      640x480
         public/assets/views/<scene-id>/<orbit|walk>-<1..3>-lg.webp  1600x1200

The masters are 1.2-4 MB PNGs at inconsistent aspect ratios. Two derivatives are
written per shot because the two jobs are so far apart: a tile that six-up on
screen never exceeds ~260 CSS px, and a full frame that only moves when someone
opens the lightbox. Both are cropped to the same 4:3 so the enlarged view is the
tile, not a differently framed image.

Usage:  SRC=/path/to/Image_Renamed scripts/build-view-stills.py
Needs:  python3 with Pillow.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image

# scene-id -> folder name under Orbit/ and Walk/.
# The delivered folders carry internal scene names; this is the same mapping
# scripts/encode-clips.sh verified against public/assets/layouts/<scene-id>.webp.
SCENES = {
    "frontier-mosaic": "village",
    "snowline-village": "snowy",
    "painted-dunes": "desert",
    "island-settlement": "sland",
    "grand-canyon": "realmount",
    "azure-archipelago": "largesland",
    "ember-caldera": "volcanic",
    "desert-frontier": "oasis",
    "frontier-mine": "realrock",
    "verdant-valley": "flatmount",
    "snowbound-outpost": "realsnowy",
}

VIEWS = {"orbit": "Orbit", "walk": "Walk"}
SHOTS = ("1", "2", "3")

TILE = (640, 480)
FULL = (1600, 1200)
TILE_QUALITY = 78
FULL_QUALITY = 76

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "public" / "assets" / "views"


def cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Center-crops to the target ratio, then resizes. Sources run 1.25-1.37."""
    target = size[0] / size[1]
    width, height = image.size
    if width / height > target:
        crop_w = round(height * target)
        box = ((width - crop_w) // 2, 0, (width - crop_w) // 2 + crop_w, height)
    else:
        crop_h = round(width / target)
        box = (0, (height - crop_h) // 2, width, (height - crop_h) // 2 + crop_h)
    return image.resize(size, Image.LANCZOS, box=box)


def main() -> int:
    src = Path(os.environ.get("SRC", Path.home() / "Downloads" / "Image_Renamed"))
    if not src.is_dir():
        print(f"source dir not found: {src}", file=sys.stderr)
        return 1

    written = 0
    total = 0
    for scene_id, folder in SCENES.items():
        out_dir = DEST / scene_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for view, view_dir in VIEWS.items():
            for shot in SHOTS:
                source = src / view_dir / folder / f"{shot}.png"
                if not source.is_file():
                    print(f"MISSING {source}", file=sys.stderr)
                    continue

                with Image.open(source) as image:
                    rgb = image.convert("RGB")
                    tile = out_dir / f"{view}-{shot}.webp"
                    full = out_dir / f"{view}-{shot}-lg.webp"
                    cover(rgb, TILE).save(
                        tile, "WEBP", quality=TILE_QUALITY, method=6
                    )
                    cover(rgb, FULL).save(
                        full, "WEBP", quality=FULL_QUALITY, method=6
                    )

                written += 2
                total += tile.stat().st_size + full.stat().st_size
                print(f"ok  {scene_id}/{view}-{shot}")

    print(f"=== done ===\nfiles   : {written}\nbytes   : {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
