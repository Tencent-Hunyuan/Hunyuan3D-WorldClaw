#!/usr/bin/env python3
"""Derives the strip thumbnails from the full-size isometric layout renders.

    in   public/assets/layouts/<scene-id>.webp        1200x878, ~140 KB each
    out  public/assets/layouts/thumbs/<scene-id>.webp  300x220,  ~10 KB each

The strip under the viewer shows all eleven worlds at once. Reusing the full
renders there would pull ~1.5 MB to paint eleven ~95 CSS px tiles, so each one
gets a derivative sized for that job. Alpha is preserved: the layouts are
cut-outs and the strip tile supplies its own backdrop, exactly as the main
frame does.

Usage:  scripts/build-layout-thumbs.py
Needs:  python3 with Pillow.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

# Matches LAYOUT_THUMB_WIDTH / LAYOUT_THUMB_HEIGHT in src/data/content.ts, and
# keeps the 1200x878 aspect so the tile crops nothing off the diorama.
THUMB = (300, 220)
QUALITY = 72

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "public" / "assets" / "layouts"
DEST = SRC / "thumbs"


def main() -> int:
    if not SRC.is_dir():
        print(f"layout dir not found: {SRC}", file=sys.stderr)
        return 1

    sources = sorted(SRC.glob("*.webp"))
    if not sources:
        print(f"no layout renders in {SRC}", file=sys.stderr)
        return 1

    DEST.mkdir(parents=True, exist_ok=True)

    total = 0
    for source in sources:
        with Image.open(source) as image:
            thumb = image.convert("RGBA").resize(THUMB, Image.LANCZOS)
            out = DEST / source.name
            thumb.save(out, "WEBP", quality=QUALITY, method=6)

        total += out.stat().st_size
        print(f"ok  {out.relative_to(ROOT)}  {out.stat().st_size / 1e3:.1f} KB")

    print(f"=== done ===\nfiles   : {len(sources)}\nbytes   : {total / 1e3:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
