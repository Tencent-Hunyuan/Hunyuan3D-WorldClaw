# Video assets

## Overview clip

`worldclaw-teaser.mp4` is the 43-second reel between the hero and the results,
with `worldclaw-teaser.webp` as its poster — the clip's own first frame, so
there is no jump when playback starts.

The delivered master was 1080p at 8.4 Mbps, or 45 MB, for a frame that is never
wider than about 800 CSS px. Re-encode a replacement the same way:

```bash
ffmpeg -i master.mp4 -c:v libx264 -profile:v high -pix_fmt yuv420p \
  -crf 26 -preset slow -vf 'scale=1280:720:flags=lanczos' \
  -an -movflags +faststart worldclaw-teaser.mp4
```

That lands at about 7.7 MB with no visible loss at display size. The element
uses `preload="none"` and only plays once it scrolls into view, so a reader who
stops at the hero transfers none of it — and it stays on the poster entirely for
anyone who has asked the browser to save data.

# Result clips

The results section shows one isometric layout render per world plus four
synchronized channel clips — a single camera orbit rendered four ways. Layout
renders live in `public/assets/layouts/` as `<scene-id>.webp`; only the clips
and their posters belong here.

Layout renders are alpha cut-outs normalised onto a single 1200x878 canvas
(`LAYOUT_WIDTH` / `LAYOUT_HEIGHT` in `src/data/content.ts`) so that every world
fills the same frame and switching scenes never reflows the page. The frame
supplies the backdrop and an accent-tinted cast shadow, so renders should ship
without a baked background.

The strip under the viewer shows all eleven layouts at once, reading from
`public/assets/layouts/thumbs/<scene-id>.webp` — 300x220 derivatives totalling
~130 KB, against ~1.5 MB for the full set. After adding or replacing a layout
render, regenerate them:

```sh
scripts/build-layout-thumbs.py
```

## Directory convention

Paths are resolved by convention from the scene id in `src/data/content.ts`:

```text
scenes/
  frontier-mosaic/
    rgb.mp4       instance.mp4       normal.mp4       depth.mp4
    rgb.webp      instance.webp      normal.webp      depth.webp
```

The `.webp` next to each clip is its poster: the clip's own first frame, so the
still a visitor sees is exactly the frame the video starts on.

Then set `hasVideos: true` on that scene:

```ts
{
  id: "frontier-mosaic",
  // …
  hasVideos: true,
}
```

Use `videos` instead if a clip lives somewhere else:

```ts
videos: {
  rgb: "media/scenes/frontier-mosaic/rgb-v2.mp4",
}
```

Until a clip is available the panel shows the matching still frame from the case
figure, so the layout never breaks. `hasVideos` also gates the play/pause
control above the grid — without it there would be nothing to pause.

All eleven scenes currently ship all four channels.

## Encoding

Use `scripts/encode-clips.sh`, which also holds the mapping from the render
prefixes to the scene ids:

```bash
SRC=/path/to/renders scripts/encode-clips.sh
```

It writes H.264 High / yuv420p at 960x540, 30 fps, no audio, CRF 28 for `rgb`
and 30 for the data channels, with `-movflags +faststart`. That last flag
matters: without it the browser must download the whole file before it can show
a frame.

The delivered masters run ~11 Mbps and total around 500 MB, which is far more
than a project page should push at a reader. This pass lands all 44 clips at
about 63 MB with no visible loss at the sizes they are displayed — a ~300 px
grid tile and a ~1100 px lightbox. All four channels of a world must share the
same camera path and duration so they stay in sync while looping.

## How they load

Nothing is downloaded speculatively:

- The four posters for the first scene load with the page (~85 KB total).
- No `<video>` element exists until the render grid scrolls into view.
- Once mounted the elements use `preload="none"`, so the download starts when
  playback does, not before.
- Switching scenes unmounts the previous clips and fetches only the four for
  the new one (~3–5 MB).
- Visible clips start playing automatically. Because every video is muted and
  uses `playsInline`, this remains compatible with browser autoplay policies.

A visitor who never scrolls to Results transfers no video at all.
