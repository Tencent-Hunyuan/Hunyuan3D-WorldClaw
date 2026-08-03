# Result clips

The results section shows one isometric layout render per world plus four
synchronized channel clips. Layout renders live in `public/assets/layouts/` as
`<scene-id>.webp`; only the clips need to be added here.

Layout renders are alpha cut-outs normalised onto a single 1200x878 canvas
(`LAYOUT_WIDTH` / `LAYOUT_HEIGHT` in `src/data/content.ts`) so that every world
fills the same frame and switching scenes never reflows the page. The frame
supplies the backdrop and an accent-tinted cast shadow, so renders should ship
without a baked background.

## Directory convention

Paths are resolved by convention from the scene id in `src/data/content.ts`:

```text
scenes/
  frontier-mosaic/
    rgb.mp4
    instance.mp4
    normal.mp4
    depth.mp4
```

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

## Encoding

H.264 MP4, no audio track, 1920x1080 or 1600x1000, `-movflags +faststart`, and
under 12 MB per clip when practical. All four channels of a world should share
the same camera path and duration so they stay visually in sync while looping.
