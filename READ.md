# WorldClaw

Project page for *WorldClaw: Agentic Open-World 3D Scene Generation at Scale*.
React + TypeScript, built with Vite, deployed to GitHub Pages.

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # -> dist/
npm run preview
npm run lint
```

## Content

Almost everything on the page is data rather than markup. `src/data/content.ts`
holds the abstract, the method stages and their formulas, the eleven result
scenes, the contributor list, and the BibTeX entry. Adding a scene or reordering
the method is an edit to that file.

Two link targets are worth knowing about:

- `paperUrl` in `src/data/content.ts` is the single source for every "arXiv"
  link on the page. It currently points at the bundled PDF; change it once and
  the nav, hero, citation block, and footer all follow.
- The Open Graph and Twitter tags in `index.html` need absolute URLs, because
  social scrapers do not resolve relative ones reliably. They are the only place
  the deployed origin is hardcoded — update all four together if the page moves.

## Assets

| Path | What |
| --- | --- |
| `public/assets/worldclaw-<season>.webp` | Hero season renders |
| `public/assets/layouts/<scene-id>.webp` | Isometric layout per result scene |
| `public/assets/cases/case*.jpg` | Paper case sheets (fallback stills) |
| `public/assets/paper/*.jpg` | Figures in the method section |
| `public/media/scenes/<scene-id>/` | Result clips and posters |
| `public/favicon.svg` | Brand mark, shared with `src/components/BrandMark.tsx` |

Result clips have their own conventions and encoding recipe — see
[`public/media/README.md`](public/media/README.md) and
`scripts/encode-clips.sh`.

## Deploying to GitHub Pages

`.github/workflows/deploy-pages.yml` builds on every push to `main` and
publishes `dist/`. In the repository settings, set **Pages → Source** to
**GitHub Actions**.

`vite.config.ts` uses `base: "./"`, so the build works from any path — a
`user.github.io/repo/` project site, a user site, or a custom domain — with no
configuration change.

### Payload

The deploy is around 82 MB, of which 63 MB is video. That is well inside the
1 GB GitHub Pages limit, and no visitor downloads anything close to it:

Measured against the production build at 1440x900:

| | Transferred |
| --- | --- |
| Landing on the page, never scrolling | 1.4 MB, no video |
| Scrolling to Results | 6.5 MB — the active scene's four clips are 4 MB of that |
| Opening a second scene | +5 MB |
| Reading the whole page, two scenes viewed | 12 MB |

Video only moves when the render grid is on screen, and only for the scene
being viewed. See the loading notes in `public/media/README.md`.

GitHub Pages' bandwidth allowance is a soft 100 GB/month. At roughly 5–10 MB
for a reader who browses a few scenes, that is on the order of ten thousand
visits a month before it becomes a question.

### Two things worth knowing

**Think twice before moving the clips to Git LFS.** GitHub's docs state plainly
that Git LFS cannot be used with Pages sites: a branch-based deploy publishes
the pointer text files and every video on the page breaks. This workflow
uploads a build artifact rather than a branch, so LFS *can* work here — but
only if the checkout step is given `lfs: true`, otherwise the build copies
pointer files into `dist/`. At 63 MB total and 2.4 MB for the largest clip,
the files sit well inside ordinary Git limits, so the simplest thing is not to
use LFS at all.

**Headers are not configurable.** GitHub Pages ignores `Cache-Control` and has
no equivalent of a `_headers` file; it serves everything with its own
ten-minute cache plus ETags. Repeat visitors still revalidate cheaply, and JS
and CSS are content-hashed by Vite.

If traffic ever does become a problem, putting Cloudflare (free tier) in front
of the Pages domain caches the clips at the edge and removes the bandwidth
question entirely, without moving the files anywhere.
