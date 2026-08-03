# WorldClaw Project Page

Project page for **WorldClaw: Agentic Open-World 3D Scene Generation at Scale**.

Built with Vite, React and TypeScript. No backend — the whole site is static
files served from `dist/`.

## Local development

```bash
npm install
npm run dev
```

Production checks:

```bash
npm run lint     # eslint
npm run build    # tsc -b && vite build  (type errors fail the build)
npm run preview  # serve dist/ locally
```

## Deploying to GitHub Pages

`.github/workflows/deploy-pages.yml` builds and publishes `dist/` on every push
to `main`, and can also be run manually from the Actions tab. It uses the
official Pages actions, so there is no `gh-pages` branch and no deploy key.

One-time repository setup:

1. Open **Settings → Pages**.
2. Set **Build and deployment → Source** to **GitHub Actions**.
3. Push to `main`. The deployed URL appears on the workflow run under the
   `github-pages` environment.

The workflow runs `npm ci`, so **`package-lock.json` must be committed and in
sync with `package.json`** or the build fails before it reaches Vite.

### Base path

`vite.config.ts` sets `base: "./"`, so every asset is referenced relatively and
the build works unchanged at a repository subpath
(`https://<user>.github.io/WorldClaw/`), at a domain root, or from the local
filesystem. Nothing needs to change if the repository is renamed.

In app code, always build public-asset URLs from `import.meta.env.BASE_URL`
rather than a leading `/`:

```ts
src={`${import.meta.env.BASE_URL}assets/paper/pipeline.jpg`}
```

An absolute path like `/assets/…` resolves to the domain root and 404s once the
site lives under `/WorldClaw/`.

### The one hardcoded URL

Open Graph and Twitter cards require **absolute** image URLs — scrapers do not
resolve relative ones reliably — so the deployed origin is hardcoded in
`index.html`. Four tags carry it and must be updated together if the page moves
to a different repository, organisation or custom domain:

- `og:url`
- `og:image`
- `twitter:image`

They currently point at `https://longhz140516.github.io/WorldClaw/`. Everything
else on the page is relative.

## Linking the paper

Every "arXiv" link on the page — nav, hero, mobile menu, citation and footer —
reads a single constant in `src/data/content.ts`:

```ts
export const paperUrl = `${import.meta.env.BASE_URL}WorldClaw.pdf`;
```

It currently serves the PDF bundled in `public/`. Once the preprint is live,
replace the whole expression with the arXiv abstract URL:

```ts
export const paperUrl = "https://arxiv.org/abs/2601.00000";
```

Drop the `BASE_URL` prefix when doing so — it only applies to files served from
this site.

## Content and media

Copy, scene metadata, method stages and the BibTeX entry all live in
`src/data/content.ts`.

| Path | Contents |
| --- | --- |
| `public/WorldClaw.pdf` | Bundled paper, target of `paperUrl` until arXiv is live |
| `public/favicon.svg` | Brand mark, same geometry as `src/components/BrandMark.tsx` |
| `public/assets/apple-touch-icon.png` | 180×180 home-screen icon |
| `public/assets/worldclaw-share.jpg` | 1200×630 social card |
| `public/assets/worldclaw-{spring,summer,autumn,winter}.webp` | Hero season renders, transparent, 2:1 |
| `public/assets/layouts/<scene-id>.webp` | 11 isometric layout renders for the Results section |
| `public/assets/cases/` | 11 case sheets from the paper, cropped for the channel stills |
| `public/assets/paper/` | Fig. 1 pipeline, Fig. 2 terrain stages, Fig. 3 scene refinement |
| `public/media/scenes/<scene-id>/` | Optional per-channel clips — see `public/media/README.md` |

Layout renders are alpha cut-outs normalised onto one 1200×878 canvas
(`LAYOUT_WIDTH` / `LAYOUT_HEIGHT` in `src/data/content.ts`) so switching scenes
never reflows the page.

Result clips are optional. Until a scene sets `hasVideos: true`, its four
channel tiles show still frames cropped from the case sheet and the play/pause
control is hidden. See `public/media/README.md` for the directory convention and
encoding settings.

### Adding a paper figure

Figures render through `src/components/FigurePlate.tsx`, which makes them
clickable and opens a full-resolution view with a fit / actual-size toggle. Pass
the image's **true** pixel dimensions — they reserve layout space before the
image loads, so wrong values cause a visible jump.

## Accessibility and themes

Light and dark themes are driven by `data-theme` on `<html>`, set before first
paint by an inline script in `index.html` and toggled by
`src/components/ThemeToggle.tsx`. The preference persists in `localStorage`
under `worldclaw-theme`. `prefers-reduced-motion` is honoured throughout.
