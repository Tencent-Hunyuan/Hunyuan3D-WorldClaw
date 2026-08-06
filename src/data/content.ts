export type SeasonId = "spring" | "summer" | "autumn" | "winter";
export type ChannelId = "rgb" | "instance" | "normal" | "depth";
export type ViewTrackId = "orbit" | "walk";

/**
 * Single source of truth for every "arXiv" link on the page.
 *
 * Replace the whole expression with the arXiv abstract URL once the preprint is
 * live, e.g. `export const paperUrl = "https://arxiv.org/abs/2601.00000";`.
 * Until then it serves the PDF bundled in `public/`, which is why it is
 * prefixed with BASE_URL — an absolute arXiv URL must not be.
 */
export const paperUrl = `${import.meta.env.BASE_URL}WorldClaw.pdf`;

/**
 * Every case figure is a 1524x1800 composite laid out on the same template.
 * Crops below are normalised rects measured from those figures.
 */
export interface CropRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

const FRAME_WIDTH = 0.236;
const FRAME_ORIGIN_X = 0.0166;
const FRAME_STRIDE_X = 0.2434;
const CHANNEL_ROW_HEIGHT = 0.1345;
const CHANNEL_ROW_ORIGIN_Y = 0.4252;
const CHANNEL_ROW_STRIDE_Y = 0.1436;

/** Trims the coloured frame the paper figure draws around each render tile. */
const FRAME_INSET_X = 0.008;
const FRAME_INSET_Y = 0.0068;

/**
 * Still frame used while a channel video is unavailable or still buffering.
 * `figureRow` is the channel's row inside the case figure.
 */
export function channelStill(figureRow: number, column = 0): CropRect {
  return {
    x: FRAME_ORIGIN_X + column * FRAME_STRIDE_X + FRAME_INSET_X,
    y: CHANNEL_ROW_ORIGIN_Y + figureRow * CHANNEL_ROW_STRIDE_Y + FRAME_INSET_Y,
    w: FRAME_WIDTH - FRAME_INSET_X * 2,
    h: CHANNEL_ROW_HEIGHT - FRAME_INSET_Y * 2,
  };
}

/**
 * Isometric layout renders are cut out on alpha and normalised to one canvas,
 * so every scene fills the same frame and switching worlds never reflows.
 */
export const LAYOUT_WIDTH = 1200;
export const LAYOUT_HEIGHT = 878;

/**
 * Strip thumbnails share that aspect but are their own derivative: eleven of
 * the full renders would cost ~1.5 MB to paint tiles under 100 CSS px.
 * Regenerate with `scripts/build-layout-thumbs.py` after replacing a layout.
 */
export const LAYOUT_THUMB_WIDTH = 300;
export const LAYOUT_THUMB_HEIGHT = 220;

export interface Season {
  id: SeasonId;
  name: string;
  accent: string;
  accentSoft: string;
  canvas: string;
  image: string;
  particle: "petal" | "mote" | "leaf" | "snow";
  assetNote: string;
}

export interface Scene {
  id: string;
  name: string;
  type: string;
  image: string;
  accent: string;
  icon:
    | "flower"
    | "sun"
    | "leaf"
    | "snow"
    | "fire"
    | "waves"
    | "canyon"
    | "tree"
    | "wind"
    | "ruins";
  prompt: string;
  /** Optional override for a layout render that does not follow the id convention. */
  layout?: string;
  /** Turn on once the four channel clips exist under public/media/scenes/<id>/. */
  hasVideos?: boolean;
  /** Optional per-channel override for non-conventional paths. */
  videos?: Partial<Record<ChannelId, string>>;
}

export const seasons: Season[] = [
  {
    id: "spring",
    name: "Spring",
    accent: "#c85c78",
    accentSoft: "#f3c8d1",
    canvas: "#f4eee9",
    image: "assets/worldclaw-spring.webp",
    particle: "petal",
    assetNote:
      "Scatter assets are built through 3D coding; all other objects are produced by 3D generative models.",
  },
  {
    id: "summer",
    name: "Summer",
    accent: "#2b7a57",
    accentSoft: "#bdd9c3",
    canvas: "#eef1e8",
    image: "assets/worldclaw-summer.webp",
    particle: "mote",
    assetNote:
      "Scatter assets are sourced from Sketchfab; all other objects are produced by 3D generative models.",
  },
  {
    id: "autumn",
    name: "Autumn",
    accent: "#b5552f",
    accentSoft: "#e6bea2",
    canvas: "#f1e9df",
    image: "assets/worldclaw-autumn.webp",
    particle: "leaf",
    assetNote:
      "Scatter assets and all other objects are produced by 3D generative models.",
  },
  {
    id: "winter",
    name: "Winter",
    accent: "#527da0",
    accentSoft: "#c7d8e5",
    canvas: "#edf0f1",
    image: "assets/worldclaw-winter.webp",
    particle: "snow",
    assetNote:
      "Scatter assets and all other objects are produced by 3D generative models.",
  },
];

export interface Person {
  name: string;
  url?: string;
}

export interface Contributor {
  role: string;
  people: Person[];
}

const authors: Record<string, Person> = {
  chunchaoGuo: { name: "Chunchao Guo", url: "https://scholar.google.com/citations?user=8wGH7IsAAAAJ" },
  yangLi: { name: "Yang Li", url: "https://yang-l1.github.io/" },
  jinpengLi: { name: "Jinpeng Li", url: "https://github.com/Lijp411" },
  zilongHuang: { name: "Zilong Huang", url: "https://scholar.google.com/citations?user=Nq2HLEUAAAAJ" },
};

export const contributors: Contributor[] = [
  {
    role: "Project leaders",
    people: [authors.chunchaoGuo, authors.yangLi],
  },
  {
    role: "Local planning",
    people: [authors.jinpengLi, authors.yangLi, authors.zilongHuang],
  },
  {
    role: "Terrain generation",
    people: [authors.zilongHuang, authors.yangLi, authors.jinpengLi],
  },
];

export const sections = [
  { index: "01", id: "results", label: "Results" },
  { index: "02", id: "abstract", label: "Abstract" },
  { index: "03", id: "method", label: "Method" },
  { index: "04", id: "conclusion", label: "Conclusion" },
  { index: "05", id: "citation", label: "Citation" },
];

export const methodStages = [
  {
    anchor: "method-planning",
    title: "Intent Analysis & Planning",
    formula: "P = F_{\\mathrm{plan}}(q)",
    input: {
      label: "Open-ended prompt",
      formula: "q",
    },
    description:
      "An intent analysis agent extracts and normalizes only the constraints the prompt states explicitly, without inventing content or filling gaps. A scene planning agent then resolves ambiguous descriptions and completes the attributes downstream modules require, following a predefined specification schema.",
    equation: {
      formula: "P = (R,\\; C_{\\mathrm{terrain}},\\; C_{\\mathrm{object}})",
      caption:
        "The specification is the shared semantic interface: regions, terrain constraints, and object constraints are read the same way by every later stage.",
    },
    artifacts: [
      {
        symbol: "R",
        title: "Scene regions",
        detail: "Major regions, their attributes, and spatial relationships",
        icon: "regions" as const,
      },
      {
        symbol: "C_{\\mathrm{terrain}}",
        title: "Terrain constraints",
        detail:
          "Terrain types, landform traits, surface appearance, and terrain assets",
        icon: "terrain" as const,
      },
      {
        symbol: "C_{\\mathrm{object}}",
        title: "Object constraints",
        detail:
          "Object categories, appearance, densities, and spatial relations",
        icon: "objects" as const,
      },
    ],
    output: {
      label: "Structured scene specification",
      formula: "P",
    },
    icon: "plan" as const,
  },
  {
    anchor: "method-terrain",
    title: "Global Terrain Generation",
    formula: "T = F_{\\mathrm{terrain}}(P)",
    input: {
      label: "Scene specification",
      formula: "P",
    },
    description:
      "Terrain planning first turns the high-level constraints into an executable terrain specification. Asset generation then produces a semantic layout map, reusable 3D prototypes, and surface materials, which a region-aware height field composes into continuous but irregular landforms. A render-and-inspect loop corrects regional transitions, material scales, and scattering while preserving the semantic layout.",
    equation: {
      formula:
        "H(x) = \\sum_{r} \\tilde{m}_{r}(x) \\left[\\, h_{r} + \\sum_{k} w_{r,k} N_{r,k}(x) + \\sum_{j} \\alpha_{r,j} G_{r,j}(x) \\right]",
      caption:
        "Soft region weights blend each region's base elevation with multi-frequency noise and geomorphic operators — peak, dune, terrace, erosion — so distinct landforms meet along irregular yet continuous boundaries. The same weights later blend the surface materials.",
    },
    artifacts: [
      {
        symbol: "I_{\\mathrm{layout}}",
        title: "Semantic layout map",
        detail: "Color-coded 2D partition of the terrain categories",
        icon: "layout" as const,
      },
      {
        symbol: "O_{\\mathrm{asset}}",
        title: "Asset prototypes",
        detail: "Reusable rocks, vegetation clusters, landform attachments",
        icon: "assets" as const,
      },
      {
        symbol: "M_{\\mathrm{terrain}}",
        title: "Surface materials",
        detail: "Generative textures plus procedural node materials",
        icon: "materials" as const,
      },
    ],
    output: {
      label: "Global terrain representation",
      formula: "T",
    },
    icon: "terrain" as const,
    figure: {
      src: "assets/paper/terrain-stages.jpg",
      width: 1800,
      height: 289,
      label: "Fig. 2",
      alt: "Three terrain sub-stages: height-field generation from a layout map and terrain parameters, 3D asset scattering driven by samplers, and terrain refinement of parameters, scatter, material, and skybox",
      caption:
        "Global terrain in three passes. (a) A composite height field built from the semantic layout map, region parameters, and geomorphic operators, with materials assigned per region. (b) Terrain-asset scattering guided by regional semantics and local surface conditions. (c) A render–inspect–edit loop over parameters, scatter, materials, and environment lighting.",
    },
  },
  {
    anchor: "method-regional",
    title: "Regional Object Generation & Placement",
    formula: "O = F_{\\mathrm{region}}(P, T)",
    input: {
      label: "Specification and terrain",
      formula: "(P,\\; T)",
    },
    description:
      "A regional planning agent selects only the regions whose local terrain can support the requested functions. Each one is rendered from a recorded camera, turned into a terrain-conditioned composition image, segmented into instances, and reconstructed as textured meshes. Corresponding rays through the reconstruction and terrain cameras recover every placement, then a refinement agent checks pose, mesh quality, scale, and object–terrain contact.",
    equation: {
      formula:
        "O_{r} = \\left\\{ \\left( M_{i},\\; U_{i},\\; T_{\\mathrm{place}}^{i} \\right) \\right\\}_{i=1}^{n_{r}}",
      caption:
        "Every selected region returns instance-level content: geometry, appearance attributes, and a terrain-aligned placement transform per object, so each asset stays independently editable and reusable.",
    },
    artifacts: [
      {
        symbol: "I_{\\mathrm{comp}}^{r}",
        title: "Region composition",
        detail: "2D layout prior for object appearance and arrangement",
        icon: "composition" as const,
      },
      {
        symbol: "M_i",
        title: "Editable meshes",
        detail: "Independently reconstructed textured instances",
        icon: "mesh" as const,
      },
      {
        symbol: "T_{\\mathrm{place}}^{i}",
        title: "Placement transforms",
        detail: "Terrain-aligned position, scale, and orientation",
        icon: "placement" as const,
      },
    ],
    output: {
      label: "Editable regional object set",
      formula: "O",
    },
    icon: "region" as const,
    figure: {
      src: "assets/paper/scene-refinement.jpg",
      width: 2600,
      height: 474,
      label: "Fig. 3",
      alt: "Two render-guided refinement loops: an object loop that logs each instance, reads its semantic and geometric attributes alongside the region design, and re-renders after correcting pose, size, and orientation; and a terrain loop that checks mesh quality and object-terrain contact before re-rendering",
      caption:
        "Render-guided refinement runs as a closed loop per instance. (a) An object agent inspects each placement against its semantic and geometric attributes and the region design, then rewrites pose, size, and orientation until the check passes. (b) A terrain agent measures object–terrain contact and re-seats anything left colliding or suspended. Both loops re-render after every edit, so failures are caught on the image rather than in the scene graph.",
    },
  },
];

export const scenes: Scene[] = [
  {
    id: "frontier-mosaic",
    name: "Frontier Mosaic",
    type: "Multi-biome village",
    image: "assets/cases/case0_plain_small.jpg",
    accent: "#45a870",
    icon: "tree",
    prompt:
      "A medieval-style village scene with diverse terrain, including snow-capped mountains, plains, bodies of water, and a desert, populated with animals.",
    hasVideos: true,
  },
  {
    id: "snowline-village",
    name: "Snowline Village",
    type: "Compact snow world",
    image: "assets/cases/case1_snowy_small.jpg",
    accent: "#7295ad",
    icon: "snow",
    prompt:
      "A snow-covered village scene set in a frozen landscape, with the village distributed along both sides of a river.",
    hasVideos: true,
  },
  {
    id: "painted-dunes",
    name: "Painted Dunes",
    type: "Desert landforms",
    image: "assets/cases/case2_dunes_small.jpg",
    accent: "#bd7d42",
    icon: "wind",
    prompt:
      "A desert adventure camp surrounded by several massive dragons coiling around the landscape.",
    hasVideos: true,
  },
  {
    id: "island-settlement",
    name: "Island Settlement",
    type: "Compact island",
    image: "assets/cases/case3_island_small.jpg",
    accent: "#2c91a1",
    icon: "waves",
    prompt:
      "A tropical island that serves as a pirate stronghold, inspired by the adventurous atmosphere of One Piece.",
    hasVideos: true,
  },
  {
    id: "grand-canyon",
    name: "Grand Canyon",
    type: "Large canyon world",
    image: "assets/cases/case4_canyon_large.jpg",
    accent: "#ad613f",
    icon: "canyon",
    prompt:
      "A canyon scene with a river flowing through the entire canyon. Primitive tribal villages are scattered along the surrounding cliffs and valley floor.",
    hasVideos: true,
  },
  {
    id: "azure-archipelago",
    name: "Azure Archipelago",
    type: "Large island world",
    image: "assets/cases/case5_island_large.jpg",
    accent: "#198b9a",
    icon: "waves",
    prompt:
      "An island scene with multiple Japanese-style towns scattered across the island, surrounded by the ocean",
    hasVideos: true,
  },
  {
    id: "ember-caldera",
    name: "Ember Caldera",
    type: "Large volcanic world",
    image: "assets/cases/case6_volcano_large.jpg",
    accent: "#c34f2e",
    icon: "fire",
    prompt:
      "A volcanic landscape filled with glowing lava, where the entire volcano resembles the lair of a powerful demon.",
    hasVideos: true,
  },
  {
    id: "desert-frontier",
    name: "Desert Frontier",
    type: "Large desert world",
    image: "assets/cases/case7_desert_large.jpg",
    accent: "#b78047",
    icon: "sun",
    prompt:
      "A desert battlefield inspired by PUBG's desert maps, designed as an open environment suitable for large-scale PvP combat.",
    hasVideos: true,
  },
  {
    id: "frontier-mine",
    name: "Frontier Mine",
    type: "Industrial terrain",
    image: "assets/cases/case8_mine_large.jpg",
    accent: "#8c7557",
    icon: "ruins",
    prompt:
      "A mining site filled with rich gemstones deposits, with excavation equipment and construction areas actively extracting the resources.",
    hasVideos: true,
  },
  {
    id: "verdant-valley",
    name: "Verdant Valley",
    type: "Large mountain valley",
    image: "assets/cases/case9_valley_large.jpg",
    accent: "#587b50",
    icon: "flower",
    prompt:
      "A realistic mountain valley with scattered Hobbit-style villages nestled beneath the surrounding hills.",
    hasVideos: true,
  },
  {
    id: "snowbound-outpost",
    name: "Snowbound Outpost",
    type: "Large snow world",
    image: "assets/cases/case10_snowy_large.jpg",
    accent: "#667d93",
    icon: "snow",
    prompt:
      "A realistic snow-covered mountain valley inspired by the style of Command & Conquer: Red Alert, featuring a variety of futuristic high-tech buildings scattered throughout the landscape.",
    hasVideos: true,
  },
];

export interface Channel {
  id: ChannelId;
  label: string;
  detail: string;
  /** Row of this channel inside the case figure, used for the still frame. */
  figureRow: number;
}

export const channels: Channel[] = [
  { id: "rgb", label: "RGB", detail: "Shaded appearance", figureRow: 0 },
  {
    id: "instance",
    label: "Instance",
    detail: "Editable asset masks",
    figureRow: 1,
  },
  { id: "normal", label: "Normal", detail: "Surface orientation", figureRow: 3 },
  { id: "depth", label: "Depth", detail: "Metric scene distance", figureRow: 2 },
];

/**
 * Channel clips are resolved by convention, so dropping files into
 * `public/media/scenes/<scene-id>/<channel>.mp4` is enough to publish them.
 * An explicit `videos` entry on a scene overrides the convention.
 */
export function channelVideo(scene: Scene, channel: ChannelId) {
  return (
    scene.videos?.[channel] ??
    (scene.hasVideos ? `media/scenes/${scene.id}/${channel}.mp4` : null)
  );
}

/**
 * Poster frame sitting next to each clip: the clip's own first frame, so the
 * tile shows the right image for ~25 KB before a single byte of video moves.
 */
export function channelPoster(scene: Scene, channel: ChannelId) {
  return scene.hasVideos ? `media/scenes/${scene.id}/${channel}.webp` : null;
}

/** Layout renders follow the same convention: `assets/layouts/<scene-id>.webp`. */
export function layoutRender(scene: Scene) {
  return scene.layout ?? `assets/layouts/${scene.id}.webp`;
}

/**
 * Thumbnails sit in a `thumbs/` subfolder under the same name. A scene with a
 * hand-placed `layout` override has no generated derivative, so it falls back
 * to the full render rather than 404ing on a path that was never written.
 */
export function layoutThumb(scene: Scene) {
  return scene.layout ?? `assets/layouts/thumbs/${scene.id}.webp`;
}

/* ------------------------------------------------------------------ *
 * Camera views
 *
 * Every world is photographed twice: an orbit that reads the whole
 * composition from above, and a walk that puts the camera on the ground
 * inside it. Three shots per track, written by scripts/build-view-stills.py
 * as `assets/views/<scene-id>/<track>-<n>.webp` plus a `-lg` full frame.
 * ------------------------------------------------------------------ */

export interface ViewTrack {
  id: ViewTrackId;
  label: string;
  detail: string;
}

export interface ViewShot {
  id: string;
  track: ViewTrack;
  /** 1-based position within the track, as shown on the tile. */
  shot: number;
  tile: string;
  full: string;
  alt: string;
}

export const viewTracks: ViewTrack[] = [
  {
    id: "orbit",
    label: "Orbit",
    detail: "Aerial pass over the finished world",
  },
  {
    id: "walk",
    label: "Walk",
    detail: "Ground-level camera inside the scene",
  },
];

export const VIEW_SHOTS_PER_TRACK = 3;
/** Both derivatives are cropped to 4:3, so the enlarged view reframes nothing. */
export const VIEW_TILE_WIDTH = 640;
export const VIEW_TILE_HEIGHT = 480;
export const VIEW_FULL_WIDTH = 1600;
export const VIEW_FULL_HEIGHT = 1200;

export function sceneViews(scene: Scene, track: ViewTrack): ViewShot[] {
  return Array.from({ length: VIEW_SHOTS_PER_TRACK }, (_, index) => {
    const shot = index + 1;
    const base = `assets/views/${scene.id}/${track.id}-${shot}`;

    return {
      id: `${track.id}-${shot}`,
      track,
      shot,
      tile: `${base}.webp`,
      full: `${base}-lg.webp`,
      alt: `${scene.name}, ${track.label.toLowerCase()} view ${shot} of ${VIEW_SHOTS_PER_TRACK}`,
    };
  });
}

/** Flat orbit-then-walk order, which is what the lightbox steps through. */
export function sceneViewSequence(scene: Scene): ViewShot[] {
  return viewTracks.flatMap((track) => sceneViews(scene, track));
}

export const bibtex = `@article{worldclaw2026,
  title   = {WorldClaw: Agentic Open-World 3D Scene Generation at Scale},
  author  = {Tencent Hunyuan3D Team},
  year    = {2026},
  month   = {July}
}`;
