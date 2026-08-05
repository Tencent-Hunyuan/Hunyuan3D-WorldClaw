/**
 * Third-party assets embedded in the renders on this site.
 *
 * Every entry here is CC BY 4.0, which requires the title, the creator, and a
 * link to both the work and the licence to travel with the work. Keep that
 * shape when adding rows — dropping a field breaks the licence terms, not just
 * the layout.
 */
export interface Attribution {
  title: string;
  /** Canonical page for the work itself. */
  url: string;
  author: string;
}

/**
 * https, not the http form Sketchfab generates: the page is served over TLS and
 * an http licence link would be downgraded or blocked. Same licence, same URI
 * path, and creativecommons.org redirects http to https regardless.
 */
export const CC_BY_URL = "https://creativecommons.org/licenses/by/4.0/";
export const CC_BY_NAME = "Creative Commons Attribution";

export const attributions: Attribution[] = [
  {
    title: "Low Poly Grass Pack",
    url: "https://skfb.ly/6SGGt",
    author: "Anskar",
  },
  {
    title: "Low Poly Flowers",
    url: "https://skfb.ly/6WtsA",
    author: "Anskar",
  },
  {
    title: "Reed Plants Pack",
    url: "https://skfb.ly/ov6zp",
    author: "Nicholas-3D",
  },
  {
    title: "Lilac bush pack (12 vars, LODs, game ready)",
    url: "https://skfb.ly/pwC67",
    author: "LOLIPOP",
  },
  {
    title: "Realistic Tree",
    url: "https://skfb.ly/oWWCX",
    author: "Daniel",
  },
  {
    title: "Cattail Plant",
    url: "https://skfb.ly/6wInZ",
    author: "SXuno",
  },
  {
    title: "Stylized Fence",
    url: "https://skfb.ly/opGES",
    author: "Mr. Compotchino",
  },
  {
    title: "Deciduous Tree with Leaves (medium-Poly)",
    url: "https://skfb.ly/p6pYE",
    author: "Sereib",
  },
  {
    title: "Old Wooden Bench",
    url: "https://skfb.ly/pAqCH",
    author: "Nikoleta.Zhecheva",
  },
  {
    title: "Stylized Wooden Sign",
    url: "https://skfb.ly/pCXsr",
    author: "FrieDev",
  },
  {
    title: "Wooden Picnic Table - 4096px²",
    url: "https://skfb.ly/pGztM",
    author: "Mark Peters",
  },
  {
    title: "Stylized Pine Tree Tree",
    url: "https://skfb.ly/oo89z",
    author: "Batuhan13",
  },
];

/** Relative so the page resolves under a project path such as /WorldClaw/. */
export const attributionsUrl = `${import.meta.env.BASE_URL}attributions.html`;
export const homeUrl = import.meta.env.BASE_URL;
