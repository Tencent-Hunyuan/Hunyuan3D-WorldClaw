import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    target: "es2022",
    sourcemap: true,
    rollupOptions: {
      /*
       * Two real HTML entries rather than a client-side router: GitHub Pages
       * serves static files with no rewrite rule, so attributions.html has to
       * exist on disk to survive a direct hit or a refresh. Paths stay relative
       * to avoid pulling in @types/node just for __dirname.
       */
      input: {
        main: "index.html",
        attributions: "attributions.html",
      },
    },
  },
});
