import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";

export default defineConfig({
  root: "demo",
  plugins: [wasm()],
  build: {
    outDir: "../dist-demo",
    emptyOutDir: true,
  },
});
