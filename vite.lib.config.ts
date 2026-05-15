import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [
    wasm(),
  ],
  build: {
    lib: {
      entry: resolve(__dirname, "ts/index.ts"),
      formats: ["es"],
      fileName: "index",
    },
    outDir: "dist",
  },
});
