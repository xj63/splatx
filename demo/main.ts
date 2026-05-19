import "./style.css";
import { createRenderer } from "../ts/index.ts";

function getCanvas(): HTMLCanvasElement {
  const canvas = document.querySelector<HTMLCanvasElement>("#app");
  if (!canvas) throw new Error("missing #app canvas");
  return canvas;
}

const canvas = getCanvas();
await createRenderer(canvas);
