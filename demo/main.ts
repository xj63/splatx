import "./style.css";
import { createRenderer } from "../ts/index.ts";

const bundledModels = [
  "coffee_martini_S.npz",
  "coffee_martini_L.npz",
  "cook_spinach_S.npz",
  "cook_spinach_L.npz",
];

function getCanvas(): HTMLCanvasElement {
  const canvas = document.querySelector<HTMLCanvasElement>("#app");
  if (!canvas) throw new Error("missing #app canvas");
  return canvas;
}

function getHashResource(): string | null {
  const hash = window.location.hash.slice(1).trim();
  if (!hash) return null;
  return decodeURIComponent(hash);
}

function resourceToFetchUrl(resource: string): string {
  if (/^https?:\/\//i.test(resource)) {
    return resource;
  }

  return resource.startsWith("/") ? resource : `/${resource}`;
}

function setResourceHash(resource: string) {
  window.location.hash = encodeURI(resource);
}

function createModelPicker() {
  const picker = document.createElement("nav");
  picker.className = "model-picker";
  picker.setAttribute("aria-label", "Model resources");

  for (const filename of bundledModels) {
    const resource = `/model/${filename}`;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = filename.replace(/\.npz$/i, "");
    button.addEventListener("click", () => setResourceHash(resource));
    picker.append(button);
  }

  document.body.append(picker);
  return picker;
}

const canvas = getCanvas();
const picker = createModelPicker();
const renderer = await createRenderer(canvas);

renderer.onend = () => {
  renderer.play({ from: 0, to: 1, duration: 6 });
};

async function loadResource(resource: string) {
  const url = resourceToFetchUrl(resource);
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`failed to load model: ${response.status} ${response.statusText}`);
  }

  renderer.loadModel(await response.arrayBuffer());
  renderer.play({ from: 0, to: 1, duration: 6 });
}

async function syncHashResource() {
  const resource = getHashResource();
  picker.hidden = resource !== null;

  if (!resource) {
    renderer.pause();
    return;
  }

  try {
    await loadResource(resource);
  } catch (error) {
    console.error(error);
  }
}

window.addEventListener("hashchange", () => {
  void syncHashResource();
});

void syncHashResource();
