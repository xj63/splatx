import "./style.css";
import { createRenderer, type Camera } from "../ts/index.ts";

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

const presetCamera: Camera = {
  position: [0.44396468423162905, -1.1035034400953323, -0.3499272977464685],
  target: [0.5478754702925966, -1.0966363123253844, 0.6446356168528959],
  up: [-0.021732170138006792, 0.9997530962884785, -0.004632412189448103],
  fovyRadians: 1.0,
  znear: 8.831384232013642,
  zfar: 109.77542390000633,
};

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
let frame: number | null = null;
let playbackStart = 0;
const MODEL_DURATION_SECONDS = 10;
const MODEL_DURATION_MS = MODEL_DURATION_SECONDS * 1000;
const SMALL_MODEL_FPS = 15;
const LARGE_MODEL_FPS = 5;
let lastRenderTime = 0;
let frameIntervalMs = 1000 / SMALL_MODEL_FPS;

function updatePlaybackProfile(resource: string) {
  const name = resource.split("/").pop() ?? resource;
  frameIntervalMs = name.includes("_L.")
    ? 1000 / LARGE_MODEL_FPS
    : 1000 / SMALL_MODEL_FPS;
}

function startPlayback() {
  stopPlayback();
  playbackStart = performance.now();
  lastRenderTime = 0;
  frame = requestAnimationFrame(renderFrame);
}

function stopPlayback() {
  if (frame !== null) {
    cancelAnimationFrame(frame);
    frame = null;
  }
}

function renderFrame(now: number) {
  if (lastRenderTime !== 0 && now - lastRenderTime < frameIntervalMs) {
    frame = requestAnimationFrame(renderFrame);
    return;
  }

  const time = ((now - playbackStart) % MODEL_DURATION_MS) / 1000;
  lastRenderTime = now;
  renderer.render(time);
  frame = requestAnimationFrame(renderFrame);
}

async function loadResource(resource: string) {
  const url = resourceToFetchUrl(resource);
  updatePlaybackProfile(resource);
  renderer.setCamera(presetCamera);
  const modelResponse = await fetch(url);

  if (!modelResponse.ok) {
    throw new Error(
      `failed to load model: ${modelResponse.status} ${modelResponse.statusText}`,
    );
  }

  renderer.loadModel(await modelResponse.arrayBuffer());
  startPlayback();
}

async function syncHashResource() {
  const resource = getHashResource();
  picker.hidden = resource !== null;

  if (!resource) {
    stopPlayback();
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
