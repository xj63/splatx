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

function modelResourceToCameraJsonResource(resource: string): string | null {
  const match = resource.match(/^(.*?)(?:_[SL])?\.npz$/i);
  if (!match) {
    return null;
  }

  return `${match[1]}_poses_bounds.json`;
}

type JsonCameraSet = {
  cameras: JsonCamera[];
};

type JsonCamera = {
  id: number;
  intrinsics: {
    height: number;
    width: number;
    focal_length: number;
  };
  bounds: {
    near: number;
    far: number;
  };
  rotation: [
    [number, number, number],
    [number, number, number],
    [number, number, number],
  ];
  position: [number, number, number];
};

function defaultCameraFromJson(camera: JsonCamera): Camera {
  const position = camera.position;
  const up: [number, number, number] = [
    camera.rotation[0][1],
    camera.rotation[1][1],
    camera.rotation[2][1],
  ];
  const forward: [number, number, number] = [
    -camera.rotation[0][2],
    -camera.rotation[1][2],
    -camera.rotation[2][2],
  ];
  const target: [number, number, number] = [
    position[0] + forward[0],
    position[1] + forward[1],
    position[2] + forward[2],
  ];
  const fovyRadians =
    2 * Math.atan((camera.intrinsics.height * 0.5) / camera.intrinsics.focal_length);

  return {
    position,
    target,
    up,
    fovyRadians,
    znear: Math.max(0.001, camera.bounds.near),
    zfar: Math.max(camera.bounds.far, camera.bounds.near + 0.001),
  };
}

async function loadDefaultCamera(resource: string): Promise<void> {
  const cameraResource = modelResourceToCameraJsonResource(resource);
  if (!cameraResource) {
    return;
  }

  const response = await fetch(resourceToFetchUrl(cameraResource));
  if (!response.ok) {
    throw new Error(
      `failed to load camera json: ${response.status} ${response.statusText}`,
    );
  }

  const cameraSet = (await response.json()) as JsonCameraSet;
  const camera = cameraSet.cameras.find((item) => item.id === 0);
  if (!camera) {
    throw new Error("camera id 0 not found");
  }

  renderer.setCamera(defaultCameraFromJson(camera));
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
let frame: number | null = null;
let playbackStart = 0;

function startPlayback() {
  stopPlayback();
  playbackStart = performance.now();
  frame = requestAnimationFrame(renderFrame);
}

function stopPlayback() {
  if (frame !== null) {
    cancelAnimationFrame(frame);
    frame = null;
  }
}

function renderFrame(now: number) {
  const duration = 6_000;
  const time = ((now - playbackStart) % duration) / duration;
  renderer.render(time);
  frame = requestAnimationFrame(renderFrame);
}

async function loadResource(resource: string) {
  const url = resourceToFetchUrl(resource);
  const [modelResponse] = await Promise.all([fetch(url), loadDefaultCamera(resource)]);

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
