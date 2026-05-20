import type { WebRenderer } from "../pkg/splatx.js";

type WorkerMessage =
  | { type: "init"; canvas: OffscreenCanvas; width: number; height: number }
  | { type: "resize"; width: number; height: number }
  | { type: "set-running"; running: boolean }
  | { type: "load-model"; data: Uint8Array }
  | { type: "set-time"; time: number }
  | { type: "set-time-range"; minTime: number; maxTime: number }
  | { type: "look-at"; camera: LookAtCamera }
  | { type: "set-camera"; camera: Camera }
  | { type: "destroy" };

interface LookAtCamera {
  position: Vec3Tuple;
  target: Vec3Tuple;
  up: Vec3Tuple;
}

interface Camera extends LookAtCamera {
  fovyRadians: number;
  znear: number;
  zfar: number;
}

type Vec3Tuple = [number, number, number];

let renderer: WebRenderer | null = null;
let webRenderer: typeof import("../pkg/splatx.js").WebRenderer | null = null;
let canvas: OffscreenCanvas | null = null;
let running = false;
let frameHandle: ReturnType<typeof setTimeout> | null = null;
let pendingSize: { width: number; height: number } | null = null;

async function loadWebRenderer() {
  webRenderer ??= (await import("../pkg/splatx.js")).WebRenderer;
  return webRenderer;
}

function applyResize(width: number, height: number) {
  if (!canvas) return;

  canvas.width = width;
  canvas.height = height;
  renderer?.resize(width, height);
}

function resize(width: number, height: number) {
  pendingSize = { width, height };

  if (!running) {
    applyPendingResize();
  }
}

function applyPendingResize() {
  if (!pendingSize) {
    return;
  }

  const { width, height } = pendingSize;
  pendingSize = null;
  applyResize(width, height);
}

function render() {
  applyPendingResize();
  renderer?.render();
}

function scheduleFrame() {
  if (!running || frameHandle !== null) return;

  frameHandle = setTimeout(() => {
    frameHandle = null;
    render();
    scheduleFrame();
  }, 16);
}

function setRunning(nextRunning: boolean) {
  if (running === nextRunning) {
    return;
  }

  running = nextRunning;

  if (running) {
    scheduleFrame();
    return;
  }

  if (frameHandle !== null) {
    clearTimeout(frameHandle);
    frameHandle = null;
  }
}

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;

  switch (message.type) {
    case "init":
      canvas = message.canvas;
      applyResize(message.width, message.height);
      renderer = await (await loadWebRenderer()).create_offscreen(canvas);
      renderer.resize(message.width, message.height);
      break;
    case "resize":
      resize(message.width, message.height);
      break;
    case "set-running":
      setRunning(message.running);
      break;
    case "load-model":
      renderer?.load_npz_bytes(message.data);
      break;
    case "set-time":
      renderer?.set_time(message.time);
      break;
    case "set-time-range":
      renderer?.set_time_range(message.minTime, message.maxTime);
      break;
    case "look-at":
      renderer?.look_at(
        message.camera.position[0],
        message.camera.position[1],
        message.camera.position[2],
        message.camera.target[0],
        message.camera.target[1],
        message.camera.target[2],
        message.camera.up[0],
        message.camera.up[1],
        message.camera.up[2],
      );
      break;
    case "set-camera":
      renderer?.set_camera(
        message.camera.position[0],
        message.camera.position[1],
        message.camera.position[2],
        message.camera.target[0],
        message.camera.target[1],
        message.camera.target[2],
        message.camera.up[0],
        message.camera.up[1],
        message.camera.up[2],
        message.camera.fovyRadians,
        message.camera.znear,
        message.camera.zfar,
      );
      break;
    case "destroy":
      setRunning(false);
      renderer?.free();
      renderer = null;
      canvas = null;
      pendingSize = null;
      break;
  }
};
