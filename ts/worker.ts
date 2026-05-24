import type { WebRenderer } from "../pkg/splatx.js";

type WorkerMessage =
  | { type: "init"; canvas: OffscreenCanvas; width: number; height: number }
  | { type: "resize"; width: number; height: number }
  | { type: "load-model"; data: Uint8Array }
  | { type: "render"; time: number }
  | { type: "set-camera"; camera: Required<Camera> }
  | { type: "destroy" };

interface Camera {
  position: Vec3Tuple;
  target: Vec3Tuple;
  up: Vec3Tuple;
  fovyRadians: number;
  znear: number;
  zfar: number;
}

type Vec3Tuple = [number, number, number];

type WebRendererRuntime = WebRenderer & {
  render(time: number): void;
};

let renderer: WebRendererRuntime | null = null;
let webRenderer: typeof import("../pkg/splatx.js").WebRenderer | null = null;
let canvas: OffscreenCanvas | null = null;
let pendingSize: { width: number; height: number } | null = null;

async function loadWebRenderer() {
  webRenderer ??= (await import("../pkg/splatx.js")).WebRenderer;
  return webRenderer;
}

function resize(width: number, height: number) {
  pendingSize = { width, height };
}

function applyPendingResize() {
  if (!canvas || !pendingSize) {
    return;
  }

  const { width, height } = pendingSize;
  pendingSize = null;
  canvas.width = width;
  canvas.height = height;
  renderer?.resize(width, height);
}

function render(time: number) {
  applyPendingResize();
  renderer?.render(time);
}

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;

  switch (message.type) {
    case "init": {
      canvas = message.canvas;
      resize(message.width, message.height);
      applyPendingResize();
      renderer = (await (await loadWebRenderer()).create_offscreen(
        canvas,
      )) as WebRendererRuntime;
      renderer.resize(message.width, message.height);
      break;
    }
    case "resize":
      resize(message.width, message.height);
      break;
    case "load-model":
      renderer?.load_npz_bytes(message.data);
      break;
    case "render":
      render(message.time);
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
      renderer?.free();
      renderer = null;
      canvas = null;
      pendingSize = null;
      break;
  }
};
