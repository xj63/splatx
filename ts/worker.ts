import type { WebRenderer } from "../pkg/splatx.js";

type WorkerMessage =
  | { type: "init"; canvas: OffscreenCanvas; width: number; height: number }
  | { type: "resize"; width: number; height: number }
  | { type: "set-running"; running: boolean }
  | { type: "destroy" };

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
    case "destroy":
      setRunning(false);
      renderer?.free();
      renderer = null;
      canvas = null;
      pendingSize = null;
      break;
  }
};
