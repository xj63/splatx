export interface Renderer {
  readonly canvas: HTMLCanvasElement;
  pause(): void;
  resume(): void;
  dispose(): void;
}

type WorkerMessage =
  | { type: "init"; canvas: OffscreenCanvas; width: number; height: number }
  | { type: "resize"; width: number; height: number }
  | { type: "set-running"; running: boolean }
  | { type: "destroy" };

function getCanvasSize(canvas: HTMLCanvasElement) {
  const scale = window.devicePixelRatio || 1;
  return {
    width: Math.max(1, Math.floor(canvas.clientWidth * scale)),
    height: Math.max(1, Math.floor(canvas.clientHeight * scale)),
  };
}

export async function createRenderer(
  canvas: HTMLCanvasElement,
): Promise<Renderer> {
  if (!("transferControlToOffscreen" in canvas)) {
    throw new Error("OffscreenCanvas is not supported by this browser");
  }

  const worker = new Worker(new URL("./worker.ts", import.meta.url), {
    type: "module",
  });
  const offscreen = canvas.transferControlToOffscreen();
  let size = getCanvasSize(canvas);
  let visible = false;
  let paused = false;
  let disposed = false;
  let running = false;
  let resizeFrame: number | null = null;

  function post(message: WorkerMessage) {
    if (!disposed) {
      worker.postMessage(message);
    }
  }

  function flushResize() {
    resizeFrame = null;

    const nextSize = getCanvasSize(canvas);
    if (nextSize.width === size.width && nextSize.height === size.height) {
      return;
    }

    size = nextSize;
    post({ type: "resize", ...size });
  }

  function scheduleResize() {
    if (resizeFrame !== null) {
      return;
    }

    resizeFrame = requestAnimationFrame(flushResize);
  }

  function updateRunning() {
    const nextRunning = visible && !document.hidden && !paused;
    if (nextRunning === running) {
      return;
    }

    running = nextRunning;
    post({ type: "set-running", running });
  }

  const resizeObserver = new ResizeObserver(() => {
    scheduleResize();
  });
  resizeObserver.observe(canvas);

  const intersectionObserver = new IntersectionObserver(([entry]) => {
    visible = Boolean(entry?.isIntersecting);
    updateRunning();
  });
  intersectionObserver.observe(canvas);

  document.addEventListener("visibilitychange", updateRunning);
  window.addEventListener("resize", scheduleResize);

  worker.postMessage(
    {
      type: "init",
      canvas: offscreen,
      width: size.width,
      height: size.height,
    } satisfies WorkerMessage,
    [offscreen],
  );

  function dispose() {
    if (disposed) return;

    disposed = true;
    if (resizeFrame !== null) {
      cancelAnimationFrame(resizeFrame);
      resizeFrame = null;
    }
    resizeObserver.disconnect();
    intersectionObserver.disconnect();
    document.removeEventListener("visibilitychange", updateRunning);
    window.removeEventListener("resize", scheduleResize);
    worker.postMessage({ type: "destroy" } satisfies WorkerMessage);
    worker.terminate();
  }

  scheduleResize();
  updateRunning();

  return {
    canvas,
    pause() {
      paused = true;
      updateRunning();
    },
    resume() {
      paused = false;
      updateRunning();
    },
    dispose,
  };
}
