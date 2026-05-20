export interface Renderer {
  readonly canvas: HTMLCanvasElement;
  onend: (() => void) | null;
  loadModel(data: ArrayBuffer | Uint8Array): void;
  setTime(time: number): void;
  setTimeRange(minTime: number, maxTime: number): void;
  play(options?: PlayOptions): void;
  lookAt(camera: LookAtCamera): void;
  setCamera(camera: Camera): void;
  pause(): void;
  resume(): void;
  dispose(): void;
}

export interface LookAtCamera {
  position: Vec3Tuple;
  target: Vec3Tuple;
  up?: Vec3Tuple;
}

export interface Camera extends LookAtCamera {
  fovyRadians?: number;
  znear?: number;
  zfar?: number;
}

export type Vec3Tuple = [number, number, number];

export interface PlayOptions {
  from?: number;
  to?: number;
  duration?: number;
}

type WorkerMessage =
  | { type: "init"; canvas: OffscreenCanvas; width: number; height: number }
  | { type: "resize"; width: number; height: number }
  | { type: "set-running"; running: boolean }
  | { type: "load-model"; data: Uint8Array }
  | { type: "set-time"; time: number }
  | { type: "set-time-range"; minTime: number; maxTime: number }
  | { type: "look-at"; camera: Required<LookAtCamera> }
  | { type: "set-camera"; camera: Required<Camera> }
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
  let playbackFrame: number | null = null;
  let playbackFrom = 0;
  let playbackTo = 1;
  let playbackDuration = 1;
  let playbackStart = 0;
  let onend: (() => void) | null = null;
  let yaw = 0.7;
  let pitch = 0.35;
  let distance = 3.0;
  let dragging = false;
  let lastPointerX = 0;
  let lastPointerY = 0;

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

  function stopPlayback() {
    if (playbackFrame !== null) {
      cancelAnimationFrame(playbackFrame);
      playbackFrame = null;
    }
  }

  function playbackTick(now: number) {
    const progress = Math.min(1, (now - playbackStart) / (playbackDuration * 1000));
    const time = playbackFrom + (playbackTo - playbackFrom) * progress;
    post({ type: "set-time", time });

    if (progress >= 1) {
      playbackFrame = null;
      paused = true;
      updateRunning();
      onend?.();
      return;
    }

    playbackFrame = requestAnimationFrame(playbackTick);
  }

  function updateCamera() {
    const clampedPitch = Math.max(-1.45, Math.min(1.45, pitch));
    const cp = Math.cos(clampedPitch);
    const position: Vec3Tuple = [
      Math.sin(yaw) * cp * distance,
      Math.sin(clampedPitch) * distance,
      Math.cos(yaw) * cp * distance,
    ];

    post({
      type: "look-at",
      camera: {
        position,
        target: [0, 0, 0],
        up: [0, 1, 0],
      },
    });
  }

  function handlePointerDown(event: PointerEvent) {
    dragging = true;
    lastPointerX = event.clientX;
    lastPointerY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(event: PointerEvent) {
    if (!dragging) {
      return;
    }

    const dx = event.clientX - lastPointerX;
    const dy = event.clientY - lastPointerY;
    lastPointerX = event.clientX;
    lastPointerY = event.clientY;

    yaw -= dx * 0.008;
    pitch += dy * 0.008;
    updateCamera();
  }

  function handlePointerUp(event: PointerEvent) {
    dragging = false;
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
  }

  function handleWheel(event: WheelEvent) {
    event.preventDefault();
    distance = Math.max(1.2, Math.min(20, distance * Math.exp(event.deltaY * 0.001)));
    updateCamera();
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
  canvas.addEventListener("pointerdown", handlePointerDown);
  canvas.addEventListener("pointermove", handlePointerMove);
  canvas.addEventListener("pointerup", handlePointerUp);
  canvas.addEventListener("pointercancel", handlePointerUp);
  canvas.addEventListener("wheel", handleWheel, { passive: false });

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
    stopPlayback();
    resizeObserver.disconnect();
    intersectionObserver.disconnect();
    document.removeEventListener("visibilitychange", updateRunning);
    window.removeEventListener("resize", scheduleResize);
    canvas.removeEventListener("pointerdown", handlePointerDown);
    canvas.removeEventListener("pointermove", handlePointerMove);
    canvas.removeEventListener("pointerup", handlePointerUp);
    canvas.removeEventListener("pointercancel", handlePointerUp);
    canvas.removeEventListener("wheel", handleWheel);
    worker.postMessage({ type: "destroy" } satisfies WorkerMessage);
    worker.terminate();
  }

  scheduleResize();
  updateCamera();
  updateRunning();

  const renderer: Renderer = {
    canvas,
    get onend() {
      return onend;
    },
    set onend(callback) {
      onend = callback;
    },
    loadModel(data) {
      const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
      post({ type: "load-model", data: bytes });
    },
    setTime(time) {
      stopPlayback();
      post({ type: "set-time", time });
    },
    setTimeRange(minTime, maxTime) {
      post({ type: "set-time-range", minTime, maxTime });
    },
    play(options = {}) {
      stopPlayback();
      playbackFrom = options.from ?? 0;
      playbackTo = options.to ?? 1;
      playbackDuration = Math.max(0.001, options.duration ?? 6);
      playbackStart = performance.now();
      paused = false;
      updateRunning();
      post({ type: "set-time-range", minTime: playbackFrom, maxTime: playbackTo });
      post({ type: "set-time", time: playbackFrom });
      playbackFrame = requestAnimationFrame(playbackTick);
    },
    lookAt(camera) {
      post({
        type: "look-at",
        camera: {
          position: camera.position,
          target: camera.target,
          up: camera.up ?? [0, 1, 0],
        },
      });
    },
    setCamera(camera) {
      post({
        type: "set-camera",
        camera: {
          position: camera.position,
          target: camera.target,
          up: camera.up ?? [0, 1, 0],
          fovyRadians: camera.fovyRadians ?? Math.PI / 4,
          znear: camera.znear ?? 0.01,
          zfar: camera.zfar ?? 10000,
        },
      });
    },
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

  return renderer;
}
