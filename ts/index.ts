export interface Renderer {
  readonly canvas: HTMLCanvasElement;
  loadModel(data: ArrayBuffer | Uint8Array): void;
  render(time: number): void;
  setCamera(camera: Camera): void;
  dispose(): void;
}

export interface Camera {
  position: Vec3Tuple;
  target: Vec3Tuple;
  up?: Vec3Tuple;
  fovyRadians?: number;
  znear?: number;
  zfar?: number;
}

export type Vec3Tuple = [number, number, number];

type WorkerMessage =
  | { type: "init"; canvas: OffscreenCanvas; width: number; height: number }
  | { type: "resize"; width: number; height: number }
  | { type: "load-model"; data: Uint8Array }
  | { type: "render"; time: number }
  | { type: "set-camera"; camera: Required<Camera> }
  | { type: "destroy" };

const defaultCamera: Required<Camera> = {
  position: [
    0.44396468423162905,
    -1.1035034400953323,
    -0.3499272977464685,
  ],
  target: [
    0.5478754702925966,
    -1.0966363123253844,
    0.6446356168528959,
  ],
  up: [
    0.9943491646907965,
    0.021132652881061354,
    -0.10403436769127798,
  ],
  fovyRadians: 1.2135940334461468,
  znear: 8.831384232013642,
  zfar: 109.77542390000633,
};

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
  const visibility = createVisibilityTracker(canvas);
  const resize = createResizeTracker(canvas, (size) => {
    post({ type: "resize", ...size });
  });
  const cameraControls = createOrbitCameraControls(canvas, setCamera);

  let disposed = false;

  function post(message: WorkerMessage) {
    if (!disposed) {
      worker.postMessage(message);
    }
  }

  function setCamera(camera: Camera) {
    post({
      type: "set-camera",
      camera: normalizeCamera(camera),
    });
  }

  worker.postMessage(
    {
      type: "init",
      canvas: offscreen,
      ...resize.current(),
    } satisfies WorkerMessage,
    [offscreen],
  );
  cameraControls.sync();

  return {
    canvas,
    loadModel(data) {
      const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
      post({ type: "load-model", data: bytes });
    },
    render(time) {
      if (visibility.visible()) {
        post({ type: "render", time });
      }
    },
    setCamera,
    dispose() {
      if (disposed) {
        return;
      }

      disposed = true;
      resize.dispose();
      visibility.dispose();
      cameraControls.dispose();
      worker.postMessage({ type: "destroy" } satisfies WorkerMessage);
      worker.terminate();
    },
  };
}

function normalizeCamera(camera: Camera): Required<Camera> {
  return {
    position: camera.position,
    target: camera.target,
    up: camera.up ?? [0, 1, 0],
    fovyRadians: camera.fovyRadians ?? Math.PI / 4,
    znear: camera.znear ?? 0.01,
    zfar: camera.zfar ?? 10000,
  };
}

function getCanvasSize(canvas: HTMLCanvasElement) {
  const scale = window.devicePixelRatio || 1;
  return {
    width: Math.max(1, Math.floor(canvas.clientWidth * scale)),
    height: Math.max(1, Math.floor(canvas.clientHeight * scale)),
  };
}

function createResizeTracker(
  canvas: HTMLCanvasElement,
  onResize: (size: { width: number; height: number }) => void,
) {
  let size = getCanvasSize(canvas);
  let frame: number | null = null;

  function flush() {
    frame = null;
    const nextSize = getCanvasSize(canvas);
    if (nextSize.width === size.width && nextSize.height === size.height) {
      return;
    }

    size = nextSize;
    onResize(size);
  }

  function schedule() {
    if (frame === null) {
      frame = requestAnimationFrame(flush);
    }
  }

  const resizeObserver = new ResizeObserver(schedule);
  resizeObserver.observe(canvas);
  window.addEventListener("resize", schedule);

  return {
    current: () => size,
    dispose() {
      if (frame !== null) {
        cancelAnimationFrame(frame);
      }
      resizeObserver.disconnect();
      window.removeEventListener("resize", schedule);
    },
  };
}

function createVisibilityTracker(canvas: HTMLCanvasElement) {
  let visible = true;

  const intersectionObserver = new IntersectionObserver(([entry]) => {
    visible = Boolean(entry?.isIntersecting);
  });
  intersectionObserver.observe(canvas);

  return {
    visible: () => visible && !document.hidden,
    dispose() {
      intersectionObserver.disconnect();
    },
  };
}

function createOrbitCameraControls(
  canvas: HTMLCanvasElement,
  setCamera: (camera: Camera) => void,
) {
  const target = defaultCamera.target;
  const up = defaultCamera.up;
  const fovyRadians = defaultCamera.fovyRadians;
  const znear = defaultCamera.znear;
  const zfar = defaultCamera.zfar;
  const initialOffset = subtract(defaultCamera.position, target);
  let distance = Math.max(0.0001, length(initialOffset));
  let yaw = Math.atan2(initialOffset[0], initialOffset[2]);
  let pitch = Math.asin(clamp(initialOffset[1] / distance, -1, 1));
  let dragging = false;
  let lastPointerX = 0;
  let lastPointerY = 0;

  function sync() {
    const clampedPitch = clamp(pitch, -1.45, 1.45);
    const cp = Math.cos(clampedPitch);
    setCamera({
      position: [
        target[0] + Math.sin(yaw) * cp * distance,
        target[1] + Math.sin(clampedPitch) * distance,
        target[2] + Math.cos(yaw) * cp * distance,
      ],
      target,
      up,
      fovyRadians,
      znear,
      zfar,
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
    sync();
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
    sync();
  }

  canvas.addEventListener("pointerdown", handlePointerDown);
  canvas.addEventListener("pointermove", handlePointerMove);
  canvas.addEventListener("pointerup", handlePointerUp);
  canvas.addEventListener("pointercancel", handlePointerUp);
  canvas.addEventListener("wheel", handleWheel, { passive: false });

  return {
    sync,
    dispose() {
      canvas.removeEventListener("pointerdown", handlePointerDown);
      canvas.removeEventListener("pointermove", handlePointerMove);
      canvas.removeEventListener("pointerup", handlePointerUp);
      canvas.removeEventListener("pointercancel", handlePointerUp);
      canvas.removeEventListener("wheel", handleWheel);
    },
  };
}

function subtract(a: Vec3Tuple, b: Vec3Tuple): Vec3Tuple {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function length(value: Vec3Tuple): number {
  return Math.hypot(value[0], value[1], value[2]);
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
