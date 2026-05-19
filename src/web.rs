use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, OffscreenCanvas};

use crate::triangle_render::TriangleRenderer;

#[wasm_bindgen(start)]
pub fn start() {
    init_logger();
}

fn init_logger() {
    // we can call the function at least once during initialization,
    // and then we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    console_error_panic_hook::set_once();

    tracing_wasm::set_as_global_default();
}

#[wasm_bindgen]
pub fn hello() {
    tracing::info!("hello");
}

#[wasm_bindgen]
pub struct WebRenderer {
    renderer: TriangleRenderer<'static>,
}

#[wasm_bindgen]
impl WebRenderer {
    pub async fn create(canvas: HtmlCanvasElement) -> Result<WebRenderer, JsValue> {
        let width = canvas.width();
        let height = canvas.height();
        let renderer = TriangleRenderer::new(wgpu::SurfaceTarget::Canvas(canvas), width, height)
            .await
            .map_err(|err| JsValue::from_str(&err))?;

        Ok(Self { renderer })
    }

    pub async fn create_offscreen(canvas: OffscreenCanvas) -> Result<WebRenderer, JsValue> {
        let width = canvas.width();
        let height = canvas.height();
        let renderer =
            TriangleRenderer::new(wgpu::SurfaceTarget::OffscreenCanvas(canvas), width, height)
                .await
                .map_err(|err| JsValue::from_str(&err))?;

        Ok(Self { renderer })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
    }

    pub fn render(&mut self) -> Result<(), JsValue> {
        self.renderer
            .render()
            .map_err(|err| JsValue::from_str(&err))
    }
}
