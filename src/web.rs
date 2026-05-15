use wasm_bindgen::prelude::*;

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
