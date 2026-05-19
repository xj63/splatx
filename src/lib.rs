pub mod model;
pub mod triangle_render;

#[cfg(feature = "web")]
#[cfg(target_arch = "wasm32")]
pub mod web;
