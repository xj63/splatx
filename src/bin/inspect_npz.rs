use std::env;

use splatx::model::SplatxModel;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: cargo run --bin inspect-npz -- <path-to-model.npz>")?;
    let model = SplatxModel::load_npz(path)?;
    print!("{}", model);
    Ok(())
}
