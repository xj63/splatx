use std::{env, process};

use half::f16;
use ndarray::ArrayD;
use splatx::model::{F16Array, SplatxModel};

struct Stats {
    len: usize,
    finite: usize,
    mean: f64,
    variance: f64,
    min: f32,
    max: f32,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: cargo run --bin inspect-npz -- <path-to-model.npz>")?;
    let model = SplatxModel::load_npz(path)?;

    println!("gaussians: {}", model.len());
    print_array("means", Some(&model.means));
    print_array("times", model.times.as_ref());
    print_array("scales", model.scales.as_ref());
    print_array("quats", model.quats.as_ref());
    print_array("durations", model.durations.as_ref());
    print_array("velocities", model.velocities.as_ref());
    print_array("features_static", model.features_static.as_ref());
    print_array("features_view", model.features_view.as_ref());
    print_array("mlp_cont", model.mlp_cont.as_ref());
    print_array("mlp_dc", model.mlp_dc.as_ref());
    print_array("mlp_sh", model.mlp_sh.as_ref());
    print_array("mlp_opacity", model.mlp_opacity.as_ref());

    for (name, array) in &model.extra {
        print_array(name, Some(array));
    }

    Ok(())
}

fn print_array(name: &str, array: Option<&F16Array>) {
    match array {
        Some(array) => {
            let stats = stats(array);
            println!(
                "{name}: shape={:?} dtype=f16 len={} finite={} mean={:.9} var={:.9} min={:.9} max={:.9}",
                array.shape(),
                stats.len,
                stats.finite,
                stats.mean,
                stats.variance,
                stats.min,
                stats.max,
            );
        }
        None => {
            println!("{name}: missing");
        }
    }
}

fn stats(array: &ArrayD<f16>) -> Stats {
    let mut len = 0_usize;
    let mut finite = 0_usize;
    let mut mean = 0_f64;
    let mut m2 = 0_f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for value in array.iter().map(|value| value.to_f32()) {
        len += 1;
        if !value.is_finite() {
            continue;
        }

        finite += 1;
        min = min.min(value);
        max = max.max(value);

        let value = value as f64;
        let delta = value - mean;
        mean += delta / finite as f64;
        let delta2 = value - mean;
        m2 += delta * delta2;
    }

    let variance = if finite > 0 { m2 / finite as f64 } else { f64::NAN };
    if finite == 0 {
        min = f32::NAN;
        max = f32::NAN;
    }

    Stats {
        len,
        finite,
        mean,
        variance,
        min,
        max,
    }
}
