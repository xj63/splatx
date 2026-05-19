#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def load_converter():
    path = Path(__file__).with_name("convert-omg4ftgs-to-splatx.py")
    spec = importlib.util.spec_from_file_location("convert_omg4ftgs_to_splatx", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load converter from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats(array: np.ndarray):
    data = array.astype(np.float32, copy=False).reshape(-1)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {
            "len": data.size,
            "finite": 0,
            "mean": np.nan,
            "var": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "len": data.size,
        "finite": finite.size,
        "mean": float(np.mean(finite, dtype=np.float64)),
        "var": float(np.var(finite, dtype=np.float64)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def print_array(name: str, array: np.ndarray | None):
    if array is None:
        print(f"{name}: missing")
        return

    s = stats(array)
    print(
        f"{name}: shape={array.shape} dtype={array.dtype} "
        f"len={s['len']} finite={s['finite']} "
        f"mean={s['mean']:.9f} var={s['var']:.9f} "
        f"min={s['min']:.9f} max={s['max']:.9f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect decoded OMG4 FTGS .xz checkpoint arrays."
    )
    parser.add_argument("input", type=Path, help="Path to an OMG4 FTGS .xz checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    converter = load_converter()
    save_dict = converter.load_xz_pickle(args.input)
    arrays = converter.convert(save_dict)

    means = arrays.get("means")
    if means is not None:
        print(f"gaussians: {means.shape[0] if means.ndim > 0 else 0}")

    preferred_order = [
        "means",
        "times",
        "scales",
        "quats",
        "durations",
        "velocities",
        "features_static",
        "features_view",
        "mlp_cont",
        "mlp_dc",
        "mlp_sh",
        "mlp_opacity",
    ]

    printed = set()
    for name in preferred_order:
        print_array(name, arrays.get(name))
        printed.add(name)

    for name in sorted(set(arrays) - printed):
        print_array(name, arrays[name])


if __name__ == "__main__":
    main()
