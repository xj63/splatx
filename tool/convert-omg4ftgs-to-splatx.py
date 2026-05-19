#!/usr/bin/env python3

import argparse
import lzma
import pickle
from pathlib import Path
from typing import Any

import dahuffman
import numpy as np


HUFFMAN_GROUPS = {
    "scale": "scales",
    "rotation": "quats",
    "durations": "durations",
    "velocities": "velocities",
}

DIRECT_RENAMES = {
    "xyz": "means",
    "means": "means",
    "t": "times",
    "times": "times",
    "scaling_t": "scaling_t",
    "rotation_r": "rotation_r",
    "MLP_cont": "mlp_cont",
    "MLP_dc": "mlp_dc",
    "MLP_sh": "mlp_sh",
    "MLP_opacity": "mlp_opacity",
}


def load_xz_pickle(path: Path) -> dict[str, Any]:
    with lzma.open(path, "rb") as file:
        data = pickle.load(file)

    if not isinstance(data, dict):
        raise TypeError(f"expected checkpoint dict, got {type(data)!r}")

    return data


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def as_fp16(value: Any) -> np.ndarray:
    return to_numpy(value).astype(np.float16, copy=False)


def huffman_decode(encoded_bytes: bytes, huffman_table: Any) -> np.ndarray:
    codec = dahuffman.HuffmanCodec(code_table=huffman_table)
    return np.asarray(codec.decode(encoded_bytes), dtype=np.uint16)


def decode_group(save_dict: dict[str, Any], prefix: str) -> np.ndarray | None:
    code_key = f"{prefix}_code"
    index_key = f"{prefix}_index"
    table_key = f"{prefix}_htable"

    if code_key not in save_dict:
        return None

    missing = [key for key in (index_key, table_key) if key not in save_dict]
    if missing:
        raise KeyError(f"{prefix!r} is missing compressed fields: {', '.join(missing)}")

    decoded_parts: list[np.ndarray] = []
    codes = save_dict[code_key]
    indices = save_dict[index_key]
    tables = save_dict[table_key]

    if not (len(codes) == len(indices) == len(tables)):
        raise ValueError(
            f"{prefix!r} field lengths differ: "
            f"{len(codes)} codes, {len(indices)} indices, {len(tables)} tables"
        )

    for codebook, encoded, table in zip(codes, indices, tables):
        labels = huffman_decode(encoded, table)
        centers = to_numpy(codebook)
        decoded_parts.append(centers[labels])

    return np.concatenate(decoded_parts, axis=-1).astype(np.float16, copy=False)


def convert(save_dict: dict[str, Any]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    consumed: set[str] = set()

    for prefix, output_name in HUFFMAN_GROUPS.items():
        decoded = decode_group(save_dict, prefix)
        if decoded is None:
            continue

        output[output_name] = decoded
        consumed.update(
            {
                f"{prefix}_code",
                f"{prefix}_index",
                f"{prefix}_htable",
            }
        )

    appearance = decode_group(save_dict, "app")
    if appearance is not None:
        if appearance.shape[-1] < 6:
            raise ValueError(
                f"decoded app feature must have at least 6 channels, got {appearance.shape}"
            )

        output["features_static"] = appearance[:, 0:3].astype(np.float16, copy=False)
        output["features_view"] = appearance[:, 3:6].astype(np.float16, copy=False)
        consumed.update({"app_code", "app_index", "app_htable"})

    for source_name, output_name in DIRECT_RENAMES.items():
        if source_name in save_dict:
            output[output_name] = as_fp16(save_dict[source_name])
            consumed.add(source_name)

    for name, value in save_dict.items():
        if name in consumed or name.endswith("_index") or name.endswith("_htable"):
            continue

        if name.endswith("_code"):
            continue

        try:
            output[name] = as_fp16(value)
        except Exception:
            # Keep the npz dense and numeric. Non-array metadata can be added later
            # under an explicit schema instead of pickled object arrays.
            pass

    return output


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    name = input_path.name
    if name.endswith(".xz"):
        name = name[:-3]
    if not name.endswith(".npz"):
        name = f"{name}.npz"
    return output_dir / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an OMG4 FTGS .xz checkpoint to a dense splatx .npz file."
    )
    parser.add_argument("input", type=Path, help="Path to the OMG4 FTGS .xz checkpoint")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for converted .npz files (default: output)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for(input_path, output_dir)

    save_dict = load_xz_pickle(input_path)
    converted = convert(save_dict)

    if not converted:
        raise ValueError("conversion produced no arrays")

    np.savez(output_path, **converted)
    print(f"wrote {output_path}")
    print("arrays:")
    for name, array in converted.items():
        print(f"  {name}: shape={array.shape}, dtype={array.dtype}")


if __name__ == "__main__":
    main()
