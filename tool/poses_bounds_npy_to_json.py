import numpy as np
import json
import argparse
import sys
import os

def poses_to_json(npy_path):
    """
    Converts LLFF poses_bounds.npy to a structured JSON format.
    
    LLFF format:
    - Each row is 17 elements.
    - First 15 elements: 3x5 matrix (Rotation, Translation, [H, W, Focal])
    - Last 2 elements: [Near, Far] bounds
    """
    try:
        poses_bounds = np.load(npy_path)
    except Exception as e:
        print(f"Error loading {npy_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Reshape to (N, 17) if it's a flat array (though LLFF usually isn't)
    if len(poses_bounds.shape) == 1:
        poses_bounds = poses_bounds.reshape(-1, 17)

    output = {
        "description": "Camera parameters extracted from LLFF poses_bounds.npy",
        "fields": {
            "rotation": "3x3 rotation matrix (World-to-Camera or Camera-to-World depending on conventions)",
            "position": "3x1 camera position in world space",
            "intrinsics": {
                "height": "Image height in pixels",
                "width": "Image width in pixels",
                "focal_length": "Focal length in pixels"
            },
            "bounds": {
                "near": "Nearest depth of the scene from this camera",
                "far": "Farthest depth of the scene from this camera"
            },
        },
        "cameras": []
    }

    for i in range(len(poses_bounds)):
        # Extract the 3x5 matrix and the 2 bounds
        pose_15 = poses_bounds[i, :15].reshape(3, 5)
        bounds = poses_bounds[i, 15:]

        # Extract components from 3x5
        # LLFF poses are [R | T | [H, W, fl]]
        rotation_matrix = pose_15[:, :3]
        translation = pose_15[:, 3]
        hwf = pose_15[:, 4]

        h, w, fl = hwf

        cam_info = {
            "id": i,
            "intrinsics": {
                "height": float(h),
                "width": float(w),
                "focal_length": float(fl)
            },
            "bounds": {
                "near": float(bounds[0]),
                "far": float(bounds[1])
            },
            "rotation": rotation_matrix.tolist(),
            "position": translation.tolist(),
        }
        output["cameras"].append(cam_info)

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LLFF 'poses_bounds.npy' to a readable JSON format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Field Explanations:
  - rotation: The orientation of the camera (3x3 matrix).
  - position: The (x, y, z) coordinates of the camera in the world.
  - intrinsics:
      - height/width: Resolution of the image.
      - focal_length: Focal length in pixels.
  - bounds: The depth range [near, far] relevant for rendering.
        """
    )
    parser.add_argument("input", help="Path to the 'poses_bounds.npy' file")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    json_data = poses_to_json(args.input)
    print(json.dumps(json_data, indent=4))
