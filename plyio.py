# Modified from https://antimatter15.com/splat

from plyfile import PlyData
import numpy as np
import argparse
from io import BytesIO


def splat_to_numpy(file_path):
    with open(file_path, 'rb') as f:
        splat_data = f.read()

    splat_dtype = np.dtype([
        ('position', np.float32, 3),
        ('scale', np.float32, 3),
        ('color', np.uint8, 4),
        ('rotation', np.uint8, 4)
    ])

    
    splat_array = np.frombuffer(splat_data, dtype=splat_dtype)

    points = splat_array["position"]
    scales = splat_array["scale"]
    rots = (splat_array["rotation"]/255)*2 - 1
    color = splat_array["color"]/255
    
    
    return points, scales, rots.astype(np.float32), color.astype(np.float32)


def ply_to_numpy(ply_file_path):
    plydata = PlyData.read(ply_file_path)
    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    buffer = BytesIO()

    positions = np.zeros((len(sorted_indices), 3), dtype=np.float32)
    scales = np.zeros((len(sorted_indices), 3), dtype=np.float32)
    rots = np.zeros((len(sorted_indices), 4), dtype=np.float32)
    colors = np.zeros((len(sorted_indices), 4), dtype=np.float32)

    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        scale = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        SH_C0 = 0.28209479177387814
        color = np.array(
            [
                0.5 + SH_C0 * v["f_dc_0"],
                0.5 + SH_C0 * v["f_dc_1"],
                0.5 + SH_C0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"])),
            ]
        )
        
        
        positions[idx] = position
        scales[idx] = scale
        rots[idx] = rot
        colors[idx] = color
    return positions, scales, rots, colors




def numpy_to_splat(positions, scales, rots, colors, output_path, file_type):
    buffer = BytesIO()
    if file_type == 'ply':
        
        for idx in range(len(positions)):
            position = positions[idx]
            scale = scales[idx]
            rot = rots[idx]
            color = colors[idx]
            buffer.write(position.tobytes())
            buffer.write(scale.tobytes())
            buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write(((rot / np.linalg.norm(rot)) * 128 + 128).clip(0, 255).astype(np.uint8).tobytes()
            )

        splat_data = buffer.getvalue()
        with open(output_path, "wb") as f:
            f.write(splat_data)
    else:
        for idx in range(len(positions)):
            position = positions[idx]
            scale = scales[idx]
            rot = rots[idx]
            color = colors[idx]
            buffer.write(position.tobytes())
            buffer.write(scale.tobytes())
            buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write(((rot / np.linalg.norm(rot)) * 128 + 128).clip(0, 255).astype(np.uint8).tobytes()
            )

        splat_data = buffer.getvalue()
        with open(output_path, "wb") as f:
            f.write(splat_data)
        
    return splat_data




def main():
    parser = argparse.ArgumentParser(description="Convert PLY files to SPLAT format.")
    parser.add_argument(
        "input_files", nargs="+", help="The input PLY files to process."
    )
    parser.add_argument(
        "--output", "-o", default="output.splat", help="The output SPLAT file."
    )
    args = parser.parse_args()
    for input_file in args.input_files:
        print(f"Processing {input_file}...")
        positions, scales, rotations, colors = ply_to_numpy(input_file)

        numpy_to_splat(positions, scales, rotations, colors, args.output)



if __name__ == "__main__":
    main()