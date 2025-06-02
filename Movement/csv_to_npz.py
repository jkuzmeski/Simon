import numpy as np
import pandas as pd
import os
import argparse


def csv_to_npz(input_dir, output_file):
    """
    Convert CSV files to NPZ format motion file.

    Args:
        input_dir: Directory containing the CSV files
        output_file: Path for the output NPZ file
    """
    # Dictionary to store all data that will be saved to NPZ
    npz_data = {}

    # Load fps (scalar)
    fps_files = [f for f in os.listdir(input_dir) if f.endswith('_fps.csv')]
    if fps_files:
        with open(os.path.join(input_dir, fps_files[0]), 'r') as f:
            npz_data['fps'] = np.array(int(f.read().strip()))

    # Load string arrays
    for key in ['dof_names', 'body_names']:
        files = [f for f in os.listdir(input_dir) if f.endswith(f'_{key}.csv')]
        if files:
            df = pd.read_csv(os.path.join(input_dir, files[0]))
            npz_data[key] = np.array(df[key].values, dtype='U')

    # Load 2D arrays
    for key in ['dof_positions', 'dof_velocities']:
        files = [f for f in os.listdir(input_dir) if f.endswith(f'_{key}.csv')]
        if files:
            df = pd.read_csv(os.path.join(input_dir, files[0]))
            npz_data[key] = df.values.astype(np.float32)

    # Determine dimensions from body_positions if available
    body_pos_files = [f for f in os.listdir(input_dir) if f.endswith('_body_positions.csv')]
    if body_pos_files:
        df = pd.read_csv(os.path.join(input_dir, body_pos_files[0]))
        N = len(df)  # Number of frames

        # Determine number of bodies based on column names
        # Assuming format is body_name_x, body_name_y, body_name_z
        # cols = df.columns
        # B = len(cols) // 3  # Each body has x, y, z

        # Load and reshape 3D arrays
        for key, dims in [
            ('body_positions', 3),
            ('body_rotations', 4),
            ('body_linear_velocities', 3),
            ('body_angular_velocities', 3)
        ]:
            files = [f for f in os.listdir(input_dir) if f.endswith(f'_{key}.csv')]
            if files:
                df = pd.read_csv(os.path.join(input_dir, files[0]))
                if key == 'body_rotations':
                    B_actual = len(df.columns) // 4  # Each body has w, x, y, z for rotations
                else:
                    B_actual = len(df.columns) // 3  # Each body has x, y, z

                # Reshape to 3D
                npz_data[key] = df.values.reshape(N, B_actual, dims).astype(np.float32)

    # Save to NPZ file
    np.savez(output_file, **npz_data)
    print(f"Motion data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV files to NPZ motion file")
    parser.add_argument("--input-dir", required=True, help="Directory containing CSV files")
    parser.add_argument("--output", required=True, help="Output NPZ file path")
    args = parser.parse_args()

    csv_to_npz(args.input_dir, args.output)
