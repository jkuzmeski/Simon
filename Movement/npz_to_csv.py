import numpy as np
import pandas as pd
import os
import argparse
import json

def npz_to_csv(npz_file_path, output_dir=None):
    """
    Convert motion data in NPZ file to separate CSV files for each key.
    
    Args:
        npz_file_path: Path to the NPZ motion file
        output_dir: Directory to save CSV files (default is same directory as NPZ)
    """
    # Load the NPZ file
    data = np.load(npz_file_path, allow_pickle=True)
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.dirname(npz_file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    basename = os.path.splitext(os.path.basename(npz_file_path))[0]
    
    # Create metadata dictionary to store information about the data
    metadata = {}
    
    # Get body and DOF names for better column naming
    if 'body_names' in data:
        body_names = data['body_names']
    else:
        body_names = [f"body{i}" for i in range(data['body_positions'].shape[1])]
        
    if 'dof_names' in data:
        dof_names = data['dof_names']
    else:
        dof_names = [f"dof{i}" for i in range(data['dof_positions'].shape[1])]
    
    # Process each key in the data
    for key in data.keys():
        print(f"Processing {key}...")
        value = data[key]
        output_path = os.path.join(output_dir, f"{basename}_{key}.csv")
        
        # Store metadata about this key
        metadata[key] = {
            'dtype': str(value.dtype),
            'shape': str(value.shape),
            'file': os.path.basename(output_path)
        }
        
        # Handle different data types and shapes
        if key == 'fps':
            # Save scalar as simple file
            with open(output_path, 'w') as f:
                f.write(f"{value}")
                
        elif key in ['dof_names', 'body_names']:
            # Save string arrays
            pd.DataFrame(value, columns=[key]).to_csv(output_path, index=False)
            
        elif key == 'dof_positions':
            # 2D array (N, D) with named columns from dof_names
            df = pd.DataFrame(value, columns=dof_names)
            df.to_csv(output_path, index=False)
            
        elif key == 'dof_velocities':
            # 2D array (N, D) with named columns from dof_names
            df = pd.DataFrame(value, columns=[f"{name}_vel" for name in dof_names])
            df.to_csv(output_path, index=False)
            
        elif key == 'body_positions':
            # 3D array (N, B, 3) - position of each body
            N, B, dims = value.shape
            # Create column headers using body names
            columns = []
            for b in range(B):
                for d in range(dims):
                    component = ['x', 'y', 'z'][d]
                    columns.append(f"{body_names[b]}_{component}")
            
            # Reshape to 2D for CSV
            reshaped = value.reshape(N, B * dims)
            df = pd.DataFrame(reshaped, columns=columns)
            df.to_csv(output_path, index=False)
            
        elif key == 'body_rotations':
            # 3D array (N, B, 4) - quaternions
            N, B, dims = value.shape
            # Create column headers
            columns = []
            for b in range(B):
                for d in range(dims):
                    component = ['w', 'x', 'y', 'z'][d]
                    columns.append(f"{body_names[b]}_{component}")
            
            # Reshape to 2D for CSV
            reshaped = value.reshape(N, B * dims)
            df = pd.DataFrame(reshaped, columns=columns)
            df.to_csv(output_path, index=False)
            
        elif key in ['body_linear_velocities', 'body_angular_velocities']:
            # 3D array (N, B, 3)
            N, B, dims = value.shape
            suffix = 'lin_vel' if key == 'body_linear_velocities' else 'ang_vel'
            
            # Create column headers
            columns = []
            for b in range(B):
                for d in range(dims):
                    component = ['x', 'y', 'z'][d]
                    columns.append(f"{body_names[b]}_{suffix}_{component}")
            
            # Reshape to 2D for CSV
            reshaped = value.reshape(N, B * dims)
            df = pd.DataFrame(reshaped, columns=columns)
            df.to_csv(output_path, index=False)
        
        print(f"Saved {output_path}")
    
    # Save metadata.json for reference
    metadata_path = os.path.join(output_dir, f"{basename}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    print(f"All data exported to CSV files in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert motion NPZ file to CSV files")
    parser.add_argument("--file", required=True, help="Input NPZ motion file")
    parser.add_argument("--output-dir", help="Output directory for CSV files")
    args = parser.parse_args()
    
    npz_to_csv(args.file, args.output_dir)
