#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to convert biomechanics CSV data to torque motion NPZ format for AMP training.

This script takes biomechanics CSV files collected from trained agents and converts them
to NPZ format that can be used with TorqueMotionLoader for torque-based imitation learning.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def extract_torque_data_from_csv(csv_file: str) -> dict:
    """Extract torque and motion data from biomechanics CSV file.
    
    Args:
        csv_file: Path to the biomechanics CSV file
        
    Returns:
        Dictionary containing extracted data arrays
    """
    print(f"Loading biomechanics data from: {csv_file}")
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} data points")
    
    # Print available columns for debugging
    print("Available columns:")
    torque_cols = [col for col in df.columns if 'torque' in col.lower()]
    pos_cols = [col for col in df.columns if 'dof_pos' in col.lower() or 'joint_pos' in col.lower()]
    vel_cols = [col for col in df.columns if 'dof_vel' in col.lower() or 'joint_vel' in col.lower()]
    
    print(f"  Torque columns ({len(torque_cols)}): {torque_cols[:5]}{'...' if len(torque_cols) > 5 else ''}")
    print(f"  Position columns ({len(pos_cols)}): {pos_cols[:5]}{'...' if len(pos_cols) > 5 else ''}")
    print(f"  Velocity columns ({len(vel_cols)}): {vel_cols[:5]}{'...' if len(vel_cols) > 5 else ''}")
    
    if not torque_cols:
        raise ValueError("No torque columns found in CSV file. Make sure the CSV contains torque data.")
    
    # Extract time information
    timestep_col = 'timestep' if 'timestep' in df.columns else None
    time_col = 'time' if 'time' in df.columns else None
    
    if timestep_col:
        timesteps = df[timestep_col].values
        if time_col:
            times = df[time_col].values
            dt = np.mean(np.diff(times)) if len(times) > 1 else 1.0 / 60.0
        else:
            # Estimate dt from timesteps (assuming 60 FPS default)
            dt = 1.0 / 60.0 if len(timesteps) > 1 else 1.0 / 60.0
            times = timesteps * dt
    else:
        raise ValueError("No timestep column found in CSV file.")
    
    # Extract joint torques
    torque_data = df[torque_cols].values.astype(np.float32)
    num_dofs = len(torque_cols)
    
    # Try to extract DOF positions and velocities
    dof_positions = None
    dof_velocities = None
    
    if pos_cols:
        dof_positions = df[pos_cols].values.astype(np.float32)
        if dof_positions.shape[1] != num_dofs:
            print(f"Warning: Position columns ({dof_positions.shape[1]}) don't match torque columns ({num_dofs})")
    
    if vel_cols:
        dof_velocities = df[vel_cols].values.astype(np.float32)
        if dof_velocities.shape[1] != num_dofs:
            print(f"Warning: Velocity columns ({dof_velocities.shape[1]}) don't match torque columns ({num_dofs})")
    
    # If we don't have positions/velocities, create dummy data
    if dof_positions is None:
        print("Warning: No DOF position data found. Creating dummy data.")
        dof_positions = np.zeros((len(df), num_dofs), dtype=np.float32)
    
    if dof_velocities is None:
        print("Warning: No DOF velocity data found. Creating dummy data.")
        dof_velocities = np.zeros((len(df), num_dofs), dtype=np.float32)
    
    # Try to extract body pose data
    body_pos_cols = [col for col in df.columns if 'body_pos' in col.lower()]
    body_rot_cols = [col for col in df.columns if 'body_rot' in col.lower() or 'body_quat' in col.lower()]
    body_lin_vel_cols = [col for col in df.columns if 'body_lin_vel' in col.lower()]
    body_ang_vel_cols = [col for col in df.columns if 'body_ang_vel' in col.lower()]
    
    # Estimate number of bodies from body position columns
    if body_pos_cols:
        # Assuming format like body_0_pos_x, body_0_pos_y, body_0_pos_z
        num_bodies = len(body_pos_cols) // 3
        body_positions = df[body_pos_cols].values.reshape(-1, num_bodies, 3).astype(np.float32)
    else:
        print("Warning: No body position data found. Creating dummy data.")
        num_bodies = 15  # Default for humanoid
        body_positions = np.zeros((len(df), num_bodies, 3), dtype=np.float32)
    
    if body_rot_cols and len(body_rot_cols) >= num_bodies * 4:
        body_rotations = df[body_rot_cols].values.reshape(-1, num_bodies, 4).astype(np.float32)
    else:
        print("Warning: No body rotation data found. Creating dummy data.")
        body_rotations = np.zeros((len(df), num_bodies, 4), dtype=np.float32)
        body_rotations[:, :, 0] = 1.0  # Set w=1 for identity quaternions
    
    if body_lin_vel_cols and len(body_lin_vel_cols) >= num_bodies * 3:
        body_linear_velocities = df[body_lin_vel_cols].values.reshape(-1, num_bodies, 3).astype(np.float32)
    else:
        print("Warning: No body linear velocity data found. Creating dummy data.")
        body_linear_velocities = np.zeros((len(df), num_bodies, 3), dtype=np.float32)
    
    if body_ang_vel_cols and len(body_ang_vel_cols) >= num_bodies * 3:
        body_angular_velocities = df[body_ang_vel_cols].values.reshape(-1, num_bodies, 3).astype(np.float32)
    else:
        print("Warning: No body angular velocity data found. Creating dummy data.")
        body_angular_velocities = np.zeros((len(df), num_bodies, 3), dtype=np.float32)
    
    # Create DOF and body names
    dof_names = [f"joint_{i}" for i in range(num_dofs)]
    body_names = [f"body_{i}" for i in range(num_bodies)]
    
    # Try to extract actual names if available
    # Check if there are any columns that might contain joint names
    for col in df.columns:
        if 'dof_name' in col.lower() or 'joint_name' in col.lower():
            # This would require additional parsing logic
            pass
    
    fps = 1.0 / dt
    
    return {
        'fps': fps,
        'dt': dt,
        'joint_torques': torque_data,
        'dof_positions': dof_positions,
        'dof_velocities': dof_velocities,
        'body_positions': body_positions,
        'body_rotations': body_rotations,
        'body_linear_velocities': body_linear_velocities,
        'body_angular_velocities': body_angular_velocities,
        'dof_names': dof_names,
        'body_names': body_names,
        'num_frames': len(df),
        'duration': times[-1] - times[0] if len(times) > 1 else 0.0
    }


def save_torque_motion_npz(data: dict, output_file: str):
    """Save extracted data to NPZ format for TorqueMotionLoader.
    
    Args:
        data: Dictionary containing motion and torque data
        output_file: Path to save the NPZ file
    """
    print(f"Saving torque motion data to: {output_file}")
    
    np.savez(
        output_file,
        fps=np.array(data['fps']),
        dof_names=np.array(data['dof_names'], dtype='U'),
        body_names=np.array(data['body_names'], dtype='U'),
        joint_torques=data['joint_torques'],
        dof_positions=data['dof_positions'],
        dof_velocities=data['dof_velocities'],
        body_positions=data['body_positions'],
        body_rotations=data['body_rotations'],
        body_linear_velocities=data['body_linear_velocities'],
        body_angular_velocities=data['body_angular_velocities']
    )
    
    print(f"Torque motion data saved successfully!")
    print(f"Summary:")
    print(f"  Duration: {data['duration']:.2f} seconds")
    print(f"  Frames: {data['num_frames']}")
    print(f"  FPS: {data['fps']:.1f}")
    print(f"  DOFs: {len(data['dof_names'])}")
    print(f"  Bodies: {len(data['body_names'])}")
    print(f"  Joint torques shape: {data['joint_torques'].shape}")


def main():
    parser = argparse.ArgumentParser(description="Convert biomechanics CSV to torque motion NPZ")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Path to biomechanics CSV file or directory containing CSV files")
    parser.add_argument("--output", "-o", type=str, 
                        help="Output NPZ file path (default: auto-generated based on input)")
    parser.add_argument("--output-dir", type=str, default="./torque_motions",
                        help="Output directory for NPZ files when processing multiple CSVs")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix.lower() == '.csv':
        # Single CSV file
        csv_files = [input_path]
    elif input_path.is_dir():
        # Directory containing CSV files
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {input_path}")
    else:
        raise ValueError(f"Input path must be a CSV file or directory: {input_path}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(csv_files)} CSV file(s) to process")
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        
        try:
            # Extract data from CSV
            data = extract_torque_data_from_csv(str(csv_file))
            
            # Determine output file name
            if args.output and len(csv_files) == 1:
                output_file = args.output
            else:
                base_name = csv_file.stem
                output_file = output_dir / f"{base_name}_torque_motion.npz"
            
            # Save to NPZ
            save_torque_motion_npz(data, str(output_file))
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    print(f"\nProcessing complete! Torque motion files saved to: {output_dir}")


if __name__ == "__main__":
    main()
