#!/usr/bin/env python3

"""
Test script for the torque-based motion workflow.
This script validates the complete pipeline without requiring a full simulation.
"""

import numpy as np
import os
import tempfile
from pathlib import Path

def create_mock_torque_data():
    """Create mock torque motion data for testing."""
    print("Creating mock torque motion data...")
    
    # Mock parameters
    fps = 60.0
    duration = 2.0  # 2 seconds
    num_frames = int(fps * duration)
    num_dofs = 28  # Typical humanoid DOF count
    num_bodies = 15  # Typical humanoid body count
    
    # Generate synthetic motion data
    time_array = np.linspace(0, duration, num_frames)
    
    # Mock joint positions (simple sine wave)
    dof_positions = np.zeros((num_frames, num_dofs))
    for i in range(num_dofs):
        dof_positions[:, i] = 0.1 * np.sin(2 * np.pi * (i + 1) * time_array / duration)
    
    # Mock joint velocities (derivative of positions)
    dof_velocities = np.zeros((num_frames, num_dofs))
    for i in range(num_dofs):
        dof_velocities[:, i] = 0.1 * 2 * np.pi * (i + 1) / duration * np.cos(2 * np.pi * (i + 1) * time_array / duration)
    
    # Mock joint torques (correlated with velocities)
    joint_torques = np.zeros((num_frames, num_dofs))
    for i in range(num_dofs):
        # Torque roughly proportional to velocity with some damping
        joint_torques[:, i] = -0.5 * dof_velocities[:, i] + 0.1 * np.random.normal(0, 0.01, num_frames)
    
    # Mock body poses
    body_positions = np.random.normal(0, 0.1, (num_frames, num_bodies, 3))
    body_rotations = np.zeros((num_frames, num_bodies, 4))
    body_rotations[:, :, 0] = 1.0  # w=1 for identity quaternions
    body_linear_velocities = np.random.normal(0, 0.1, (num_frames, num_bodies, 3))
    body_angular_velocities = np.random.normal(0, 0.1, (num_frames, num_bodies, 3))
    
    # Create names
    dof_names = [f"joint_{i}" for i in range(num_dofs)]
    body_names = [f"body_{i}" for i in range(num_bodies)]
    
    return {
        'fps': fps,
        'dof_names': dof_names,
        'body_names': body_names,
        'joint_torques': joint_torques.astype(np.float32),
        'dof_positions': dof_positions.astype(np.float32),
        'dof_velocities': dof_velocities.astype(np.float32),
        'body_positions': body_positions.astype(np.float32),
        'body_rotations': body_rotations.astype(np.float32),
        'body_linear_velocities': body_linear_velocities.astype(np.float32),
        'body_angular_velocities': body_angular_velocities.astype(np.float32)
    }

def test_torque_motion_loader():
    """Test the TorqueMotionLoader class."""
    print("\n=== Testing TorqueMotionLoader ===")
    
    # Create mock data
    data = create_mock_torque_data()
    
    # Save to temporary NPZ file
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
        np.savez(tmp_file.name, **data)
        temp_npz_path = tmp_file.name
    
    try:
        # Test importing TorqueMotionLoader
        from torque_motion_loader import TorqueMotionLoader
        
        # Load the mock data
        loader = TorqueMotionLoader(temp_npz_path, device="cpu")
        
        print(f"✓ Loaded torque motion: {loader.num_frames} frames, {loader.duration:.2f}s")
        print(f"✓ DOFs: {loader.num_dofs}, Bodies: {loader.num_bodies}")
        
        # Test sampling with torques
        dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel, torques = loader.sample_with_torques(5)
        
        print(f"✓ Sampled data shapes:")
        print(f"  - DOF positions: {dof_pos.shape}")
        print(f"  - Joint torques: {torques.shape}")
        
        # Test single time sampling
        single_torques = loader.get_torques_at_time(0.5)
        print(f"✓ Single time torques shape: {single_torques.shape}")
        
        # Test backward compatibility
        regular_sample = loader.sample(3)
        print(f"✓ Regular sampling (6 outputs): {len(regular_sample)} outputs")
        
        print("✓ TorqueMotionLoader tests passed!")
        
    except ImportError as e:
        print(f"✗ Failed to import TorqueMotionLoader: {e}")
        return False
    except Exception as e:
        print(f"✗ TorqueMotionLoader test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(temp_npz_path):
            os.unlink(temp_npz_path)
    
    return True

def test_csv_converter():
    """Test the CSV to torque motion converter."""
    print("\n=== Testing CSV to Torque Motion Converter ===")
    
    # Create mock CSV data
    mock_data = create_mock_torque_data()
    num_frames = mock_data['joint_torques'].shape[0]
    num_dofs = mock_data['joint_torques'].shape[1]
    
    # Create CSV-like data structure
    csv_data = {}
    csv_data['timestep'] = np.arange(num_frames)
    csv_data['time'] = np.arange(num_frames) / mock_data['fps']
    
    # Add torque columns
    for i in range(num_dofs):
        csv_data[f'joint_torque_{i}'] = mock_data['joint_torques'][:, i]
        csv_data[f'dof_pos_{i}'] = mock_data['dof_positions'][:, i]
        csv_data[f'dof_vel_{i}'] = mock_data['dof_velocities'][:, i]
    
    # Create temporary CSV file
    import pandas as pd
    df = pd.DataFrame(csv_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as tmp_file:
        df.to_csv(tmp_file.name, index=False)
        temp_csv_path = tmp_file.name
    
    try:
        # Test the converter
        from csv_to_torque_motion import extract_torque_data_from_csv, save_torque_motion_npz
        
        # Extract data
        extracted_data = extract_torque_data_from_csv(temp_csv_path)
        
        print(f"✓ Extracted data from CSV:")
        print(f"  - Frames: {extracted_data['num_frames']}")
        print(f"  - Duration: {extracted_data['duration']:.2f}s")
        print(f"  - DOFs: {len(extracted_data['dof_names'])}")
        print(f"  - Torques shape: {extracted_data['joint_torques'].shape}")
        
        # Save to NPZ
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_npz:
            save_torque_motion_npz(extracted_data, tmp_npz.name)
            temp_npz_path = tmp_npz.name
        
        print("✓ CSV to NPZ conversion successful!")
        
        # Verify the NPZ can be loaded
        npz_data = np.load(temp_npz_path)
        expected_keys = ['fps', 'dof_names', 'body_names', 'joint_torques', 'dof_positions', 'dof_velocities']
        for key in expected_keys:
            if key not in npz_data:
                print(f"✗ Missing key in NPZ: {key}")
                return False
        
        print("✓ NPZ validation passed!")
        
        # Clean up NPZ
        if os.path.exists(temp_npz_path):
            os.unlink(temp_npz_path)
            
    except ImportError as e:
        print(f"✗ Failed to import converter modules: {e}")
        return False
    except Exception as e:
        print(f"✗ CSV converter test failed: {e}")
        return False
    finally:
        # Clean up CSV
        if os.path.exists(temp_csv_path):
            os.unlink(temp_csv_path)
    
    return True

def main():
    """Run all tests."""
    print("=== Torque-Based Motion Workflow Test Suite ===")
    
    # Change to Movement directory
    script_dir = Path(__file__).parent
    movement_dir = script_dir / "Movement"
    
    if movement_dir.exists():
        os.chdir(movement_dir)
        print(f"Working directory: {os.getcwd()}")
    else:
        print(f"Warning: Movement directory not found at {movement_dir}")
        print(f"Current directory: {os.getcwd()}")
    
    # Run tests
    tests_passed = 0
    total_tests = 2
    
    if test_torque_motion_loader():
        tests_passed += 1
    
    if test_csv_converter():
        tests_passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! The torque workflow is ready to use.")
        print("\nNext steps:")
        print("1. Run biomechanics.py with --save_torque_profiles to collect real data")
        print("2. Use train_torque_amp.py to train agents with the collected torque profiles")
    else:
        print("✗ Some tests failed. Check the errors above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    main()
