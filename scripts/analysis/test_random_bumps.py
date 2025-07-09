#!/usr/bin/env python3
"""
Test script for random force bump functionality.

This script runs a short simulation with random bumps enabled to verify
the implementation is working correctly.

Usage:
    python test_random_bumps.py --task Simon-Biomech-Run-v0 --checkpoint path/to/checkpoint.pt
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_bump_test(task: str, checkpoint: str):
    """Run a test simulation with random bumps."""
    
    print("Testing random force bump functionality...")
    print(f"Task: {task}")
    print(f"Checkpoint: {checkpoint}")
    
    # Construct command
    bump_script = Path(__file__).parent.parent / "skrl" / "bump.py"
    
    cmd = [
        sys.executable,
        str(bump_script),
        "--task", task,
        "--num_envs", "1",
        "--checkpoint", checkpoint,
        "--use_distance_termination",
        "--max_distance", "25.0",  # Short distance for quick test
        "--enable_random_bumps",
        "--bump_force_magnitude", "50.0",
        "--bump_interval_range", "2.0", "5.0",  # Frequent bumps for testing
        "--save_biomechanics_data"
    ]
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    print("\nExpected behavior:")
    print("- Simulation should start normally")
    print("- Random bump events should be logged with [BUMP] messages")
    print("- Two CSV files should be created (biomechanics + bump events)")
    print("- Simulation should terminate after traveling ~25 meters")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\nTest completed successfully! Exit code: {result.returncode}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTest failed with exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test random force bump functionality')
    parser.add_argument('--task', default='Simon-Biomech-Run-v0', help='Environment task to test')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Run test
    success = run_bump_test(args.task, args.checkpoint)
    
    if success:
        print("\n" + "="*50)
        print("BUMP TEST SUCCESSFUL!")
        print("="*50)
        print("Next steps:")
        print("1. Check the logs directory for CSV files")
        print("2. Run the analysis script on the generated data")
        print("3. Experiment with different bump parameters")
    else:
        print("\n" + "="*50)
        print("BUMP TEST FAILED!")
        print("="*50)
        print("Check the error messages above and:")
        print("1. Verify your checkpoint path is correct")
        print("2. Ensure the task name is valid")
        print("3. Check that all dependencies are installed")
        
        sys.exit(1)


if __name__ == "__main__":
    main()