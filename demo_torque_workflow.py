#!/usr/bin/env python3
"""
Torque Workflow Demonstration Script

This script demonstrates the complete torque-based motion imitation workflow.
It can be used to validate that all components are working correctly.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the workspace root to the Python path
WORKSPACE_ROOT = Path(__file__).parent
sys.path.append(str(WORKSPACE_ROOT))

def check_torque_motion_loader():
    """Check if TorqueMotionLoader is available and working."""
    print("=== Testing TorqueMotionLoader ===")
    try:
        from Movement.torque_motion_loader import MotionLoader as TorqueMotionLoader
        print("✅ TorqueMotionLoader imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import TorqueMotionLoader: {e}")
        return False

def check_environment_config():
    """Check if torque environment configuration is working."""
    print("\n=== Testing Environment Configuration ===")
    try:
        from source.Simon.Simon.tasks.direct.simon_torque.simon_torque_env_cfg import SimonTorqueEnvCfg
        cfg = SimonTorqueEnvCfg()
        print(f"✅ Environment config loaded")
        print(f"   - Enable torque AMP: {cfg.enable_torque_amp}")
        print(f"   - Motion file: {cfg.motion_file}")
        print(f"   - AMP observation space: {cfg.amp_observation_space}")
        return True
    except Exception as e:
        print(f"❌ Failed to load environment config: {e}")
        return False

def check_converter_script():
    """Check if the CSV to NPZ converter exists."""
    print("\n=== Testing CSV to NPZ Converter ===")
    converter_path = WORKSPACE_ROOT / "Movement" / "csv_to_torque_motion.py"
    if converter_path.exists():
        print(f"✅ Converter script found: {converter_path}")
        return True
    else:
        print(f"❌ Converter script not found: {converter_path}")
        return False

def check_training_script():
    """Check if the torque AMP training script exists."""
    print("\n=== Testing Training Script ===")
    training_path = WORKSPACE_ROOT / "scripts" / "skrl" / "train_torque_amp.py"
    if training_path.exists():
        print(f"✅ Training script found: {training_path}")
        return True
    else:
        print(f"❌ Training script not found: {training_path}")
        return False

def check_directories():
    """Check if required directories exist."""
    print("\n=== Checking Directories ===")
    required_dirs = [
        "Movement/torque_motions",
        "Movement",
        "scripts/skrl",
        "source/Simon/Simon/tasks/direct/simon_torque",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = WORKSPACE_ROOT / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (missing)")
            all_exist = False
    
    return all_exist

def list_available_motions():
    """List available motion files."""
    print("\n=== Available Motion Files ===")
    
    # Check regular motions
    motion_dir = WORKSPACE_ROOT / "Movement"
    regular_motions = list(motion_dir.glob("*.npz"))
    print(f"Regular motions ({len(regular_motions)}):")
    for motion in regular_motions:
        print(f"  - {motion.name}")
    
    # Check torque motions
    torque_motion_dir = WORKSPACE_ROOT / "Movement" / "torque_motions"
    if torque_motion_dir.exists():
        torque_motions = list(torque_motion_dir.glob("*.npz"))
        print(f"\nTorque motions ({len(torque_motions)}):")
        for motion in torque_motions:
            print(f"  - {motion.name}")
    else:
        print("\nTorque motions directory does not exist yet.")

def analyze_motion_file(motion_file_path):
    """Analyze a specific motion file."""
    print(f"\n=== Analyzing Motion File: {motion_file_path} ===")
    try:
        import numpy as np
        data = np.load(motion_file_path)
        
        print("Available data arrays:")
        for key in data.files:
            array = data[key]
            if isinstance(array, np.ndarray):
                print(f"  - {key}: {array.shape} ({array.dtype})")
            else:
                print(f"  - {key}: {type(array)} - {array}")
        
        # Check for torque data specifically
        if "joint_torques" in data.files:
            print("✅ This is a TORQUE motion file")
            torques = data["joint_torques"]
            print(f"   Torque shape: {torques.shape}")
            print(f"   Torque range: [{np.min(torques):.3f}, {np.max(torques):.3f}]")
        else:
            print("ℹ️  This is a REGULAR motion file (no torque data)")
            
        print(f"Motion duration: {data['fps']} fps, {len(data['dof_positions'])} frames")
        
    except Exception as e:
        print(f"❌ Failed to analyze motion file: {e}")

def demo_workflow_commands():
    """Show example commands for the complete workflow."""
    print("\n" + "="*60)
    print("TORQUE WORKFLOW DEMONSTRATION COMMANDS")
    print("="*60)
    
    print("\n1. COLLECT TORQUE PROFILES:")
    print("   python scripts\\skrl\\biomechanics.py --checkpoint [YOUR_MODEL.pt] --save_torque_profiles")
    
    print("\n2. CONVERT TO NPZ FORMAT:")
    print("   python Movement\\csv_to_torque_motion.py --csv_file outputs\\[DATE]\\[TIME]\\data.csv --motion_name demo_torques")
    
    print("\n3. TRAIN TORQUE-BASED AMP:")
    print("   python scripts\\skrl\\train_torque_amp.py --use_torque_amp --num_envs 4096")
    
    print("\n4. VALIDATE MOTION FILE:")
    print("   python demo_torque_workflow.py --analyze Movement\\torque_motions\\demo_torques.npz")

def main():
    parser = argparse.ArgumentParser(description="Torque workflow demonstration and validation")
    parser.add_argument("--analyze", type=str, help="Analyze a specific motion file")
    parser.add_argument("--full-check", action="store_true", help="Run all system checks")
    
    args = parser.parse_args()
    
    print("Torque-Based Motion Imitation Workflow Demo")
    print("==========================================")
    
    if args.analyze:
        analyze_motion_file(args.analyze)
        return
    
    if args.full_check:
        # Run all checks
        checks = [
            check_directories(),
            check_torque_motion_loader(),
            check_environment_config(),
            check_converter_script(),
            check_training_script(),
        ]
        
        list_available_motions()
        
        print(f"\n=== Summary ===")
        passed = sum(checks)
        total = len(checks)
        print(f"System checks: {passed}/{total} passed")
        
        if passed == total:
            print("✅ All systems ready for torque-based motion imitation!")
            demo_workflow_commands()
        else:
            print("❌ Some components are missing. Please check the errors above.")
    else:
        # Quick check
        print("Running quick system check...")
        print("(Use --full-check for detailed validation)")
        
        loader_ok = check_torque_motion_loader()
        config_ok = check_environment_config()
        
        if loader_ok and config_ok:
            print("\n✅ Core components are working!")
            print("Run with --full-check for complete validation.")
        else:
            print("\n❌ Some core components have issues.")

if __name__ == "__main__":
    main()
