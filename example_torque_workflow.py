#!/usr/bin/env python3
"""
Complete Torque Workflow Example

This script demonstrates the entire torque-based motion imitation process:
1. Shows how to collect torque profiles from a trained model (when available)
2. Converts CSV data to NPZ torque motion files
3. Trains a new agent using torque-based AMP

Usage:
    # Step 1: Collect torque profiles (when you have a trained model)
    python example_torque_workflow.py --step collect --model_path logs/skrl/path/to/your_model.pt
    
    # Step 2: Convert CSV to NPZ
    python example_torque_workflow.py --step convert --csv_file outputs/2025-XX-XX/XX-XX-XX/data.csv
    
    # Step 3: Train with torques
    python example_torque_workflow.py --step train

Requirements:
    - A trained model checkpoint (.pt file)
    - Isaac Lab environment properly set up
    - All torque workflow components in place
"""

import argparse
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def step_collect_torques(model_path):
    """Step 1: Collect torque profiles from a trained model."""
    print("="*60)
    print("STEP 1: COLLECTING TORQUE PROFILES")
    print("="*60)
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("\nTo get a trained model:")
        print("1. Train an agent first using regular AMP")
        print("2. Find the checkpoint in logs/skrl/[experiment_name]/checkpoints/")
        print("3. Use that checkpoint path with this script")
        return False
    
    # Run biomechanics with torque collection
    command = f'python scripts\\skrl\\biomechanics.py --checkpoint "{model_path}" --save_torque_profiles'
    
    return run_command(command, "Collecting torque profiles from trained agent")

def step_convert_csv_to_npz(csv_file):
    """Step 2: Convert CSV torque data to NPZ motion file."""
    print("="*60)
    print("STEP 2: CONVERTING CSV TO NPZ")
    print("="*60)
    
    if not Path(csv_file).exists():
        print(f"❌ CSV file not found: {csv_file}")
        print("\nTo get CSV files:")
        print("1. Run step 1 (collect) first to generate torque profiles")
        print("2. Check the outputs/ directory for the generated CSV files")
        return False
    
    # Generate motion name based on CSV file
    csv_path = Path(csv_file)
    motion_name = f"torque_motion_{csv_path.stem}"
    
    command = f'python Movement\\csv_to_torque_motion.py --csv_file "{csv_file}" --motion_name "{motion_name}"'
    
    success = run_command(command, f"Converting CSV to NPZ motion file: {motion_name}")
    
    if success:
        npz_file = WORKSPACE_ROOT / "Movement" / "torque_motions" / f"{motion_name}.npz"
        print(f"\n📁 Created torque motion file: {npz_file}")
        
        # Verify the file
        if npz_file.exists():
            print("✅ NPZ file created successfully")
            return True
        else:
            print("❌ NPZ file was not created")
            return False
    
    return False

def step_train_torque_amp():
    """Step 3: Train an agent using torque-based AMP."""
    print("="*60)
    print("STEP 3: TRAINING TORQUE-BASED AMP")
    print("="*60)
    
    # Check if we have any torque motion files
    torque_motions_dir = WORKSPACE_ROOT / "Movement" / "torque_motions"
    if not torque_motions_dir.exists():
        print("❌ Torque motions directory doesn't exist")
        print("Run steps 1 and 2 first to create torque motion files")
        return False
    
    torque_files = list(torque_motions_dir.glob("*.npz"))
    if not torque_files:
        print("❌ No torque motion files found")
        print("Run steps 1 and 2 first to create torque motion files")
        return False
    
    print(f"Found {len(torque_files)} torque motion files:")
    for file in torque_files:
        print(f"  - {file.name}")
    
    # Run torque-based AMP training
    command = 'python scripts\\skrl\\train_torque_amp.py --use_torque_amp --num_envs 2048'
    
    return run_command(command, "Training agent with torque-based AMP")

def list_available_files():
    """List available files for the workflow."""
    print("="*60)
    print("AVAILABLE FILES")
    print("="*60)
    
    # Check for trained models
    print("\n🤖 TRAINED MODELS:")
    model_patterns = ["**/*.pt", "**/checkpoint_*.pt", "**/model.pt"]
    found_models = []
    for pattern in model_patterns:
        models = list(WORKSPACE_ROOT.glob(pattern))
        found_models.extend(models)
    
    if found_models:
        for model in found_models:
            print(f"  ✅ {model}")
    else:
        print("  ❌ No trained models found")
        print("     Train an agent first using regular AMP workflow")
    
    # Check for CSV files
    print("\n📊 CSV DATA FILES:")
    csv_files = list((WORKSPACE_ROOT / "outputs").glob("**/*.csv"))
    if csv_files:
        for csv_file in csv_files[-5:]:  # Show last 5
            print(f"  ✅ {csv_file}")
        if len(csv_files) > 5:
            print(f"  ... and {len(csv_files) - 5} more")
    else:
        print("  ❌ No CSV files found")
        print("     Run biomechanics with --save_torque_profiles first")
    
    # Check for torque motion files
    print("\n🎯 TORQUE MOTION FILES:")
    torque_motions_dir = WORKSPACE_ROOT / "Movement" / "torque_motions"
    if torque_motions_dir.exists():
        torque_files = list(torque_motions_dir.glob("*.npz"))
        if torque_files:
            for torque_file in torque_files:
                print(f"  ✅ {torque_file.name}")
        else:
            print("  ❌ No torque motion files found")
            print("     Convert CSV files to NPZ first")
    else:
        print("  ❌ Torque motions directory doesn't exist")
    
    # Check for regular motion files
    print("\n📋 REGULAR MOTION FILES:")
    regular_motions = list((WORKSPACE_ROOT / "Movement").glob("*.npz"))
    if regular_motions:
        for motion in regular_motions:
            print(f"  ✅ {motion.name}")
    else:
        print("  ❌ No regular motion files found")

def main():
    parser = argparse.ArgumentParser(description="Complete torque workflow example")
    parser.add_argument("--step", choices=["collect", "convert", "train", "all"], 
                       help="Which step to run")
    parser.add_argument("--model_path", type=str, 
                       help="Path to trained model for torque collection")
    parser.add_argument("--csv_file", type=str, 
                       help="Path to CSV file for conversion")
    parser.add_argument("--list", action="store_true", 
                       help="List available files")
    
    args = parser.parse_args()
    
    print("Complete Torque-Based Motion Imitation Workflow")
    print("=" * 60)
    
    if args.list:
        list_available_files()
        return
    
    if not args.step:
        print("Usage examples:")
        print("  python example_torque_workflow.py --list")
        print("  python example_torque_workflow.py --step collect --model_path logs/skrl/.../checkpoint.pt")
        print("  python example_torque_workflow.py --step convert --csv_file outputs/.../data.csv")
        print("  python example_torque_workflow.py --step train")
        print("  python example_torque_workflow.py --step all")
        print("\nUse --list to see available files")
        return
    
    success = True
    
    if args.step in ["collect", "all"]:
        if not args.model_path:
            print("❌ --model_path required for collect step")
            print("Use --list to see available models")
            return
        success &= step_collect_torques(args.model_path)
    
    if args.step in ["convert", "all"]:
        if args.step == "all":
            # For "all" mode, find the most recent CSV file
            csv_files = list((WORKSPACE_ROOT / "outputs").glob("**/*.csv"))
            if not csv_files:
                print("❌ No CSV files found for conversion")
                success = False
            else:
                # Use the most recent CSV file
                csv_file = max(csv_files, key=lambda f: f.stat().st_mtime)
                print(f"📁 Using most recent CSV file: {csv_file}")
                success &= step_convert_csv_to_npz(str(csv_file))
        else:
            if not args.csv_file:
                print("❌ --csv_file required for convert step")
                print("Use --list to see available CSV files")
                return
            success &= step_convert_csv_to_npz(args.csv_file)
    
    if args.step in ["train", "all"]:
        success &= step_train_torque_amp()
    
    print("\n" + "="*60)
    if success:
        print("🎉 Workflow completed successfully!")
        if args.step == "all":
            print("\nYou now have a torque-based AMP agent training!")
            print("Check the logs/skrl/ directory for training progress.")
    else:
        print("❌ Workflow encountered errors")
        print("Check the output above for details")

if __name__ == "__main__":
    main()
