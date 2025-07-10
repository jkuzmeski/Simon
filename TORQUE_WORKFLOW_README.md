# Torque-Based Motion Imitation Workflow

This document explains how to use the complete torque-based Adversarial Motion Prior (AMP) system that has been implemented in your Isaac Lab workspace.

## Overview

The torque-based AMP workflow allows you to:
1. **Collect torque profiles** from trained agents during evaluation
2. **Convert torque data** to NPZ motion files
3. **Train new agents** to imitate both kinematic motion and joint torques

## System Components

### 1. Enhanced Biomechanics Script
- **File**: `scripts/skrl/biomechanics.py`
- **Purpose**: Collects joint torque profiles from trained agents
- **New Features**: `--save_torque_profiles` flag for torque data collection

### 2. Torque Motion Loader
- **File**: `Movement/torque_motion_loader.py` (also copied to environment: `source/Simon/Simon/tasks/direct/simon_torque/motions/motion_torque_loader.py`)
- **Purpose**: Loads and samples motion data including joint torques
- **Key Methods**: `sample_with_torques()`, `has_torque_data`, `get_torques_at_time()`

### 3. CSV to NPZ Converter
- **File**: `Movement/csv_to_torque_motion.py`
- **Purpose**: Converts biomechanics CSV output to NPZ torque motion files
- **Output**: Creates NPZ files in `Movement/torque_motions/` directory

### 4. Torque AMP Training
- **File**: `scripts/skrl/train_torque_amp.py`
- **Purpose**: Trains agents using torque-based AMP
- **Features**: Automatic torque motion file detection, enhanced observation space

### 5. Torque Environment
- **Files**: 
  - `source/Simon/Simon/tasks/direct/simon_torque/simon_torque_env.py`
  - `source/Simon/Simon/tasks/direct/simon_torque/simon_torque_env_cfg.py`
- **Purpose**: Environment that supports torque observations for AMP discriminator
- **Features**: TorqueMotionLoader integration, torque-enhanced observations

## Complete Workflow

### Step 1: Collect Torque Profiles

Run the biomechanics script with torque collection enabled:

```bash
cd d:\Isaac\Simon
python scripts\skrl\biomechanics.py --checkpoint /path/to/your/trained/model.pt --save_torque_profiles
```

**Expected Output:**
- CSV files with joint torque data in the `outputs/` directory
- Console output showing torque collection progress

### Step 2: Convert to NPZ Motion Files

Convert the collected CSV data to NPZ format:

```bash
cd d:\Isaac\Simon
python Movement\csv_to_torque_motion.py --csv_file outputs\[DATE]\[TIME]\data.csv --motion_name my_torque_motion
```

**Expected Output:**
- NPZ file in `Movement/torque_motions/my_torque_motion.npz`
- Console output showing conversion details and data validation

### Step 3: Train Torque-Based AMP Agent

Train a new agent using the torque motion data:

```bash
cd d:\Isaac\Simon
python scripts\skrl\train_torque_amp.py --use_torque_amp --num_envs 4096
```

**Expected Output:**
- Training progress with torque-enhanced AMP discriminator
- Model checkpoints in `logs/skrl/` directory
- Wandb logging (if configured)

## Advanced Usage

### Multiple Torque Motion Files

To use multiple torque motion files, place them all in `Movement/torque_motions/` and the training script will automatically detect and use them:

```bash
# Place multiple NPZ files:
# Movement/torque_motions/walk_torques.npz
# Movement/torque_motions/run_torques.npz
# Movement/torque_motions/dance_torques.npz

python scripts\skrl\train_torque_amp.py --use_torque_amp
```

### Custom Motion Names

```bash
python Movement\csv_to_torque_motion.py --csv_file outputs\data.csv --motion_name custom_walk_v2 --output_dir Movement\torque_motions
```

### Training with Specific Motion Files

You can specify exact motion files by modifying the config in `simon_torque_env_cfg.py`:

```python
# In the config file, update the motion_files list:
motion_files = [
    "Movement/torque_motions/specific_motion.npz",
    "Movement/motion_data/regular_motion.npz",  # Mix with regular motions
]
```

## Data Format

### NPZ Torque Motion Files

The NPZ files contain the following arrays:

- `fps`: Frame rate (float)
- `dof_names`: Joint names (array of strings)
- `body_names`: Body names (array of strings)
- `dof_positions`: Joint positions (frames, num_joints)
- `dof_velocities`: Joint velocities (frames, num_joints)
- `body_positions`: Body positions (frames, num_bodies, 3)
- `body_rotations`: Body rotations as quaternions (frames, num_bodies, 4)
- `body_linear_velocities`: Body linear velocities (frames, num_bodies, 3)
- `body_angular_velocities`: Body angular velocities (frames, num_bodies, 3)
- **`joint_torques`**: Joint torques (frames, num_joints) - **NEW for torque-based AMP**

### Observation Space

When using torque-based AMP, the discriminator receives enhanced observations:

**Standard AMP observations (50 dimensions):**
- Joint positions (14) + Joint velocities (14) + Root state (41) = 50 total

**Torque-enhanced AMP observations (64 dimensions):**
- Joint positions (14) + Joint velocities (14) + Root state (22) + **Joint torques (14)** = 64 total

## Troubleshooting

### Common Issues

1. **ImportError for TorqueMotionLoader**
   - The system automatically falls back to regular MotionLoader
   - Check that `Movement/torque_motion_loader.py` exists

2. **No torque data in NPZ files**
   - Ensure you used `--save_torque_profiles` when running biomechanics
   - Check that CSV files contain torque columns

3. **Training fails with observation size mismatch**
   - Verify that your motion files have consistent DOF counts
   - Check that torque dimensions match your robot's joint count

4. **No torque motion files found**
   - Ensure files are in `Movement/torque_motions/` directory
   - Check file permissions and naming conventions

### Validation Commands

Check if your setup is working:

```bash
# Test motion loader
python -c "
from Movement.torque_motion_loader import MotionLoader
motion = MotionLoader('Movement/torque_motions/your_file.npz', 'cpu')
print(f'Has torque data: {motion.has_torque_data}')
print(f'Motion frames: {motion.num_frames}')
"

# Check environment config
python -c "
from source.Simon.Simon.tasks.direct.simon_torque.simon_torque_env_cfg import SimonTorqueEnvCfg
cfg = SimonTorqueEnvCfg()
print(f'Enable torque AMP: {cfg.enable_torque_amp}')
print(f'Motion file: {cfg.motion_file}')
"
```

## Next Steps

1. **Collect your first torque profiles** using the biomechanics script
2. **Convert to NPZ format** and validate the motion files
3. **Start training** with torque-based AMP
4. **Compare performance** between standard AMP and torque-based AMP agents
5. **Experiment with different motion types** (walking, running, complex movements)

The system is designed to be backward compatible - you can always fall back to regular AMP if needed, and you can mix torque and non-torque motion files in your training.
