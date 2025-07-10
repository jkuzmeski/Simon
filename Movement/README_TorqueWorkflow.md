# Torque-Based Motion Imitation Workflow

This document describes the new torque-based AMP (Adversarial Motion Prior) workflow that extends the standard motion imitation to learn from joint torque profiles collected from trained agents.

## Overview

The workflow consists of several stages:

1. **Data Collection**: Collect torque profiles from trained agents using the enhanced `biomechanics.py` script
2. **Data Conversion**: Convert collected CSV data to NPZ format using the motion loader
3. **Motion Learning**: Train new agents using torque-based AMP with the collected torque profiles

## Components

### 1. Enhanced Biomechanics Script (`scripts/skrl/biomechanics.py`)

The biomechanics script has been enhanced with a new `--save_torque_profiles` flag that collects:
- Joint torques (applied torques from the robot)
- DOF positions and velocities
- Body positions, rotations, and velocities
- Temporal information (timesteps and time)

**Usage:**
```bash
# Collect torque profiles during evaluation
python scripts/skrl/biomechanics.py \
    --task Simon.tasks.locomotion.HumanoidTorqueTrainEnv \
    --checkpoint path/to/checkpoint.pt \
    --save_torque_profiles \
    --use_distance_termination \
    --max_distance 50.0
```

**Output**: Creates NPZ files in `{log_dir}/torque_profiles/` containing motion and torque data.

### 2. TorqueMotionLoader (`Movement/torque_motion_loader.py`)

Enhanced motion loader that supports joint torque data alongside traditional motion data.

**Features:**
- Loads NPZ files containing joint torques
- Provides torque sampling with temporal interpolation
- Compatible with existing MotionLoader interface
- Supports both torque-aware and standard sampling methods

**Usage:**
```python
from Movement.torque_motion_loader import TorqueMotionLoader

loader = TorqueMotionLoader("path/to/torque_motion.npz", device="cuda")
dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel, torques = loader.sample_with_torques(num_samples=100)
```

### 3. CSV to Torque Motion Converter (`Movement/csv_to_torque_motion.py`)

Converts biomechanics CSV files to the NPZ format required by TorqueMotionLoader.

**Usage:**
```bash
# Convert single CSV file
python Movement/csv_to_torque_motion.py \
    --input path/to/biomechanics_data.csv \
    --output torque_motion.npz

# Convert all CSV files in a directory
python Movement/csv_to_torque_motion.py \
    --input path/to/csv_directory/ \
    --output-dir path/to/output_directory/
```

### 4. Torque-Based AMP Training (`scripts/skrl/train_torque_amp.py`)

Training script that uses torque profiles for AMP-based imitation learning.

**Usage:**
```bash
python scripts/skrl/train_torque_amp.py \
    --task Simon.tasks.locomotion.HumanoidTorqueAMPEnv \
    --use_torque_amp \
    --torque_motion_files torque_motion1.npz torque_motion2.npz \
    --torque_weight 1.0 \
    --num_envs 4096
```

## Data Format

### Torque Motion NPZ Format

The NPZ files contain the following arrays:

- `fps`: Scalar, frames per second
- `dof_names`: Array of DOF names (strings)
- `body_names`: Array of body names (strings) 
- `joint_torques`: Shape (num_frames, num_dofs) - **Key addition for torque learning**
- `dof_positions`: Shape (num_frames, num_dofs)
- `dof_velocities`: Shape (num_frames, num_dofs)
- `body_positions`: Shape (num_frames, num_bodies, 3)
- `body_rotations`: Shape (num_frames, num_bodies, 4) - wxyz quaternions
- `body_linear_velocities`: Shape (num_frames, num_bodies, 3)
- `body_angular_velocities`: Shape (num_frames, num_bodies, 3)

## Complete Workflow Example

### Step 1: Collect Torque Data from Trained Agent

```bash
# Evaluate a trained agent and collect torque profiles
python scripts/skrl/biomechanics.py \
    --task Simon.tasks.locomotion.HumanoidWalkEvalEnv \
    --checkpoint logs/skrl/humanoid_walk/2025-07-10_10-30-00/checkpoints/checkpoint_2000.pt \
    --save_torque_profiles \
    --use_distance_termination \
    --max_distance 100.0 \
    --num_envs 1
```

### Step 2: Convert to Motion Format (if using CSV intermediate)

```bash
# If you collected CSV data first, convert it to NPZ
python Movement/csv_to_torque_motion.py \
    --input logs/skrl/humanoid_walk/2025-07-10_10-30-00/biomechanics/ \
    --output-dir Movement/torque_motions/
```

### Step 3: Train Torque-Based AMP Agent

```bash
# Train a new agent using the collected torque profiles
python scripts/skrl/train_torque_amp.py \
    --task Simon.tasks.locomotion.HumanoidTorqueAMPEnv \
    --use_torque_amp \
    --torque_motion_files Movement/torque_motions/torque_motion_20250710_103000.npz \
    --torque_weight 1.0 \
    --num_envs 4096 \
    --max_iterations 2000
```

## Key Features and Benefits

### 1. Torque-Level Imitation
- Learn control policies at the joint torque level
- Capture more nuanced motor control patterns
- Potential for more natural and efficient movement

### 2. Data Requirements
**Minimum Required:**
- Joint torques (essential)
- DOF positions (important for context)
- DOF velocities (important for dynamics)

**Helpful Additional Data:**
- Body poses (for reference motion tracking)
- Temporal consistency (for proper sequencing)

### 3. Flexibility
- Can work with just torque data alone
- Better results with full kinematic context
- Supports multiple motion files for diverse behaviors

## Implementation Notes

### Environment Requirements

Your task environment needs to support torque-based AMP features:
- Access to applied joint torques during simulation
- AMP discriminator that can process torque features
- Proper integration with TorqueMotionLoader

### Performance Considerations

- Torque data collection uses more memory than standard biomechanics
- Consider using `--collect_sensors` flag only when needed
- Multiple environments can speed up data collection

### Troubleshooting

1. **No torque data in CSV**: Ensure the environment exposes torque information in the `extras` dict
2. **Dimension mismatches**: Check that torque, position, and velocity arrays have consistent DOF counts
3. **Missing motion data**: The converter will create dummy data for missing fields, but this may affect training quality

## Future Enhancements

1. **Automatic feature extraction**: Identify the most important torque patterns
2. **Multi-modal learning**: Combine torque and kinematic features intelligently
3. **Torque pattern analysis**: Visualize and analyze collected torque profiles
4. **Adaptive weighting**: Automatically balance torque vs. kinematic losses

## Questions to Consider

Based on your original question about whether you need all the information:

**Can you learn from just joint torques?** 
- Theoretically yes, but results may be limited
- Joint positions provide crucial context for when to apply torques
- Joint velocities are essential for dynamic torque application

**Recommended minimal set:**
- Joint torques (required)
- DOF positions (strongly recommended)
- DOF velocities (strongly recommended)

**For best results, include:**
- All of the above plus body poses for full motion context
