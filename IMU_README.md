# IMU Integration for Simon Biomech Environment

This document explains how to use the IMU sensor integration in your Simon biomech training environment.

## Overview

The IMU (Inertial Measurement Unit) sensor has been added to the pelvis of your humanoid agent. It collects:
- **Linear acceleration** (3-axis accelerometer data in body frame)
- **Angular velocity** (3-axis gyroscope data in body frame) 
- **Orientation** (quaternion representation in world frame)

## Files Modified

### 1. Robot Configuration (`source/isaaclab_assets/isaaclab_assets/robots/simon_IMU.py`)
- Added IMU sensor configuration attached to the pelvis
- Sensor updates every simulation step with no history buffering

### 2. Environment (`source/Simon/Simon/tasks/direct/simon/simon_biomech_env.py`)
- Modified `_get_observations()` method to collect IMU data
- IMU data is stored in `self.extras` dictionary with keys:
  - `'imu_acceleration'`: Linear acceleration in body frame (m/s²)
  - `'imu_angular_velocity'`: Angular velocity in body frame (rad/s)
  - `'imu_orientation'`: Orientation quaternion in world frame

## Using IMU Data

### During Training
The IMU data is automatically collected and stored in the environment's `extras` dictionary. You can access it in your training loop or callbacks:

```python
# In your training script or callback
obs, reward, done, info = env.step(action)
if hasattr(env.unwrapped, 'extras') and env.unwrapped.extras:
    imu_accel = env.unwrapped.extras['imu_acceleration']
    imu_gyro = env.unwrapped.extras['imu_angular_velocity']
    imu_orient = env.unwrapped.extras['imu_orientation']
```

### For Evaluation and Analysis

#### Option 1: Use the Evaluation Script
Run the provided evaluation script to collect and analyze IMU data:

```bash
cd d:\Isaac\Simon\scripts\skrl
python evaluate_imu.py --task Simon-Biomech-Run-v0 --checkpoint path/to/your/checkpoint.pt --num_episodes 5 --save_data
```

This will:
- Run evaluation episodes with your trained policy
- Collect IMU data throughout each episode
- Save data as `.npz` files
- Generate plots automatically

#### Option 2: Use the Analysis Script
If you already have saved IMU data files:

```bash
cd d:\Isaac\Simon\Analysis
python plot_imu_analysis.py path/to/your/imu_data.npz
```

Or simply run it in a directory containing `imu_data_*.npz` files:

```bash
python plot_imu_analysis.py
```

## Data Format

The saved IMU data is in NumPy `.npz` format with the following arrays:
- `timestamps`: Time array (seconds)
- `acceleration`: Shape (timesteps, num_envs, 3) - Linear acceleration
- `angular_velocity`: Shape (timesteps, num_envs, 3) - Angular velocity  
- `orientation`: Shape (timesteps, num_envs, 4) - Orientation quaternions

## Coordinate Frames

- **Acceleration & Angular Velocity**: Body frame (relative to pelvis orientation)
- **Orientation**: World frame quaternions

## Visualization

The analysis scripts generate plots showing:
1. **Linear Acceleration** (X, Y, Z components over time)
2. **Angular Velocity** (X, Y, Z components over time)
3. **Orientation** (Quaternion W, X, Y, Z components over time)

## Integration with Your Workflow

### Custom Analysis
You can create your own analysis scripts by loading the `.npz` files:

```python
import numpy as np

# Load data
data = np.load('imu_data_episode_1_timestamp.npz')
timestamps = data['timestamps']
acceleration = data['acceleration'][:, 0, :]  # First environment
angular_velocity = data['angular_velocity'][:, 0, :]
orientation = data['orientation'][:, 0, :]

# Your custom analysis here
# e.g., compute step detection, gait analysis, etc.
```

### Real-time Monitoring
For real-time monitoring during training, you can modify the training script to log IMU data:

```python
# In your training loop
if step % log_interval == 0:
    extras = env.unwrapped.extras
    if 'imu_acceleration' in extras:
        # Log IMU data to tensorboard, wandb, etc.
        logger.log('imu/accel_magnitude', torch.norm(extras['imu_acceleration'], dim=-1).mean())
```

## Troubleshooting

### No IMU Data in Extras
- Ensure the robot configuration includes the IMU sensor
- Check that the sensor is properly attached to the pelvis body
- Verify the sensor is being updated each timestep

### Empty Data Arrays
- Make sure the environment is running for multiple steps
- Check that `collect()` is being called in your evaluation loop

### Plotting Issues
- Ensure matplotlib is installed: `pip install matplotlib`
- Check that the data file exists and contains the expected arrays

## Next Steps

This IMU integration provides the foundation for:
- Gait analysis and step detection
- Balance and stability assessment
- Motion quality metrics
- Biomechanical analysis of your trained policies
- Real-time feedback during training
