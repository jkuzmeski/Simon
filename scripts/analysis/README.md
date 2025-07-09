# Random Force Bump Analysis for Isaac Lab Humanoid Simulation

This directory contains tools for applying random force bumps to a humanoid robot's pelvis during simulation and analyzing the resulting biomechanical and IMU responses.

## Features

1. **Random Force Bumps**: Apply random magnitude and direction forces to the pelvis at configurable intervals
2. **Bump Event Logging**: Save detailed information about each bump event (timing, force vector, robot position)
3. **IMU Response Analysis**: Correlate bump events with changes in IMU data
4. **Visualization Tools**: Generate plots showing bump timeline, IMU responses, and recovery patterns

## Usage

### 1. Running Simulation with Random Bumps

```bash
# Basic bump simulation with distance termination
python d:\Isaac\Simon\scripts\skrl\bump.py \
    --task Simon-Biomech-Run-v0 \
    --num_envs 1 \
    --checkpoint path/to/your/checkpoint.pt \
    --use_distance_termination \
    --max_distance 100.0 \
    --enable_random_bumps \
    --bump_force_magnitude 75.0 \
    --bump_interval_range 3.0 8.0 \
    --save_biomechanics_data

# Parameters explanation:
# --enable_random_bumps: Enable the bump functionality
# --bump_force_magnitude: Force magnitude in Newtons (50-100 N typical for human-scale)
# --bump_interval_range: Min and max seconds between bumps [min_seconds max_seconds]
# --save_biomechanics_data: Save detailed sensor data for analysis
```

### 2. Analyzing Bump Effects

```bash
# Run the analysis script
python d:\Isaac\Simon\scripts\analysis\bump_analysis.py \
    --biomechanics_csv logs/skrl/experiment_name/biomechanics/biomechanics_data_TIMESTAMP.csv \
    --bump_csv logs/skrl/experiment_name/bump_events/bump_events_TIMESTAMP.csv \
    --output_dir bump_analysis_results
```

## Output Files

### Bump Events CSV
Contains one row per bump event with columns:
- `timestep`: When the bump occurred
- `force_x`, `force_y`, `force_z`: Force vector components (N)
- `force_magnitude`: Total force magnitude (N)
- `distance_from_origin`: Distance traveled when bump occurred (if using distance mode)
- `pelvis_x`, `pelvis_y`, `pelvis_z`: Robot pelvis position at bump time

### Biomechanics CSV
Contains one row per timestep with all sensor data:
- `timestep`: Simulation timestep
- `policy_obs_*`: All policy observations (joint positions, velocities, IMU data, etc.)
- `imu_*`: IMU sensor data (linear acceleration, angular velocity, orientation)
- `contact_*`: Contact sensor data from feet
- `distance_from_origin`, `pelvis_*`: Position tracking data

### Analysis Outputs
- `bump_timeline.png`: Timeline showing bump events and key IMU responses
- `bump_response_analysis.png`: Correlation plots between bump force and IMU changes
- `bump_direction_analysis.png`: Analysis of how bump direction affects response
- `bump_analysis_results.csv`: Quantitative analysis results

## Configuration Options

### Bump Parameters
- `--bump_force_magnitude`: Force strength (Newtons)
  - Light bumps: 25-50 N
  - Moderate bumps: 50-100 N  
  - Strong bumps: 100-200 N
  
- `--bump_interval_range`: Time between bumps (seconds)
  - Frequent: [1.0, 3.0]
  - Moderate: [3.0, 8.0]
  - Sparse: [8.0, 15.0]

### Force Direction
Random force direction is automatically generated with:
- Primarily horizontal components (X, Y directions)
- Small vertical component (Z direction, ±20% of horizontal)
- Forces are applied to the pelvis body

## Expected Results

### Typical IMU Responses to Bumps
1. **Linear Acceleration**: Sharp spike at bump time, followed by oscillatory response
2. **Angular Velocity**: Rotational response depending on bump direction
3. **Recovery Time**: Usually 1-3 seconds for moderate bumps
4. **Direction Sensitivity**: Forward/backward bumps affect different axes than lateral bumps

### Analysis Insights
- Correlation between bump magnitude and IMU response amplitude
- Direction-dependent response patterns
- Recovery dynamics and stability measures
- Comparison with natural gait patterns

## Troubleshooting

### Common Issues
1. **No bump events recorded**: Check that `--enable_random_bumps` is specified
2. **IMU data missing**: Ensure your environment has IMU sensors configured
3. **Force too weak/strong**: Adjust `--bump_force_magnitude` based on robot scale
4. **Analysis fails**: Verify CSV file paths and that both files exist

### Performance Tips
- Use `--num_envs 1` for bump analysis (multiple envs complicate analysis)
- Enable `--save_biomechanics_data` only when needed (large files)
- Use distance termination for consistent analysis windows

## Example Workflow

```bash
# 1. Run simulation with bumps
python d:\Isaac\Simon\scripts\skrl\bump.py \
    --task Simon-Biomech-Run-v0 \
    --num_envs 1 \
    --checkpoint your_checkpoint.pt \
    --use_distance_termination \
    --max_distance 50.0 \
    --enable_random_bumps \
    --bump_force_magnitude 75.0 \
    --bump_interval_range 4.0 10.0 \
    --save_biomechanics_data

# 2. Analyze results
python d:\Isaac\Simon\scripts\analysis\bump_analysis.py \
    --biomechanics_csv logs/skrl/your_experiment/biomechanics/biomechanics_data_20240101_120000.csv \
    --bump_csv logs/skrl/your_experiment/bump_events/bump_events_20240101_120000.csv \
    --output_dir analysis_results

# 3. Review outputs
# - Check analysis_results/ for plots and quantitative results
# - Look for correlations between bump timing and IMU responses
# - Analyze recovery patterns and stability
```

## Research Applications

This bump analysis system is useful for:
- Studying balance recovery strategies
- Validating control robustness
- Analyzing perturbation responses
- Comparing human-like vs. robotic responses
- Developing better stability controllers