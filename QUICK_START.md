# Quick Start: Torque-Based Motion Imitation

## ⚡ Quick Commands

### 1. Check Available Files
```bash
torque_workflow.bat list
```

### 2. Collect Torque Profiles
```bash
# Use your best trained model
torque_workflow.bat collect "logs\skrl\Biomech\2025-07-10_15-12-07_Test1\checkpoints\best_agent.pt"
```

### 3. Convert to NPZ Format
```bash
# After step 2, check outputs directory and convert
torque_workflow.bat convert "outputs\[DATE]\[TIME]\data.csv" "my_first_torques"
```

### 4. Train Torque-Based AMP
```bash
torque_workflow.bat train
```

## 🔧 Setup Required

1. **Edit Isaac Path**: Open `torque_workflow.bat` and set your Isaac Lab path:
   ```batch
   set ISAAC_PATH=C:\your\isaac\lab\path
   ```

2. **Common Isaac Lab Paths**:
   - `C:\isaac-sim` (standalone)
   - `C:\isaacsim` (alternative)
   - `C:\Users\%USERNAME%\.local\share\ov\pkg\isaac-sim-2023.1.1` (Omniverse)

## 📊 What Each Step Does

### Step 1: Collect Torque Profiles
- **Input**: Trained model (.pt file)
- **Process**: Runs the model and records joint torques during motion
- **Output**: CSV file with torque data in `outputs/[DATE]/[TIME]/`

### Step 2: Convert CSV to NPZ
- **Input**: CSV file from step 1
- **Process**: Converts to NPZ format for motion learning
- **Output**: `.npz` file in `Movement/torque_motions/`

### Step 3: Train with Torques
- **Input**: NPZ torque motion files
- **Process**: Trains new agent to imitate both motion and torques
- **Output**: New trained model in `logs/skrl/`

## 🚀 Complete Example

Starting from a trained model, here's the complete workflow:

```bash
# 1. List what we have
torque_workflow.bat list

# 2. Collect torque profiles (this will take a few minutes)
torque_workflow.bat collect "logs\skrl\Biomech\2025-07-10_15-12-07_Test1\checkpoints\best_agent.pt"

# 3. Find the generated CSV file (check outputs directory)
# Look for newest directory in outputs\2025-XX-XX\XX-XX-XX\
# Then convert it:
torque_workflow.bat convert "outputs\2025-07-10\16-45-30\data.csv" "walk_torques"

# 4. Train the torque-based agent
torque_workflow.bat train
```

## 🎯 Expected Results

After running the complete workflow:

1. **Better motion quality**: Torque-based agents should produce more natural movements
2. **Improved efficiency**: Learning from torques can be more sample-efficient
3. **Physical realism**: Movements should better respect physical constraints

## 🔍 Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'isaaclab'"**
   - Make sure you're using the correct Isaac Lab path in `torque_workflow.bat`

2. **"No CSV files found"**
   - Check that biomechanics completed successfully
   - Look in `outputs/[DATE]/[TIME]/` for generated files

3. **"No torque motion files found"**
   - Make sure step 2 (convert) completed successfully
   - Check `Movement/torque_motions/` directory

4. **Training fails**
   - Ensure you have sufficient GPU memory (training uses 2048 environments by default)
   - Reduce `--num_envs` if needed

### Getting Help

- Run `torque_workflow.bat help` for command reference
- Check `TORQUE_WORKFLOW_README.md` for detailed documentation
- Use `torque_workflow.bat demo` to validate system setup
