# load data from log folder .csv and plot the results
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import os
import argparse

# create argparse parser
parser = argparse.ArgumentParser(description="Plot simulation results from log folder.")
parser.add_argument("--log_folder", type=str, help="Path to the log folder containing CSV files.")
parser.add_argument("--variables", type=str, nargs='+', help="List of observation variables to plot.")
parser.add_argument("--bump_folder", type=str, help="Path to the folder containing bump events CSV files.")


# list of the observations in the csv files - CORRECTED MAPPING
# 14 DOF pos + 14 DOF vel + 1 root height + 6 tangent/normal + 3 lin vel + 3 ang vel + 9 key body pos
obs = {
    "timestep": 'timestep',
    # Joint Positions (14 values)
    'policy_obs_0': 'right_hip_frontal',
    'policy_obs_1': 'right_hip_sagittal',
    'policy_obs_2': 'right_hip_transverse',
    'policy_obs_3': 'left_hip_frontal',
    'policy_obs_4': 'left_hip_sagittal',
    'policy_obs_5': 'left_hip_transverse',
    'policy_obs_6': 'right_knee',
    'policy_obs_7': 'left_knee',
    'policy_obs_8': 'right_ankle_frontal',
    'policy_obs_9': 'right_ankle_sagittal',
    'policy_obs_10': 'right_ankle_transverse',
    'policy_obs_11': 'left_ankle_frontal',
    'policy_obs_12': 'left_ankle_sagittal',
    'policy_obs_13': 'left_ankle_transverse',
    # Joint Velocities (14 values)
    'policy_obs_14': 'right_hip_frontal_vel',
    'policy_obs_15': 'right_hip_sagittal_vel',
    'policy_obs_16': 'right_hip_transverse_vel',
    'policy_obs_17': 'left_hip_frontal_vel',
    'policy_obs_18': 'left_hip_sagittal_vel',
    'policy_obs_19': 'left_hip_transverse_vel',
    'policy_obs_20': 'right_knee_vel',
    'policy_obs_21': 'left_knee_vel',
    'policy_obs_22': 'right_ankle_frontal_vel',
    'policy_obs_23': 'right_ankle_sagittal_vel',
    'policy_obs_24': 'right_ankle_transverse_vel',
    'policy_obs_25': 'left_ankle_frontal_vel',
    'policy_obs_26': 'left_ankle_sagittal_vel',
    'policy_obs_27': 'left_ankle_transverse_vel',
    # Root State (1 value)
    'policy_obs_28': 'root_height',
    # Root Orientation - Tangent and Normal vectors (6 values)
    'policy_obs_29': 'root_tangent_x',
    'policy_obs_30': 'root_tangent_y',
    'policy_obs_31': 'root_tangent_z',
    'policy_obs_32': 'root_normal_x',
    'policy_obs_33': 'root_normal_y',
    'policy_obs_34': 'root_normal_z',
    # Root Linear Velocities (3 values)
    'policy_obs_35': 'root_lin_vel_x',
    'policy_obs_36': 'root_lin_vel_y',
    'policy_obs_37': 'root_lin_vel_z',
    # Root Angular Velocities (3 values)
    'policy_obs_38': 'root_ang_vel_x',
    'policy_obs_39': 'root_ang_vel_y',
    'policy_obs_40': 'root_ang_vel_z',
    # Key Body Positions relative to root (9 values)
    'policy_obs_41': 'right_foot_rel_x',
    'policy_obs_42': 'right_foot_rel_y',
    'policy_obs_43': 'right_foot_rel_z',
    'policy_obs_44': 'left_foot_rel_x',
    'policy_obs_45': 'left_foot_rel_y',
    'policy_obs_46': 'left_foot_rel_z',
    'policy_obs_47': 'pelvis_rel_x',
    'policy_obs_48': 'pelvis_rel_y',
    'policy_obs_49': 'pelvis_rel_z',
    'imu_acceleration_0': 'imu_acceleration_ap',
    'imu_acceleration_1': 'imu_acceleration_ml',
    'imu_acceleration_2': 'imu_acceleration_vertical',
    'imu_angular_velocity_0': 'imu_angular_velocity_ap',
    'imu_angular_velocity_1': 'imu_angular_velocity_ml',
    'imu_angular_velocity_2': 'imu_angular_velocity_vertical',
    'imu_orientation_0': 'imu_orientation_x',
    'imu_orientation_1': 'imu_orientation_y',
    'imu_orientation_2': 'imu_orientation_z',
    'imu_orientation_3': 'imu_orientation_w',
    'net_force_left_foot_0': 'net_force_left_foot_ap',
    'net_force_left_foot_1': 'net_force_left_foot_ml',
    'net_force_left_foot_2': 'net_force_left_foot_vertical',
    'net_force_right_foot_0': 'net_force_right_foot_ap',
    'net_force_right_foot_1': 'net_force_right_foot_ml',
    'net_force_right_foot_2': 'net_force_right_foot_vertical',
    'pelvis_angular_velocity_global_0': 'pelvis_angular_velocity_global_x',
    'pelvis_angular_velocity_global_1': 'pelvis_angular_velocity_global_y',
    'pelvis_angular_velocity_global_2': 'pelvis_angular_velocity_global_z',
    'pelvis_linear_velocity_global_0': 'pelvis_linear_velocity_global_x',
    'pelvis_linear_velocity_global_1': 'pelvis_linear_velocity_global_y',
    'pelvis_linear_velocity_global_2': 'pelvis_linear_velocity_global_z',
    'pelvis_orientation_global_0': 'pelvis_orientation_global_x',
    'pelvis_orientation_global_1': 'pelvis_orientation_global_y',
    'pelvis_orientation_global_2': 'pelvis_orientation_global_z',
    'pelvis_orientation_global_3': 'pelvis_orientation_global_w',
    'pelvis_position_global_0': 'pelvis_position_global_x',
    'pelvis_position_global_1': 'pelvis_position_global_y',
    'pelvis_position_global_2': 'pelvis_position_global_z',
    'distance_from_origin': 'distance_from_origin',
    'pelvis_x': 'pelvis_ap',
    'pelvis_y': 'pelvis_ml',
    'pelvis_z': 'pelvis_vertical'
}


def preprocess_data(log_folder, obs=obs):
    '''Preprocess the data from the log folder and return a DataFrame with the specified variables.'''
    
    # Load data from CSV files in the log folder
    csv_files = [f for f in os.listdir(log_folder) if f.endswith(".csv")]
    data = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(log_folder, csv_file))
        data.append(df)

    # Concatenate all dataframes
    all_data = pd.concat(data, ignore_index=True)
    
    # remove any columns that contain 'amp' in their name
    all_data = all_data.loc[:, ~all_data.columns.str.contains('amp', case=False, na=False)]
    
    # Ensure the data has the correct number of columns
    expected_columns = len(obs)
    if all_data.shape[1] != expected_columns:
        # print the columns for debugging
        # print(f"Columns found: {all_data.columns.tolist()}")
        # print the names of the columns in all_data dataframe that are not in the obs dict
        raise ValueError(f"Expected {expected_columns} columns, but got {all_data.shape[1]}.")
    
    # # Map the variable names to their corresponding indices
    all_data = all_data.rename(columns=obs)
    # # get rid of the columns that are not in the obs dict
    # missing_columns = [col for col in all_data.columns if col not in obs.values()]
    # print(missing_columns)
    # print(all_data.columns)

    return all_data

def rad2deg(data):
    '''Convert radians to degrees for the specified columns in the DataFrame.'''
    # Convert radians to degrees for specific columns
    rad_columns = ['right_hip_frontal', 'right_hip_sagittal', 'right_hip_transverse', 'right_knee',
                   'right_ankle_frontal', 'right_ankle_sagittal', 'right_ankle_transverse',
                   'left_hip_frontal', 'left_hip_sagittal', 'left_hip_transverse', 'left_knee',
                   'left_ankle_frontal', 'left_ankle_sagittal', 'left_ankle_transverse']

    for col in rad_columns:
        if col in data.columns:
            data[col] = np.rad2deg(data[col])
    
    return data


def filter_force_data(data, cutoff_freq=20, sampling_rate=480, filter_order=4):
    """
    Apply a lowpass Butterworth filter to force and IMU data columns.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the force and IMU data
    cutoff_freq : float
        Cutoff frequency in Hz (default 20)
    sampling_rate : float
        Sampling rate in Hz (default 480)
    filter_order : int
        Filter order (default 4)
        
    Returns:
    --------
    pd.DataFrame : DataFrame with filtered force and IMU data
    """
    # Create a copy to avoid modifying original data
    filtered_data = data.copy()
    
    # Define force columns to filter
    force_columns = [
        'net_force_left_foot_ap',
        'net_force_left_foot_ml', 
        'net_force_left_foot_vertical',
        'net_force_right_foot_ap',
        'net_force_right_foot_ml',
        'net_force_right_foot_vertical'
    ]
    
    # Define IMU columns to filter
    imu_columns = [
        'imu_acceleration_ap',
        'imu_acceleration_ml',
        'imu_acceleration_vertical',
        'imu_angular_velocity_ap',
        'imu_angular_velocity_ml',
        'imu_angular_velocity_vertical'
    ]
    
    # Calculate normalized cutoff frequency (Nyquist frequency = sampling_rate/2)
    nyquist_freq = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist_freq
    
    # Design the Butterworth filter
    b, a = signal.butter(filter_order, normalized_cutoff, btype='low', analog=False)
    
    # Apply filter to each force column
    for col in force_columns:
        if col in filtered_data.columns:
            # Apply zero-phase forward and backward filter
            filtered_data[col] = signal.filtfilt(b, a, filtered_data[col])
            # Set floor at 0N - any negative values become 0
            filtered_data[col] = np.maximum(filtered_data[col], 0)
            print(f"Applied {cutoff_freq}Hz lowpass filter to {col}")
    
    # Apply filter to each IMU column
    for col in imu_columns:
        if col in filtered_data.columns:
            # Apply zero-phase forward and backward filter
            filtered_data[col] = signal.filtfilt(b, a, filtered_data[col])
            print(f"Applied {cutoff_freq}Hz lowpass filter to {col}")
    
    return filtered_data


def find_heel_strikes(data, threshold=50.0):
    """
    Find heel strike events in the data based on vertical ground reaction forces.
    Note: This function now expects pre-filtered force data.
    
    A heel strike is defined as when the net_force_left_foot_vertical or 
    net_force_right_foot_vertical rises above the threshold (default 10 N) 
    while the waveform is increasing.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the filtered force data
    threshold : float
        Force threshold in Newtons (default 10.0)
        
    Returns:
    --------
    dict : Dictionary containing heel strike indices for left and right feet
    """
    heel_strikes = {'left_foot': [], 'right_foot': []}
    
    # Check if the required columns exist
    left_force_col = 'net_force_left_foot_vertical'
    right_force_col = 'net_force_right_foot_vertical'
    
    if left_force_col not in data.columns or right_force_col not in data.columns:
        print(f"Warning: Required force columns not found in data")
        return heel_strikes
    
    # Find heel strikes for left foot
    left_force = data[left_force_col].values
    for i in range(1, len(left_force)):
        # Check if force crosses threshold while increasing
        if (left_force[i] > threshold
            and left_force[i - 1] <= threshold
            and left_force[i] > left_force[i - 1]):
            # Mark the frame before the threshold is crossed (i-1)
            heel_strike_frame = i - 3
            # Check if enough time has passed since last heel strike (minimum 25 timesteps)
            if not heel_strikes['left_foot'] or heel_strike_frame - heel_strikes['left_foot'][-1] > 250:
                heel_strikes['left_foot'].append(heel_strike_frame)
    
    # Find heel strikes for right foot
    right_force = data[right_force_col].values
    for i in range(1, len(right_force)):
        # Check if force crosses threshold while increasing
        if (right_force[i] > threshold
            and right_force[i - 1] <= threshold
            and right_force[i] > right_force[i - 1]):
            # Mark the frame before the threshold is crossed (i-1)
            heel_strike_frame = i - 3
            # Check if enough time has passed since last heel strike (minimum 25 timesteps)
            if not heel_strikes['right_foot'] or heel_strike_frame - heel_strikes['right_foot'][-1] > 250:
                heel_strikes['right_foot'].append(heel_strike_frame)
    
    print(f"Found {len(heel_strikes['left_foot'])} left heel strikes")
    print(f"Found {len(heel_strikes['right_foot'])} right heel strikes")
    
    return heel_strikes


def chop_data_by_gait_cycles(data, heel_strikes, foot='right_foot'):
    """
    Chop data into gait cycles based on heel strikes.
    
    A gait cycle is defined as the period from one heel strike to the next 
    heel strike of the same foot.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the time series data
    heel_strikes : dict
        Dictionary containing heel strike indices for left and right feet
    foot : str
        Which foot to use for gait cycle definition ('left_foot' or 'right_foot')
        
    Returns:
    --------
    list : List of DataFrames, each containing one gait cycle
    """
    if foot not in heel_strikes:
        print(f"Warning: {foot} not found in heel strikes data")
        return []
    
    strikes = heel_strikes[foot]
    if len(strikes) < 2:
        print(f"Warning: Need at least 2 heel strikes for {foot} to define gait cycles")
        return []
    
    gait_cycles = []
    for i in range(len(strikes) - 1):
        start_idx = strikes[i]
        end_idx = strikes[i + 1]
        cycle_data = data.iloc[start_idx:end_idx].copy()
        # Reset index to start from 0 for each cycle
        cycle_data.reset_index(drop=True, inplace=True)
        gait_cycles.append(cycle_data)
    
    # Remove first and last gait cycles to eliminate potentially misaligned strides
    if len(gait_cycles) > 2:
        gait_cycles = gait_cycles[1:-1]
        print(f"Created {len(gait_cycles)} gait cycles for {foot} (removed first and last cycles)")
    else:
        print(f"Warning: Only {len(gait_cycles)} gait cycles available for {foot}, keeping all")
    
    return gait_cycles


def plot_results(all_data, variables, bump_events=None, heel_strikes=None):
    """
    Plots each variable from the DataFrame with optional bump event markers.
    The DataFrame index is used as the x-axis.
    
    Parameters:
    -----------
    all_data : pd.DataFrame
        DataFrame containing the time series data
    variables : list
        List of variable names to plot
    bump_events : pd.DataFrame, optional
        DataFrame containing bump events with 'timestep' column
    heel_strikes : dict, optional
        Dictionary containing heel strike indices for trimming data
    """
    # Trim data to match gait cycle analysis if heel strikes are provided
    plot_data = all_data.copy()
    if heel_strikes is not None and 'right_foot' in heel_strikes:
        strikes = heel_strikes['right_foot']
        if len(strikes) > 2:
            # Use the same trimming logic as gait cycle analysis
            # Skip first strike and use until second-to-last strike
            start_idx = strikes[1]
            end_idx = strikes[-2]
            plot_data = all_data.iloc[start_idx:end_idx].copy()
            print(f"Trimmed timeseries data from heel strike {start_idx} to {end_idx}")

    for var in variables:
        if var in plot_data.columns:
            plt.figure(figsize=(12, 8))
            # pandas' .plot() method uses the DataFrame index for the x-axis by default.
            plot_data[var].plot(grid=True)
            
            # Add bump event markers if available
            if bump_events is not None and not bump_events.empty:
                bump_timesteps = bump_events['timestep'].values
                # Get the y-values at bump timesteps for marker placement
                for timestep in bump_timesteps:
                    # Adjust timestep to trimmed data range
                    if heel_strikes is not None and 'right_foot' in heel_strikes:
                        strikes = heel_strikes['right_foot']
                        if len(strikes) > 2:
                            adjusted_timestep = timestep - strikes[1]
                            if 0 <= adjusted_timestep < len(plot_data):
                                y_val = plot_data[var].iloc[adjusted_timestep]
                                plt.axvline(x=adjusted_timestep, color='red', linestyle='--', alpha=0.7, linewidth=2)
                                plt.scatter(adjusted_timestep, y_val, color='red', s=100, marker='v',
                                          zorder=5, label='Bump Event' if timestep == bump_timesteps[0] else "")
                    else:
                        # Original behavior if no heel strikes
                        if timestep < len(plot_data):
                            y_val = plot_data[var].iloc[timestep]
                            plt.axvline(x=timestep, color='red', linestyle='--', alpha=0.7, linewidth=2)
                            plt.scatter(timestep, y_val, color='red', s=100, marker='v',
                                      zorder=5, label='Bump Event' if timestep == bump_timesteps[0] else "")
                
                # Add legend if bump events were plotted
                if len(bump_timesteps) > 0:
                    plt.legend()
            
            plt.title(f'{var} over Time')
            plt.xlabel("Time Step")
            plt.ylabel(var)
        else:
            print(f"Warning: Variable '{var}' not found in data. Skipping plot.")

    # Adjust layout and display all generated plots.
    plt.tight_layout()
    plt.show()


def plot_gait_cycles(gait_cycles, variables, foot='right_foot', overlay=True):
    """
    Plot variables across gait cycles.
    
    Parameters:
    -----------
    gait_cycles : list
        List of DataFrames, each containing one gait cycle
    variables : list
        List of variable names to plot
    foot : str
        Which foot the gait cycles are based on
    overlay : bool
        If True, overlay all cycles on same plot. If False, plot separately.
    """
    if not gait_cycles:
        print("No gait cycles to plot")
        return
    
    for var in variables:
        if var not in gait_cycles[0].columns:
            print(f"Warning: Variable '{var}' not found in data. Skipping plot.")
            continue
            
        plt.figure(figsize=(12, 8))
        
        if overlay:
            # Plot all cycles on the same axes
            for i, cycle in enumerate(gait_cycles):
                cycle[var].plot(alpha=0.7, label=f'Cycle {i+1}', grid=True)
            plt.title(f'{var} across {foot} gait cycles (Overlaid)')
        else:
            # Plot cycles in subplots
            n_cycles = len(gait_cycles)
            cols = min(3, n_cycles)
            rows = (n_cycles + cols - 1) // cols
            
            for i, cycle in enumerate(gait_cycles):
                plt.subplot(rows, cols, i + 1)
                cycle[var].plot(grid=True)
                plt.title(f'Cycle {i+1}')
                plt.xlabel('Time Step within Cycle')
                plt.ylabel(var)
            
            plt.suptitle(f'{var} across {foot} gait cycles')
        
        plt.xlabel('Time Step within Gait Cycle')
        plt.ylabel(var)
        plt.tight_layout()
    
    plt.show()


def plot_average_gait_cycle(gait_cycles, variables, foot='right_foot'):
    """
    Plot average gait cycle with standard deviation shading.
    
    Parameters:
    -----------
    gait_cycles : list
        List of DataFrames, each containing one gait cycle
    variables : list
        List of variable names to plot
    foot : str
        Which foot the gait cycles are based on
    """
    if not gait_cycles:
        print("No gait cycles to plot")
        return
    
    # Find the minimum cycle length to normalize all cycles
    min_length = min(len(cycle) for cycle in gait_cycles)
    
    for var in variables:
        if var not in gait_cycles[0].columns:
            print(f"Warning: Variable '{var}' not found in data. Skipping plot.")
            continue
        
        # Create matrix where each row is a cycle, truncated to min_length
        cycle_matrix = np.zeros((len(gait_cycles), min_length))
        
        for i, cycle in enumerate(gait_cycles):
            cycle_matrix[i, :] = cycle[var].iloc[:min_length].values
        
        # Calculate mean and standard deviation across cycles
        mean_cycle = np.mean(cycle_matrix, axis=0)
        std_cycle = np.std(cycle_matrix, axis=0)
        
        # Create x-axis as percentage of gait cycle
        x_percent = np.linspace(0, 100, min_length)
        
        plt.figure(figsize=(12, 8))
        
        # Plot mean line
        plt.plot(x_percent, mean_cycle, 'b-', linewidth=2, label='Mean')
        
        # Plot standard deviation shading
        plt.fill_between(x_percent,
                         mean_cycle - std_cycle,
                         mean_cycle + std_cycle,
                         alpha=0.3, color='blue', label='±1 SD')
        
        plt.title(f'Average {var} across {foot} gait cycles (n={len(gait_cycles)})')
        plt.xlabel('Gait Cycle (%)')
        plt.ylabel(var)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    plt.show()


def load_bump_events(bump_folder):
    """
    Load bump events from the specified bump folder.
    
    Parameters:
    -----------
    bump_folder : str
        Path to the folder containing bump events CSV files
        
    Returns:
    --------
    pd.DataFrame or None : DataFrame containing bump events or None if not found
    """
    if bump_folder is None:
        print("No bump folder specified")
        return None
    
    if not os.path.exists(bump_folder):
        print(f"Bump folder '{bump_folder}' does not exist")
        return None
    
    # Look for CSV files in bump folder
    csv_files = [f for f in os.listdir(bump_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print("No bump events CSV files found")
        return None
    
    # Load the first CSV file (assuming there's only one)
    bump_file = csv_files[0]
    bump_path = os.path.join(bump_folder, bump_file)
    
    try:
        bump_events = pd.read_csv(bump_path)
        print(f"Loaded {len(bump_events)} bump events from {bump_file}")
        return bump_events
    except Exception as e:
        print(f"Error loading bump events: {e}")
        return None


def main():
    args = parser.parse_args()
    log_folder = args.log_folder
    variables = args.variables
    bump_folder = args.bump_folder

    # Handle variables passed as a single string like "[var1+var2]"
    if variables and len(variables) == 1 and variables[0].startswith('[') and variables[0].endswith(']'):
        vars_str = variables[0].strip('[]')
        variables = vars_str.split('+')

    if not os.path.exists(log_folder):
        print(f"Log folder '{log_folder}' does not exist.")
        return

    all_data = preprocess_data(log_folder, obs=obs)
    all_data = rad2deg(all_data)  # Convert radians to degrees for specific columns
    
    # Load bump events if available
    bump_events = load_bump_events(bump_folder)
    
    # Apply lowpass filter to force data before heel strike detection
    filtered_data = filter_force_data(all_data, cutoff_freq=20, sampling_rate=1000, filter_order=4)
    
    heelstrikes = find_heel_strikes(filtered_data)  # Find heel strikes using filtered data
    cycles = chop_data_by_gait_cycles(filtered_data, heelstrikes, foot='right_foot')  # Use filtered data for cycles
    plot_gait_cycles(cycles, variables, foot='right_foot', overlay=True)  # Plot gait cycles
    plot_average_gait_cycle(cycles, variables, foot='right_foot')  # Plot average gait cycle with std
    plot_results(filtered_data, variables, bump_events, heelstrikes)  # Plot filtered time series with bump markers

if __name__ == "__main__":
    main()