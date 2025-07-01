# load data from log folder .csv and plot the results
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# create argparse parser
import os
import argparse

parser = argparse.ArgumentParser(description="Plot simulation results from log folder.")
parser.add_argument("--log_folder", type=str, help="Path to the log folder containing CSV files.")
parser.add_argument("--variables", type=str, nargs='+', help="List of observation variables to plot.")


# list of the observations in the csv files
# 14 DOF pos + 14 DOF vel + 1 root height + 6 tangent/normal + 3 lin vel + 3 ang vel + 6 key body pos
obs = {
    "time_step": 'time_step',
    'policy_obs_0': 'right_hip_x',
    'policy_obs_1': 'right_hip_y',
    'policy_obs_2': 'right_hip_z',
    'policy_obs_3': 'right_knee',
    'policy_obs_4': 'right_ankle_x',
    'policy_obs_5': 'right_ankle_y',
    'policy_obs_6': 'right_ankle_z',
    'policy_obs_7': 'left_hip_x',
    'policy_obs_8': 'left_hip_y',
    'policy_obs_9': 'left_hip_z',
    'policy_obs_10': 'left_knee',
    'policy_obs_11': 'left_ankle_x',
    'policy_obs_12': 'left_ankle_y',
    'policy_obs_13': 'left_ankle_z',
    'policy_obs_14': 'right_hip_vel_x',
    'policy_obs_15': 'right_hip_vel_y',
    'policy_obs_16': 'right_hip_vel_z',
    'policy_obs_17': 'right_knee_vel',
    'policy_obs_18': 'right_ankle_vel_x',
    'policy_obs_19': 'right_ankle_vel_y',
    'policy_obs_20': 'right_ankle_vel_z',
    'policy_obs_21': 'left_hip_vel_x',
    'policy_obs_22': 'left_hip_vel_y',
    'policy_obs_23': 'left_hip_vel_z',
    'policy_obs_24': 'left_knee_vel',
    'policy_obs_25': 'left_ankle_vel_x',
    'policy_obs_26': 'left_ankle_vel_y',
    'policy_obs_27': 'left_ankle_vel_z',
    'policy_obs_28': 'root_height',
    'policy_obs_29': 'tangent_x',
    'policy_obs_30': 'tangent_y',
    'policy_obs_31': 'tangent_z',
    'policy_obs_32': 'normal_x',
    'policy_obs_33': 'normal_y',
    'policy_obs_34': 'normal_z',
    'policy_obs_35': 'linear_velocity_x',
    'policy_obs_36': 'linear_velocity_y',
    'policy_obs_37': 'linear_velocity_z',
    'policy_obs_38': 'angular_velocity_x',
    'policy_obs_39': 'angular_velocity_y',
    'policy_obs_40': 'angular_velocity_z',
    'policy_obs_41': 'right_foot_pos_x',
    'policy_obs_42': 'right_foot_pos_y',
    'policy_obs_43': 'right_foot_pos_z',
    'policy_obs_44': 'left_foot_pos_x',
    'policy_obs_45': 'left_foot_pos_y',
    'policy_obs_46': 'left_foot_pos_z',
    'policy_obs_47': 'pelvis_pos_x',
    'policy_obs_48': 'pelvis_pos_y',
    'policy_obs_49': 'pelvis_pos_z',
    'imu_acceleration_0': 'imu_acceleration_x',
    'imu_acceleration_1': 'imu_acceleration_y',
    'imu_acceleration_2': 'imu_acceleration_z',
    'imu_angular_velocity_0': 'imu_angular_velocity_x',
    'imu_angular_velocity_1': 'imu_angular_velocity_y',
    'imu_angular_velocity_2': 'imu_angular_velocity_z',
    'imu_orientation_0': 'imu_orientation_x',
    'imu_orientation_1': 'imu_orientation_y',
    'imu_orientation_2': 'imu_orientation_z',
    'imu_orientation_3': 'imu_orientation_w',
    'net_force_left_foot_0': 'net_force_left_foot_x',
    'net_force_left_foot_1': 'net_force_left_foot_y',
    'net_force_left_foot_2': 'net_force_left_foot_z',
    'net_force_right_foot_0': 'net_force_right_foot_x',
    'net_force_right_foot_1': 'net_force_right_foot_y',
    'net_force_right_foot_2': 'net_force_right_foot_z',
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
    'pelvis_x': 'pelvis_x',
    'pelvis_y': 'pelvis_y',
    'pelvis_z': 'pelvis_z'
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
    print(all_data.columns)

    return all_data


def stride_segmentation(data):
    '''Segment the data into strides based on the right foot contacts.'''

    #find the indices where the right foot contacts are made
    right_foot_contacts = data[data['net_force_right_foot_z'] > 10].index.tolist()
    segments = []
    for i in range(0, len(right_foot_contacts), stride_length):
        segment = data.iloc[right_foot_contacts[i:i + stride_length]]
        if not segment.empty:
            segments.append(segment)


def plot_results(all_data, variables):
    """
    Plots each variable from the DataFrame.
    The DataFrame index is used as the x-axis.
    """
    # The error "KeyError: 'time_step'" suggests 'time_step' is not a column.
    # It is likely the index of your DataFrame.
    # You can uncomment the following line to inspect your DataFrame's structure.
    # print(all_data.info())

    for var in variables:
        if var in all_data.columns:
            plt.figure(figsize=(10, 6))
            # pandas' .plot() method uses the DataFrame index for the x-axis by default.
            all_data[var].plot(grid=True)
            plt.title(f'{var} over Time')
            plt.xlabel("Time Step")
            plt.ylabel(var)
            plt.legend()
        else:
            print(f"Warning: Variable '{var}' not found in data. Skipping plot.")

    # Adjust layout and display all generated plots.
    plt.tight_layout()
    plt.show()


def main():
    args = parser.parse_args()
    log_folder = args.log_folder
    variables = args.variables

    # Handle variables passed as a single string like "[var1+var2]"
    if variables and len(variables) == 1 and variables[0].startswith('[') and variables[0].endswith(']'):
        vars_str = variables[0].strip('[]')
        variables = vars_str.split('+')

    if not os.path.exists(log_folder):
        print(f"Log folder '{log_folder}' does not exist.")
        return

    all_data = preprocess_data(log_folder, obs=obs)
    plot_results(all_data, variables)

if __name__ == "__main__":
    main()