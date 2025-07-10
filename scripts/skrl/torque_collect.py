# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task. If not provided, will try to load from checkpoint metadata.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="AMP",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--save_biomechanics_data",
    action="store_true",
    default=True,
    help="Save biomechanics data (observations and extras) to a CSV file in the log directory.",
)
parser.add_argument(
    "--save_torque_profiles",
    action="store_true",
    default=False,
    help="Save torque profiles for motion imitation (includes joint torques, positions, velocities, and body poses).",
)
parser.add_argument(
    "--use_distance_termination",
    action="store_true",
    default=False,
    help="Terminate simulation based on distance traveled rather than timesteps.",
)
parser.add_argument(
    "--max_distance",
    type=float,
    default=100.0,
    help="Maximum distance (in meters) the agent should travel from its starting position before terminating simulation when using distance termination.",
)
parser.add_argument(
    "--max_timesteps_distance",
    type=int,
    default=100000,
    help="Maximum timesteps to run when using distance termination (safety limit to prevent infinite runs).",
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import csv  # Add this import
import yaml
import numpy as np  # Add this import

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import Simon.tasks  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()


def load_task_from_metadata(checkpoint_path):
    """Load task name from metadata if available."""
    if not checkpoint_path:
        return None
    
    # Try to find metadata in the checkpoint directory structure
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Look for metadata files in common locations
    possible_metadata_paths = [
        os.path.join(checkpoint_dir, "..", "params", "metadata.yaml"),
        os.path.join(checkpoint_dir, "..", "params", "metadata.pkl"),
        os.path.join(checkpoint_dir, "params", "metadata.yaml"), 
        os.path.join(checkpoint_dir, "params", "metadata.pkl"),
    ]
    
    for metadata_path in possible_metadata_paths:
        abs_metadata_path = os.path.abspath(metadata_path)
        if os.path.exists(abs_metadata_path):
            try:
                if abs_metadata_path.endswith('.yaml'):
                    with open(abs_metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                        if metadata and 'task' in metadata:
                            print(f"[INFO] Auto-detected task '{metadata['task']}' from metadata: {abs_metadata_path}")
                            return metadata['task']
                elif abs_metadata_path.endswith('.pkl'):
                    import pickle
                    with open(abs_metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                        if metadata and 'task' in metadata:
                            print(f"[INFO] Auto-detected task '{metadata['task']}' from metadata: {abs_metadata_path}")
                            return metadata['task']
            except Exception as e:
                print(f"[WARNING] Could not load metadata from {abs_metadata_path}: {e}")
                continue
    
    print("[WARNING] Could not find task metadata. Please specify --task manually.")
    return None


def main():
    """Play with skrl agent."""
    # If task is not provided, try to auto-detect from checkpoint metadata
    task_name = args_cli.task
    if not task_name and args_cli.checkpoint:
        base_task = load_task_from_metadata(args_cli.checkpoint)
        if base_task:
            # Smart task switching: prefer eval version for biomechanics analysis
            if "Train" in base_task and args_cli.save_biomechanics_data:
                eval_task = base_task.replace("Train", "Eval")
                print(f"[INFO] Auto-switching from training task '{base_task}' to evaluation task '{eval_task}' for biomechanics analysis")
                task_name = eval_task
            else:
                task_name = base_task
            args_cli.task = task_name
        else:
            print("[ERROR] Task name not provided and could not be auto-detected from checkpoint metadata.")
            print("Please specify --task manually.")
            return
    elif not task_name:
        print("[ERROR] Task name must be provided when not using a checkpoint with metadata.")
        print("Please specify --task.")
        return

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    env.unwrapped.set_eval_mode(True)
    print("[INFO] Environment set to evaluation mode - sensor data collection enabled")


    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # --- Biomechanics data saving setup ---
    biomechanics_data_to_save = []
    current_episode_data = []  # Temporary storage for current episode
    biomechanics_csv_path = None
    episode_terminated_unsuccessfully = False  # Track if current episode failed
    if args_cli.save_biomechanics_data:
        biomechanics_dir = os.path.join(log_dir, "biomechanics")
        os.makedirs(biomechanics_dir, exist_ok=True)
        # Create a unique filename with a timestamp
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        biomechanics_csv_path = os.path.join(biomechanics_dir, f"biomechanics_data_{timestamp_str}.csv")
        print(f"[INFO] Saving biomechanics data to: {biomechanics_csv_path}")
        print("[INFO] Only successful runs (not terminated due to failure) will be saved.")
    # --- End Biomechanics data saving setup ---

    # --- Torque profile saving setup ---
    torque_profiles_to_save = []
    current_torque_episode_data = []  # Temporary storage for current episode torque data
    torque_profiles_dir = None
    if args_cli.save_torque_profiles:
        torque_profiles_dir = os.path.join(log_dir, "torque_profiles")
        os.makedirs(torque_profiles_dir, exist_ok=True)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        print(f"[INFO] Saving torque profiles to: {torque_profiles_dir}")
        print("[INFO] Torque profiles will include joint torques, positions, velocities, and body poses for motion imitation.")
    # --- End Torque profile saving setup ---

    # reset environment
    current_obs, current_info = env.reset()  # Get initial obs and info
    timestep = 0
    
    # --- Distance tracking setup ---
    initial_pelvis_x = None
    current_distance_from_origin = 0.0
    max_distance_reached = False
    
    # We'll set the initial pelvis position after the first step, not from reset
    if args_cli.use_distance_termination:
        print(f"[INFO] Distance termination enabled. Will run until agent moves {args_cli.max_distance}m from origin.")
        print("[INFO] Initial pelvis position will be captured after first simulation step.")
        print(f"[INFO] Timestep limit set to {args_cli.max_timesteps_distance} for distance-based termination.")
    else:
        print("[INFO] Timestep termination enabled. Will run for up to 1000 timesteps.")
    # --- End Distance tracking setup ---
    
    # simulate environment - terminate based on timesteps or distance
    max_timesteps = 1000 if not args_cli.use_distance_termination else args_cli.max_timesteps_distance
    while timestep < max_timesteps and not max_distance_reached:
        start_time = time.time()

        # --- Save biomechanics data for current_obs and current_info ---
        if args_cli.save_biomechanics_data and not episode_terminated_unsuccessfully:
            num_active_envs = 0
            # Determine number of active environments based on observation structure
            if isinstance(current_obs, dict) and "policy" in current_obs:
                if current_obs["policy"] is not None:
                    num_active_envs = current_obs["policy"].shape[0]
            elif isinstance(current_obs, torch.Tensor):  # Handles single tensor observations
                num_active_envs = current_obs.shape[0]

            for i in range(num_active_envs):
                data_row = {"timestep": timestep}
                
                # Add distance from origin if available
                if args_cli.use_distance_termination and initial_pelvis_x is not None:
                    data_row["distance_from_origin"] = current_distance_from_origin
                    # Also save current pelvis position for analysis
                    try:
                        unwrapped_env = env.unwrapped
                        if hasattr(unwrapped_env, 'robot') and hasattr(unwrapped_env.robot, 'data'):
                            pelvis_pos_3d = unwrapped_env.robot.data.body_pos_w[i, unwrapped_env.ref_body_index].cpu().numpy()
                            data_row["pelvis_x"] = pelvis_pos_3d[0]
                            data_row["pelvis_y"] = pelvis_pos_3d[1]
                            data_row["pelvis_z"] = pelvis_pos_3d[2]
                    except Exception:
                        pass  # Skip if can't access robot data
                
                # Add policy observations
                policy_obs_tensor = None
                if isinstance(current_obs, dict) and "policy" in current_obs:
                    policy_obs_tensor = current_obs["policy"][i]
                elif isinstance(current_obs, torch.Tensor):  # single tensor observation
                    policy_obs_tensor = current_obs[i]
                
                if policy_obs_tensor is not None:
                    policy_obs_np = policy_obs_tensor.cpu().numpy()
                    for j, val in enumerate(policy_obs_np):
                        data_row[f"policy_obs_{j}"] = val
                
                # Add extras from current_info
                if current_info and "extras" in current_info and current_info["extras"]:
                    # Debug: Print available extras keys on first timestep
                    if timestep == 0:
                        print(f"[DEBUG] Available extras keys: {list(current_info['extras'].keys())}")
                    
                    for key, tensor_val_all_envs in current_info["extras"].items():
                        # Skip if the value is None
                        if tensor_val_all_envs is None:
                            continue
                            
                        # Check if it's a tensor and has the right shape
                        if hasattr(tensor_val_all_envs, 'numel') and hasattr(tensor_val_all_envs, 'shape'):
                            if tensor_val_all_envs.numel() > 0 and i < tensor_val_all_envs.shape[0]:
                                item_data_tensor = tensor_val_all_envs[i]
                                if item_data_tensor is not None:
                                    item_data = item_data_tensor.cpu().numpy()
                                    if item_data.ndim == 0:  # scalar
                                        data_row[key] = item_data.item()
                                    else:  # vector/tensor, flatten it
                                        flat_item_data = item_data.flatten()
                                        for k_idx, k_val in enumerate(flat_item_data):
                                            data_row[f"{key}_{k_idx}"] = k_val
                        else:
                            # Handle non-tensor values (like nested dicts or other types)
                            if timestep == 0:  # Only print debug info on first timestep
                                print(f"[DEBUG] Non-tensor extra '{key}': type={type(tensor_val_all_envs)}")
                
                if len(data_row) > 1:  # Only add if there's more than just timestep
                    current_episode_data.append(data_row)
        elif args_cli.save_biomechanics_data and episode_terminated_unsuccessfully:
            # Debug: Print why data is not being saved
            if timestep % 500 == 0:  # Print every 500 timesteps to avoid spam
                print(f"[DEBUG] Timestep {timestep}: Not saving data because episode_terminated_unsuccessfully = True")
        # --- End Save biomechanics data ---

        # --- Save torque profile data ---
        if args_cli.save_torque_profiles and not episode_terminated_unsuccessfully:
            # Collect torque profile data for motion imitation
            try:
                # Access robot state directly from environment
                robot = env.unwrapped.scene["robot"]
                
                # Get current time for this sample
                current_time = timestep * dt
                
                # Extract joint torques (applied torques)
                joint_torques = robot.data.applied_torque.cpu().numpy()  # Shape: (num_envs, num_dofs)
                
                # Extract DOF positions and velocities
                dof_positions = robot.data.joint_pos.cpu().numpy()  # Shape: (num_envs, num_dofs)
                dof_velocities = robot.data.joint_vel.cpu().numpy()  # Shape: (num_envs, num_dofs)
                
                # Extract body positions and orientations
                body_positions = robot.data.body_pos_w.cpu().numpy()  # Shape: (num_envs, num_bodies, 3)
                body_rotations = robot.data.body_quat_w.cpu().numpy()  # Shape: (num_envs, num_bodies, 4) - wxyz quaternion
                
                # Extract body velocities
                body_linear_vels = robot.data.body_lin_vel_w.cpu().numpy()  # Shape: (num_envs, num_bodies, 3)
                body_angular_vels = robot.data.body_ang_vel_w.cpu().numpy()  # Shape: (num_envs, num_bodies, 3)
                
                # Collect data for each active environment
                num_active_envs = joint_torques.shape[0]
                for env_idx in range(num_active_envs):
                    torque_data_point = {
                        'timestep': timestep,
                        'time': current_time,
                        'env_idx': env_idx,
                        'joint_torques': joint_torques[env_idx],
                        'dof_positions': dof_positions[env_idx], 
                        'dof_velocities': dof_velocities[env_idx],
                        'body_positions': body_positions[env_idx],
                        'body_rotations': body_rotations[env_idx],
                        'body_linear_velocities': body_linear_vels[env_idx],
                        'body_angular_velocities': body_angular_vels[env_idx]
                    }
                    current_torque_episode_data.append(torque_data_point)
                    
                if timestep == 0:
                    print(f"[INFO] Torque profile collection started. Shape info:")
                    print(f"  - Joint torques: {joint_torques.shape}")
                    print(f"  - DOF positions: {dof_positions.shape}")
                    print(f"  - DOF velocities: {dof_velocities.shape}")
                    print(f"  - Body positions: {body_positions.shape}")
                    print(f"  - Body rotations: {body_rotations.shape}")
                    
            except Exception as e:
                if timestep == 0:  # Only print error once
                    print(f"[WARNING] Could not collect torque profile data: {e}")
        # --- End Save torque profile data ---

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # Pass current timestep for potential use by the agent (e.g., in recurrent policies)
            outputs = runner.agent.act(current_obs, timestep=timestep, timesteps=timestep)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                # Default to outputs[0] if mean_actions not present, common for eval mode
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            next_obs, rewards, terminated, truncated, next_info = env.step(actions)
        
        # Check for episode termination
        episode_ended = False
        if hasattr(terminated, 'any'):  # Handle tensor/array termination signals
            if terminated.any() or truncated.any():
                episode_ended = True
                # Check if termination was due to failure (e.g., height termination, falling, etc.)
                # Check for failure indicators regardless of whether terminated or truncated
                failure_detected = False
                if next_info and "extras" in next_info and next_info["extras"]:
                    # Debug: Print available extras when episode ends
                    print(f"[DEBUG] Episode ended. Available extras: {list(next_info['extras'].keys())}")
                    
                    # Common failure indicators in Isaac Lab environments
                    failure_keys = ["height_termination", "body_height", "fallen", "failure", "terminate"]
                    for failure_key in failure_keys:
                        if failure_key in next_info["extras"]:
                            failure_tensor = next_info["extras"][failure_key]
                            print(f"[DEBUG] Checking {failure_key}: {failure_tensor}")
                            if hasattr(failure_tensor, 'any') and failure_tensor.any():
                                failure_detected = True
                                print(f"[INFO] Episode terminated due to failure ({failure_key}). Data will not be saved.")
                                break
                            elif hasattr(failure_tensor, 'item') and failure_tensor.item():
                                failure_detected = True
                                print(f"[INFO] Episode terminated due to failure ({failure_key}). Data will not be saved.")
                                break
                
                # Additional check: if episode ended very early (< 100 timesteps), likely a failure
                # unless we're in distance mode and reached the target
                if not failure_detected:
                    episode_length = len(current_episode_data)
                    if episode_length < 1000 and not (args_cli.use_distance_termination and max_distance_reached):
                        failure_detected = True
                        print(f"[INFO] Episode ended early ({episode_length} timesteps), treating as failure. Data will not be saved.")
                
                if failure_detected:
                    episode_terminated_unsuccessfully = True
                    # Clear current episode data as it's not successful
                    current_episode_data = []
                else:
                    # Successful termination (e.g., time limit reached, distance goal achieved)
                    print(f"[INFO] Episode completed successfully. Saving {len(current_episode_data)} data points.")
                    biomechanics_data_to_save.extend(current_episode_data)
                    current_episode_data = []
                    # Also save torque profile data for successful episodes
                    if args_cli.save_torque_profiles and current_torque_episode_data:
                        print(f"[INFO] Saving {len(current_torque_episode_data)} torque profile data points.")
                        torque_profiles_to_save.extend(current_torque_episode_data)
                        current_torque_episode_data = []
                    episode_terminated_unsuccessfully = False
        elif isinstance(terminated, bool):  # Handle single boolean termination
            if terminated or truncated:
                episode_ended = True
                # Check for failure indicators regardless of termination type
                failure_detected = False
                if next_info and "extras" in next_info and next_info["extras"]:
                    # Debug: Print available extras when episode ends
                    print(f"[DEBUG] Episode ended (single env). Available extras: {list(next_info['extras'].keys())}")
                    
                    failure_keys = ["height_termination", "body_height", "fallen", "failure", "terminate"]
                    for failure_key in failure_keys:
                        if failure_key in next_info["extras"]:
                            failure_tensor = next_info["extras"][failure_key]
                            print(f"[DEBUG] Checking {failure_key}: {failure_tensor}")
                            if hasattr(failure_tensor, 'item'):
                                if failure_tensor.item():
                                    failure_detected = True
                                    print(f"[INFO] Episode terminated due to failure ({failure_key}). Data will not be saved.")
                                    break
                            elif hasattr(failure_tensor, 'any') and failure_tensor.any():
                                failure_detected = True
                                print(f"[INFO] Episode terminated due to failure ({failure_key}). Data will not be saved.")
                                break
                
                # Additional check: if episode ended very early (< 100 timesteps), likely a failure
                if not failure_detected:
                    episode_length = len(current_episode_data)
                    if episode_length < 1000 and not (args_cli.use_distance_termination and max_distance_reached):
                        failure_detected = True
                        print(f"[INFO] Episode ended early ({episode_length} timesteps), treating as failure. Data will not be saved.")
                
                if failure_detected:
                    episode_terminated_unsuccessfully = True
                    current_episode_data = []
                    # Clear torque data for failed episodes
                    if args_cli.save_torque_profiles:
                        current_torque_episode_data = []
                else:
                    print(f"[INFO] Episode completed successfully. Saving {len(current_episode_data)} data points.")
                    biomechanics_data_to_save.extend(current_episode_data)
                    current_episode_data = []
                    # Also save torque profile data for successful episodes
                    if args_cli.save_torque_profiles and current_torque_episode_data:
                        print(f"[INFO] Saving {len(current_torque_episode_data)} torque profile data points.")
                        torque_profiles_to_save.extend(current_torque_episode_data)
                        current_torque_episode_data = []
                    episode_terminated_unsuccessfully = False
        
        # Update obs and info for the next iteration
        current_obs = next_obs
        current_info = next_info
        
        # Handle environment reset if episode ended
        if episode_ended:
            print(f"[INFO] Resetting environment after episode termination at timestep {timestep}")
            current_obs, current_info = env.reset()
            # Reset episode tracking for new episode
            episode_terminated_unsuccessfully = False  # Reset failure flag for new episode
            # Reset torque episode data for new episode
            if args_cli.save_torque_profiles:
                current_torque_episode_data = []
            # Reset distance tracking for new episode
            if args_cli.use_distance_termination:
                initial_pelvis_x = None
                current_distance_from_origin = 0.0

        # --- Check distance from origin if using distance termination ---
        if args_cli.use_distance_termination:
            # Access pelvis position directly from the environment's robot data
            try:
                # Get the unwrapped environment to access robot data
                unwrapped_env = env.unwrapped
                if hasattr(unwrapped_env, 'robot') and hasattr(unwrapped_env.robot, 'data'):
                    # Get pelvis position (3D: x, y, z) for first environment
                    pelvis_pos_3d = unwrapped_env.robot.data.body_pos_w[0, unwrapped_env.ref_body_index].cpu().numpy()
                    current_pelvis_x = pelvis_pos_3d[0]  # Extract X coordinate
                    
                    # Set initial position after first step (timestep 1)
                    if initial_pelvis_x is None and timestep >= 1:
                        initial_pelvis_x = current_pelvis_x
                        print(f"[INFO] Initial pelvis X position captured: {initial_pelvis_x:.3f}m")
                    
                    # Calculate distance from initial position
                    if initial_pelvis_x is not None:
                        current_distance_from_origin = abs(current_pelvis_x - initial_pelvis_x)

                        # Print distance progress every 100 timesteps
                        if timestep % 100 == 0:
                            print(f"[INFO] Timestep {timestep}: Distance traveled: {current_distance_from_origin:.2f}m")
                        
                        if current_distance_from_origin >= args_cli.max_distance:
                            max_distance_reached = True
                            print(f"[INFO] Distance termination reached: {current_distance_from_origin:.2f}m >= {args_cli.max_distance}m")
                            # Save current episode data as successful since distance goal was reached
                            print(f"[DEBUG] Current episode data length: {len(current_episode_data)}")
                            print(f"[DEBUG] Episode terminated_unsuccessfully: {episode_terminated_unsuccessfully}")
                            if current_episode_data:
                                print(f"[INFO] Distance goal achieved! Saving {len(current_episode_data)} data points from successful episode.")
                                biomechanics_data_to_save.extend(current_episode_data)
                                print(f"[DEBUG] Total data points now in biomechanics_data_to_save: {len(biomechanics_data_to_save)}")
                                current_episode_data = []
                                # Also save torque profile data for distance goal success
                                if args_cli.save_torque_profiles and current_torque_episode_data:
                                    print(f"[INFO] Distance goal achieved! Saving {len(current_torque_episode_data)} torque profile data points.")
                                    torque_profiles_to_save.extend(current_torque_episode_data)
                                    current_torque_episode_data = []
                                episode_terminated_unsuccessfully = False
                            else:
                                print("[WARNING] Distance goal reached but no current episode data to save!")
                else:
                    # Fallback: disable distance termination if robot data not accessible
                    if timestep == 1:  # Only print this once
                        print("[WARNING] Cannot access robot data. Distance termination disabled.")
                        args_cli.use_distance_termination = False
            except Exception as e:
                # Fallback: disable distance termination on any error
                if timestep == 1:  # Only print this once
                    print(f"[WARNING] Error accessing robot data: {e}. Distance termination disabled.")
                    args_cli.use_distance_termination = False
        # --- End distance checking ---

        if args_cli.video:
            # exit the play loop after recording one video
            if timestep == args_cli.video_length - 1:  # Adjust for 0-indexed timestep
                break
        
        timestep += 1  # Increment timestep for all iterations

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # --- Final distance report ---
    if args_cli.use_distance_termination and initial_pelvis_x is not None:
        print(f"[INFO] Simulation ended. Final distance from start: {current_distance_from_origin:.2f}m after {timestep} timesteps")
        if max_distance_reached:
            print(f"[INFO] Distance target of {args_cli.max_distance}m was reached!")
        else:
            print(f"[INFO] Simulation ended before reaching distance target of {args_cli.max_distance}m")
    else:
        print(f"[INFO] Simulation ended after {timestep} timesteps")
    # --- End final distance report ---

    # --- Write biomechanics data to CSV ---
    if args_cli.save_biomechanics_data:
        print(f"[DEBUG] Final check - current_episode_data length: {len(current_episode_data)}")
        print(f"[DEBUG] Final check - episode_terminated_unsuccessfully: {episode_terminated_unsuccessfully}")
        print(f"[DEBUG] Final check - biomechanics_data_to_save length: {len(biomechanics_data_to_save)}")
        
        # Save any remaining successful episode data
        if current_episode_data and not episode_terminated_unsuccessfully:
            print(f"[INFO] Saving remaining {len(current_episode_data)} data points from ongoing successful episode.")
            biomechanics_data_to_save.extend(current_episode_data)
        
        # Save any remaining successful torque episode data  
        if args_cli.save_torque_profiles and current_torque_episode_data and not episode_terminated_unsuccessfully:
            print(f"[INFO] Saving remaining {len(current_torque_episode_data)} torque profile data points from ongoing successful episode.")
            torque_profiles_to_save.extend(current_torque_episode_data)
        
        if biomechanics_data_to_save:
            if biomechanics_csv_path:
                fieldnames = []
                if biomechanics_data_to_save:
                    all_keys = set()
                    for row in biomechanics_data_to_save:
                        all_keys.update(row.keys())
                    # Ensure timestep is first
                    if "timestep" in all_keys:
                        fieldnames = ["timestep"] + sorted([k for k in all_keys if k != "timestep"])
                    else:
                        fieldnames = sorted(list(all_keys))  # Should ideally not happen if timestep is always added

                with open(biomechanics_csv_path, "w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(biomechanics_data_to_save)
                print(f"[INFO] Biomechanics data saved to {biomechanics_csv_path}")
                print(f"[INFO] Total successful data points saved: {len(biomechanics_data_to_save)}")
        else:
            print("[INFO] No successful episode data to save.")
    # --- End Write biomechanics data to CSV ---

    # --- Write torque profile data to NPZ files ---
    if args_cli.save_torque_profiles:
        if torque_profiles_to_save:
            # Convert torque profile data to NPZ format
            print(f"[INFO] Converting {len(torque_profiles_to_save)} torque profile data points to NPZ format...")
            
            # Extract robot and body names from environment
            try:
                robot = env.unwrapped.scene["robot"]
                dof_names = robot.data.joint_names
                body_names = robot.data.body_names
                print(f"[INFO] Found {len(dof_names)} DOFs and {len(body_names)} bodies")
            except Exception as e:
                print(f"[WARNING] Could not extract DOF/body names from environment: {e}")
                # Fallback to generic names
                num_dofs = torque_profiles_to_save[0]['joint_torques'].shape[0]
                num_bodies = torque_profiles_to_save[0]['body_positions'].shape[0]
                dof_names = [f"joint_{i}" for i in range(num_dofs)]
                body_names = [f"body_{i}" for i in range(num_bodies)]
            
            # Convert data to motion loader format
            num_frames = len(torque_profiles_to_save)
            num_dofs = len(dof_names)
            num_bodies = len(body_names)
            
            # Initialize arrays
            joint_torques = np.zeros((num_frames, num_dofs), dtype=np.float32)
            dof_positions = np.zeros((num_frames, num_dofs), dtype=np.float32)
            dof_velocities = np.zeros((num_frames, num_dofs), dtype=np.float32)
            body_positions = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
            body_rotations = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
            body_linear_velocities = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
            body_angular_velocities = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
            
            # Fill arrays with data
            for i, data_point in enumerate(torque_profiles_to_save):
                joint_torques[i] = data_point['joint_torques']
                dof_positions[i] = data_point['dof_positions']
                dof_velocities[i] = data_point['dof_velocities']
                body_positions[i] = data_point['body_positions']
                body_rotations[i] = data_point['body_rotations']
                body_linear_velocities[i] = data_point['body_linear_velocities']
                body_angular_velocities[i] = data_point['body_angular_velocities']
            
            # Calculate FPS (frames per second) from timestep data
            if len(torque_profiles_to_save) > 1:
                time_diff = torque_profiles_to_save[1]['time'] - torque_profiles_to_save[0]['time']
                fps = 1.0 / time_diff if time_diff > 0 else 60.0  # fallback to 60 FPS
            else:
                fps = 60.0  # fallback to 60 FPS
            
            # Save to NPZ file
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            npz_filename = os.path.join(torque_profiles_dir, f"torque_motion_{timestamp_str}.npz")
            
            np.savez(
                npz_filename,
                fps=np.array(fps),
                dof_names=np.array(dof_names, dtype='U'),
                body_names=np.array(body_names, dtype='U'),
                joint_torques=joint_torques,  # This is the key addition for torque-based learning
                dof_positions=dof_positions,
                dof_velocities=dof_velocities,
                body_positions=body_positions,
                body_rotations=body_rotations,
                body_linear_velocities=body_linear_velocities,
                body_angular_velocities=body_angular_velocities
            )
            
            print(f"[INFO] Torque motion data saved to {npz_filename}")
            print(f"[INFO] Motion duration: {num_frames / fps:.2f} seconds ({num_frames} frames at {fps:.1f} FPS)")
            print(f"[INFO] Shape summary:")
            print(f"  - Joint torques: {joint_torques.shape}")
            print(f"  - DOF positions: {dof_positions.shape}")
            print(f"  - DOF velocities: {dof_velocities.shape}")
            print(f"  - Body positions: {body_positions.shape}")
            print(f"  - Body rotations: {body_rotations.shape}")
            
        else:
            print("[INFO] No successful torque profile data to save.")
    # --- End Write torque profile data to NPZ files ---

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
