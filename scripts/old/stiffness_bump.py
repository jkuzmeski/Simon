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
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
parser.add_argument(
    "--modulation_type",
    type=str,
    default="stiffness",
    choices=["stiffness", "effort"],
    help="Type of joint parameter to modulate over distance: 'stiffness' or 'effort'.",
)
parser.add_argument(
    "--modulation_percent",
    type=float,
    default=0.0,
    help="Percentage change in the selected parameter over max_distance (e.g., 20.0 for 20% increase in stiffness or 20% decrease in effort limit).",
)
parser.add_argument(
    "--enable_random_bumps",
    action="store_true",
    default=False,
    help="Enable random force bumps at the pelvis during simulation.",
)
parser.add_argument(
    "--bump_force_magnitude",
    type=float,
    default=50.0,
    help="Magnitude of the random bump force (in Newtons).",
)
parser.add_argument(
    "--bump_interval_range",
    type=float,
    nargs=2,
    default=[2.0, 8.0],
    help="Range for random intervals between bumps (in seconds).",
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
import random

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
import isaaclab.assets.articulation

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
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
        biomechanics_dir = os.path.join(log_dir, "stiffness")
        os.makedirs(biomechanics_dir, exist_ok=True)
        # Create a unique filename with a timestamp
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        biomechanics_csv_path = os.path.join(biomechanics_dir, f"stiffness_data_{timestamp_str}.csv")
        print(f"[INFO] Saving stiffness data to: {biomechanics_csv_path}")
        print("[INFO] Only successful runs (not terminated due to failure) will be saved.")
    # --- End Biomechanics data saving setup ---

    # --- Random bump setup ---
    bump_data_to_save = []
    bump_csv_path = None
    next_bump_time = None
    
    if args_cli.enable_random_bumps:
        bump_dir = os.path.join(log_dir, "bump_events")
        os.makedirs(bump_dir, exist_ok=True)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        bump_csv_path = os.path.join(bump_dir, f"bump_events_{timestamp_str}.csv")
        print(f"[INFO] Random bumps enabled with force magnitude: {args_cli.bump_force_magnitude}N")
        print(f"[INFO] Bump interval range: {args_cli.bump_interval_range[0]}-{args_cli.bump_interval_range[1]} seconds")
        print(f"[INFO] Saving bump event data to: {bump_csv_path}")
        
        # Schedule first bump
        first_bump_delay = random.uniform(args_cli.bump_interval_range[0], args_cli.bump_interval_range[1])
        next_bump_time = first_bump_delay * 480  # Convert to timesteps (assuming 480 Hz)
        print(f"[INFO] First bump scheduled at timestep {int(next_bump_time)}")
    # --- End Random bump setup ---

    # reset environment
    current_obs, current_info = env.reset()  # Get initial obs and info
    timestep = 0
    
    # --- Distance tracking setup ---
    initial_pelvis_x = None
    current_distance_from_origin = 0.0
    max_distance_reached = False
    initial_knee_stiffness = None
    initial_knee_damping = None
    initial_knee_effort_limit = None
    knee_joint_ids = []
    
    # We'll set the initial pelvis position after the first step, not from reset
    if args_cli.use_distance_termination:
        print(f"[INFO] Distance termination enabled. Will run until agent moves {args_cli.max_distance}m from origin.")
        print("[INFO] Initial pelvis position will be captured after first simulation step.")
        print(f"[INFO] Timestep limit set to {args_cli.max_timesteps_distance} for distance-based termination.")
        print(f"[INFO] Modulation mode: {args_cli.modulation_type} will change by {args_cli.modulation_percent}% over distance.")
    else:
        print("[INFO] Timestep termination enabled. Will run for up to 1000 timesteps.")
    # --- End Distance tracking setup ---
    
    # simulate environment - terminate based on timesteps or distance
    max_timesteps = 1000 if not args_cli.use_distance_termination else args_cli.max_timesteps_distance
    while timestep < max_timesteps and not max_distance_reached:
        
        # --- Apply random bump if enabled and scheduled ---
        if args_cli.enable_random_bumps and next_bump_time is not None and timestep >= next_bump_time:
            try:
                # Get the unwrapped environment to access the robot
                unwrapped_env = env.unwrapped
                if hasattr(unwrapped_env, 'robot') and hasattr(unwrapped_env.robot, 'data'):
                    # Apply random force bump to pelvis
                    num_envs = unwrapped_env.robot.num_instances
                    
                    # Find pelvis body index
                    pelvis_body_ids = [unwrapped_env.robot.data.body_names.index("pelvis")]
                    
                    # Create random force direction (mostly horizontal)
                    force_direction = torch.zeros(3, device=unwrapped_env.device)
                    force_direction[0] = random.uniform(-0.01, 0.01)  # X component
                    force_direction[1] = random.uniform(-1.0, 1.0)  # Y component
                    force_direction[2] = random.uniform(-0.01, 0.01)  # Small Z component
                    force_direction = force_direction / torch.norm(force_direction)  # Normalize
                    
                    # Scale by magnitude
                    bump_force = force_direction * args_cli.bump_force_magnitude
                    
                    # Apply force to all environments
                    forces = torch.zeros(num_envs, len(pelvis_body_ids), 3, device=unwrapped_env.device)
                    torques = torch.zeros(num_envs, len(pelvis_body_ids), 3, device=unwrapped_env.device)
                    forces[:, 0, :] = bump_force
                    
                    unwrapped_env.robot.set_external_force_and_torque(forces, torques, body_ids=pelvis_body_ids)
                    
                    # Record bump event
                    bump_event = {
                        'timestep': timestep,
                        'force_x': bump_force[0].item(),
                        'force_y': bump_force[1].item(),
                        'force_z': bump_force[2].item(),
                        'force_magnitude': args_cli.bump_force_magnitude,
                    }
                    
                    # Add distance info if available
                    if args_cli.use_distance_termination and initial_pelvis_x is not None:
                        bump_event['distance_from_origin'] = current_distance_from_origin
                        pelvis_pos_3d = unwrapped_env.robot.data.body_pos_w[0, unwrapped_env.ref_body_index].cpu().numpy()
                        bump_event['pelvis_x'] = pelvis_pos_3d[0]
                        bump_event['pelvis_y'] = pelvis_pos_3d[1]
                        bump_event['pelvis_z'] = pelvis_pos_3d[2]
                    
                    bump_data_to_save.append(bump_event)
                    
                    print(f"[BUMP] Applied force at timestep {timestep}: [{bump_force[0]:.1f}, {bump_force[1]:.1f}, {bump_force[2]:.1f}] N")
                    
                    # Schedule next bump
                    next_interval = random.uniform(args_cli.bump_interval_range[0], args_cli.bump_interval_range[1])
                    next_bump_time = timestep + (next_interval * 480)  # Convert seconds to timesteps
                    
            except Exception as e:
                print(f"[WARNING] Failed to apply bump force: {e}")
                # Schedule next bump anyway
                next_interval = random.uniform(args_cli.bump_interval_range[0], args_cli.bump_interval_range[1])
                next_bump_time = timestep + (next_interval * 480)
        # --- End bump application ---
        
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
                else:
                    print(f"[INFO] Episode completed successfully. Saving {len(current_episode_data)} data points.")
                    biomechanics_data_to_save.extend(current_episode_data)
                    current_episode_data = []
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
                    # On the first step, get knee joint info
                    if timestep == 1 and initial_knee_stiffness is None:
                        robot = unwrapped_env.robot
                        # Find knee joint indices by name. Adjust names if needed.
                        knee_joint_names = ["right_knee", "left_knee"]
                        robot_joint_names = [name.strip() for name in robot.joint_names]
                        for name in knee_joint_names:
                            if name in robot_joint_names:
                                knee_joint_ids.append(robot_joint_names.index(name))
                        
                        if knee_joint_ids:
                            # Store the initial values for the knee joints
                            initial_knee_stiffness = robot.data.joint_stiffness[0, knee_joint_ids].clone()
                            initial_knee_damping = robot.data.joint_damping[0, knee_joint_ids].clone()
                            initial_knee_effort_limit = robot.data.joint_effort_limits[0, knee_joint_ids].clone()
                            
                            print(f"[INFO] Found knee joints at indices: {knee_joint_ids}")
                            print(f"[INFO] Initial knee stiffness: {initial_knee_stiffness.cpu().numpy()}")
                            print(f"[INFO] Initial knee damping: {initial_knee_damping.cpu().numpy()}")
                            print(f"[INFO] Initial knee effort limits: {initial_knee_effort_limit.cpu().numpy()}")
                            print(f"[INFO] Modulation type: {args_cli.modulation_type}")
                            print(f"[INFO] Modulation percent: {args_cli.modulation_percent}%")
                            
                            # Check if initial stiffness is zero and set a default value (only for stiffness mode)
                            if args_cli.modulation_type == "stiffness":
                                stiffness_was_zero = torch.all(initial_knee_stiffness == 0)
                                damping_was_zero = torch.all(initial_knee_damping == 0)
                                
                                if stiffness_was_zero:
                                    default_stiffness_value = 50.0
                                    print(f"[WARNING] Initial knee stiffness was 0. Setting to a default of {default_stiffness_value}.")
                                    initial_knee_stiffness = torch.full_like(initial_knee_stiffness, default_stiffness_value)
                                    # Apply this default stiffness to the simulation immediately
                                    unwrapped_env.robot.write_joint_stiffness_to_sim(initial_knee_stiffness, joint_ids=knee_joint_ids)
                                
                                if damping_was_zero:
                                    default_damping_value = 5.0  # Typical damping is usually lower than stiffness
                                    print(f"[WARNING] Initial knee damping was 0. Setting to a default of {default_damping_value}.")
                                    initial_knee_damping = torch.full_like(initial_knee_damping, default_damping_value)
                                    # Apply this default damping to the simulation immediately
                                    unwrapped_env.robot.write_joint_damping_to_sim(initial_knee_damping, joint_ids=knee_joint_ids)
                            
                            # Check effort limits for effort mode
                            elif args_cli.modulation_type == "effort":
                                effort_was_inf = torch.all(torch.isinf(initial_knee_effort_limit))
                                if effort_was_inf:
                                    default_effort_value = 200.0  # Default knee effort limit
                                    print(f"[WARNING] Initial knee effort limit was infinite. Setting to a default of {default_effort_value}.")
                                    initial_knee_effort_limit = torch.full_like(initial_knee_effort_limit, default_effort_value)
                                    # Apply this default effort limit to the simulation immediately
                                    unwrapped_env.robot.write_joint_effort_limit_to_sim(initial_knee_effort_limit, joint_ids=knee_joint_ids)

                        else:
                            print(f"[WARNING] Could not find knee joints. {args_cli.modulation_type.capitalize()} will not be adjusted.")

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
                        
                        # Dynamically adjust knee parameters based on modulation type
                        if knee_joint_ids and initial_knee_stiffness is not None:
                            # Calculate progress factor (0.0 to 1.0 over max_distance)
                            progress_factor = min(current_distance_from_origin / args_cli.max_distance, 1.0)
                            
                            if args_cli.modulation_type == "stiffness":
                                # Increase stiffness and damping by modulation_percent over distance
                                change_factor = 1.0 + (progress_factor * args_cli.modulation_percent / 100.0)
                                
                                new_stiffness = initial_knee_stiffness * change_factor
                                new_damping = initial_knee_damping * change_factor
                                
                                # Apply the new stiffness and damping to the simulation
                                unwrapped_env.robot.write_joint_stiffness_to_sim(new_stiffness, joint_ids=knee_joint_ids)
                                unwrapped_env.robot.write_joint_damping_to_sim(new_damping, joint_ids=knee_joint_ids)

                                if timestep % 100 == 0:  # Print every 100 timesteps
                                    print(f"[INFO] Timestep {timestep}: Distance: {current_distance_from_origin:.2f}m | Knee Stiffness: {new_stiffness.cpu().numpy()} | Knee Damping: {new_damping.cpu().numpy()}")
                            
                            elif args_cli.modulation_type == "effort":
                                # Decrease effort limit by modulation_percent over distance
                                change_factor = 1.0 - (progress_factor * args_cli.modulation_percent / 100.0)
                                # Ensure we don't go below 10% of original effort limit
                                change_factor = max(change_factor, 0.1)
                                
                                new_effort_limit = initial_knee_effort_limit * change_factor
                                
                                # Apply the new effort limit to the simulation
                                unwrapped_env.robot.write_joint_effort_limit_to_sim(new_effort_limit, joint_ids=knee_joint_ids)

                                if timestep % 100 == 0:  # Print every 100 timesteps
                                    print(f"[INFO] Timestep {timestep}: Distance: {current_distance_from_origin:.2f}m | Knee Effort Limit: {new_effort_limit.cpu().numpy()}")
                        
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

    # close the simulator
    env.close()

    # --- Write bump data to CSV ---
    if args_cli.enable_random_bumps and bump_data_to_save:
        print(f"[INFO] Writing {len(bump_data_to_save)} bump event records to CSV...")
        try:
            with open(bump_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = bump_data_to_save[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(bump_data_to_save)
                print(f"[INFO] Bump event data successfully saved to: {bump_csv_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save bump event data: {e}")
    elif args_cli.enable_random_bumps:
        print("[INFO] No bump events occurred during simulation.")
    # --- End Write bump data to CSV ---


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
