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
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
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
    default=False,
    help="Save biomechanics data (observations and extras) to a CSV file in the log directory.",
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
    biomechanics_csv_path = None
    if args_cli.save_biomechanics_data:
        biomechanics_dir = os.path.join(log_dir, "biomechanics")
        os.makedirs(biomechanics_dir, exist_ok=True)
        # Create a unique filename with a timestamp
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        biomechanics_csv_path = os.path.join(biomechanics_dir, f"biomechanics_data_{timestamp_str}.csv")
        print(f"[INFO] Saving biomechanics data to: {biomechanics_csv_path}")
    # --- End Biomechanics data saving setup ---

    # reset environment
    current_obs, current_info = env.reset()  # Get initial obs and info
    timestep = 0
    # simulate environment time step is less than 1000
    while timestep < 1000:
        start_time = time.time()

        # --- Save biomechanics data for current_obs and current_info ---
        if args_cli.save_biomechanics_data:
            num_active_envs = 0
            # Determine number of active environments based on observation structure
            if isinstance(current_obs, dict) and "policy" in current_obs:
                if current_obs["policy"] is not None:
                    num_active_envs = current_obs["policy"].shape[0]
            elif isinstance(current_obs, torch.Tensor):  # Handles single tensor observations
                num_active_envs = current_obs.shape[0]

            for i in range(num_active_envs):
                data_row = {"timestep": timestep}
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
                    for key, tensor_val_all_envs in current_info["extras"].items():
                        if tensor_val_all_envs is not None and tensor_val_all_envs.numel() > 0 and i < tensor_val_all_envs.shape[0]:
                            item_data_tensor = tensor_val_all_envs[i]
                            if item_data_tensor is not None:
                                item_data = item_data_tensor.cpu().numpy()
                                if item_data.ndim == 0:  # scalar
                                    data_row[key] = item_data.item()
                                else:  # vector/tensor, flatten it
                                    flat_item_data = item_data.flatten()
                                    for k_idx, k_val in enumerate(flat_item_data):
                                        data_row[f"{key}_{k_idx}"] = k_val
                
                if len(data_row) > 1:  # Only add if there's more than just timestep
                    biomechanics_data_to_save.append(data_row)
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
        
        # Update obs and info for the next iteration
        current_obs = next_obs
        current_info = next_info

        if args_cli.video:
            # exit the play loop after recording one video
            if timestep == args_cli.video_length - 1:  # Adjust for 0-indexed timestep
                break
        
        timestep += 1  # Increment timestep for all iterations

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # --- Write biomechanics data to CSV ---
    if args_cli.save_biomechanics_data and biomechanics_data_to_save:
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
    # --- End Write biomechanics data to CSV ---

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
