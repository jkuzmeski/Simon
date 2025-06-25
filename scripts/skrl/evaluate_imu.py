# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to evaluate trained biomech policy and extract IMU data for plotting.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate trained biomech policy and extract IMU data.")
parser.add_argument("--task", type=str, default="Simon-Half-Run-Biomech", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to evaluate.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--save_data", action="store_true", default=True, help="Save IMU data to file.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import Simon.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import DirectRLEnvCfg


class IMUDataCollector:
    """Helper class to collect and store IMU data during evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset data storage."""
        self.acceleration_data = []
        self.angular_velocity_data = []
        self.orientation_data = []
        self.timestamps = []
        self.step_count = 0
    
    def collect(self, extras_dict, dt=1 / 60):
        """Collect IMU data from environment extras."""
        if 'imu_acceleration' in extras_dict:
            # Convert to CPU and numpy for storage
            self.acceleration_data.append(extras_dict['imu_acceleration'].cpu().numpy())
            self.angular_velocity_data.append(extras_dict['imu_angular_velocity'].cpu().numpy())
            self.orientation_data.append(extras_dict['imu_orientation'].cpu().numpy())
            self.timestamps.append(self.step_count * dt)
            self.step_count += 1
    
    def get_data_arrays(self):
        """Convert collected data to numpy arrays."""
        if not self.acceleration_data:
            return None, None, None, None
        
        return (
            np.array(self.timestamps),
            np.array(self.acceleration_data),
            np.array(self.angular_velocity_data),
            np.array(self.orientation_data)
        )
    
    def plot_imu_data(self, save_path=None):
        """Plot collected IMU data."""
        timestamps, accel_data, gyro_data, quat_data = self.get_data_arrays()
        
        if timestamps is None:
            print("No IMU data collected!")
            return
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot acceleration data (take first environment if multiple)
        if len(accel_data.shape) > 2:
            accel_data = accel_data[:, 0, :]  # First environment
        
        axes[0].plot(timestamps, accel_data[:, 0], label='Accel X', color='r')
        axes[0].plot(timestamps, accel_data[:, 1], label='Accel Y', color='g')
        axes[0].plot(timestamps, accel_data[:, 2], label='Accel Z', color='b')
        axes[0].set_ylabel('Acceleration (m/s²)')
        axes[0].set_title('IMU Linear Acceleration')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot angular velocity data
        if len(gyro_data.shape) > 2:
            gyro_data = gyro_data[:, 0, :]  # First environment
        
        axes[1].plot(timestamps, gyro_data[:, 0], label='Gyro X', color='r')
        axes[1].plot(timestamps, gyro_data[:, 1], label='Gyro Y', color='g')
        axes[1].plot(timestamps, gyro_data[:, 2], label='Gyro Z', color='b')
        axes[1].set_ylabel('Angular Velocity (rad/s)')
        axes[1].set_title('IMU Angular Velocity')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot orientation (quaternion magnitude and components)
        if len(quat_data.shape) > 2:
            quat_data = quat_data[:, 0, :]  # First environment
        
        axes[2].plot(timestamps, quat_data[:, 0], label='Quat W', color='k')
        axes[2].plot(timestamps, quat_data[:, 1], label='Quat X', color='r')
        axes[2].plot(timestamps, quat_data[:, 2], label='Quat Y', color='g')
        axes[2].plot(timestamps, quat_data[:, 3], label='Quat Z', color='b')
        axes[2].set_ylabel('Quaternion Components')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_title('IMU Orientation (Quaternion)')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"IMU plot saved to: {save_path}")
        
        plt.show()
    
    def save_data(self, save_path):
        """Save collected IMU data to file."""
        timestamps, accel_data, gyro_data, quat_data = self.get_data_arrays()
        
        if timestamps is None:
            print("No IMU data to save!")
            return
        
        np.savez(save_path,
                 timestamps=timestamps,
                 acceleration=accel_data,
                 angular_velocity=gyro_data,
                 orientation=quat_data)
        print(f"IMU data saved to: {save_path}")


@hydra_task_config(args_cli.task, "skrl_amp_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    """Evaluate trained policy and collect IMU data."""
    
    # override configurations
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # wrap for video recording if requested
    if args_cli.video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_kwargs = {
            "video_folder": f"./imu_evaluation_videos_{timestamp}",
            "step_trigger": lambda step: True,  # Record all episodes
            "video_length": 1000,  # Max episode length
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # initialize data collector
    imu_collector = IMUDataCollector()
    
    print(f"Evaluating policy for {args_cli.num_episodes} episodes...")
    
    total_rewards = []
    
    for episode in range(args_cli.num_episodes):
        print(f"\nEpisode {episode + 1}/{args_cli.num_episodes}")
        
        # reset environment and data collector
        obs, info = env.reset()
        imu_collector.reset()
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # for evaluation, we can use random actions or load a trained policy
            # here using random actions as an example
            action = env.action_space.sample()
            
            # step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward.mean().item() if hasattr(reward, 'mean') else reward
            
            # collect IMU data from extras
            if hasattr(env.unwrapped, 'extras') and env.unwrapped.extras:
                imu_collector.collect(env.unwrapped.extras)
            
            step += 1
            
            # print progress occasionally
            if step % 100 == 0:
                print(f"  Step {step}, Reward: {episode_reward:.2f}")
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} completed. Total reward: {episode_reward:.2f}")
        
        # plot and save data for each episode
        if args_cli.save_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_filename = f"imu_data_episode_{episode+1}_{timestamp}.npz"
            plot_filename = f"imu_plot_episode_{episode+1}_{timestamp}.png"
            
            imu_collector.save_data(data_filename)
            imu_collector.plot_imu_data(plot_filename)
    
    # print summary
    print(f"\nEvaluation completed!")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")
    
    # close environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
