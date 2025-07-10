# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate
from isaaclab.sensors import Imu, ImuCfg, ContactSensor, ContactSensorCfg


from .simon_torque_env_cfg import SimonTorqueEnvCfg
from .motions import MotionLoader
# Try to import TorqueMotionLoader if available, fallback to regular MotionLoader
try:
    from Movement.torque_motion_loader import TorqueMotionLoader
    TORQUE_MOTION_LOADER_AVAILABLE = True
except ImportError:
    TorqueMotionLoader = MotionLoader  # Fallback
    TORQUE_MOTION_LOADER_AVAILABLE = False


class SimonTorqueEnv(DirectRLEnv):
    cfg: SimonTorqueEnvCfg

    def __init__(self, cfg: SimonTorqueEnvCfg, render_mode: str | None = None, **kwargs):
        # create IMU sensor configuration only if enabled
        if cfg.enable_imu_sensor:
            self.imu_cfg = ImuCfg(
                prim_path="/World/envs/env_.*/Robot/simon/pelvis",
                update_period=0.001,
                history_length=5,
                debug_vis=True,
            )
        else:
            self.imu_cfg = None

        # create contact sensor configurations for feet only if enabled
        if cfg.enable_contact_sensors:
            self.left_foot_contact_cfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/simon/left_foot",
                update_period=0.001,
                history_length=5,
                debug_vis=False,
            )

            self.right_foot_contact_cfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/simon/right_foot",
                update_period=0.001,
                history_length=5,
                debug_vis=False,
            )
        else:
            self.left_foot_contact_cfg = None
            self.right_foot_contact_cfg = None

        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale for torque control
        # For torque control, we typically want actions in range [-1, 1] to map to reasonable torque values
        # Estimate reasonable torque limits based on joint characteristics
        max_torque = 300.0  # Maximum torque in N⋅m (adjust based on your robot's capabilities)
        self.action_offset = torch.zeros(self.robot.num_joints, device=self.device)  # No offset for torques
        self.action_scale = torch.full((self.robot.num_joints,), max_torque, device=self.device)  # Scale to max torque

        # load motion - use TorqueMotionLoader if available and torque features enabled
        try:
            if TORQUE_MOTION_LOADER_AVAILABLE and cfg.enable_torque_amp:
                self._motion_loader = TorqueMotionLoader(motion_file=self.cfg.motion_file, device=self.device)
                print("[INFO] Using TorqueMotionLoader for torque-based AMP")
            else:
                self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)
                if cfg.enable_torque_amp:
                    print("[WARNING] TorqueMotionLoader not available, falling back to regular MotionLoader")
        except ValueError as e:
            # Fallback to regular MotionLoader if torque data not found
            print(f"[WARNING] Failed to load as torque motion file: {e}")
            print("[INFO] Falling back to regular MotionLoader")
            self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # Check if we actually have torque data and adjust observation space accordingly
        self.has_torque_data = hasattr(self._motion_loader, 'has_torque_data') and self._motion_loader.has_torque_data
        if cfg.enable_torque_amp and self.has_torque_data:
            # Use torque-enhanced observation space
            actual_amp_obs_space = 64  # 50 standard + 14 torques
            actual_policy_obs_space = 64  # Policy observations should match AMP for consistency
            print(f"[INFO] Torque data available - using expanded observation spaces: AMP={actual_amp_obs_space}, Policy={actual_policy_obs_space}")
        else:
            # Use standard observation space
            actual_amp_obs_space = 50
            actual_policy_obs_space = 50
            if cfg.enable_torque_amp:
                print(f"[WARNING] Torque AMP requested but no torque data available - using standard observation spaces: AMP={actual_amp_obs_space}, Policy={actual_policy_obs_space}")
            else:
                print(f"[INFO] Using standard observation spaces: AMP={actual_amp_obs_space}, Policy={actual_policy_obs_space}")
        
        # Update the configuration to match actual capabilities
        self.cfg.amp_observation_space = actual_amp_obs_space
        self.cfg.observation_space = actual_policy_obs_space

        # DOF and key body indexes
        key_body_names = ["right_foot", "left_foot", "pelvis"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        # Debug information
        print("=== JOINT ORDER DEBUG ===")
        print(f"Robot joint names: {self.robot.data.joint_names}")
        print(f"Motion DOF indexes: {self.motion_dof_indexes}")
        print(f"Motion DOF names: {self._motion_loader.dof_names}")

        # Print the complete observation mapping
        print("\n=== COMPLETE OBSERVATION MAPPING ===")
        obs_index = 0

        # Joint positions (14 values)
        print("Joint Positions:")
        for i, joint_name in enumerate(self.robot.data.joint_names):
            print(f"policy_obs_{obs_index}: {joint_name} (position)")
            obs_index += 1

        # Joint velocities (14 values)
        print("\nJoint Velocities:")
        for i, joint_name in enumerate(self.robot.data.joint_names):
            print(f"policy_obs_{obs_index}: {joint_name} (velocity)")
            obs_index += 1

        # Root height (1 value)
        print("\nRoot State:")
        print(f"policy_obs_{obs_index}: root_height (Z position)")
        obs_index += 1

        # Quaternion to tangent and normal (6 values)
        print("\nRoot Orientation (Tangent + Normal vectors):")
        print(f"policy_obs_{obs_index}: root_tangent_x")
        print(f"policy_obs_{obs_index+1}: root_tangent_y")
        print(f"policy_obs_{obs_index+2}: root_tangent_z")
        print(f"policy_obs_{obs_index+3}: root_normal_x")
        print(f"policy_obs_{obs_index+4}: root_normal_y")
        print(f"policy_obs_{obs_index+5}: root_normal_z")
        obs_index += 6

        # Root linear velocities (3 values)
        print("\nRoot Linear Velocities:")
        print(f"policy_obs_{obs_index}: root_lin_vel_x")
        print(f"policy_obs_{obs_index+1}: root_lin_vel_y")
        print(f"policy_obs_{obs_index+2}: root_lin_vel_z")
        obs_index += 3

        # Root angular velocities (3 values)
        print("\nRoot Angular Velocities:")
        print(f"policy_obs_{obs_index}: root_ang_vel_x")
        print(f"policy_obs_{obs_index+1}: root_ang_vel_y")
        print(f"policy_obs_{obs_index+2}: root_ang_vel_z")
        obs_index += 3

        # Key body positions relative to root (9 values: 3 bodies × 3 coordinates)
        key_body_names = ["right_foot", "left_foot", "pelvis"]
        print("\nKey Body Positions (relative to root):")
        for body_name in key_body_names:
            print(f"policy_obs_{obs_index}: {body_name}_rel_x")
            print(f"policy_obs_{obs_index+1}: {body_name}_rel_y")
            print(f"policy_obs_{obs_index+2}: {body_name}_rel_z")
            obs_index += 3

        print(f"\nTotal observation size: {obs_index}")
        if self.has_torque_data and self.cfg.include_torque_in_amp:
            print("\nTorque Observations (for AMP discriminator):")
            for i, joint_name in enumerate(self.robot.data.joint_names):
                print(f"amp_obs_{obs_index + i}: {joint_name} (torque)")
            obs_index += len(self.robot.data.joint_names)
            print(f"Total AMP observation size with torques: {obs_index}")
        print("=" * 50)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # create IMU sensor after robot is added to scene
        if self.cfg.enable_imu_sensor and self.imu_cfg is not None:
            try:
                self.imu_sensor = Imu(self.imu_cfg)
                self.scene.sensors["pelvis_imu"] = self.imu_sensor
            except Exception as e:
                print(f"Warning: Failed to create IMU sensor: {e}")
                self.imu_sensor = None
        else:
            self.imu_sensor = None

        # create contact sensors only if enabled
        if self.cfg.enable_contact_sensors and self.left_foot_contact_cfg is not None:
            try:
                self.left_foot_contact = ContactSensor(self.left_foot_contact_cfg)
                self.scene.sensors["left_foot_contact"] = self.left_foot_contact
            except Exception as e:
                print(f"Warning: Failed to create left foot contact sensor: {e}")
                self.left_foot_contact = None
        else:
            self.left_foot_contact = None

        if self.cfg.enable_contact_sensors and self.right_foot_contact_cfg is not None:
            try:
                self.right_foot_contact = ContactSensor(self.right_foot_contact_cfg)
                self.scene.sensors["right_foot_contact"] = self.right_foot_contact
            except Exception as e:
                print(f"Warning: Failed to create right foot contact sensor: {e}")
                self.right_foot_contact = None
        else:
            self.right_foot_contact = None

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        # Use torque control for the agent's learning
        # Actions represent desired joint torques directly
        applied_torques = self.action_scale * self.actions  # Scale actions to appropriate torque range
        self.robot.set_joint_effort_target(applied_torques)
        
        # Store the applied torques for observations and AMP
        self.current_applied_torques = applied_torques.clone()

    def _get_observations(self) -> dict:
        # build task observation - include torques in AMP observations if available
        if self.has_torque_data and self.cfg.include_torque_in_amp:
            # Use torque-enhanced observations for AMP buffer
            obs = compute_obs_with_torques(
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
                self.robot.data.body_pos_w[:, self.ref_body_index],
                self.robot.data.body_quat_w[:, self.ref_body_index],
                self.robot.data.body_lin_vel_w[:, self.ref_body_index],
                self.robot.data.body_ang_vel_w[:, self.ref_body_index],
                self.robot.data.body_pos_w[:, self.key_body_indexes],
                self.robot.data.applied_torque,  # Current robot torques
            )
        else:
            # Standard observations
            obs = compute_obs(
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
                self.robot.data.body_pos_w[:, self.ref_body_index],
                self.robot.data.body_quat_w[:, self.ref_body_index],
                self.robot.data.body_lin_vel_w[:, self.ref_body_index],
                self.robot.data.body_ang_vel_w[:, self.ref_body_index],
                self.robot.data.body_pos_w[:, self.key_body_indexes],
            )
        
        # add additional torques to observation if configured (separate from AMP)
        if self.cfg.include_torque_in_obs and not (self.has_torque_data and self.cfg.include_torque_in_amp):
            # get applied torques (from robot actions) - only if not already included above
            applied_torques = self.robot.data.applied_torque
            obs = torch.cat([obs, applied_torques], dim=-1)

        # collect IMU data
        imu_data = {}
        if hasattr(self, 'imu_sensor') and self.imu_sensor is not None:
            try:
                imu_data = {
                    'imu_acceleration': self.imu_sensor.data.lin_acc_b.clone(),
                    'imu_angular_velocity': self.imu_sensor.data.ang_vel_b.clone(),
                    'imu_orientation': self.imu_sensor.data.quat_w.clone(),
                }
            except Exception as e:
                print(f"Warning: Failed to get IMU data: {e}")

        # collect contact sensor data
        left_contact_data = {}
        if hasattr(self, 'left_foot_contact') and self.left_foot_contact is not None:
            try:
                left_contact_data = {
                    'net_force_left_foot': self.left_foot_contact.data.net_forces_w.clone(),
                }
            except Exception as e:
                print(f" Failed to get left foot contact data: {e}")
        right_contact_data = {}
        if hasattr(self, 'right_foot_contact') and self.right_foot_contact is not None:
            try:
                right_contact_data = {
                    'net_force_right_foot': self.right_foot_contact.data.net_forces_w.clone(),
                }
            except Exception as e:
                print(f"Failed to get right foot contact data: {e}")

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = obs.clone()

        # combine all sensor data in extras
        self.extras = {
            "amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size),
            "pelvis_position_global": self.robot.data.body_pos_w[:, self.ref_body_index].clone(),
            "pelvis_orientation_global": self.robot.data.body_quat_w[:, self.ref_body_index].clone(),
            "pelvis_linear_velocity_global": self.robot.data.body_lin_vel_w[:, self.ref_body_index].clone(),
            "pelvis_angular_velocity_global": self.robot.data.body_ang_vel_w[:, self.ref_body_index].clone(),
            **imu_data,
            **left_contact_data,
            **right_contact_data,
        }

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # get root transforms (the humanoid torso)
        motion_torso_index = self._motion_loader.get_body_index(["pelvis"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.15  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # env methods

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        if hasattr(self._motion_loader, 'sample_with_torques') and self.has_torque_data and self.cfg.include_torque_in_amp:
            # use torque-enabled motion loader
            (
                dof_positions,
                dof_velocities,
                body_positions,
                body_rotations,
                body_linear_velocities,
                body_angular_velocities,
                joint_torques,
            ) = self._motion_loader.sample_with_torques(num_samples=num_samples, times=times)
            
            # compute AMP observation with torques
            amp_observation = compute_obs_with_torques(
                dof_positions[:, self.motion_dof_indexes],
                dof_velocities[:, self.motion_dof_indexes],
                body_positions[:, self.motion_ref_body_index],
                body_rotations[:, self.motion_ref_body_index],
                body_linear_velocities[:, self.motion_ref_body_index],
                body_angular_velocities[:, self.motion_ref_body_index],
                body_positions[:, self.motion_key_body_indexes],
                joint_torques[:, self.motion_dof_indexes],
            )
        else:
            # fallback to regular motion loader
            (
                dof_positions,
                dof_velocities,
                body_positions,
                body_rotations,
                body_linear_velocities,
                body_angular_velocities,
            ) = self._motion_loader.sample(num_samples=num_samples, times=times)
            
            # compute AMP observation without torques
            amp_observation = compute_obs(
                dof_positions[:, self.motion_dof_indexes],
                dof_velocities[:, self.motion_dof_indexes],
                body_positions[:, self.motion_ref_body_index],
                body_rotations[:, self.motion_ref_body_index],
                body_linear_velocities[:, self.motion_ref_body_index],
                body_angular_velocities[:, self.motion_ref_body_index],
                body_positions[:, self.motion_key_body_indexes],
            )

        # Debug information to understand the shape mismatch
        # print(f"Debug - AMP observation shape: {amp_observation.shape}")
        # print(f"Debug - Expected amp_observation_size: {self.amp_observation_size}")
        # print(f"Debug - Total elements: {amp_observation.numel()}")
        # print(f"Debug - num_samples: {num_samples}")
        # print(f"Debug - cfg.num_amp_observations: {self.cfg.num_amp_observations}")
        # print(f"Debug - cfg.amp_observation_space: {self.cfg.amp_observation_space}")

        # Check if reshaping is possible
        total_elements = amp_observation.numel()
        if total_elements % self.amp_observation_size != 0:
            print(f"Warning: Cannot reshape tensor of size {total_elements} to [-1, {self.amp_observation_size}]")
            # Calculate correct number of samples that fit
            correct_samples = total_elements // self.amp_observation_size
            print(f"Adjusting to {correct_samples} samples")
            # Trim to make it divisible
            elements_to_keep = correct_samples * self.amp_observation_size
            amp_observation = amp_observation.flatten()[:elements_to_keep]

        return amp_observation.view(-1, self.amp_observation_size)

    # Add this method to ensure extras are passed to info
    def step(self, actions):
        obs, rewards, terminated, truncated, info = super().step(actions)
        # Add extras to info so they can be accessed in the biomechanics script
        if hasattr(self, 'extras') and self.extras:
            info["extras"] = self.extras
        return obs, rewards, terminated, truncated, info

    def set_train_mode(self):
        """Set environment to training mode (disable sensors)"""
        self.cfg.enable_imu_sensor = False
        self.cfg.enable_contact_sensors = False
        self.cfg.save_biomechanics_data = False
        print("[INFO] Environment set to training mode - sensors disabled for performance")

    def set_eval_mode(self, enable_sensors=True):
        """Set environment to evaluation mode (enable sensors)"""
        self.cfg.enable_imu_sensor = enable_sensors
        self.cfg.enable_contact_sensors = enable_sensors
        self.cfg.save_biomechanics_data = enable_sensors
        print(f"[INFO] Environment set to evaluation mode - sensors {'enabled' if enable_sensors else 'disabled'}")


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_rotate(q, ref_tangent)
    normal = quat_rotate(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_obs_with_torques(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
    joint_torques: torch.Tensor,
) -> torch.Tensor:
    """Compute observations including joint torques for AMP discriminator."""
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
            joint_torques,  # Add joint torques for torque-based AMP
        ),
        dim=-1,
    )
    return obs
