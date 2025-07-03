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


from .simon_biomech_stiffness_env_cfg import SimonBiomechStiffnessEnvCfg
from .motions import MotionLoader


class SimonBiomechStiffnessEnv(DirectRLEnv):
    cfg: SimonBiomechStiffnessEnvCfg

    def __init__(self, cfg: SimonBiomechStiffnessEnvCfg, render_mode: str | None = None, **kwargs):
        # create sensor configurations first
        self.imu_cfg = ImuCfg(
            prim_path="/World/envs/env_.*/Robot/simon/pelvis",
            update_period=0.0,
            # history_length=8,
            debug_vis=True,
        )

        # create contact sensor configurations for feet
        self.left_foot_contact_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/simon/left_foot",  # Verify this path exists
            update_period=0.0,
            # history_length=8,
            debug_vis=False,
        )

        self.right_foot_contact_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/simon/right_foot",  # Verify this path exists
            update_period=0.0,
            # history_length=8,
            debug_vis=False,
        )

        # call the parent constructor. This will call _setup_scene() internally
        # and create the self.robot.actuators dictionary.
        super().__init__(cfg, render_mode, **kwargs)

        # -- FIX: Now that super().__init__() is done, self.robot.actuators exists. --
        # Iterate through the configured actuators and write their properties to the sim.
        for actuator in self.robot.actuators.values():
            if hasattr(actuator, "stiffness"):
                self.robot.write_joint_stiffness_to_sim(actuator.stiffness)
            if hasattr(actuator, "damping"):
                self.robot.write_joint_damping_to_sim(actuator.damping)

        # (Optional) Add a debug print here to confirm the values are now set
        print(f"[INFO] Joint stiffness after writing to sim: {self.robot.data.joint_stiffness[0]}")

        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

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

    def _setup_scene(self):
        # This method now only handles creating and adding assets to the scene.
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
        try:
            self.imu_sensor = Imu(self.imu_cfg)
            self.scene.sensors["pelvis_imu"] = self.imu_sensor
        except Exception as e:
            print(f"Warning: Failed to create IMU sensor: {e}")
            self.imu_sensor = None

        try:
            self.left_foot_contact = ContactSensor(self.left_foot_contact_cfg)
            self.scene.sensors["left_foot_contact"] = self.left_foot_contact
        except Exception as e:
            print(f"Warning: Failed to create left foot contact sensor: {e}")
            self.left_foot_contact = None

        try:
            self.right_foot_contact = ContactSensor(self.right_foot_contact_cfg)
            self.scene.sensors["right_foot_contact"] = self.right_foot_contact
        except Exception as e:
            print(f"Warning: Failed to create right foot contact sensor: {e}")
            self.right_foot_contact = None

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        # With IdealPDActuator, actions represent desired joint positions
        # The actuator handles PD control and torque limiting automatically
        # Scale actions from [-1, 1] to joint position limits
        joint_limits = self.robot.data.soft_joint_pos_limits  # Shape: (num_envs, num_joints, 2)
        joint_range = joint_limits[..., 1] - joint_limits[..., 0]  # Shape: (num_envs, num_joints)
        joint_center = (joint_limits[..., 1] + joint_limits[..., 0]) / 2.0  # Shape: (num_envs, num_joints)
        
        # Debug: Print shapes on first call
        if not hasattr(self, '_debug_printed'):
            print(f"Debug - Action shape: {self.actions.shape}")
            print(f"Debug - Joint limits shape: {joint_limits.shape}")
            print(f"Debug - Joint range shape: {joint_range.shape}")
            print(f"Debug - Joint center shape: {joint_center.shape}")
            print(f"Debug - Joint stiffness: {self.robot.data.joint_stiffness}")
            self._debug_printed = True
        
        # Convert normalized actions to joint positions
        # All tensors now have shape: (num_envs, num_joints)
        target_positions = joint_center + 0.5 * joint_range * self.actions
        
        # Set the target positions (actuator will compute torques)
        self.robot.set_joint_position_target(target_positions)

    def _get_observations(self) -> dict:
        # build task observation
        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        # Check if we're in evaluation mode
        if not hasattr(self, '_is_eval_mode'):
            # Auto-detect on first call if not explicitly set
            self._is_eval_mode = self._auto_detect_eval_mode()
        is_eval_mode = self._is_eval_mode
        
        # Only collect sensor data during evaluation
        if is_eval_mode:
            # collect IMU data (avoid unnecessary cloning)
            imu_data = {}
            if hasattr(self, 'imu_sensor') and self.imu_sensor is not None:
                try:
                    # Use detach() instead of clone() to save memory
                    imu_data = {
                        'imu_acceleration': self.imu_sensor.data.lin_acc_b.detach(),
                        'imu_angular_velocity': self.imu_sensor.data.ang_vel_b.detach(),
                        'imu_orientation': self.imu_sensor.data.quat_w.detach(),
                    }
                except Exception as e:
                    print(f"Warning: Failed to get IMU data: {e}")

            # collect contact sensor data (avoid unnecessary cloning)
            left_contact_data = {}
            if hasattr(self, 'left_foot_contact') and self.left_foot_contact is not None:
                try:
                    left_contact_data = {
                        'net_force_left_foot': self.left_foot_contact.data.net_forces_w.detach(),
                    }
                except Exception as e:
                    print(f" Failed to get left foot contact data: {e}")
            right_contact_data = {}
            if hasattr(self, 'right_foot_contact') and self.right_foot_contact is not None:
                try:
                    right_contact_data = {
                        'net_force_right_foot': self.right_foot_contact.data.net_forces_w.detach(),
                    }
                except Exception as e:
                    print(f"Failed to get right foot contact data: {e}")
        else:
            # During training, use empty dictionaries to save memory
            imu_data = {}
            left_contact_data = {}
            right_contact_data = {}

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = obs.clone()

        # Memory monitoring (add this for debugging)
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 0
            
        # Print memory usage every 1000 steps
        if self._step_count % 1000 == 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                print(f"Step {self._step_count}: GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        # combine all sensor data in extras (reduce cloning to save memory)
        if is_eval_mode:
            self.extras = {
                "amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size),
                # Only add pelvis data during eval
                "pelvis_position_global": self.robot.data.body_pos_w[:, self.ref_body_index],
                "pelvis_orientation_global": self.robot.data.body_quat_w[:, self.ref_body_index],
                "pelvis_linear_velocity_global": self.robot.data.body_lin_vel_w[:, self.ref_body_index],
                "pelvis_angular_velocity_global": self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            }
            
            # Only add sensor data if it exists (avoid empty dict creation)
            if imu_data:
                self.extras.update(imu_data)
            if left_contact_data:
                self.extras.update(left_contact_data)
            if right_contact_data:
                self.extras.update(right_contact_data)
        else:
            # During training, minimal extras to save memory
            self.extras = {
                "amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size),
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

        # Clear any accumulated gradients or cached tensors
        if hasattr(self, '_step_count') and self._step_count % 5000 == 0:
            torch.cuda.empty_cache()  # Occasional cleanup

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
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        
        # compute AMP observation
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )

        # Ensure tensor is detached to prevent gradient accumulation
        amp_observation = amp_observation.detach()

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

    def set_eval_mode(self, eval_mode: bool = True):
        """Set the environment to evaluation mode to enable sensor data collection.
        
        Args:
            eval_mode: If True, enables sensor data collection. If False, disables it for training.
        """
        self._is_eval_mode = eval_mode
        if eval_mode:
            print("Environment set to EVALUATION mode - sensor data will be collected")
        else:
            print("Environment set to TRAINING mode - sensor data collection disabled")
    
    def set_train_mode(self):
        """Set the environment to training mode (disables sensor data collection)."""
        self.set_eval_mode(False)
    
    def _auto_detect_eval_mode(self) -> bool:
        """Automatically detect if we're in evaluation mode based on various indicators."""
        
        # Method 1: Check if we're being called from an evaluation script
        import inspect
        frame = inspect.currentframe()
        try:
            while frame:
                filename = frame.f_code.co_filename
                if 'eval' in filename.lower() or 'test' in filename.lower():
                    return True
                frame = frame.f_back
        finally:
            del frame
        
        # Method 2: Check if environment is in deterministic mode (often used in eval)
        if hasattr(self, 'is_deterministic') and self.is_deterministic:
            return True
            
        # Method 3: Check render mode (evaluation often renders)
        if hasattr(self, 'render_mode') and self.render_mode is not None:
            return True
            
        # Default to training mode
        return False
    
    def _update_eval_mode(self):
        """Update the evaluation mode automatically based on certain conditions."""
        detected_mode = self._auto_detect_eval_mode()
        if detected_mode != getattr(self, '_is_eval_mode', False):
            self.set_eval_mode(detected_mode)
            mode_str = "EVALUATION" if detected_mode else "TRAINING"
            print(f"=== Automatic Mode Switch: {mode_str} ===")


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
