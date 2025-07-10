# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import torch
from typing import Optional
from .motion_loader import MotionLoader


class TorqueMotionLoader(MotionLoader):
    """
    Enhanced motion loader that supports joint torque data for torque-based imitation learning.
    Extends the standard MotionLoader to include joint torques alongside traditional motion data.
    """

    def __init__(self, motion_file: str, device: torch.device) -> None:
        """Load a motion file with torque data and initialize the internal variables.

        Args:
            motion_file: Motion file path to load (NPZ format with torque data).
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist or doesn't contain torque data.
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)

        # Check if this is a torque motion file
        if "joint_torques" not in data:
            raise ValueError(f"Motion file {motion_file} does not contain joint torque data. "
                             "Please use MotionLoader for standard motion files.")

        self.device = device
        self._dof_names = data["dof_names"].tolist()
        self._body_names = data["body_names"].tolist()

        # Load standard motion data
        self.dof_positions = torch.tensor(data["dof_positions"], dtype=torch.float32, device=self.device)
        self.dof_velocities = torch.tensor(data["dof_velocities"], dtype=torch.float32, device=self.device)
        self.body_positions = torch.tensor(data["body_positions"], dtype=torch.float32, device=self.device)
        self.body_rotations = torch.tensor(data["body_rotations"], dtype=torch.float32, device=self.device)
        self.body_linear_velocities = torch.tensor(
            data["body_linear_velocities"], dtype=torch.float32, device=self.device
        )
        self.body_angular_velocities = torch.tensor(
            data["body_angular_velocities"], dtype=torch.float32, device=self.device
        )

        # Load torque data
        self.joint_torques = torch.tensor(data["joint_torques"], dtype=torch.float32, device=self.device)

        self.dt = 1.0 / data["fps"]
        self.num_frames = self.dof_positions.shape[0]
        self.duration = self.dt * (self.num_frames - 1)
        print(f"Torque motion loaded ({motion_file}): duration: {self.duration} sec, frames: {self.num_frames}")
        print(f"Joint torques shape: {self.joint_torques.shape}")

    def sample_with_torques(
        self, num_samples: int, times: Optional[np.ndarray] = None, duration: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data including joint torques.

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

        Returns:
            Sampled motion DOF positions (with shape (N, num_dofs)), DOF velocities (with shape (N, num_dofs)),
            body positions (with shape (N, num_bodies, 3)), body rotations (with shape (N, num_bodies, 4), as wxyz quaternion),
            body linear velocities (with shape (N, num_bodies, 3)), body angular velocities (with shape (N, num_bodies, 3)),
            and joint torques (with shape (N, num_dofs)).
        """
        times = self.sample_times(num_samples, duration) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        return (
            self._interpolate(self.dof_positions, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_positions, blend=blend, start=index_0, end=index_1),
            self._slerp(self.body_rotations, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_linear_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_angular_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.joint_torques, blend=blend, start=index_0, end=index_1),  # Joint torques
        )

    def sample(
        self, num_samples: int, times: Optional[np.ndarray] = None, duration: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data (standard interface without torques for compatibility).

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

        Returns:
            Sampled motion DOF positions (with shape (N, num_dofs)), DOF velocities (with shape (N, num_dofs)),
            body positions (with shape (N, num_bodies, 3)), body rotations (with shape (N, num_bodies, 4), as wxyz quaternion),
            body linear velocities (with shape (N, num_bodies, 3)) and body angular velocities (with shape (N, num_bodies, 3)).
        """
        # Just call the parent class method for standard sampling
        return super().sample(num_samples, times, duration)

    def get_torques_at_time(self, time: float) -> torch.Tensor:
        """Get joint torques at a specific time.

        Args:
            time: Time to sample torques at.

        Returns:
            Joint torques at the specified time with shape (num_dofs,).
        """
        times = np.array([time])
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)
        
        torques = self._interpolate(self.joint_torques, blend=blend, start=index_0, end=index_1)
        return torques[0]  # Return single frame

    def get_motion_and_torques_at_time(self, time: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get full motion state and torques at a specific time.

        Args:
            time: Time to sample motion and torques at.

        Returns:
            Motion DOF positions, DOF velocities, body positions, body rotations,
            body linear velocities, body angular velocities, and joint torques at the specified time.
            All tensors have first dimension removed (single frame).
        """
        dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel, joint_torques = self.sample_with_torques(1, times=np.array([time]))
        return dof_pos[0], dof_vel[0], body_pos[0], body_rot[0], body_lin_vel[0], body_ang_vel[0], joint_torques[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Torque motion file")
    args, _ = parser.parse_known_args()

    torque_motion = TorqueMotionLoader(args.file, "cpu")

    print("- number of frames:", torque_motion.num_frames)
    print("- number of DOFs:", torque_motion.num_dofs)
    print("- number of bodies:", torque_motion.num_bodies)
    print("- duration:", torque_motion.duration, "seconds")
    print("- joint torques shape:", torque_motion.joint_torques.shape)

    # Test sampling with torques
    dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel, joint_torques = torque_motion.sample_with_torques(5)
    print("\nSample data shapes:")
    print("- DOF positions:", dof_pos.shape)
    print("- Joint torques:", joint_torques.shape)
