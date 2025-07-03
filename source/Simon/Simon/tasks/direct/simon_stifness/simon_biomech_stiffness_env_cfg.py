# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets.robots.simon_IMU import simon_IMU  # Changed import

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class SimonBiomechStiffnessEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 30
    decimation = 1

    # spaces
    observation_space = 50  # 14 DOF pos + 14 DOF vel + 1 root height + 6 tangent/normal + 3 lin vel + 3 ang vel + 9 key body pos = 50
    action_space = 14  # 14 DOF actions (matching your DOF count)
    state_space = 0
    num_amp_observations = 4
    amp_observation_space = 50  # Should match observation_space for AMP

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "pelvis"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 480,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)  # Reduced from 2048 to 1024 to save memory

    # robot
    robot: ArticulationCfg = simon_IMU.replace(prim_path="/World/envs/env_.*/Robot").replace(
        actuators={
            "body": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                stiffness={
                    # Hip joints - Primary movement planes stronger
                    "right_hip_x": 60.0, "right_hip_y": 120.0, "right_hip_z": 40.0,  # Abd/Flex/Rot
                    "left_hip_x": 60.0, "left_hip_y": 120.0, "left_hip_z": 40.0,
                    
                    # Knee joints - High stiffness for weight bearing
                    "right_knee": 180.0, "left_knee": 180.0,  # Slightly reduced but still strong
                    
                    # Ankle joints - Lower stiffness, foot is more compliant
                    "right_ankle_x": 40.0, "right_ankle_y": 80.0, "right_ankle_z": 30.0,  # Inv/Dorsi/Rot
                    "left_ankle_x": 40.0, "left_ankle_y": 80.0, "left_ankle_z": 30.0,
                },
                damping={
                    # Hip damping - Proportional to stiffness (typically 10-15% of stiffness)
                    "right_hip_x": 6.0, "right_hip_y": 12.0, "right_hip_z": 4.0,
                    "left_hip_x": 6.0, "left_hip_y": 12.0, "left_hip_z": 4.0,
                    
                    # Knee damping - Higher for stability
                    "right_knee": 18.0, "left_knee": 18.0,
                    
                    # Ankle damping - Lower for foot compliance
                    "right_ankle_x": 4.0, "right_ankle_y": 8.0, "right_ankle_z": 3.0,
                    "left_ankle_x": 4.0, "left_ankle_y": 8.0, "left_ankle_z": 3.0,
                },
                effort_limit={
                    # These are good - based on human muscle strength data
                    "right_hip_x": 80.0, "right_hip_y": 150.0, "right_hip_z": 50.0,
                    "left_hip_x": 80.0, "left_hip_y": 150.0, "left_hip_z": 50.0,
                    "right_knee": 200.0, "left_knee": 200.0,
                    "right_ankle_x": 80.0, "right_ankle_y": 100.0, "right_ankle_z": 50.0,
                    "left_ankle_x": 80.0, "left_ankle_y": 100.0, "left_ankle_z": 50.0,
                },
            ),
        },
    )
    
    
@configclass
class SimonBiomechStiffnessRunEnvCfg(SimonBiomechStiffnessEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_half_run.npz")


@configclass
class SimonBiomechStiffnessWalkEnvCfg(SimonBiomechStiffnessEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
