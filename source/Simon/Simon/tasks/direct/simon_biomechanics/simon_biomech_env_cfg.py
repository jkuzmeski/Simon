# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets.robots.simon_IMU import simon_IMU  # Changed import

from isaaclab.actuators import ImplicitActuatorCfg
# from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class SimonBiomechEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 10
    decimation = 1
    
    # sensor configuration
    enable_imu_sensor = True
    enable_contact_sensors = True
    save_biomechanics_data = False

    # spaces
    observation_space = 50  # 14 DOF pos + 14 DOF vel + 1 root height + 6 tangent/normal + 3 lin vel + 3 ang vel + 9 key body pos = 50
    action_space = 14  # 14 DOF actions (matching your DOF count)
    state_space = 0
    num_amp_observations = 8
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
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=3072, env_spacing=2.5, replicate_physics=True)  # Increased num_envs from 2048 to 3072

    # robot
    robot: ArticulationCfg = simon_IMU.replace(prim_path="/World/envs/env_.*/Robot").replace(  # Changed to simon_half_CFG
        spawn=simon_IMU.spawn.replace(activate_contact_sensors=True),  # Add this line
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
            ),
        },
    )


@configclass
class SimonBiomechRunEnvCfg(SimonBiomechEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_half_run.npz")


@configclass
class SimonBiomechWalkEnvCfg(SimonBiomechEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")


@configclass
class SimonBiomechTrainEnvCfg(SimonBiomechEnvCfg):
    """Training environment - optimized for speed"""
    # More environments for training
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=3072, env_spacing=5.0, replicate_physics=True)
    
    # Disable expensive sensors during training
    enable_imu_sensor = False
    enable_contact_sensors = False
    save_biomechanics_data = False
    
    # Shorter episodes for training efficiency
    episode_length_s = 10.0


@configclass
class SimonBiomechEvalEnvCfg(SimonBiomechEnvCfg):
    """Evaluation environment - full sensor suite for biomechanics analysis"""
    # Fewer environments for detailed analysis
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=True)
    
    # Enable all sensors for evaluation
    enable_imu_sensor = True
    enable_contact_sensors = True
    save_biomechanics_data = True
    
    # Longer episodes for evaluation
    episode_length_s = 20.0


@configclass
class SimonBiomechTrainRunEnvCfg(SimonBiomechTrainEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_half_run.npz")


@configclass
class SimonBiomechTrainWalkEnvCfg(SimonBiomechTrainEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")


@configclass
class SimonBiomechEvalRunEnvCfg(SimonBiomechEvalEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_half_run.npz")


@configclass
class SimonBiomechEvalWalkEnvCfg(SimonBiomechEvalEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
