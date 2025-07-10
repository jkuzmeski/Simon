# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets.robots.simon_IMU import simon_IMU  # Changed import

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")
# Path to torque motions collected from trained agents
TORQUE_MOTIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "Movement", "torque_motions")


@configclass
class SimonTorqueEnvCfg(DirectRLEnvCfg):
    """Simon Torque-based AMP environment config (base class)."""

    # env
    episode_length_s = 10.0
    decimation = 2
    
    # sensor configuration - reduced for torque-focused training
    enable_imu_sensor = False  # Not needed for torque AMP
    enable_contact_sensors = False  # Not needed for torque AMP
    save_biomechanics_data = False

    # torque-specific configuration
    enable_torque_amp = True
    torque_motion_files: list[str] = []
    torque_weight = 1.0
    include_torque_in_obs = False  # Whether to include torques in observations
    include_torque_in_amp = True   # Whether to include torques in AMP discriminator

    # spaces
    observation_space = 64  # 14 DOF pos + 14 DOF vel + 1 root height + 6 tangent/normal + 3 lin vel + 3 ang vel + 9 key body pos + 14 torques = 64
    action_space = 14  # 14 DOF actions (matching your DOF count)
    state_space = 0
    num_amp_observations = 4
    # AMP observation space - matches policy observation space for torque-based training
    amp_observation_space = 64  # Standard 50 + 14 torques = 64

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING  # This will be a torque motion file
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=3072, env_spacing=5.0, replicate_physics=True)  # Increased num_envs from 2048 to 3072

    # robot
    robot: ArticulationCfg = simon_IMU.replace(prim_path="/World/envs/env_.*/Robot").replace(  # Changed to simon_half_CFG
        spawn=simon_IMU.spawn.replace(activate_contact_sensors=True),  # Add this line
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit_sim=100.0,  # Changed from velocity_limit
                stiffness=None,
                damping=None,
            ),
        },
    )


@configclass
class SimonTorqueRunEnvCfg(SimonTorqueEnvCfg):
    """Torque-based AMP training for running motions"""
    # Use torque motion file when available, fallback to regular motion file
    motion_file = os.path.join(TORQUE_MOTIONS_DIR, "torque_motion_run.npz") if os.path.exists(os.path.join(TORQUE_MOTIONS_DIR, "torque_motion_run.npz")) else os.path.join(MOTIONS_DIR, "humanoid_half_run.npz")


@configclass
class SimonTorqueWalkEnvCfg(SimonTorqueEnvCfg):
    """Torque-based AMP training for walking motions"""
    # Use torque motion file when available, fallback to regular motion file
    motion_file = os.path.join(TORQUE_MOTIONS_DIR, "torque_motion_walk.npz") if os.path.exists(os.path.join(TORQUE_MOTIONS_DIR, "torque_motion_walk.npz")) else os.path.join(MOTIONS_DIR, "humanoid_walk.npz")


@configclass
class SimonTorqueTrainEnvCfg(SimonTorqueEnvCfg):
    """Training environment for torque-based AMP - optimized for speed"""
    # More environments for training
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)
    
    # Optimized for training
    episode_length_s = 15.0
    decimation = 2
    
    # Torque-specific training settings
    include_torque_in_obs = False  # Don't add torques to observations for efficiency
    include_torque_in_amp = True   # Include torques in AMP discriminator


@configclass
class SimonTorqueEvalEnvCfg(SimonTorqueEnvCfg):
    """Evaluation environment for torque-based AMP - detailed analysis"""
    # Fewer environments for detailed analysis
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=5.0, replicate_physics=True)
    
    # Longer episodes for evaluation
    episode_length_s = 30.0
    decimation = 1
    
    # Enable additional features for evaluation
    include_torque_in_obs = True   # Include torques in observations for analysis
    include_torque_in_amp = True   # Include torques in AMP discriminator


@configclass
class SimonTorqueTrainRunEnvCfg(SimonTorqueTrainEnvCfg):
    """Training environment for torque-based running"""
    # Use torque motion file when available, fallback to regular motion file
    motion_file = os.path.join(TORQUE_MOTIONS_DIR, "torque_motion_run.npz") if os.path.exists(os.path.join(TORQUE_MOTIONS_DIR, "torque_motion_run.npz")) else os.path.join(MOTIONS_DIR, "humanoid_half_run.npz")


@configclass
class SimonTorqueTrainWalkEnvCfg(SimonTorqueTrainEnvCfg):
    """Training environment for torque-based walking"""
    # Use torque motion file when available, fallback to regular motion file
    motion_file = os.path.join(TORQUE_MOTIONS_DIR, "torque_motion_walk.npz") if os.path.exists(os.path.join(TORQUE_MOTIONS_DIR, "torque_motion_walk.npz")) else os.path.join(MOTIONS_DIR, "humanoid_walk.npz")


@configclass
class SimonTorqueEvalRunEnvCfg(SimonTorqueEvalEnvCfg):
    """Evaluation environment for torque-based running"""
    # Use torque motion file when available, fallback to regular motion file
    motion_file = os.path.join(TORQUE_MOTIONS_DIR, "torque_motion_run.npz") if os.path.exists(os.path.join(TORQUE_MOTIONS_DIR, "torque_motion_run.npz")) else os.path.join(MOTIONS_DIR, "humanoid_half_run.npz")


@configclass
class SimonTorqueEvalWalkEnvCfg(SimonTorqueEvalEnvCfg):
    """Evaluation environment for torque-based walking"""
    # Use torque motion file when available, fallback to regular motion file
    motion_file = os.path.join(TORQUE_MOTIONS_DIR, "torque_motion_walk.npz") if os.path.exists(os.path.join(TORQUE_MOTIONS_DIR, "torque_motion_walk.npz")) else os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
