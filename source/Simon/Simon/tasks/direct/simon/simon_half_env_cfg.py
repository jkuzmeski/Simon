# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets.robots.simon_half import simon_half_CFG  # Changed import

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class SimonHalfEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 10.0
    decimation = 2

    # spaces
    observation_space = 55  # 14 DOF pos + 14 DOF vel + 13 root state + 14 prev actions
    action_space = 14  # 14 DOF actions (matching your DOF count)
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 55  # Should match observation_space for AMP

    early_termination = True
    termination_height = 0.4

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=True)  # Increased num_envs from 2048 to 3072

    # robot
    robot: ArticulationCfg = simon_half_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(  # Changed to simon_half_CFG
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                stiffness=None,
                damping=None,
            ),
        },
    )


@configclass
class SimonHalfRunEnvCfg(SimonHalfEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_half_run.npz")


@configclass
class SimonWalkEnvCfg(SimonHalfEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
