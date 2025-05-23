# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets.robots.simonSHOES import simon_CFG, LEFT_FOOT_SOFT_CFG, RIGHT_FOOT_SOFT_CFG  # Changed import

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, DeformableObjectCfg  # Added DeformableObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class SimonShoesEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 10.0
    decimation = 2

    # spaces
    observation_space = 81
    action_space = 28
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 81

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "torso"
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=False)  # Increased num_envs from 2048 to 3072, set replicate_physics to False

    # robot
    robot: ArticulationCfg = simon_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(  # Changed to simon_CFG
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                stiffness=None,
                damping=None,
            ),
        },
    )

    # soft bodies for feet
    left_foot_soft: DeformableObjectCfg = LEFT_FOOT_SOFT_CFG.replace(prim_path="/World/envs/env_.*/Robot/left_foot/SoftFootMesh")
    right_foot_soft: DeformableObjectCfg = RIGHT_FOOT_SOFT_CFG.replace(prim_path="/World/envs/env_.*/Robot/right_foot/SoftFootMesh")

    def __post_init__(self):
        """Post-initialization to add soft bodies to the scene."""
        super().__post_init__()  # Call parent's post-init

        # Add deformable bodies to the scene configuration
        # scene_objects is a dict and should be initialized by InteractiveSceneCfg
        if self.scene.scene_objects is None:  # Defensive check
            self.scene.scene_objects = {}
        self.scene.scene_objects["left_shoe"] = self.left_foot_soft
        self.scene.scene_objects["right_shoe"] = self.right_foot_soft


@configclass
class SimonShoesRunEnvCfg(SimonShoesEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_run.npz")


@configclass
class SimonShoesWalkEnvCfg(SimonShoesEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
