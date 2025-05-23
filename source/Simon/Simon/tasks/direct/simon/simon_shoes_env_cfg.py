# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets.robots.simonSHOES import simon_CFG, LEFT_FOOT_SOFT_CFG, RIGHT_FOOT_SOFT_CFG  # Changed import

from isaaclab.actuators import ImplicitActuatorCfg
#from isaaclab.assets import ArticulationCfg, DeformableObjectCfg  # Removed AssetBaseCfg as it's not used
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

# Define helper configs at the module level for clarity
_simon_robot_cfg_modified = simon_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            stiffness=None,
            damping=None,
        ),
    },
)

_left_shoe_cfg_modified = LEFT_FOOT_SOFT_CFG.replace(prim_path="/World/envs/env_.*/Robot/left_foot/SoftFootMesh")
_right_shoe_cfg_modified = RIGHT_FOOT_SOFT_CFG.replace(prim_path="/World/envs/env_.*/Robot/right_foot/SoftFootMesh")


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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,  # Default, overridden by CLI if --num_envs is used
        env_spacing=5.0,
        replicate_physics=False,  # As in the original config
        # Add robot config directly to the scene config
        robot=_simon_robot_cfg_modified,
    )

    # The robot ArticulationCfg is now part of `scene.robot`.
    # These specific DeformableObjectCfg are for reference in __post_init__
    # to populate scene.deformable_objects. They are not directly used by DirectRLEnv framework as top-level asset cfgs.

    def __post_init__(self):
        """Post-initialization to add soft bodies to the scene's config."""
        super().__post_init__()  # Call parent's post-init

        # Add deformable bodies to the InteractiveSceneCfg instance (self.scene.deformable_objects)
        # This ensures they are part of the scene config that InteractiveScene processes.
        if not hasattr(self.scene, "deformable_objects"):
            self.scene.deformable_objects = {}
        self.scene.deformable_objects["left_shoe"] = _left_shoe_cfg_modified
        self.scene.deformable_objects["right_shoe"] = _right_shoe_cfg_modified


@configclass
class SimonShoesRunEnvCfg(SimonShoesEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_run.npz")


@configclass
class SimonShoesWalkEnvCfg(SimonShoesEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
