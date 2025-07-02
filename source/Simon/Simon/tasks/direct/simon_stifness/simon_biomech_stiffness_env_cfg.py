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


@configclass
class SimonBiomechStiffnessEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 10000
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=3072, env_spacing=5.0, replicate_physics=True)  # Increased num_envs from 2048 to 3072

    # robot
    robot: ArticulationCfg = simon_IMU.replace(prim_path="/World/envs/env_.*/Robot").replace(  # Changed to simon_half_CFG
        spawn=simon_IMU.spawn.replace(activate_contact_sensors=True),  # Add this line
        actuators={
            # Hip joints - based on human muscle groups
            "hip_x": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_x"],
                velocity_limit_sim=100.0,
                stiffness=120.0,  # Hip abduction/adduction (glute medius)
                damping=12.0,
            ),
            "hip_y": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_y"],
                velocity_limit_sim=100.0,
                stiffness=180.0,  # Hip flexion/extension (strongest - glutes/hip flexors)
                damping=18.0,
            ),
            "hip_z": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_z"],
                velocity_limit_sim=100.0,
                stiffness=100.0,  # Hip rotation (deep rotators)
                damping=10.0,
            ),
            # Knee joints - quadriceps/hamstrings
            "knee": ImplicitActuatorCfg(
                joint_names_expr=[".*knee"],
                velocity_limit_sim=100.0,
                stiffness=150.0,  # Knee flexion/extension
                damping=15.0,
            ),
            # Ankle joints - lower stiffness for natural compliance
            "ankle_x": ImplicitActuatorCfg(
                joint_names_expr=[".*ankle_x"],
                velocity_limit_sim=100.0,
                stiffness=60.0,  # Ankle dorsi/plantarflexion (calf muscles)
                damping=6.0,
            ),
            "ankle_y": ImplicitActuatorCfg(
                joint_names_expr=[".*ankle_y"],
                velocity_limit_sim=100.0,
                stiffness=40.0,  # Ankle inversion/eversion (peroneals)
                damping=4.0,
            ),
            "ankle_z": ImplicitActuatorCfg(
                joint_names_expr=[".*ankle_z"],
                velocity_limit_sim=100.0,
                stiffness=30.0,  # Ankle rotation (minimal natural stiffness)
                damping=3.0,
            ),
        },
    )


@configclass
class SimonBiomechStiffnessRunEnvCfg(SimonBiomechStiffnessEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_half_run.npz")


@configclass
class SimonBiomechStiffnessWalkEnvCfg(SimonBiomechStiffnessEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
