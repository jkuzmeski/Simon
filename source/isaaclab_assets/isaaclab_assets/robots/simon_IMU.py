# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the 28-DOFs Mujoco Humanoid robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg


##
# Configuration
##

simon_IMU = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="D:\\Isaac\\Simon\\models\\humanoid_28\\simon_half.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,

        ),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "body": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                # Hip joints - Primary movement planes stronger
                "right_hip_x": 1200.0, "right_hip_y": 600.0, "right_hip_z": 400.0,  # Flex/Abd/Rot
                "left_hip_x": 1200.0, "left_hip_y": 600.0, "left_hip_z": 400.0,
                
                # Knee joints - High stiffness for weight bearing
                "right_knee": 1800.0, "left_knee": 1800.0,  # Slightly reduced but still strong
                
                # Ankle joints - Lower stiffness, foot is more compliant
                "right_ankle_x": 800.0, "right_ankle_y": 400.0, "right_ankle_z": 300.0,  # Dorsi/Inv/Rot
                "left_ankle_x": 800.0, "left_ankle_y": 400.0, "left_ankle_z": 300.0,
            },
            damping={
                # Hip damping - Proportional to stiffness (typically 10-15% of stiffness)
                "right_hip_x": 120.0, "right_hip_y": 60.0, "right_hip_z": 40.0,
                "left_hip_x": 120.0, "left_hip_y": 60.0, "left_hip_z": 40.0,
                
                # Knee damping - Higher for stability
                "right_knee": 180.0, "left_knee": 180.0,
                
                # Ankle damping - Lower for foot compliance
                "right_ankle_x": 80.0, "right_ankle_y": 40.0, "right_ankle_z": 30.0,
                "left_ankle_x": 80.0, "left_ankle_y": 40.0, "left_ankle_z": 30.0,
            },
            effort_limit={
                # These are good - based on human muscle strength data
                "right_hip_x": 150.0, "right_hip_y": 80.0, "right_hip_z": 50.0,
                "left_hip_x": 150.0, "left_hip_y": 80.0, "left_hip_z": 50.0,
                "right_knee": 200.0, "left_knee": 200.0,
                "right_ankle_x": 100.0, "right_ankle_y": 80.0, "right_ankle_z": 50.0,
                "left_ankle_x": 100.0, "left_ankle_y": 80.0, "left_ankle_z": 50.0,
            },
        ),
    },
)
"""Configuration for the 14-DOFs Mujoco Humanoid robot."""

