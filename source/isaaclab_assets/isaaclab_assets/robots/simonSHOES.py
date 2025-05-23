# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the 28-DOFs Mujoco Humanoid robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, DeformableObjectCfg  # Added DeformableObjectCfg

##
# Configuration
##

simon_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="D:\\Isaac\\Simon\\models\\simon.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
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
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
        ),
    },
)
"""Configuration for the 28-DOFs Mujoco Humanoid robot."""

##
# Deformable Feet Configurations
##

# Common properties for shoe material and physics
_DEFORMABLE_PROPS_CFG = sim_utils.DeformableBodyPropertiesCfg(
    rest_offset=0.0,
    contact_offset=0.001,
    # deform_stiffness, damping_stiffness, volume_stiffness were moved to _PHYSICS_MATERIAL_CFG
)

_PHYSICS_MATERIAL_CFG = sim_utils.DeformableBodyMaterialCfg(
    poissons_ratio=0.4,
    youngs_modulus=1e5,  # Lower value for softer material, adjust as needed
)

# Assuming foot dimensions: length=0.2, width=0.1, height=0.05
# Shoe dimensions: length=0.22, width=0.12, thickness=0.03
_SHOE_SIZE = (0.22, 0.12, 0.03)
# Position relative to foot center: -(foot_height/2) - (shoe_thickness/2)
# -(0.05/2) - (0.03/2) = -0.025 - 0.015 = -0.04
_SHOE_INIT_POS_Z = -0.04


LEFT_FOOT_SOFT_CFG = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_foot/SoftFootMesh",  # Default, will be replaced in env_cfg
    spawn=sim_utils.MeshCuboidCfg(
        size=_SHOE_SIZE,
        deformable_props=_DEFORMABLE_PROPS_CFG,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),  # Blueish color for left shoe
        physics_material=_PHYSICS_MATERIAL_CFG,
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _SHOE_INIT_POS_Z)),
    debug_vis=True,
)
"""Configuration for the left soft shoe."""

RIGHT_FOOT_SOFT_CFG = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_foot/SoftFootMesh",  # Default, will be replaced in env_cfg
    spawn=sim_utils.MeshCuboidCfg(
        size=_SHOE_SIZE,
        deformable_props=_DEFORMABLE_PROPS_CFG,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),  # Reddish color for right shoe
        physics_material=_PHYSICS_MATERIAL_CFG,
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _SHOE_INIT_POS_Z)),
    debug_vis=True,
)
"""Configuration for the right soft shoe."""
