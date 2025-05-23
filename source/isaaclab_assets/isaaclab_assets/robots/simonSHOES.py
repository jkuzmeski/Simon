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
# Soft Body Configurations for Feet
##

# Define foot size (X, Y, Z) based on typical humanoid foot proportions or your model's specifics
# Extracted from previous USD: extent = [(-0.0885, -0.045, -0.0275), (0.0885, 0.045, 0.0275)]
# This means size is (0.177, 0.09, 0.055)
FOOT_SIZE = (0.177, 0.09, 0.055)

# Stiffness can be controlled by youngs_modulus. Higher is stiffer.
# Tessellation is default for MeshCuboidCfg. For more control, replace MeshCuboidCfg
# with UsdFileCfg pointing to a custom mesh USD.
COMMON_DEFORMABLE_MATERIAL = sim_utils.DeformableBodyMaterialCfg(
    youngs_modulus=1.0e5,  # Example: 100 kPa. Adjust for desired stiffness.
    poissons_ratio=0.45    # Example value.
)

COMMON_DEFORMABLE_PROPS = sim_utils.DeformableBodyPropertiesCfg(
    rest_offset=0.0,
    contact_offset=0.001,  # From Isaac Lab tutorial
    # enable_self_collisions=False, # Default
    # solver_type="explicit", # Default
)

COMMON_VISUAL_MATERIAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.2, 0.2, 0.9),  # Blueish
    opacity=0.7
)

LEFT_FOOT_SOFT_CFG = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_foot/SoftFootMesh",  # Will be child of the left_foot Xform
    spawn=sim_utils.MeshCuboidCfg(
        size=FOOT_SIZE,
        visual_material=COMMON_VISUAL_MATERIAL,
        physics_material=COMMON_DEFORMABLE_MATERIAL,
        deformable_props=COMMON_DEFORMABLE_PROPS,
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0)  # Local position relative to the parent (left_foot prim)
    ),
    debug_vis=True  # Shows markers for deformable bodies if enabled in viewport
)
"""Configuration for the soft left foot of the Simon robot."""

RIGHT_FOOT_SOFT_CFG = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_foot/SoftFootMesh",  # Will be child of the right_foot Xform
    spawn=sim_utils.MeshCuboidCfg(
        size=FOOT_SIZE,  # Assuming same size for the right foot
        visual_material=COMMON_VISUAL_MATERIAL,
        physics_material=COMMON_DEFORMABLE_MATERIAL,
        deformable_props=COMMON_DEFORMABLE_PROPS,
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0)  # Local position relative to the parent (right_foot prim)
    ),
    debug_vis=True
)
"""Configuration for the soft right foot of the Simon robot."""
