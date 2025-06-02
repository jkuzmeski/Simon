# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn deformable shoes attached to the simon model feet.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p examples/shoe_testing.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="This script demonstrates deformable shoes attached to simon model feet.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, DeformableObject, DeformableObjectCfg
from isaaclab_assets.robots.simon_half import simon_CFG


def design_scene() -> tuple[dict, dict]:
    """Designs the scene with Simon model and deformable shoes."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light.func("/World/light", cfg_light)

    # Configure and spawn Simon robot
    simon_cfg = simon_CFG.replace(prim_path="/World/Robot")
    simon_cfg.func("/World/Robot", simon_cfg)

    # Create Simon articulation
    simon = Articulation(simon_cfg)

    # Configure deformable shoe objects
    cfg_shoe = sim_utils.UsdFileCfg(
        usd_path="D:\\Isaac\\Simon\\models\\Shoe.usda",
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(
            rest_offset=0.001,
            contact_offset=0.005,
            solver_position_iteration_count=16,
            vertex_velocity_damping=0.005,
        ),
    )    # Spawn left shoe
    cfg_shoe.func("/World/LeftShoe", cfg_shoe, translation=(0.0, 0.15, 0.0))

    # Spawn right shoe
    cfg_shoe.func("/World/RightShoe", cfg_shoe, translation=(0.0, -0.15, 0.0))

    # Create deformable shoe objects
    left_shoe_cfg = DeformableObjectCfg(
        prim_path="/World/LeftShoe",
        spawn=None,
        init_state=DeformableObjectCfg.InitialStateCfg(),
    )
    left_shoe = DeformableObject(left_shoe_cfg)

    right_shoe_cfg = DeformableObjectCfg(
        prim_path="/World/RightShoe",
        spawn=None,
        init_state=DeformableObjectCfg.InitialStateCfg(),
    )
    right_shoe = DeformableObject(right_shoe_cfg)    # return the scene entities
    scene_entities = {
        "simon": simon,
        "left_shoe": left_shoe,
        "right_shoe": right_shoe
    }

    foot_targets = {
        "left_foot": "left_foot",
        "right_foot": "right_foot"
    }

    return scene_entities, foot_targets


def run_simulator(sim: sim_utils.SimulationContext, entities: dict, foot_targets: dict):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Get simon articulation
    simon = entities["simon"]
    left_shoe = entities["left_shoe"]
    right_shoe = entities["right_shoe"]

    # Find foot body indices
    try:
        left_foot_idx = simon.body_names.index("left_foot")
        right_foot_idx = simon.body_names.index("right_foot")
        print(f"[INFO]: Found left foot at index {left_foot_idx}, right foot at index {right_foot_idx}")
    except ValueError as e:
        print(f"[ERROR]: Could not find foot bodies: {e}")
        print(f"[INFO]: Available body names: {simon.body_names}")
        return

    # Simulate physics
    while simulation_app.is_running():        # reset
        if count % 1000 == 0:
            # reset counters
            sim_time = 0.0
            count = 0

            # reset simon state
            simon.reset()

            # reset deformable shoe states
            left_shoe_nodal_state = left_shoe.data.default_nodal_state_w.clone()
            right_shoe_nodal_state = right_shoe.data.default_nodal_state_w.clone()

            left_shoe.write_nodal_state_to_sim(left_shoe_nodal_state)
            right_shoe.write_nodal_state_to_sim(right_shoe_nodal_state)            
            left_shoe.reset()
            right_shoe.reset()

            print("[INFO]: Resetting simulation state...")

        # Update entities
        simon.update(sim_dt)
        left_shoe.update(sim_dt)
        right_shoe.update(sim_dt)

        # Get foot positions from simon
        foot_positions = simon.data.body_pos_w

        # Update shoe positions to follow feet (simple attachment)
        if foot_positions.shape[0] > 0:  # Check if we have valid data            # Get left and right foot positions
            left_foot_pos = foot_positions[0, left_foot_idx]
            right_foot_pos = foot_positions[0, right_foot_idx]

            # Update shoe positions (simplified - just move to foot position)
            # In a full implementation, you would update the nodal positions properly
            print(f"Left foot position: {left_foot_pos}")
            print(f"Right foot position: {right_foot_pos}")

        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([4.0, 4.0, 3.0], [0.5, 0.5, 0.0])    # Design scene by adding assets to it
    scene_entities, foot_targets = design_scene()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, scene_entities, foot_targets)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
