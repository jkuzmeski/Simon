# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Simon-Half-Run-Biomech-Stiffness",
    entry_point=f"{__name__}.simon_biomech_stiffness_env:SimonBiomechStiffnessEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simon_biomech_stiffness_env_cfg:SimonBiomechStiffnessRunEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:simon_run_biomech_stiffness_cfg.yaml",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:simon_run_biomech_stiffness_cfg.yaml",
    },
)

gym.register(
    id="Simon-Half-Run-Biomech-Stiffness-Large",
    entry_point=f"{__name__}.simon_biomech_stiffness_env:SimonBiomechStiffnessEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simon_biomech_stiffness_env_cfg:SimonBiomechStiffnessRunEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:simon_run_biomech_stiffness_cfg_large.yaml",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:simon_run_biomech_stiffness_cfg_large.yaml",
    },
)


gym.register(
    id="Simon-Half-Walk-Biomech-Stiffness",
    entry_point=f"{__name__}.simon_biomech_stiffness_env:SimonBiomechStiffnessEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simon_biomech_stiffness_env_cfg:SimonBiomechStiffnessWalkEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:simon_walk_stiffness_cfg.yaml",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:simon_walk_stiffness_cfg.yaml",
    },
)