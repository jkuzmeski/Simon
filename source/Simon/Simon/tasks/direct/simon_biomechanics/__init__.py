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
    id="Simon-Half-Run-Biomech",
    entry_point=f"{__name__}.simon_biomech_env:SimonBiomechEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simon_biomech_env_cfg:SimonBiomechRunEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_simon_run_biomech_cfg.yaml",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_simon_run_biomech_cfg.yaml",
    },
)

gym.register(
    id="Simon-Half-Walk-Biomech",
    entry_point=f"{__name__}.simon_biomech_env:SimonBiomechEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simon_biomech_env_cfg:SimonBiomechWalkEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_simon_walk_cfg.yaml",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_simon_walk_cfg.yaml",
    },
)