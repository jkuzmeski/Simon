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
    id="Simon-Walk",
    entry_point=f"{__name__}.simon_env:SimonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simon_env_cfg:SimonWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_simon_walk_cfg.yaml",
    },
)


gym.register(
    id="Simon-Half-Run",
    entry_point=f"{__name__}.simon_half_env:SimonHalfEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simon_half_env_cfg:SimonHalfRunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_simon_run_cfg.yaml",
    },
)


