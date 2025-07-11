# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Original environments (full sensors enabled by default)
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



gym.register(
    id="Simon-Train-Walk-Biomech",
    entry_point=f"{__name__}.simon_biomech_env:SimonBiomechEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simon_biomech_env_cfg:SimonBiomechTrainWalkEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_simon_walk_cfg.yaml",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_simon_walk_cfg.yaml",
    },
)

# Training environments (sensors disabled for performance)
gym.register(
    id="Simon-Train-Run-Biomech",
    entry_point=f"{__name__}.simon_biomech_env:SimonBiomechEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simon_biomech_env_cfg:SimonBiomechTrainRunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_simon_run_biomech_cfg.yaml",
    },
)

# Evaluation environments (full sensors enabled for biomechanics analysis)
gym.register(
    id="Simon-Eval-Run-Biomech",
    entry_point=f"{__name__}.simon_biomech_env:SimonBiomechEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simon_biomech_env_cfg:SimonBiomechEvalRunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_simon_run_biomech_cfg.yaml",
    },
)

# Training environments (sensors disabled for performance)
gym.register(
    id="Simon-Train-Run-Biomech-Large",
    entry_point=f"{__name__}.simon_biomech_env:SimonBiomechEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simon_biomech_env_cfg:SimonBiomechTrainRunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_simon_run_biomech_large_cfg.yaml",
    },
)

# Evaluation environments (full sensors enabled for biomechanics analysis)
gym.register(
    id="Simon-Eval-Run-Biomech-Large",
    entry_point=f"{__name__}.simon_biomech_env:SimonBiomechEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simon_biomech_env_cfg:SimonBiomechEvalRunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_simon_run_biomech_large_cfg.yaml",
    },
)

gym.register(
    id="Simon-Eval-Walk-Biomech",
    entry_point=f"{__name__}.simon_biomech_env:SimonBiomechEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simon_biomech_env_cfg:SimonBiomechEvalWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_simon_walk_cfg.yaml",
    },
)