# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
from .fault_locomotion_env import FaultLocomotionEnv


# Go2 environments
from .fault_locomotion_env import Go2FlatEnvCfg, Go2RoughVisionEnvCfg, Go2RoughBlindEnvCfg

gym.register(
    id="FaultLocomotion-Go2-Flat",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Flat-EMLP",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatSymmPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Flat-CDAE",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatCDAEPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Flat-RFF",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatRFFKoopmanPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Flat-EMLP-ECDAE",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatSymmECDAEPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Flat-EMLP-ERFF",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatSymmERFFKoopmanPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Rough-Blind",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Rough-Blind-EMLP",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughSymmPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Rough-Blind-CDAE",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughCDAEPPORunnerCfg",
    },
)


gym.register(
    id="FaultLocomotion-Go2-Rough-Blind-RFF",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughRFFKoopmanPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Rough-Blind-EMLP-ECDAE",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughSymmECDAEPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Rough-Blind-EMLP-ERFF",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughVisionSymmERFFKoopmanPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Rough-Vision",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Rough-Vision-EMLP",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughVisionSymmPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Rough-Vision-CDAE",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughCDAEPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Rough-Vision-RFF",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughRFFKoopmanPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Rough-Vision-EMLP-ECDAE",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughVisionSymmECDAEPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Rough-Vision-EMLP-ERFF",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughVisionSymmERFFKoopmanPPORunnerCfg",
    },
)


# Pegasus environments
from .fault_locomotion_env import PegasusFlatEnvCfg, PegasusRoughVisionEnvCfg, PegasusRoughBlindEnvCfg

gym.register(
    id="FaultLocomotion-Pegasus-Flat",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegasusFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Pegasus-Rough-Blind",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegasusRoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Pegasus-Rough-Vision",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegasusRoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)
