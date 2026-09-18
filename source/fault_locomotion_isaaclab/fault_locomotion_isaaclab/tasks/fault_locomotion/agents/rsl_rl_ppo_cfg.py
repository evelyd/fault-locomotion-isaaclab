# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from pathlib import Path
from dataclasses import MISSING

from copy import deepcopy
from . import moe_cfg
from . import morphosymm_cfg
from . import koopman_model_cfg


@configclass
class FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2500
    save_interval = 1000
    experiment_name = "flat_direct"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}  # rsl_rl 5.4 port
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive", #fixed, adaptive
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # Mixture of Expert Stuff
    moe_cfg = moe_cfg.moe_cfg

    # Morphosymm-rl Related Stuff
    morphologycal_symmetries_cfg = morphosymm_cfg.morphologycal_symmetries_cfg



@configclass
class RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 1000
    experiment_name = "rough_direct"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}  # rsl_rl 5.4 port
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        #distribution_cfg=RslRlMLPModelCfg.BetaDistribution(action_range=[-3.0, 3.0])
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # Mixture of Expert Stuff
    moe_cfg = moe_cfg.moe_cfg

    # Morphosymm-rl Related Stuff
    morphologycal_symmetries_cfg = morphosymm_cfg.morphologycal_symmetries_cfg

@configclass
class RoughPPORFFRunnerCfg(RoughPPORunnerCfg):
    class_name = "DAEOnPolicyRunner"
    experiment_name = "rough_vision_rff_direct"
    algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPORFF",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    koopman_cfg = koopman_model_cfg.rff_koopman_cfg

@configclass
class RoughPPORFFKoopmanRunnerCfg(RoughPPORFFRunnerCfg):
    experiment_name = "rough_vision_rff_koopman_direct"

@configclass
class RoughPPOSymmRunnerCfg(RoughPPORunnerCfg):
    class_name = "SymmOnPolicyRunner"
    experiment_name = "rough_vision_symm_direct"
    actor = RslRlMLPModelCfg(
        class_name="SymmModel",
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        #distribution_cfg=RslRlMLPModelCfg.BetaDistribution(action_range=[-3.0, 3.0])
    )
    critic = RslRlMLPModelCfg(
        class_name="SymmModel",
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg( # class name assigned by runner
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    morphologycal_symmetries_cfg = morphosymm_cfg.vision_morphologycal_symmetries_cfg

@configclass
class RoughPPOSymmDAERunnerCfg(RoughPPOSymmRunnerCfg):
    class_name = "SymmDAEOnPolicyRunner"
    experiment_name = "rough_vision_symm_ecdae_direct"
    algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPOSymmDAEOnline",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    morphologycal_symmetries_cfg = morphosymm_cfg.vision_dae_morphologycal_symmetries_cfg
    koopman_cfg = koopman_model_cfg.ecdae_koopman_cfg

@configclass
class RoughPPOSymmRFFRunnerCfg(RoughPPOSymmDAERunnerCfg):
    experiment_name = "rough_vision_symm_erff_direct"
    algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPOSymmERFF",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    koopman_cfg = koopman_model_cfg.erff_koopman_cfg

@configclass
class RoughPPOSymmRFFKoopmanRunnerCfg(RoughPPOSymmRFFRunnerCfg):
    experiment_name = "rough_vision_symm_erff_koopman_direct"