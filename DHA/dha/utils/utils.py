# This version is modified for your case:
# - 35-dimensional observation
# - DAE-aug model (no action used)
# - Not using IMU task

import torch
import os
import re
import math
import numpy as np
import dha
from dha.utils.mysc import class_from_name
from dha.nn.DynamicsAutoEncoder import DAE
from dha.nn.EquivDynamicsAutoencoder import EquivDAE
from dha.nn.ControlledDynamicsAutoEncoder import ControlledDAE
from dha.nn.ControlledEquivDynamicsAutoencoder import ControlledEquivDAE
from morpho_symm.utils.robot_utils import load_symmetric_system
from hydra import initialize, compose
import escnn
from escnn.nn import FieldType
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens
from typing import Union

def extract_trained_model_info(state_dict, model_dir) -> (int, int, bool, int):
    """Extracts model information from a state_dict."""
    layers = 0
    hidden_units = 0
    obs_state_dim = 0
    has_bias = False

    for key in state_dict.keys():
        if ".obs_fn.net" in key:
            if "model.obs_fn.net.block_" in key and "weight" in key:
                layers += 1
            if "E-DAE" in model_dir or "EC-DAE" in model_dir:
                if "model.obs_fn.net.block_0.linear_0" in key and "matrix" in key:
                    state_dim = state_dict[key].shape[1]
            else:
                if "model.obs_fn.net.block_0" in key and "weight" in key:
                    state_dim = state_dict[key].shape[1]
            if "linear_0" in key and ("weight" in key or "matrix" in key):
                hidden_units = state_dict[key].shape[0]
            if 'bias' in key and not has_bias:
                has_bias = True
            if "head" in key and ("weight" in key or "matrix" in key):
                obs_state_dim = state_dict[key].shape[0]

    layers += 1  # Add one for the head layer

    return layers, hidden_units, has_bias, obs_state_dim, state_dim

def remove_state_dict_prefix(state_dict, prefix):
    return {key[len(prefix):] if key.startswith(prefix) else key: value for key, value in state_dict.items()}

def load_normalization_stats(model_path: str, device: torch.device):
    """
    从给定 model_path 下加载 state_mean_var.npy 文件，返回 PyTorch 格式的 mean 和 std。
    """
    norm_path = os.path.join(model_path, "state_mean_var.npy")
    if not os.path.exists(norm_path):
        print(f"[Warning] Normalization file not found at: {norm_path}")
        # fallback to default
        return torch.zeros(35, device=device), torch.ones(35, device=device)

    norm_data = np.load(norm_path, allow_pickle=True).item()
    state_mean = torch.tensor(norm_data["state_mean"], device=device).float()
    state_var = torch.tensor(norm_data["state_var"], device=device).float()
    state_std = torch.sqrt(state_var)
    return state_mean, state_std

def safe_standardize(x_normed: Union[torch.Tensor, np.ndarray], mean: Union[torch.Tensor, np.ndarray], std: Union[torch.Tensor, np.ndarray]):
    mask = std > 0
    if isinstance(x_normed, torch.Tensor):
        x_normed = x_normed.clone()
    if x_normed.ndim == 2:
        x_normed[:, mask] = (x_normed[:, mask] - mean[mask]) / std[mask]
    elif x_normed.ndim == 3:
        x_normed[:, :, mask] = (x_normed[:, :, mask] - mean[mask]) / std[mask]
    return x_normed


import math
import torch
import escnn
from escnn.nn import FieldType

# Assuming configure_observation_space_representations is available in your imports
from morphosymm_rl.symm_utils import configure_observation_space_representations

def initialize_dae_model(morphologycal_symmetries_cfg, koopman_cfg, task: str, state_dim: int, action_dim: int, dt: float, device: torch.device, G = None) -> torch.nn.Module:
    """
    Initializes a Koopman model based on the provided configuration (from train_koopman_cfg.koopman_model).
    Can also load pre-trained weights if koopman_cfg.load_path is specified.

    Args:
        koopman_cfg: The Koopman model configuration object from train_koopman_cfg.
        task (str): The specific DAE task type (e.g., "edae", "ecdae").
        dt (float): The environment's delta time (from environment).
        device (torch.device): The torch device (e.g., 'cuda:0', 'cpu').
        G: Optional predefined group. If None, it will be loaded automatically.

    Returns:
        torch.nn.Module: An initialized Koopman model (new or loaded).
    """

    # 1. Extract observation and action names dynamically from koopman_cfg
    # (Assuming these match your new actor/action space definitions)
    state_obs_names = morphologycal_symmetries_cfg["obs_space_names_actor"]
    action_obs_names = morphologycal_symmetries_cfg["action_space_names"]
    joints_order = morphologycal_symmetries_cfg["joints_order"]
    robot_name = koopman_cfg.get("name", "a1")

    # 2. Load all unique representations together
    all_space_names = list(dict.fromkeys([*state_obs_names, *action_obs_names]))
    loaded_G, representations = configure_observation_space_representations(
        robot_name, all_space_names, joints_order
    )

    if G is None:
        G = loaded_G

    gspace = escnn.gspaces.no_base_space(G)

    # 3. Create FieldTypes by retrieving representations for each observation/action sequence
    state_type = FieldType(gspace, representations=[representations[name] for name in state_obs_names])
    action_type = FieldType(gspace, representations=[representations[name] for name in action_obs_names])

    if G is not None:
        # Ensure that with duplicate reps the size matches the expected dimensions
        state_type.size = state_dim
        action_type.size = action_dim

    obs_state_dim = math.ceil(koopman_cfg["obs_state_ratio"] * state_dim)
    num_hidden_neurons = koopman_cfg["num_hidden_units"]
    if obs_state_dim > num_hidden_neurons:
        num_hidden_neurons = 2 ** math.ceil(math.log2(obs_state_dim))

    activation = koopman_cfg["activation"]
    if not koopman_cfg["equivariant"]:
        # Ensure class_from_name is imported or defined
        activation = class_from_name("torch.nn", activation)

    obs_fn_params = {
        'num_layers': koopman_cfg["num_layers"],
        'num_hidden_units': koopman_cfg["num_hidden_units"],
        'activation': activation,
        'bias': koopman_cfg["bias"],
        'batch_norm': koopman_cfg["batch_norm"]
    }

    initial_rng_state = torch.get_rng_state()

    # 5. Initialize the appropriate model
    if "edae" in task:
        model = EquivDAE(
            state_rep=state_type.representation,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=koopman_cfg["orth_w"],
            obs_fn_params=obs_fn_params,
            group_avg_trick=koopman_cfg["group_avg_trick"],
            state_dependent_obs_dyn=koopman_cfg["state_dependent_obs_dyn"],
            enforce_constant_fn=koopman_cfg["constant_function"],
        )
    elif "ecdae" in task:
        model = ControlledEquivDAE(
            state_rep=state_type.representation,
            action_rep=action_type.representation,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=koopman_cfg["orth_w"],
            obs_fn_params=obs_fn_params,
            group_avg_trick=koopman_cfg["group_avg_trick"],
            state_dependent_obs_dyn=koopman_cfg["state_dependent_obs_dyn"],
            enforce_constant_fn=koopman_cfg["constant_function"],
        )
    elif "cdae" in task:
        model = ControlledDAE(
            state_dim=state_dim,
            action_dim=action_dim,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=koopman_cfg["orth_w"],
            obs_fn_params=obs_fn_params,
            enforce_constant_fn=koopman_cfg["constant_function"],
        )
    elif "dae" in task:
        model = DAE(
            state_dim=state_dim,
            obs_state_dim=obs_state_dim,
            dt=dt,
            obs_pred_w=koopman_cfg["obs_pred_w"],
            orth_w=koopman_cfg["orth_w"],
            obs_fn_params=obs_fn_params,
            enforce_constant_fn=koopman_cfg["constant_function"],
        )
    else:
        raise ValueError(f"Trying to create DAE model with unsupported task: {task}")

    torch.set_rng_state(initial_rng_state)

    # Put the model on the specified device
    model.to(device)

    return model