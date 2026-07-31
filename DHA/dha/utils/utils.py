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
from morpho_symm.nn.test_EMLP import get_kinematic_three_rep_two
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

def get_trained_dae_model(model_dir):
    """
    Load the trained DAE model.

    Args:
        model_path (str): Path to the trained model.

    Returns:
        torch.nn.Module: The trained model.
    """

    ckpt_path = os.path.join(model_dir, "best.ckpt")

    # Load the model from the checkpoint
    checkpoint = torch.load(ckpt_path)

    # Extract the state_dict from the checkpoint
    state_dict = checkpoint['state_dict']

    # Define the state representation
    # G is the symmetry group of the system
    robot, G = load_symmetric_system(robot_name="a1")

    # Create the state representations
    gspace = escnn.gspaces.no_base_space(G)
    # Extract the representations from G.representations.items()
    rep_QJ = G.representations["Q_js"]  # Used to transform joint-space position coordinates q_js ∈ Q_js
    rep_TqQJ = G.representations["TqQ_js"]  # Used to transform joint-space velocity coordinates v_js ∈ TqQ_js
    rep_O3 = G.representations["Rd"]  # Used to transform the linear momentum l ∈ R3
    rep_O3_pseudo = G.representations["Rd_pseudo"]  # Used to transform the angular momentum k ∈ R3
    trivial_rep = G.trivial_representation
    rep_kin_three = get_kinematic_three_rep_two(G)

    # Define the state and action type using the extracted representations
    if "push_door" in model_dir:
        state_reps = [rep_O3, rep_O3, rep_TqQJ, rep_TqQJ, rep_kin_three, rep_O3, rep_O3, rep_O3, rep_kin_three]  #['projected_gravity', 'projected_forward_vec', 'joint_pos', 'prev_actions', 'phase_input', 'base_pos', 'door_bottom_corner_pos', 'door_normal_vec', 'lr_vec']
        state_type = FieldType(gspace, representations=state_reps)
        state_type.size = sum(rep.size for rep in state_reps) + 4 * rep_O3.size + rep_TqQJ.size + rep_kin_three.size # Count duplicates twice
    else:
        state_reps = [rep_O3, rep_O3, rep_O3_pseudo, rep_TqQJ, rep_TqQJ, rep_TqQJ, rep_kin_three] #['projected_gravity', 'projected_forward_vec', 'commands', 'joint_pos', 'joint_vel', 'prev_actions', 'clock_inputs'] # base pose
        state_type = FieldType(gspace, representations=state_reps)
        state_type.size = sum(rep.size for rep in state_reps) + rep_O3.size + 2 * rep_TqQJ.size  # Count duplicates twice
    state_type = FieldType(gspace, representations=state_reps)
    action_reps = [rep_QJ]  # ['actions']
    action_type = FieldType(gspace, representations=action_reps)
    action_type.size = sum(rep.size for rep in action_reps)

    num_layers, num_hidden_units, bias, obs_state_dim, state_dim = extract_trained_model_info(state_dict, model_dir)

    dt = 0.02
    orth_w_match = re.search(r"Orth_w:([\d\.]+)", model_dir)
    orth_w = float(orth_w_match.group(1)) if orth_w_match else 0.0
    obs_pred_w_match = re.search(r"Obs_w:([\d\.]+)", model_dir)
    obs_pred_w = float(obs_pred_w_match.group(1)) if obs_pred_w_match else 1.0
    group_avg_trick = True
    state_dependent_obs_dyn = False
    enforce_constant_fn = True
    act_match = re.search(r"Act:([\d\.]+)", model_dir)
    activation = obs_pred_w_match.group(1) if act_match else 'ELU'
    batch_norm = False

    if not "E-DAE" in model_dir and not "EC-DAE" in model_dir:
        activation = class_from_name("torch.nn", activation)

    obs_fn_params = {'num_layers': num_layers, 'num_hidden_units': num_hidden_units, 'activation': activation, 'bias': bias, 'batch_norm': batch_norm}

    initial_rng_state = torch.get_rng_state()

    if "E-DAE" in model_dir:
        model = EquivDAE(
            state_rep=state_type.representation,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=orth_w,
            obs_fn_params=obs_fn_params,
            group_avg_trick=group_avg_trick,
            state_dependent_obs_dyn=state_dependent_obs_dyn,
            enforce_constant_fn=enforce_constant_fn,
        )
    elif "EC-DAE" in model_dir:
        model = ControlledEquivDAE(
            state_rep=state_type.representation,
            action_rep=action_type.representation,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=orth_w,
            obs_fn_params=obs_fn_params,
            group_avg_trick=group_avg_trick,
            state_dependent_obs_dyn=state_dependent_obs_dyn,
            enforce_constant_fn=enforce_constant_fn,
        )
    elif "C-DAE" in model_dir:
        model = ControlledDAE(
            state_dim=state_type.size,
            action_dim=action_type.size,
            obs_state_dim=obs_state_dim,
            dt=dt,
            obs_pred_w=obs_pred_w,
            orth_w=orth_w,
            obs_fn_params=obs_fn_params,
            enforce_constant_fn=enforce_constant_fn,
        )
    else:
        corr_w = 0.0
        model = DAE(
            state_dim=state_type.size,
            obs_state_dim=obs_state_dim,
            dt=dt,
            obs_pred_w=obs_pred_w,
            orth_w=orth_w,
            corr_w=corr_w,
            obs_fn_params=obs_fn_params,
            enforce_constant_fn=enforce_constant_fn,
        )

    torch.set_rng_state(initial_rng_state)
    model.load_state_dict(remove_state_dict_prefix(state_dict, "model."))

    return model


def initialize_dae_model(cfg, task: str, dt: int, device: torch.device, G = None) -> torch.nn.Module:
    """
    Initializes a Koopman model based on the provided configuration (from train_cfg.koopman_model).
    Can also load pre-trained weights if cfg.load_path is specified.

    Args:
        cfg: The Koopman model configuration object from train_cfg.
        state_dim (int): The dimension of the observation space (from environment).
        action_dim (int): The dimension of the action space (from environment).
        dt (float): The environment's delta time (from environment).
        device (torch.device): The torch device (e.g., 'cuda:0', 'cpu').

    Returns:
        torch.nn.Module: An initialized Koopman model (new or loaded).
    """

    if G is not None:
        # Create the state representations
        gspace = escnn.gspaces.no_base_space(G)
        # Extract the representations from G.representations.items()
        rep_QJ = G.representations["Q_js"]  # Used to transform joint-space position coordinates q_js ∈ Q_js
        rep_TqQJ = G.representations["TqQ_js"]  # Used to transform joint-space velocity coordinates v_js ∈ TqQ_js
        rep_O3 = G.representations["Rd"]  # Used to transform the linear momentum l ∈ R3
        rep_O3_pseudo = G.representations["Rd_pseudo"]  # Used to transform the angular momentum k ∈ R3
        trivial_rep = G.trivial_representation
        rep_kin_three = get_kinematic_three_rep_two(G)

        # Create dict to define which obs match which representations
        obs_rep_dict = {
            'projected_gravity': rep_O3,
            'projected_forward_vec': rep_O3,
            'commands': rep_O3_pseudo,
            'joint_pos': rep_TqQJ,
            'joint_vel': rep_TqQJ,
            'prev_actions': rep_TqQJ,
            'clock_inputs': rep_kin_three,
            'phase_input': rep_kin_three,
            'base_pos': rep_O3,
            'door_bottom_corner_pos': rep_O3,
            'door_normal_vec': rep_O3,
            'lr_vec': rep_kin_three,
            'actions': rep_TqQJ,
        }

        state_reps = []
        action_reps = []
        for state_obs in cfg["robot"]["state_obs"]:
            if state_obs in obs_rep_dict:
                state_reps.append(obs_rep_dict[state_obs])
            else:
                raise ValueError(f"Observation '{state_obs}' not found in the defined representations.")
        for action_obs in cfg["robot"]["action_obs"]:
            if action_obs in obs_rep_dict:
                action_reps.append(obs_rep_dict[action_obs])
            else:
                raise ValueError(f"Action '{action_obs}' not found in the defined representations.")

        state_type = FieldType(gspace, representations=state_reps)
        action_type = FieldType(gspace, representations=action_reps)

    state_dim = cfg["robot"]["state_dim"]
    action_dim = cfg["robot"]["action_dim"]

    if G is not None:
        # Ensure that with duplicate reps the size matches the expected dimensions
        state_type.size = state_dim
        action_type.size = action_dim

    obs_state_dim = math.ceil(cfg["robot"]["obs_state_ratio"] * state_dim)
    num_hidden_neurons = cfg["model"]["num_hidden_units"]
    if obs_state_dim > num_hidden_neurons:
        num_hidden_neurons = 2 ** math.ceil(math.log2(obs_state_dim))

    activation = cfg["model"]["activation"]

    if not cfg["model"]["equivariant"]:
        activation = class_from_name("torch.nn", activation)

    obs_fn_params = {'num_layers': cfg["model"]["num_layers"], 'num_hidden_units': cfg["model"]["num_hidden_units"], 'activation': activation, 'bias': cfg["model"]["bias"], 'batch_norm': cfg["model"]["batch_norm"]}

    initial_rng_state = torch.get_rng_state()

    if "edae" in task:
        model = EquivDAE(
            state_rep=state_type.representation,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=cfg["model"]["orth_w"],
            obs_fn_params=obs_fn_params,
            group_avg_trick=cfg["model"]["group_avg_trick"],
            state_dependent_obs_dyn=cfg["model"]["state_dependent_obs_dyn"],
            enforce_constant_fn=cfg["model"]["constant_function"],
        )
    elif "ecdae" in task:
        model = ControlledEquivDAE(
            state_rep=state_type.representation,
            action_rep=action_type.representation,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=cfg["model"]["orth_w"],
            obs_fn_params=obs_fn_params,
            group_avg_trick=cfg["model"]["group_avg_trick"],
            state_dependent_obs_dyn=cfg["model"]["state_dependent_obs_dyn"],
            enforce_constant_fn=cfg["model"]["constant_function"],
        )
    elif "cdae" in task:
        model = ControlledDAE(
            state_dim=state_dim,
            action_dim=action_dim,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=cfg["model"]["orth_w"],
            obs_fn_params=obs_fn_params,
            enforce_constant_fn=cfg["model"]["constant_function"],
        )
    elif "dae" in task:
        model = DAE(
            state_dim=state_dim,
            obs_state_dim=obs_state_dim,
            dt=dt,
            obs_pred_w=cfg["model"]["obs_pred_w"],
            orth_w=cfg["model"]["orth_w"],
            obs_fn_params=obs_fn_params,
            enforce_constant_fn=cfg["model"]["constant_function"],
        )
    else:
        raise ValueError(f"Trying to create DAE model with unsupported task: {task}")

    torch.set_rng_state(initial_rng_state)

    # Put the model on the specified device
    model.to(device)

    return model

def main():

    import matplotlib.pyplot as plt

    # Use LaTeX for text rendering for compatibility with papercept/IEEE
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # List of model_dirs to process
    model_dirs = [
        "../../experiments/test/S:stand_dance-OS:3-G:C2-H:5-EH:5_EC-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=a1/seed=881",
        "../../experiments/test/S:stand_dance-OS:3-G:C2-H:5-EH:5_C-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=a1/seed=962",
        "../../experiments/test/S:walk_slope-OS:3-G:C2-H:5-EH:5_C-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=a1/seed=183",
        "../../experiments/test/S:walk_slope-OS:3-G:C2-H:5-EH:5_EC-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=a1/seed=002",
        "../../experiments/test/S:push_door-OS:3-G:C2-H:5-EH:5_C-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=a1/seed=847",
        "../../experiments/test/S:push_door-OS:3-G:C2-H:5-EH:5_EC-DAE-Obs_w:1.0-Orth_w:0.0-Act:ELU-B:True-BN:False-LR:0.001-L:5-128_system=a1/seed=254",
    ]

    # Extract unique tasks and dae types for subplot arrangement
    task_names = []
    dae_types = []
    model_info = []
    for model_dir in model_dirs:
        match = re.search(r"S:([^-/]+)", model_dir)
        task_name = match.group(1) if match else "unknown"
        dae_type = "EC-DAE" if "EC-DAE" in model_dir else "C-DAE"
        if task_name not in task_names:
            task_names.append(task_name)
        if dae_type not in dae_types:
            dae_types.append(dae_type)
        model_info.append((model_dir, task_name, dae_type))

    n_rows = len(task_names)
    n_cols = len(dae_types)

    # Prepare storage for matrices and eigenvalues
    a_matrices = {}
    b_matrices = {}
    eigvals_dict = {}
    controllability_ranks = {}
    controllability_svs = {}

    dha_dir = os.path.dirname(dha.__file__)

    for model_dir, task_name, dae_type in model_info:
        print(f"processing model in {model_dir}...")
        model_dir_full = os.path.join(dha_dir, model_dir)
        model = get_trained_dae_model(model_dir_full)

        if "EC-DAE" in model_dir:
            state_dim = model.obs_state_type.size
            action_dim = model.action_type.size

            a_op = model.obs_space_dynamics.transfer_op
            b_op = model.obs_space_dynamics.control_op

            identity_a = torch.eye(state_dim)
            input_tensor_a = model.obs_state_type(identity_a)
            a_matrix = a_op(input_tensor_a).tensor.detach().T

            identity_b = torch.eye(action_dim)
            input_tensor_b = model.action_type(identity_b)
            b_matrix = b_op(input_tensor_b).tensor.detach().T
        else:
            a_matrix = model.obs_space_dynamics.transfer_op.weight.detach()
            b_matrix = model.obs_space_dynamics.control_op.weight.detach()

            state_dim = a_matrix.shape[0]
            action_dim = b_matrix.shape[0]

        a_matrices[(task_name, dae_type)] = a_matrix
        b_matrices[(task_name, dae_type)] = b_matrix

        # Eigenvalues of A
        eigvals = np.linalg.eigvals(a_matrix.numpy())
        eigvals_dict[(task_name, dae_type)] = eigvals

        # Controllability matrix and singular values
        controllability_matrix = b_matrix
        term = b_matrix
        for i in range(1, state_dim):
            term = a_matrix.T @ term
            controllability_matrix = torch.cat((controllability_matrix, term), dim=1)
        rank_c = torch.linalg.matrix_rank(controllability_matrix)
        controllability_ranks[(task_name, dae_type)] = (rank_c, state_dim)
        singular_values = torch.linalg.svdvals(controllability_matrix)
        controllability_svs[(task_name, dae_type)] = singular_values

    # Plotting: 4 figures, each with n_rows x n_cols subplots
    fig_a, axs_a = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    fig_b, axs_b = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    fig_eig, axs_eig = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    fig_sv, axs_sv = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))

    # Helper to get correct axis for any shape (1d/2d)
    def get_ax(axs, i, j):
        if n_rows == 1 and n_cols == 1:
            return axs
        elif n_rows == 1:
            return axs[j]
        elif n_cols == 1:
            return axs[i]
        else:
            return axs[i, j]

    # For LaTeX-friendly task names
    task_latex = {
        "stand_dance": "stand dance",
        "walk_slope": "walk slope",
        "push_door": "push door",
        "unknown": "unknown"
    }

    for i, task_name in enumerate(task_names):
        for j, dae_type in enumerate(dae_types):
            key = (task_name, dae_type)
            # A matrix
            ax = get_ax(axs_a, i, j)
            a_matrix = a_matrices[key]
            im = ax.imshow(a_matrix.numpy(), cmap='viridis', interpolation='nearest')
            ax.set_title(rf"{dae_type} $A$ for {task_latex.get(task_name, task_name)}", fontsize=13)
            if j == 0:
                ax.set_ylabel("Output dim")
            if i == n_rows - 1:
                ax.set_xlabel("Input dim")
            fig_a.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # B matrix
            ax = get_ax(axs_b, i, j)
            b_matrix = b_matrices[key]
            im = ax.imshow(b_matrix.numpy(), cmap='viridis', interpolation='nearest', aspect='equal')
            ax.set_title(rf"{dae_type} $B$ for {task_latex.get(task_name, task_name)}", fontsize=13)
            if j == 0:
                ax.set_ylabel("Output dim")
            if i == n_rows - 1:
                ax.set_xlabel("Input dim")
            fig_b.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # Eigenvalues
            ax = get_ax(axs_eig, i, j)
            eigvals = eigvals_dict[key]
            ax.scatter(eigvals.real, eigvals.imag, color='blue', marker='o')
            ax.set_title(rf"{dae_type} $\lambda(A)$ for {task_latex.get(task_name, task_name)}", fontsize=13)
            if j == 0:
                ax.set_ylabel("Imag")
            if i == n_rows - 1:
                ax.set_xlabel("Real")
            theta = np.linspace(0, 2 * np.pi, 400)
            ax.plot(np.cos(theta), np.sin(theta), 'r--')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_aspect('equal')
            # Singular values
            ax = get_ax(axs_sv, i, j)
            svs = controllability_svs[key]
            rank_c, state_dim = controllability_ranks[key]
            ax.semilogy(svs.detach().numpy(), 'o-', label='Singular Values')
            ax.set_title(rf"{dae_type} SV(Kalman) for {task_latex.get(task_name, task_name)}" + f"\nRank: {rank_c}/{state_dim}", fontsize=13)
            if j == 0:
                ax.set_ylabel('SV Magnitude (log)')
            if i == n_rows - 1:
                ax.set_xlabel('SV Index')
            ax.grid(True, which="both", linestyle='--')
            ax.axhline(y=1e-8, color='r', linestyle='--', label='Num. Zero')
            ax.legend()

    # Set the same y-limits for all controllability SV plots
    all_svs = torch.cat([controllability_svs[key].detach() for key in controllability_svs])
    min_sv = all_svs.min().item()
    max_sv = all_svs.max().item()
    for i in range(n_rows):
        for j in range(n_cols):
            ax = get_ax(axs_sv, i, j)
            ax.set_ylim([min_sv, max_sv])

    # Save figures as PDF with tight bounding box for no extra whitespace
    fig_a.tight_layout()
    fig_b.tight_layout()
    fig_eig.tight_layout()
    fig_sv.tight_layout()
    fig_a.savefig("all_a_matrices.pdf", bbox_inches='tight', pad_inches=0.01)
    fig_b.savefig("all_b_matrices.pdf", bbox_inches='tight', pad_inches=0.01)
    fig_eig.savefig("all_a_eigenvalues.pdf", bbox_inches='tight', pad_inches=0.01)
    fig_sv.savefig("all_controllability_svs.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.close(fig_a)
    plt.close(fig_b)
    plt.close(fig_eig)
    plt.close(fig_sv)

    # Print max eigenvalue magnitude and max singular value for each model
    # Print results as a LaTeX table
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l l c c c}")
    print("\\hline")
    print("Task & DAE Type & Max $|\\lambda(A)|$ & Mean $|\\lambda(A)|$ & Max SV (Kalman) \\\\")
    print("\\hline")
    for key in eigvals_dict:
        task_name, dae_type = key
        eigvals = eigvals_dict[key]
        max_eigval_mag = np.abs(eigvals).max()
        mean_eigval_mag = np.abs(eigvals).mean()
        svs = controllability_svs[key]
        max_sv = svs.max().item()
        print(f"{task_name} & {dae_type} & {max_eigval_mag:.4f} & {mean_eigval_mag:.4f} & {max_sv:.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Summary of $A$ matrix eigenvalues and controllability matrix singular values for each model.}")
    print("\\label{tab:a_eig_sv_summary}")
    print("\\end{table}")

if __name__ == "__main__":
    main()
