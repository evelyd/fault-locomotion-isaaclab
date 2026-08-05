from isaaclab.utils import configclass

from pathlib import Path
from dataclasses import MISSING

@configclass
class KoopmanCfg:
    """Configuration for using Koopman model."""

    class_name: str = "Koopman"
    """The class name."""

    model_name: str = "cdae"
    """The Koopman model name."""

    equivariant: bool = False
    """Whether the Koopman model is equivariant."""

    activation: str = "ELU"
    """The activation function."""

    num_layers: int = 5
    """The number of layers in the Koopman model."""

    num_hidden_units: int = 128
    """The number of hidden units in each layer."""

    batch_norm: bool = False
    """Whether to use batch normalization."""

    obs_pred_w: float = 1.0
    """The weight for the observation prediction loss."""

    orth_w: float = 0.0
    """The weight for the orthogonality loss."""

    corr_w: float = 0.0
    """The weight for the correlation loss."""

    bias: bool = True
    """Whether to use a bias term."""

    constant_function: bool = True
    """Whether to use a constant function."""

    num_mini_batches: int = 8
    """The number of mini-batches for training."""

    mini_batch_size: int = 256
    """The size of each mini-batch."""

    beta_initial: float = 0.4
    """The initial value of beta for the replay buffer."""

    beta_annealing_steps: int = 20000
    """The number of steps over which to anneal the beta parameter."""

    lr = 1e-3
    """The learning rate for the Koopman model."""

    max_epochs = 200
    """The maximum number of epochs for training."""

    obs_state_ratio = 3
    """The ratio of observation to state dimensions."""

    pred_horizon = 5
    """The prediction horizon for the Koopman model."""

    frames_per_state = 1
    """The number of frames per state for the Koopman model."""

    replay_buffer_size: int = 100000
    """The size of the replay buffer."""

    group_avg_trick: bool | None = MISSING
    """Whether to use the group average trick for the Koopman model."""

    state_dependent_obs_dyn: bool | None = MISSING
    """Whether to use state-dependent observation dynamics for the Koopman model."""


activation = 'ELU'
num_layers = 5
num_hidden_units = 128
batch_norm = False
obs_pred_w = 1.0
orth_w = 0.0
corr_w = 0.0
bias = True
constant_function = True
num_mini_batches = 8
mini_batch_size = 256
beta_initial = 0.4
beta_annealing_steps = 20000
lr = 1e-3
max_epochs = 200
obs_state_ratio = 3
pred_horizon = 5
frames_per_state = 1
replay_buffer_size = 100000

cdae_koopman_cfg = KoopmanCfg(
        model_name = "cdae",
        equivariant = False,
        activation = activation,
        num_layers = num_layers,
        num_hidden_units = num_hidden_units,
        batch_norm = batch_norm,
        obs_pred_w = obs_pred_w,
        orth_w = orth_w,
        corr_w = corr_w,
        bias = bias,
        constant_function = constant_function,
        num_mini_batches = num_mini_batches,
        mini_batch_size = mini_batch_size,
        beta_initial = beta_initial,
        beta_annealing_steps = beta_annealing_steps,
        lr = lr,
        max_epochs = max_epochs,
        obs_state_ratio = obs_state_ratio,
        pred_horizon = pred_horizon,
        frames_per_state = frames_per_state,
        replay_buffer_size = replay_buffer_size
    )

group_avg_trick = True
state_dependent_obs_dyn = False

ecdae_koopman_cfg = KoopmanCfg(
        model_name = "ecdae",
        equivariant = True,
        activation = activation,
        num_layers = num_layers,
        num_hidden_units = num_hidden_units,
        batch_norm = batch_norm,
        obs_pred_w = obs_pred_w,
        orth_w = orth_w,
        corr_w = corr_w,
        bias = bias,
        constant_function = constant_function,
        num_mini_batches = num_mini_batches,
        mini_batch_size = mini_batch_size,
        beta_initial = beta_initial,
        beta_annealing_steps = beta_annealing_steps,
        lr = lr,
        max_epochs = max_epochs,
        obs_state_ratio = obs_state_ratio,
        pred_horizon = pred_horizon,
        frames_per_state = frames_per_state,
        replay_buffer_size = replay_buffer_size,
        group_avg_trick = group_avg_trick,
        state_dependent_obs_dyn = state_dependent_obs_dyn
    )