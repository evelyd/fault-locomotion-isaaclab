import logging
from typing import List, Union

import torch

log = logging.getLogger(__name__)


class VariationalMLP(torch.nn.Module):
    """
    MLP Encoder for a Variational Autoencoder (VAE).
    Outputs the mean (mu) and log-variance (log_var) of the latent distribution.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,  # Renamed out_dim to latent_dim for clarity in VAE context
        num_hidden_units: int = 64,
        num_layers: int = 3,
        bias: bool = True,
        batch_norm: bool = False,
        head_with_activation: bool = False,
        activation: Union[torch.nn.Module, List[torch.nn.Module]] = torch.nn.ReLU,
        init_mode="fan_in",
    ):
        """Constructor of a Multi-Layer Perceptron (MLP) model for a VAE encoder.

        Args:
        ----
            in_dim: Dimension of the input space.
            latent_dim: Dimension of the latent space (will be the dimension of both mu and log_var).
            num_hidden_units: Number of hidden units in the intermediate layers.
            num_layers: Number of layers in the MLP including input and output/head layers.
            activation: Activation modules.
            bias: Whether to include a bias term in the linear layers.
            init_mode: Initialization mode for weights.
        """
        super().__init__()
        logging.info("Instantiating VAEEncoderMLP (PyTorch)")
        # The final output dimension must be 2 * latent_dim to hold mu and log_var
        self.latent_dim = latent_dim
        self.out_dim = 2 * latent_dim
        self.in_dim = in_dim

        self.init_mode = init_mode if init_mode is not None else "fan_in"
        self.hidden_channels = num_hidden_units
        self.activation = activation if isinstance(activation, list) else [activation] * (num_layers - 1)

        self.num_layers = num_layers
        if self.num_layers == 1 and not head_with_activation:
            log.warning(f"{self} model with 1 layer and no activation. This is equivalent to a linear map")

        dim_in = self.in_dim
        dim_out = num_hidden_units

        self.net = torch.nn.Sequential()
        for n in range(self.num_layers - 1):
            dim_out = num_hidden_units

            block = torch.nn.Sequential()
            block.add_module(f"linear_{n}", torch.nn.Linear(dim_in, dim_out, bias=bias))
            if batch_norm:
                block.add_module(f"batchnorm_{n}", torch.nn.BatchNorm1d(dim_out))
            # Use the correct activation for the block
            block.add_module(f"act_{n}", self.activation[n]())

            self.net.add_module(f"block_{n}", block)
            dim_in = dim_out

        # Add last layer
        head_block = torch.nn.Sequential()
        # Output dimension is 2 * latent_dim
        head_block.add_module(
            f"linear_{num_layers - 1}", torch.nn.Linear(in_features=dim_in, out_features=self.out_dim, bias=bias)
        )
        # NOTE: Do NOT use an activation on the final layer for mu and log_var,
        # as they should be able to take any real value.
        if head_with_activation:
             # Remove this part or be aware that activation here is non-standard
             # log_var and mu should not be bounded.
            log.warning("Activation on the final layer for VAE encoder is non-standard for mu/log_var.")
            if batch_norm:
                head_block.add_module(f"batchnorm_{num_layers - 1}", torch.nn.BatchNorm1d(self.out_dim))
            head_block.add_module(f"act_{num_layers - 1}", activation())

        self.net.add_module("head", head_block)

        self.reset_parameters(init_mode=self.init_mode)

    def forward(self, input):
        """Forward pass of the MLP model. Returns mu and log_var."""
        # The output is a vector of size 2 * latent_dim
        output = self.net(input)

        # Split the output into mu and log_var
        mu = output[:, : self.latent_dim]
        log_var = output[:, self.latent_dim :]

        return mu, log_var # Return both mean and log-variance

    # ... (get_hparams and reset_parameters methods remain the same) ...
    def get_hparams(self):
        return {"num_layers": self.num_layers, "hidden_ch": self.hidden_channels, "init_mode": self.init_mode}

    def reset_parameters(self, init_mode=None):
        assert init_mode is not None or self.init_mode is not None
        self.init_mode = self.init_mode if init_mode is None else init_mode
        for module in self.net:
            if isinstance(module, torch.nn.Sequential):
                tensor = module[0].weight
                activation = module[-1].__class__.__name__
                activation = "linear" if activation == "Identity" else activation
            elif isinstance(module, torch.nn.Linear):
                tensor = module.weight
                activation = "Linear"
            else:
                raise NotImplementedError(module.__class__.__name__)

            if activation.lower() == "relu" or activation.lower() == "leakyrelu":
                torch.nn.init.kaiming_uniform_(tensor, mode=self.init_mode, nonlinearity=activation.lower())
            elif activation.lower() == "selu":
                torch.nn.init.kaiming_normal_(tensor, mode=self.init_mode, nonlinearity="linear")
            else:
                try:
                    torch.nn.init.kaiming_uniform_(tensor, mode=self.init_mode, nonlinearity=activation.lower())
                except ValueError:
                    log.info(
                        f"Could not initialize {module.__class__.__name__} with {self.init_mode} mode. "
                        f"Using default Pytorch initialization"
                    )

        log.info(f"MLP initialized with mode: {self.init_mode}")