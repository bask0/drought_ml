
from torch import nn
from torch import Tensor

from project.utils.torch_utils import get_activation


class FeedForwardBlock(nn.Module):
    """Implements a FeedForward block.

    Args:
        num_inputs (int): input dimensionality
        num_outputs (int): output dimensionality
        dropout (float): dropout applied after each HIDDEN layer, in range [0, 1)
        activation (str): activation function, see `get_activation`
        residual (bool): whether to add a residual connection, default is `False`.
        batch_norm (bool): wheter to use batch normalizations in all but the
            last layer. Defaults to `False`.
    """

    def __init__(
            self,
            num_inputs: int,
            num_outputs: int,
            dropout: float,
            activation: str,
            batch_norm: bool = False) -> None:

        super().__init__()

        self.block = nn.Sequential()

        self.block.add_module('linear', nn.Linear(num_inputs, num_outputs))
        if batch_norm:
            self.block.add_module('batch_norm', nn.BatchNorm1d(num_outputs))
        if dropout > 0.0:
            self.block.add_module('dropout', nn.Dropout(dropout))
        self.block.add_module('activation', get_activation(activation))

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FeedForward(nn.Module):
    """Implements a feed-forward neural networks with 1 to n layers.

    Implements a feed-forward model, each layer conisting of a linear,
    dropout and an actiavation layer. The dropout and actiavation of the
    last layer are optional (see `activation_last` and `dropout_last`).

    Args:
        num_inputs (int): input dimensionality
        num_outputs (int): output dimensionality
        num_hidden (int): number of hidden units
        num_layers (int): number of fully-connected hidden layers (not
            incliding the input and output layers)
        dropout (float): dropout applied after each HIDDEN layer, in range [0, 1)
        activation (str): activation function, see `get_activation`
        activation_last (str, optional): output activation, see `get_activation`.
            Defaults to None (=identity).
        dropout_last (bool, optional): if `True`, the dropout is also
            applied after last layer. Defaults to False.
        residual (bool): whether to add a residual connection, default is `False`.
        batch_norm (bool): wheter to use batch normalizations in all but the
            last layer. Defaults to `False`.

    """
    def __init__(
            self,
            num_inputs: int,
            num_outputs: int,
            num_hidden: int,
            num_layers: int,
            dropout: float,
            activation: str,
            activation_last: str | None = None,
            dropout_last: bool = False,
            residual: bool = False,
            batch_norm: bool = False) -> None:

        super().__init__()

        self.residual = residual
        if self.residual:
            self.residual_transform = nn.Linear(num_inputs, num_hidden)

        self.layer_in = FeedForwardBlock(
            num_inputs=num_inputs,
            num_outputs=num_hidden,
            dropout=0.0,
            activation=activation,
            batch_norm=batch_norm
        )

        self.layers_hidden = nn.ModuleDict()
        for idx in range(num_layers):
            layer_hidden = FeedForwardBlock(
                num_inputs=num_hidden,
                num_outputs=num_hidden,
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm
            )
            self.layers_hidden.update({f'layer_h{idx:02d}': layer_hidden})

        self.layer_out = FeedForwardBlock(
            num_inputs=num_hidden,
            num_outputs=num_outputs,
            dropout=dropout if dropout_last else 0.0,
            activation=activation_last,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Model forward call.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """

        if self.residual:
            x_res = self.residual_transform(x)
        else:
            x_res = 0.0

        out = self.layer_in(x)

        for _, layer in self.layers_hidden.items():
            out = layer(out)

        out = out + x_res

        out = self.layer_out(out)

        return out
