
from torch import nn
from torch import Tensor

from project.utils.torch_utils import get_activation


class FeedForward(nn.Module):
    """Implements a feed-forward neural networks with 1 to n layers.

    Implements a feed-forward model, each layer conisting of a linear,
    dropout and an actiavation layer. The dropout and actiavation of the
    last layer are optional (see `activation_last` and `dropout_last`).

    Args:
        num_inputs: input dimensionality
        num_outputs: output dimensionality
        num_hidden: number of hidden units
        num_layers: number of fully-connected hidden layers
        dropout: dropout applied after each layer, in range [0, 1)
        activation: activation function, see `get_activation`
        activation_last: output activation, see `get_activation`.
            Defaults to None (=identity).
        dropout_last: If `True`, the dropout is also
            applied after last layer. Defaults to False.
        batch_norm: Wheter to use batch normalizations in all but the
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
            batch_norm: bool = False) -> None:

        super().__init__()

        in_sizes = [num_inputs] + [num_hidden] * (num_layers - 1)
        out_sizes = [num_hidden] * (num_layers - 1) + [num_outputs]

        layers = {}
        is_last = False
        for idx, (ni, no) in enumerate([(ni, no) for ni, no in
                                        zip(in_sizes, out_sizes)]):

            layer = nn.Linear(ni, no)

            if idx == num_layers - 1:
                is_last = True
            layers.update({f'linear{idx:02d}': layer})

            if not is_last:
                layers.update({f'dropout{idx:02d}': nn.Dropout(dropout)})
                if batch_norm:
                    layers.update({f'batch_norm{idx:02d}': nn.BatchNorm1d(no)})
                layers.update({f'activation{idx:02d}': get_activation(activation)})

            if is_last and dropout_last:
                layers.update({f'dropout{idx:02d}': nn.Dropout(dropout)})

            if is_last and activation_last is not None:
                layers.update({f'activation{idx:02d}': get_activation(activation_last)})

        self.model = nn.Sequential()

        for k, v in layers.items():
            self.model.add_module(k, v)

    def forward(self, x: Tensor) -> Tensor:
        """Model forward call.

        Args:
            x: the input tensor.

        Returns:
            Tensor: the output tensor.
        """
        return self.model(x)
