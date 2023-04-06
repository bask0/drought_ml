
import torch
from torch import Tensor
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        """Residual connection, does downsampling if necessary.

        Args:
            n_inputs: layer input size (number of channels).
            n_outputs: layer output size (number of channels).
        """
        super(Residual, self).__init__()
        self.do_downsample = n_inputs != n_outputs
        self.downsample = nn.Linear(n_inputs, n_outputs) if self.do_downsample else nn.Identity()

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        """Residual connection.

        Shapes:
            x: (batch_size, num_outputs, sequence_length)
            res: (batch_size, num_static_inputs)
            output: (batch_size, num_outputs, sequence_length)

        Args:
            x: the input.
            res: the residual input.

        Returns:
            The output tensor.

        """
        if self.do_downsample:
            return x + self.downsample(res).unsqueeze(1)
        else:
            return x


class LSTM(nn.Module):
    def __init__(
            self,
            num_inputs: int,
            num_static_inputs: int,
            num_outputs: int,
            num_hidden: int,
            num_layers: int = 2,
            dropout: float = 0.0) -> None:
        """Implements a long short-term memory model (LSTM).

        Note:
            The LSTM layers are followed by a feedfoward layer to map the output channels to `num_outputs`.

        Shapes:
            Input:  (batch_size, num_inputs, sequence_length)
            Startic input:  (batch_size, num_static_inputs)
            Output: (batch_size, num_outputs, sequence_length)

        Args:
            num_inputs: the number of input features.
            num_static_inputs: the number of static input features.
            num_outputs: the number of outputs. If -1, no linear mapping is done.
            num_hidden: the hidden size (intermediate channel sizes) of the layers.
            num_layers: the number of stacked layers. Defaults to 2.
            dropout: a float value in the range [0, 1) that defines the dropout probability. Defaults to 0.0.
        """

        super().__init__()

        self.has_static = num_static_inputs > 0

        self.lstm = nn.LSTM(
            input_size=num_inputs,
            hidden_size=num_hidden,
            num_layers=num_layers,
            dropout=dropout,

            batch_first=True
        )

        if self.has_static:
            self.residual = Residual(n_inputs=num_static_inputs, n_outputs=num_hidden)

        if num_outputs == -1:
            self.linear = nn.Identity()
        else:
            self.linear = torch.nn.Conv1d(
                in_channels=num_hidden,
                out_channels=num_outputs,
                kernel_size=1
            )

    def forward(self, x: Tensor, s: Tensor | None = None) -> Tensor:
        """Run data through the model.

        Args:
            x: the input data with shape (batch, num_inputs, seq).
            s: the optional static input data with shape (batch, num_static_inputs).

        Returns:
            the model output tensor with shape (batch, seq, num_outputs).
        """

        out, h = self.lstm(x)

        if self.has_static:
            if s is None:
                raise ValueError(
                    '`num_static_inputs` was set to >0, but `s` is `None`.'
                )
            out = self.residual(out, s)

        out = self.linear(out)
        return out, h
