"""Temporal Concolutional Network for time-series

Adapted from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py under the
following license:

MIT License

Copyright (c) 2018 CMU Locus Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class Residual(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        """Residual connection, does downsampling if necessary.

        Args:
            n_inputs: layer input size (number of channels).
            n_outputs: layer output size (number of channels).
        """
        super(Residual, self).__init__()
        self.do_downsample = n_inputs != n_outputs
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if self.do_downsample else nn.Identity()

    def forward(self, x: Tensor, res: Tensor, s_res: Tensor | None = None) -> Tensor:
        """Residual connection.

        Shapes:
            x: (batch_size, num_outputs, sequence_length)
            res: (batch_size, num_inputs, sequence_length)
            s_res: (batch_size, num_static_inputs)
            output: (batch_size, num_outputs, sequence_length)

        Args:
            x: the input.
            res: the residual input.
            s_res: the optional static residual input.

        Returns:
            The output tensor.

        """
        if s_res is None:
            return x + self.downsample(res)
        else:
            res = torch.cat((
                res,
                s_res.unsqueeze(-1).expand(s_res.shape[0], s_res.shape[1], res.shape[-1]
            )), dim=-2)

            return x + self.downsample(res)

    def init_weights(self) -> None:
        if self.do_downsample:
            self.downsample.weight.data.normal_(0, 0.01)


class TemporalBlock(nn.Module):
    def __init__(
            self,
            n_inputs: int,
            n_static_inputs: int,
            n_outputs: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            padding: int,
            dropout: float = 0.2) -> None:
        """Implements a two-layered residual block.

        Args:
            n_inputs: number of inputs (channels).
            n_static_inputs: number of static inputs (channels).
            n_outputs: number of outputs (channels).
            kernel_size: the 1D convolution kernel size.
            stride: the 1D convolution stride.
            dilation: the 1D convolution dilation.
            padding: the padding.
            dropout: the dropout applied after each layer. Defaults to 0.2.
        """

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.Softplus()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.Softplus()
        self.dropout2 = nn.Dropout(dropout)

        self.res = Residual(n_inputs + n_static_inputs, n_outputs)
        self.act = nn.Softplus()
        self.init_weights()

    def init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        self.res.init_weights()

    def forward(self, x: Tensor, s = Tensor | None) -> Tensor:
        """Model forward run.

        Shapes:
            Input x:  (batch_size, x_input_size, sequence_length)
            Input s:  (batch_size, s_input_size)
            Output:   (batch_size, num_outputs, sequence_length)

        Args:
            x: the input dynamic tensor.
            s: the input static tensor. None (default) means no static inputs. Then,
                model parameter `n_static_inputs` must be `0`.

        Returns:
            The output tensor.
        """

        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.act2(out)
        out = self.dropout2(out)

        out = self.res(out, x, s)
        return self.act(out)


class TemporalConvNet(nn.Module):
    def __init__(
            self,
            num_inputs: int,
            num_static_inputs: int,
            num_outputs: int,
            num_hidden: int,
            kernel_size: int = 4,
            num_layers: int = 2,
            dropout: float = 0.0) -> None:
        """Implements a Temporal Convolutional Network (TCN).

        https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

        Note:
            The TCN layer is followed by a feedfoward layer to map the TCN output channels to `num_outputs`.

        Shapes:
            Input:  (batch_size, num_inputs, sequence_length)
            Startic input:  (batch_size, num_static_inputs)
            Output: (batch_size, num_outputs, sequence_length)

        Args:
            num_inputs: the number of input features.
            num_static_inputs: the number of static input features.
            num_outputs: the number of outputs. If -1, no linear mapping is done.
            num_hidden: the hidden size (intermediate channel sizes) of the layers.
            kernel_size: the kernel size. Defaults to 4.
            num_layers: the number of stacked layers. Defaults to 2.
            dropout: a float value in the range [0, 1) that defines the dropout probability. Defaults to 0.0.
        """

        super().__init__()

        # Used to calculate receptive field (`self.receptive_field_size`).
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.tcn_layers = nn.ModuleDict()
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_hidden
            layer = TemporalBlock(
                    n_inputs=in_channels,
                    n_static_inputs=num_static_inputs,
                    n_outputs=num_hidden,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout)

            self.tcn_layers.update({f'layer_h{i:02d}': layer})

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

        out = x
        for _, layer in self.tcn_layers.items():
            out = layer(out, s)

        out = self.linear(out)
        return out

    def receptive_field_size(self) -> int:
        """Returns the receptive field of the Module.

        The receptive field (number of steps the model looks back) of the model depends
        on the number of layers and the kernel size.

        Returns:
            the size of the receptive field.

        """

        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.num_layers - 1)

    def __repr__(self):
        receptive_field = self.receptive_field_size()
        s = super().__repr__() + f' [context_len={receptive_field}]'
        return s
