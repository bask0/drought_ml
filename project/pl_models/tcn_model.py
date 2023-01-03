
from torch import nn
from torch import Tensor

from project.pl_models.base_model import LightningNet
from modules.tcn import TemporalConvNet
from modules.feedforward import FeedForward
from utils.torch_utils import Transform


class TemporalConvNetPL(LightningNet):
    def __init__(
            self,
            num_inputs: int,
            num_geofactors: int,
            num_outputs: int,
            num_hidden: int,
            num_layers: int,
            dropout: float,
            static_dropout: float,
            kernel_size: int = 4,
            num_geofactors_enc: int = 6,
            **kwargs) -> None:
        """Implements a Temporal Convolutional Network (TCN).

        https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

        Note:
            The TCN layer is followed by a feedfoward layer to map the TCN output channels to `num_outputs`.

        Shapes:
            Input:  (batch_size, input_size, sequence_length)
            Output: (batch_size, sequence_length, num_outputs)

        Args:
            num_inputs:
                The number input dimensionality.
            num_geofactors:
                The number of static features.
            num_outputs:
                The output dimensionality.
            num_hidden:
                The number of hidden units.
            num_layers:
                The number of hidden fully-connected layers.
            dropout:
                The dropout applied after each layer, in range [0, 1).
            static_dropout:
                The dropout applied after each layer of static input mapping, in range [0, 1).
            kernel_size:
                The kernel size. Defaults to 4.
            activation:
                The activation function, defaults to 'relu'.
            num_geofactors_enc:
                The geofactor encoding dimensionality.
            **kwargs:
                Are passed to the parent class `LightningNet`.

        """

        super().__init__(**kwargs)

        self.encode_static = FeedForward(
            num_inputs=num_geofactors,
            num_outputs=num_geofactors_enc,
            num_hidden=32,
            num_layers=2,
            dropout=static_dropout,
            activation='relu',
            activation_last='tanh',
            dropout_last=False
        )

        self.flatten_time = Transform(transform_fun=lambda x: x.view(x.shape[0], -1, x.shape[-1]))
        self.to_sequence_last = Transform(transform_fun=lambda x: x.permute(0, 2, 1))

        self.tcn = TemporalConvNet(
            num_inputs=num_inputs,
            num_static_inputs=num_geofactors_enc,
            num_outputs=-1,
            num_hidden=num_hidden,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout
        )

        self.tcn_mean = TemporalConvNet(
            num_inputs=num_hidden,
            num_static_inputs=num_geofactors_enc,
            num_outputs=num_outputs,
            num_hidden=num_hidden,
            kernel_size=kernel_size,
            num_layers=1,
            dropout=dropout
        )

        self.tcn_var = TemporalConvNet(
            num_inputs=num_hidden,
            num_static_inputs=num_geofactors_enc,
            num_outputs=num_outputs,
            num_hidden=num_hidden,
            kernel_size=kernel_size,
            num_layers=1,
            dropout=dropout
        )

        self.downscale = nn.Conv1d(
            in_channels=num_outputs,
            out_channels=1,
            kernel_size=24,
            stride=24)

        self.mean_act = nn.Identity()
        self.var_act = nn.Softplus()

        self.to_channel_last = Transform(transform_fun=lambda x: x.permute(0, 2, 1))

        self.save_hyperparameters()

    def forward(self, x: Tensor, s: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Model forward call.

        Args:
            x: the sequencial tensor with shape (batch_size, sequence_length, features).
            s: the static features with shape (batch_size, features)

        Returns:
            the mean and variance estimates, both with shape (batch_size, sequence_length, num_outputs).
        """

        if s is not None:
            #  (B, FS) -> (B, FS*)
            s = self.encode_static(s)

        # (B, H, S, FH) -> (B, HxS, FH)
        out = self.flatten_time(x)
        # (B, HxS, FH) -> (B, FH, HxS)
        out = self.to_sequence_last(out)
        # (B, FH, HxS), (B, FS*) -> (B, D, HxS)
        out = self.tcn(out, s)

        # (B, D, HxS) -> (B, D, HxS)
        out_mean = self.tcn_mean(out, s)
        out_var = self.tcn_var(out, s)

        # (B, D, HxS) -> (B, O, S)
        out_mean = self.downscale(out_mean)
        out_var = self.downscale(out_var)

        # (B, O, S) -> (B, S, O)
        out_mean = self.mean_act(self.to_channel_last(out_mean))
        out_var = self.var_act(self.to_channel_last(out_var))

        return out_mean, out_var
