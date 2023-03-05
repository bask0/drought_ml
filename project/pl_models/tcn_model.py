
import torch
from torch import nn
from torch import Tensor

from project.pl_models.base_model import LightningNet
from project.modules.tcn import TemporalConvNet
from project.modules.feedforward import FeedForward
from project.utils.torch_utils import Transform, EncodeHourlyToDaily
from project.utils.types import VarStackPattern


class __TemporalConvNetPL(LightningNet):
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
            num_meteo_enc: int = 10,
            num_geofactors_enc: int = 6,
            predict_var: bool = False,
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
            num_meteo_enc:
                The meteorology encoding dimensionality.
            num_geofactors_enc:
                The geofactor encoding dimensionality.
            predict_var:
                Whether to predict uncertainty, default is False.
            **kwargs:
                Are passed to the parent class `LightningNet`.

        """

        super().__init__(**kwargs)

        self.predict_var = predict_var

        self.encode_static = FeedForward(
            num_inputs=num_geofactors,
            num_outputs=num_geofactors_enc,
            num_hidden=32,
            num_layers=2,
            dropout=static_dropout,
            activation='softplus',
            activation_last='tanh',
            dropout_last=False
        )

        self.encode_hourly = EncodeHourlyToDaily(
            num_inputs=num_inputs,
            num_hidden=num_hidden,
            num_encoding=num_meteo_enc,
            dropout=dropout,
            outpout_channel_last=False
        )

        self.tcn = TemporalConvNet(
            num_inputs=num_meteo_enc,
            num_static_inputs=num_geofactors_enc,
            num_outputs=-1,
            num_hidden=num_hidden,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout
        )

        self.mean_output_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=num_hidden,
                out_channels=num_outputs,
                kernel_size=1
            ),
            Transform(transform_fun=lambda x: x.transpose(1, 2))
        )

        if self.predict_var:
            self.var_output_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=num_hidden,
                    out_channels=num_outputs,
                    kernel_size=1
                ),
                nn.Softplus(),
                Transform(transform_fun=lambda x: x.transpose(1, 2))
            )

        self.to_channel_last = Transform(transform_fun=lambda x: x.transpose(1, 2))

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

        # (B, H, S, FH) -> (B, S, FD*)
        out = self.encode_hourly(x)

        # (B, S, FD*), (B, FS*) -> (B, D, S)
        out = self.tcn(out, s)

        # (B, D, S) -> (B, S, O)
        out_mean = self.mean_output_layer(out)

        if self.predict_var:
            # (B, D, S) -> (B, S, O)
            out_var = self.var_output_layer(out)
        else:
            out_var = None

        return out_mean, out_var


class TimeEncode(nn.Module):
    def __init__(self, max_value: int) -> None:
        """Predict periodic time-series with periodicity in dayofyear of values.

        Args:
            max_value: the maximum value of the time encoding.
        """
        super().__init__()

        self.max_value = max_value

    def forward(self, x: Tensor, time: Tensor) -> Tensor:
        """Predict periodic time-series with periodicity in dayofyear of values in range 0-365

        Args:
            x: sine amplitudes, shape (batch x num_hidden x 1)
            time: the time values, shape (batch, seq)

        Returns:
            The predicted time series with shape (batch x seq x 1)
        """

        num_hidden = x.shape[1]
        if not num_hidden % 2 == 1:
            raise ValueError(f'The encoding dimension of `x` must be an odd number, is {num_hidden}.')

        num_sine = num_hidden // 2
        num_cosine = num_hidden - num_sine

        time_enc_cos = torch.cos(time.unsqueeze(-1) / self.max_value * 2. * torch.pi * torch.arange(0., num_cosine, device=time.device).view(1, -1, num_cosine))
        time_enc_cos[..., 0] = 0.5

        time_enc_sin = torch.sin(time.unsqueeze(-1) / self.max_value * 2. * torch.pi * torch.arange(1., num_sine + 1, device=time.device).view(1, -1, num_sine))

        time_enc = torch.concatenate((time_enc_cos, time_enc_sin), -1)

        out = torch.bmm(time_enc, x)

        return out


class TemporalConvNetPL(LightningNet):
    def __init__(
            self,
            num_inputs: int,
            num_geofactors: int,
            num_hidden: int,
            num_layers: int,
            dropout: float,
            static_dropout: float,
            kernel_size: int = 4,
            num_meteo_enc: int = 10,
            num_geofactors_enc: int = 6,
            num_msc_harmonics: int = 15,
            predict_var: bool = False,
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
            num_meteo_enc:
                The meteorology encoding dimensionality.
            num_geofactors_enc:
                The geofactor encoding dimensionality.
            num_msc_harmonics:
                The number of sine basis functions to learn seasonality. Must be an odd number. Default is 15.
            predict_var:
                Whether to predict uncertainty, default is False.
            **kwargs:
                Are passed to the parent class `LightningNet`.

        """

        super().__init__(**kwargs)

        self.predict_var = predict_var

        self.encode_static = FeedForward(
            num_inputs=num_geofactors,
            num_outputs=num_geofactors_enc,
            num_hidden=16,
            num_layers=2,
            dropout=static_dropout,
            activation='softplus',
            activation_last='tanh',
            dropout_last=False
        )

        self.encode_hourly = EncodeHourlyToDaily(
            num_inputs=num_inputs,
            num_hidden=16,
            num_encoding=num_meteo_enc,
            dropout=dropout,
            outpout_channel_last=False
        )

        self.tcn = TemporalConvNet(
            num_inputs=num_meteo_enc,
            num_static_inputs=num_geofactors_enc,
            num_outputs=-1,
            num_hidden=num_hidden,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=0.0
        )

        # self.raw_output_layer = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=num_hidden,
        #         out_channels=1,
        #         kernel_size=1
        #     ),
        # )

        self.tcn_ano = TemporalConvNet(
            num_inputs=num_hidden,
            num_static_inputs=num_geofactors_enc,
            num_outputs=1,
            num_hidden=num_hidden,
            kernel_size=kernel_size,
            num_layers=2,
            dropout=0.0
        )

        self.maxpool = nn.MaxPool1d(
            kernel_size=5,
            stride=3
        )

        self.tcn_msc = TemporalConvNet(
            num_inputs=num_hidden,
            num_static_inputs=num_geofactors_enc,
            num_outputs=num_msc_harmonics,
            num_hidden=num_hidden,
            kernel_size=5,
            num_layers=2,
            dropout=0.0
        )

        self.amplitude_msc = nn.Conv1d(
            in_channels=num_msc_harmonics,
            out_channels=num_msc_harmonics,
            kernel_size=1
        )

        self.msc_reduce = Transform(transform_fun=lambda x: x.mean(-1, keepdim=True))

        self.time_encode = TimeEncode(max_value=365)

        self.ano_output_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=num_hidden + 1,
                out_channels=1,
                kernel_size=1
            ),
        )

        self.switch_sc_dim = Transform(transform_fun=lambda x: x.transpose(1, 2))

        self.save_hyperparameters()

    def forward(self, x: Tensor, x_msc: Tensor, x_ano: Tensor, time: Tensor | None, s: Tensor | None = None) -> tuple[VarStackPattern, VarStackPattern]:
        """Model forward call.

        Args:
            x: the sequencial tensor with shape (batch_size, sequence_length, features).
            msc: the mean seasonal cycle, mus be passed during training
            s: the static features with shape (batch_size, features)

        Returns:
            the predictions of type `ModelReturnPattern`, each with shape (batch_size, sequence_length, num_outputs).
        """

        # (B, S, 1) -> (B, 1, S)
        #msc = self.switch_sc_dim(msc)

        if s is not None:
            #  (B, FS) -> (B, FS*)
            s = self.encode_static(s)

        # (B, H, S, FH) -> (B, S, FD*)
        out = self.encode_hourly(x)

        # (B, S, FD*), (B, FS*) -> (B, D, S)
        out = self.tcn(out, s)

        # (B, D, S) -> (B, D, S*)
        out_reduced = self.maxpool(out)

        # (B, D, S*) -> (B, D, S*)
        msc_out = self.tcn_msc(out_reduced, s)

        # (B, D, S) -> (B, D, S*)
        msc_out = self.maxpool(msc_out)

        # (B, D, S*) -> (B, D, 1)
        msc_amp = self.amplitude_msc(msc_out).mean(-1, keepdim=True)

        # (B, D), (B, 1), (B, S) -> (B, S, 1)
        msc_out = self.time_encode(msc_amp, time)

        # (B, D, S) -> (B, 1, S)
        ano_out = self.tcn_ano(out, s)

        # (B, 1, S) -> (B, S, 1)
        ano_out = self.switch_sc_dim(ano_out)

        if self.training:
            mask = x_msc.isfinite() & (torch.randint(4, size=x_msc.shape, device=x_msc.device) < 1)
            msc_out[mask] = x_msc[mask]

        if self.training:
            mask = x_ano.isfinite() & (torch.randint(4, size=x_ano.shape, device=x_ano.device) < 1)
            ano_out[mask] = x_ano[mask]

        # (B, S, 1), (B, S, 1) -> (B, S, 1)
        raw_out = msc_out + ano_out

        daily_out = VarStackPattern(
            ts=raw_out,
            msc=msc_out,
            ano=ano_out,
            ts_var=None,
            msc_var=None,
            ano_var=None
        )

        hourly_out = VarStackPattern(
            ts=None,
            msc=None,
            ano=None,
            ts_var=None,
            msc_var=None,
            ano_var=None
        )

        return daily_out, hourly_out
