
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

        time_enc_cos = torch.cos(
            time.unsqueeze(-1) / self.max_value * 2. * torch.pi * torch.arange(
                0., num_cosine, device=time.device).view(1, -1, num_cosine))
        time_enc_cos[..., 0] = 0.5

        time_enc_sin = torch.sin(
            time.unsqueeze(-1) / self.max_value * 2. * torch.pi * torch.arange(
                1., num_sine + 1, device=time.device).view(1, -1, num_sine))

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
            predict_msc: bool = False,
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
            predict_msc:
                Whether to predict the MSC.
            predict_var:
                Whether to predict uncertainty, default is False.
            **kwargs:
                Are passed to the parent class `LightningNet`.

        """

        super().__init__(**kwargs)

        self.predict_msc = predict_msc
        self.predict_var = predict_var

        self.encode_static = FeedForward(
            num_inputs=num_geofactors,
            num_outputs=num_meteo_enc ** 2,
            num_hidden=16,
            num_layers=2,
            dropout=static_dropout,
            activation='softplus',
            activation_last='tanh',
            dropout_last=False
        )

        self.static_to_square = Transform(transform_fun=lambda x: x.view(-1, num_meteo_enc, num_meteo_enc))

        self.encode_hourly = EncodeHourlyToDaily(
            num_inputs=num_inputs,
            num_hidden=16,
            num_encoding=num_meteo_enc,
            dropout=dropout,
            outpout_channel_last=False
        )

        self.bmm = Transform(transform_fun=lambda x: torch.bmm(*x))

        self.num_out = (int(predict_var) + 1) * (int(predict_msc) + 1)
        num_hidden_tcn = num_hidden * self.num_out

        self.tcn = TemporalConvNet(
            num_inputs=num_meteo_enc,
            num_static_inputs=0,
            num_outputs=-1,
            num_hidden=num_hidden_tcn,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=0.0
        )

        self.output_layer = nn.Conv1d(
            in_channels=num_hidden_tcn,
            out_channels=self.num_out,
            kernel_size=1,
            groups=self.num_out
        )

        self.to_channel_last = Transform(transform_fun=lambda x: x.transpose(1, 2))

        if self.num_out > 1:
            self.split_outputs = Transform(transform_fun=lambda x: x.split(split_size=1, dim=-1))

        if self.predict_var:
            self.softplus = nn.Softplus()

        self.save_hyperparameters()

    def forward(
            self,
            x: Tensor,
            s: Tensor | None = None) -> tuple[VarStackPattern, VarStackPattern]:
        """Model forward call.

        Args:
            x: the sequencial tensor with shape (batch_size, sequence_length, features).
            msc: the mean seasonal cycle, mus be passed during training
            s: the static features with shape (batch_size, features)

        Returns:
            the predictions of type `ModelReturnPattern`, each with shape (batch_size, sequence_length, num_outputs).
        """

        # (B, H, S, FH) -> (B, FD*, S)
        out = self.encode_hourly(x)

        if s is not None:
            #  (B, FS) -> (B, FS*)
            s = self.encode_static(s)
            #  (B, FS) -> (B, FD*, FD*)
            s = self.static_to_square(s)

            # (B, FD*, S), (B, FD*, FD*) -> (B, FD*, S)
            out = self.bmm((s, out))

        # (B, FD*, S) -> (B, D, S)
        out = self.tcn(out)

        # (B, D, S) -> (B, O, S)
        out = self.output_layer(out)

        # (B, O, S) -> (B, S, O)
        out = self.to_channel_last(out)

        if self.num_out > 1:
            # (B, S, O) -> O x (B, S, 1)
            out = self.split_outputs(out)

        if self.predict_msc:
            if self.predict_var:
                msc, msc_var, ano, ano_var = out
                msc_var = self.softplus(msc_var)
                ano_var = self.softplus(ano_var)
            else:
                msc, ano = out
                msc_var = None
                ano_var = None
        else:
            if self.predict_var:
                ano, ano_var = out
                msc = None
                msc_var = None
                ano_var = self.softplus(ano_var)
            else:
                ano = out
                ano_var = None
                msc = None
                msc_var = None

        daily_out = VarStackPattern(
            ts=None,
            msc=msc,
            ano=ano,
            ts_var=None,
            msc_var=msc_var,
            ano_var=ano_var
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
