
from torch import nn
from torch import Tensor

from project.pl_models.base_model import LightningNet
from project.modules.lstm import LSTM
from project.modules.feedforward import FeedForward
from project.utils.torch_utils import EncodeHourlyToDaily, Transform
from project.utils.types import VarStackPattern


class LSTMPL(LightningNet):
    def __init__(
            self,
            num_inputs: int,
            num_geofactors: int,
            num_hidden: int,
            num_layers: int,
            dropout: float,
            static_dropout: float,
            num_meteo_enc: int = 10,
            num_geofactors_enc: int = 3,
            predict_msc: bool = False,
            predict_var: bool = False,
            **kwargs) -> None:
        """Implements a long short-term memory model (LSTM).

        Note:
            The LSTM layers are followed by a feedfoward layer to map the output channels to `num_outputs`.

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
            outpout_channel_last=True
        )

        self.num_out = (int(predict_var) + 1) * (int(predict_msc) + 1)
        num_hidden_lstm = num_hidden * self.num_out

        self.lstm = LSTM(
            num_inputs=num_meteo_enc,
            num_static_inputs=num_geofactors_enc,
            num_outputs=-1,
            num_hidden=num_hidden_lstm,
            num_layers=num_layers,
            dropout=0.0
        )

        self.to_sequence_last = Transform(transform_fun=lambda x: x.transpose(1, 2))

        self.output_layer = nn.Conv1d(
            in_channels=num_hidden_lstm,
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
            self, x: Tensor,
            s: Tensor | None = None) -> tuple[VarStackPattern, VarStackPattern]:
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

        # (B, S, FD*), (B, FS*) -> (B, S, D)
        out, _ = self.lstm(out, s)

        # (B, S, D) -> (B, D, S)
        out = self.to_sequence_last(out)

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
