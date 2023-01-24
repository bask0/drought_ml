
from torch import nn
from torch import Tensor

from project.pl_models.base_model import LightningNet
from modules.lstm import LSTM
from modules.feedforward import FeedForward
from utils.torch_utils import Transform, EncodeHourlyToDaily


class LSTMPL(LightningNet):
    def __init__(
            self,
            num_inputs: int,
            num_geofactors: int,
            num_outputs: int,
            num_hidden: int,
            num_layers: int,
            dropout: float,
            static_dropout: float,
            num_meteo_enc: int = 10,
            num_geofactors_enc: int = 6,
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
            outpout_channel_last=True
        )

        self.lstm = LSTM(
            num_inputs=num_meteo_enc,
            num_static_inputs=num_geofactors_enc,
            num_outputs=-1,
            num_hidden=num_hidden,
            num_layers=num_layers,
            dropout=dropout
        )

        self.mean_output_layer = nn.Linear(
            in_features=num_hidden,
            out_features=num_outputs
        )

        if self.predict_var:
            self.var_output_layer = nn.Sequential(
                nn.Linear(
                    in_features=num_hidden,
                    out_features=num_outputs
                ),
                nn.Softplus()
            )

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

        # (B, S, FD*), (B, FS*) -> (B, S, D)
        out, _ = self.lstm(out, s)

        # (B, S, D) -> (B, S, O)
        out_mean = self.mean_output_layer(out)

        if self.predict_var:
            # (B, D, HxS) -> (B, S, O)
            out_var = self.var_output_layer(out)
        else:
            out_var = None

        return out_mean, out_var
