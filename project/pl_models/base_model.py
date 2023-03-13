
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
import logging
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

from project.utils.loss_functions import RegressionLoss
from project.utils.types import ReturnPattern
from project.dataset import BatchPattern
from project.utils.types import VarStackPattern

# Ignore anticipated PL warnings.
warnings.filterwarnings('ignore', '.*infer the indices fetched for your dataloader.*')
warnings.filterwarnings('ignore', '.*You requested to overfit.*')

logger = logging.getLogger('lightning')


class LightningNet(pl.LightningModule):
    """Implements basic training routine."""
    def __init__(
            self,
            **kwargs) -> None:
        """Standard lightning module, should be subclassed.

        Note:
            * This class should take hyperparameters for the training process. Model hyperparameters should be
                handled in the PyTorch module.
            * call 'self.save_hyperparameters()' at the end of subclass `__init__()`.
            * The subclass must implement a `forward` method (see PyTorch doc) which takes the arguments
                `x`, the sequencial input, and `m`, the meta-features.

        Shape:
            The subclass must implement a `forward` method (see PyTorch doc) which takes the arguments `x`, the
            sequencial input:
            * `x`: (B, L, F)
            * return: (B, L, F)
            where B=batch size, L=sequence length, F=number of sequence features.

        Args:
            num_warmup_batches: the number of warmup steps. Does not apply to all
                schedulers (cyclic and onecycle do start at low lr anyway). No warmup is done if `0`, one full
                epoch (gradually increasing per batch) if `auto`. Defaults to `auto`.
            kwargs:
                Do not use kwargs, required as sink for exceeding arguments due to pytorch ligthning's agrparse scheme.
        """

        super().__init__()

        self.loss_nan_counter = 0
        self.acc_loss = 0.0

        self.loss_fn = RegressionLoss('l1', sample_wise=False, beta=0.5)
        self.loss_fn_nll = RegressionLoss('betanll', sample_wise=False, beta=0.5)

    def shared_step(
            self,
            batch: BatchPattern,
            step_type: str,
            batch_idx: int) -> tuple[Tensor, ReturnPattern]:
        """A single training step shared across specialized steps that returns the loss and the predictions.

        Args:
            batch: the bach.
            step_type: the step type (training mode), one of (`train`, `val`, `test`, `pred`).

        Returns:
            Tensor: the predictions, the loss dict.
        """

        if step_type not in ('train', 'val', 'test', 'pred'):
            raise ValueError(f'`step_type` must be one of (`train`, `val`, `test`, `pred`), is {step_type}.')

        daily_out: VarStackPattern
        hourly_out: VarStackPattern
        daily_out, hourly_out = self(
            x=batch.f_hourly,
            x_msc=batch.t_daily_msc,
            x_ano=batch.t_daily_ano,
            time=batch.coords.dayofyear,
            s=batch.f_static
        )
        out, var_means, var_uncertainties = self._check_return_variables(daily_out=daily_out, hourly_out=hourly_out)

        num_cut = batch.coords.num_days[0]

        losses_to_log = {}
        loss = 0.0
        if var_uncertainties is None:
            for var in var_means:
                var_loss = self.loss_fn(
                    input=out[var][:, ..., num_cut:, :],
                    target=getattr(batch, 't_' + var)[:, ..., num_cut:, :]
                )
                loss += var_loss

                losses_to_log.update({f'{step_type}/{var}': var_loss})
        else:
            for var, var_uncertainty in zip(var_means, var_uncertainties):
                var_loss = self.loss_fn_nll(
                    input=out[var][:, ..., num_cut:, :],
                    variance=out[var_uncertainty][:, ..., num_cut:, :],
                    target=getattr(batch, 't_' + var)[:, ..., num_cut:, :],
                )
                loss += var_loss
                losses_to_log.update({f'{step_type}/{var}': var_loss})

        losses_to_log.update({f'{step_type}/loss': loss})

        preds = ReturnPattern(
            daily_ts=daily_out.ts,
            daily_ts_var=daily_out.ts_var,
            daily_msc=daily_out.msc,
            daily_msc_var=daily_out.msc_var,
            daily_ano=daily_out.ano,
            daily_ano_var=daily_out.ano_var,
            hourly_ts=hourly_out.ts,
            hourly_ts_var=hourly_out.ts_var,
            hourly_msc=hourly_out.msc,
            hourly_msc_var=hourly_out.msc_var,
            hourly_ano=hourly_out.ano,
            hourly_ano_var=hourly_out.ano_var,
            coords=batch.coords
        )

        if step_type != 'pred':
            self.log_dict(
                losses_to_log,
                prog_bar=True,
                on_step=True if step_type == 'train' else False,
                on_epoch=True,
                batch_size=batch.f_hourly.shape[0]
            )

        if step_type == 'train' and batch_idx % 100 == 0:
            self.log_img('train/preds', batch, preds)
        elif step_type == 'val' and batch_idx == 0:
            self.log_img('val/preds', batch, preds)

        return loss, preds

    def training_step(
            self,
            batch: BatchPattern,
            batch_idx: int) -> Tensor:
        """A single training step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        Returns:
            Tensor: The batch loss.
        """

        loss, _ = self.shared_step(batch, step_type='train', batch_idx=batch_idx)

        # If loss is NaN or None (see MTloss), try 3 times with new batch.
        if loss is None:
            self.loss_nan_counter += 1
            loss = None
            if self.loss_nan_counter <= 3:
                logger.warning(
                    f' Training loss is NaN or None. Try {3 - self.loss_nan_counter} more times with next batch.'
                )
            else:
                raise RuntimeError(
                    'NaN or None encountered in training loss with four consecutive batches.'
                )

        return loss

    def validation_step(
            self,
            batch: BatchPattern,
            batch_idx: int) -> dict[str, Tensor]:
        """A single validation step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        """

        loss, _ = self.shared_step(batch, step_type='val', batch_idx=batch_idx)

        return {'val_loss': loss}

    def test_step(
            self,
            batch: BatchPattern,
            batch_idx: int) -> dict[str, Tensor]:
        """A single test step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        """

        loss, _ = self.shared_step(batch, step_type='test', batch_idx=batch_idx)

        return {'test_loss': loss}

    def predict_step(
            self,
            batch: BatchPattern,
            batch_idx: int,
            dataloader_idx: int = 0) -> ReturnPattern:
        """A single predict step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        """

        _, preds = self.shared_step(batch, step_type='pred', batch_idx=batch_idx)

        return preds

    def on_train_start(self) -> None:

        os.makedirs(self.logger.log_dir, exist_ok=True)
        with open(os.path.join(self.logger.log_dir, 'model_summary.txt'), 'w') as f:
            f.write(self.summarize())

        return super().on_train_start()

    @staticmethod
    def plot_preds(batch: BatchPattern, preds: ReturnPattern):

        num_vars = 0
        for var in ['daily', 'hourly']:
            for scale in ['ts', 'msc', 'ano']:
                scale_name = f'{var}_{scale}'
                if getattr(preds, scale_name) is not None:
                    num_vars += 1

        fig, axes = plt.subplots(num_vars, 1, figsize=(8, num_vars), dpi=180, sharex=False, squeeze=False)

        ax_i = -1
        for var in ['daily', 'hourly']:
            for scale in ['ts', 'msc', 'ano']:
                scale_name = f'{var}_{scale}'
                scale_name_var = f'{var}_{scale}_var'

                pred_mean = getattr(preds, scale_name)
                if pred_mean is None:
                    continue

                ax_i += 1
                ax = axes[ax_i, 0]

                pred_mean = pred_mean[0, ..., 0].detach().cpu()
                if var == 'hourly':
                    pred_mean = pred_mean.flatten()    

                t_range = range(len(pred_mean))

                pred_var = getattr(preds, scale_name_var)
                if pred_var is not None:
                    pred_var = pred_var[0, ..., 0].detach().cpu()         
                    if var == 'hourly':
                        pred_var = pred_var.flatten()           


                obs = getattr(batch, 't_' + scale_name)[0, ..., 0].cpu()

                if var == 'hourly':
                    obs = obs.flatten()

                if pred_var is not None:
                    ax.fill_between(
                        t_range,
                        pred_mean - pred_var,
                        pred_mean + pred_var,
                        alpha=0.2,
                        color='k',
                        label='var',
                        edgecolor='none'
                    )

                y_min = np.min(
                    (np.nanmin(obs), np.nanmin(pred_mean))
                )
                y_max = np.max(
                    (np.nanmax(obs), np.nanmax(pred_mean))
                )

                extra_space = (y_max - y_min) * 0.1
                y_min -= extra_space
                y_max += extra_space

                ax.set_ylim(y_min, y_max)

                ax.plot(t_range, pred_mean, color='k', lw=0.8, label='pred')
                ax.plot(t_range, obs, color='tab:red', lw=1.0, label='obs', alpha=0.8)

                ax.set_ylabel(f'{var} ({scale})')

        for ax in axes.flat:
            ax.spines[['right', 'top', 'bottom']].set_visible(False)
            ax.axes.get_xaxis().set_visible(False)

        axes[0, 0].legend(loc=2)
        fig.align_ylabels(axes[:, 0])
        plt.tight_layout()

        return fig

    def log_img(self, name: str, batch: BatchPattern, preds: ReturnPattern):
        tensorboard = self.logger.experiment
        tensorboard.add_figure(name, self.plot_preds(batch, preds), self.trainer.global_step, close=True)

    def summarize(self):
        s = f'=== Summary {"=" * 31}\n'
        s += f'{str(ModelSummary(self))}\n\n'
        s += f'=== Model {"=" * 33}\n'
        s += f'{str(self)}'

        return s

    def _check_return_variables(self, daily_out: VarStackPattern, hourly_out: VarStackPattern) -> tuple[dict, list[str], list[str] | None]:
        daily_out = {'daily_' + key: val for key, val in daily_out._asdict().items()}
        hourly_out = {'hourly_' + key: val for key, val in hourly_out._asdict().items()}
        out = dict(**daily_out, **hourly_out)

        vars_with_uncertainty = []
        vars_without_uncertainty = []

        for key, var in out.items():
            if '_var' in key:
                base_var = key.removesuffix('_var')
                if var is not None and out[base_var] is None:
                    raise AssertionError(
                        f'return variable contains variance \'{key}\' but mean \'{base_var}\' is `None`.'
                    )
            else:
                if out[key] is not None:  
                    if out[key + '_var'] is None:
                        vars_without_uncertainty.append(key)
                    else:
                        vars_with_uncertainty.append(key)

        if int(len(vars_with_uncertainty) > 0) + int(len(vars_without_uncertainty) > 0) == 2:
            raise AssertionError(
                f'there are variables with variance ({vars_with_uncertainty}) and such without ({vars_without_uncertainty}), '
                'which is not allowed. Either return variances for all variables or for none!'
            )

        has_var = len(vars_with_uncertainty) > 0

        if has_var:
            var_means = vars_with_uncertainty
            var_uncertainties = [var + '_var' for var in vars_with_uncertainty]
        else:
            var_means = vars_without_uncertainty
            var_uncertainties = None

        return out, var_means, var_uncertainties
