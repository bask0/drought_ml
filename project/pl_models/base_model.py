
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
import logging
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

from project.utils.loss_functions import RegressionLoss
from utils.types import ReturnPattern
from project.dataset import BatchPattern

# Ignore anticipated PL warnings.
warnings.filterwarnings('ignore', '.*infer the indices fetched for your dataloader.*')

logger = logging.getLogger('lightning')


class LightningNet(pl.LightningModule):
    """Implements basic training routine."""
    def __init__(
            self,
            use_mt_weighting: bool | str | None = False,
            use_n_last: int = 365 + 366,
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
            use_mt_weighting: whether to use multitask weightig. Default is `False`.
                Note that the option to pass a string is currently required due to wandb sweeps.
            num_warmup_batches: the number of warmup steps. Does not apply to all
                schedulers (cyclic and onecycle do start at low lr anyway). No warmup is done if `0`, one full
                epoch (gradually increasing per batch) if `auto`. Defaults to `auto`.
            use_n_last: use `use_n_last` elements of sequence to calculate the loss. Defaults to 366 * 24.
                This asserts a minimum temporal context length even for the first target sequence element.
            kwargs:
                Do not use kwargs, required as sink for exceeding arguments due to pytorch ligthning's agrparse scheme.
        """

        super().__init__()

        if isinstance(use_mt_weighting, str):
            if use_mt_weighting.lower() == 'true':
                self.use_mt_weighting = True
            elif use_mt_weighting.lower() == 'false':
                self.use_mt_weighting = False
            else:
                raise ValueError(
                    f'a string other than \'True\' or \'False\' '
                    f'has passed as argument \'use_mt_weighting\' (\'{use_mt_weighting}\').')
        elif isinstance(use_mt_weighting, bool):
            self.use_mt_weighting = use_mt_weighting
        else:
            raise TypeError(
                f'`use_mt_weighting` must be a strong of a boolean, is {use_mt_weighting}.'
            )

        self.use_n_last = use_n_last

        self.loss_nan_counter = 0
        self.acc_loss = 0.0

        self.loss_fn = RegressionLoss('l1', sample_wise=False, beta=0.5)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Optimizer')
        parser.add_argument(
            '--lr', type=float, default=0.001, help='The learning rate.')
        parser.add_argument(
            '--weight_decay', type=float, default=0.00, help='The weight decay.')
        parser.add_argument(  # Don't change to boolean flag!
            '--use_mt_weighting', type=str, help='\'true\' enables multitask loss weighting, default is \'true\'.', default='false')
        return parent_parser

    def mt_loss_fn(
            self,
            step_type: str,
            preds: Tensor,
            targets: Tensor) -> tuple[Tensor, dict[str, Tensor]]:

        num_preds = preds.shape[-1]
        num_tasks = targets.shape[-1]

        if not (num_preds == num_tasks == self.mt_loss.num_tasks):
            raise RuntimeError(
                f'number of predictions ({num_preds}), number of tasks ({num_tasks}), and number of '
                f'regression tasks ({self.mt_loss.num_tasks}) must be equal.'
            )

        preds_dict = {task: preds[..., t] for t, task in enumerate(self.mt_loss.reg_tasks)}
        tasks_dict = {task: targets[..., t] for t, task in enumerate(self.mt_loss.reg_tasks)}

        loss, loss_dict = self.mt_loss(step_type=step_type, pred=preds_dict, target=tasks_dict)

        return loss, loss_dict

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

        y_mean_hat, y_var_hat = self(batch.f_hourly, batch.f_static)

        # 1-ahead prediction, thus targets are shifted.
        loss = self.loss_fn(
            input=y_mean_hat,
            # variance=y_var_hat[:, self.use_n_last:-1, :],
            target=batch.t_daily
        )

        preds = ReturnPattern(mean_hat=y_mean_hat, var_hat=y_var_hat, coords=batch.coords)

        batch_size = y_mean_hat.shape[0]

        if step_type == 'train':
            self.acc_loss += loss
            if batch_idx % self.trainer.accumulate_grad_batches == 0 and batch_idx != 0:
                self.acc_loss = self.acc_loss / self.trainer.accumulate_grad_batches
                self.log(
                    name='train_loss',
                    value=self.acc_loss,
                    on_step=True,
                    on_epoch=True,
                    batch_size=batch_size
                )
                self.acc_loss = 0.0

            if batch_idx % 100 == 0:
                self.log_img('train_preds', batch, preds)

        elif step_type in ('val', 'test'):
            self.log(
                name=f'{step_type}_loss',
                value=loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size)

            if step_type == 'val' and batch_idx == 0:
                self.log_img('val_preds', batch, preds)

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
            f.write(str(self))

        return super().on_train_start()

    def plot_preds(self, batch: BatchPattern, preds: ReturnPattern):

        num_daily = batch.t_daily.shape[-1]

        fig, axes = plt.subplots(num_daily, 1, figsize=(8, num_daily), dpi=180, sharex=True, squeeze=False)

        for t in range(num_daily):
            obs = batch.t_daily[0, :, t].cpu()
            pred = preds.mean_hat[0, :, t].detach().cpu()
            t_range = range(len(obs))

            ax = axes[t, 0]

            if preds.var_hat is not None:
                var = preds.var_hat[0, :, t].detach().cpu()
                ax.fill_between(
                    t_range,
                    pred - var,
                    pred + var,
                    alpha=0.2,
                    color='k',
                    label='var'
                )

            obs_min = np.nanmin(obs)
            obs_max = np.nanmax(obs)
            extra_space = (obs_max - obs_max) * 0.2
            obs_min -= extra_space
            obs_max += extra_space

            ax.set_ylim(obs_min, obs_max)

            ax.plot(t_range, pred, color='k', lw=0.8, label='pred')
            ax.plot(t_range, obs, color='tab:red', lw=0.5, label='obs', alpha=0.8)

        for ax in axes[:, 0]:
            ax.spines[['right', 'top', 'bottom']].set_visible(False)
            ax.axes.get_xaxis().set_visible(False)

        axes[0, 0].legend(loc=2)
        plt.tight_layout()

        return fig

    def log_img(self, name: str, batch: BatchPattern, preds: ReturnPattern):
        tensorboard = self.logger.experiment
        tensorboard.add_figure(name, self.plot_preds(batch, preds), self.trainer.global_step, close=True)

    def __repr__(self) -> str:
        s = f'=== Summary {"=" * 31}\n'
        s += f'{str(ModelSummary(self))}\n\n'
        s += f'=== Model {"=" * 33}\n'
        s += f'{str(self)}'

        return s
