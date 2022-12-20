
import torch
from torch.autograd import Function
from torch import nn
from torch import cuda
from torch import Tensor

from prettytable import PrettyTable

import numpy as np

from typing import Union, Optional, Any, Callable


def get_activation(activation: Union[str, None]) -> nn.Module:
    """Get PyTorch activation function by string query.

    Args:
        activation (str, None): activation function, one of `relu`, `leakyrelu`, `selu`, `sigmoid`, `softplus`,
            `tanh`, `identity` (aka `linear` or `none`). If `None` is passed, `None` is returned.

    Raises:
        ValueError: if activation not found.

    Returns:
        PyTorch activation or None: activation function
    """

    if activation is None:
        return nn.Identity()

    a = activation.lower()

    if a == 'linear' or a == 'none':
        a = 'identity'

    activations = dict(
        relu=nn.ReLU,
        leakyrelu=nn.LeakyReLU,
        selu=nn.SELU,
        sigmoid=nn.Sigmoid,
        softplus=nn.Softplus,
        tanh=nn.Tanh,
        identity=nn.Identity
    )

    if a in activations:
        return activations[a]()
    else:
        choices = ', '.join(list(activations.keys()))
        raise ValueError(
            f'activation `{activation}` not found, chose one of: {choices}.'
        )


def is_flat(x: Tensor) -> bool:
    """Checks if tensor is in the flat format (batch, sequence x num_features)

    Args:
        x (Tensor): the tensor.
    """
    return x.ndim == 2


def is_sequence(x: Tensor) -> bool:
    """Checks if tensor is in the sequence format (batch, sequence, num_features)

    Args:
        x (Tensor): the tensor.
    """
    return x.ndim == 3


def seq2flat(x: Tensor) -> Tensor:
    """Reshapes tensor from sequence format to flat format.

    Sequence format: (batch, sequence, features)
    Flat format: (batch, sequence x features)

    Args:
        x (Tensor): a tensor in the sequence format (batch, sequence, features).

    Returns:
        Tensor: the transformed tensor in flat format (batch, sequence x features).
    """

    if not is_sequence(x):
        raise ValueError(
            'attempt to reshape tensor from sequence format to flat format failed. ',
            f'Excepted input tensor with 3 dimensions, got {x.ndim}.'
        )

    return x.flatten(start_dim=1)


def flat2seq(x: Tensor, num_features: int) -> Tensor:
    """Reshapes tensor from flat format to sequence format.

    Flat format: (batch, sequence x features)
    Sequence format: (batch, sequence, features)

    Args:
        x (Tensor): a tensor in the flat format (batch, sequence x features).
        num_features (int): number of features (last dimension) of the output tensor.

    Returns:
        Tensor: the transformed tensor in sequence format (batch, seq, features).
    """

    if not is_flat(x):
        raise ValueError(
            'attempt to reshape tensor from flat format to sequence format failed. ',
            f'Excepted input tensor with 2 dimensions, got {x.ndim}.'
        )

    return x.view(x.shape[0], -1, num_features)


class Transform(nn.Module):
    def __init__(self, transform_fun: Callable) -> None:
        """Transform layer, applies `transform_fun`.

        Example:
            >>> import torch
            >>> reshape_fun = lambda x: x.flatten(start_dim=1)
            >>> flatten_layer = Transform(reshape_fun)
            >>> flatten_layer(torch.ones(2, 3, 4)).shape
            torch.Size([2, 12])

        Args:
            transform_fun (Callable): thetransform function.
        """
        super(Transform, self).__init__()

        self.transform_fun = transform_fun

    def forward(self, x: Tensor) -> Tensor:
        return self.transform_fun(x)


class SelectItem(nn.Module):
    def __init__(self, index: int) -> None:
        """Select an item, may be used in `nn.Sequential` to select outputs from models which return tuples.

        Args:
            item_index (int): the index to select.
        """
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.index = index

    def forward(self, inputs: tuple[Tensor]) -> Tensor:
        """Returns the item at index ``self.index".

        Args:
            inputs (Tuple[Tensor]): the inputs to slect from.
        """
        return inputs[self.index]

    def __repr__(self):
        return f'SelectItem(index={self.index})'


class TimeRepCombine(nn.Module):
    def __init__(self, num_seq: int, num_meta: int) -> None:
        """Stack a sequential and a non-sequential tensor by repeating the latter in time.

        Args:
            num_seq (int): sequence features dimensionality.
            num_meta (int): meta features dimensionality.

        Shapes:
            x: (batch_size, sequence_length, features)
            m: (batch_size, meta_features)
            output: (batch_size, sequence_length, features + meta_features)
        """
        super().__init__()

        self.num_seq = num_seq
        self.num_meta = num_meta
        self.num_out = self.num_seq + self.num_meta

    def forward(self, x: Tensor, m: Tensor) -> Tensor:
        """Stack tensors.

        Args:
            x (Tensor): the sequencial tensor with shape (batch_size, sequence_length, features).
            m (Tensor): meta-feature tensor with shape (batch_size, meta_features)

        Returns:
            Tensor: the output tensor with shape (batch_size, sequence_length, features + meta_features).
        """
        m_rep = m.unsqueeze(1).expand((m.shape[0], x.shape[1], m.shape[1]))
        out = torch.cat((x, m_rep), dim=-1)
        if out.shape[-1] != self.num_out:
            raise RuntimeError(
                f'stacked tensor with shape `{out.shape}` should have last dimension equal to attributes '
                f'num_seq ({self.num_seq}) + num_meta ({self.num_meta}) = {self.num_out}.'
            )

        return out

    def __repr__(self):
        return f'TimeRepCombine(num_seq={self.num_seq}, num_meta={self.num_meta})'


class DotCombine(nn.Module):
    def __init__(self) -> None:
        """Dot product of two tensors using `torch.bmm`.

        Shapes:
            x: (batch_size, n, m)
            m: (batch_size, m, p)
            output: (batch_size, n, p)
        """
        super().__init__()

    def forward(self, x: Tensor, m: Tensor) -> Tensor:
        """Dot product.

        Args:
            x (Tensor): the sequencial tensor with shape (batch_size, n, m).
            m (Tensor): meta-feature tensor with shape (batch_size, m, p).

        Returns:
            Tensor: the output tensor with shape batch_size, n, p).
        """
        out = torch.bmm(x, m)

        return out

    def __repr__(self):
        return 'DotCombine()'


def torch2numpy(x: Union[Tensor, dict[str, Tensor]]) -> Union[np.ndarray, dict[str, np.ndarray]]:
    """Detach and convert pytorch tensors or dicts of tensors to numpy.
    Parameters
    ----------
    x
        A tensor or a dict of tensors to detach and convert to numpy.
    Returns
    ----------
    detached: numpy array or dict of numpy array (same as input).
    """
    if isinstance(x, dict):
        r = {}
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                if v.device.type == 'cuda':
                    r.update({k: v.detach().cpu().numpy()})
                else:
                    r.update({k: v.detach().numpy()})
            else:
                r.update({k: v})
        return r
    elif x is None:
        return x
    else:
        if x.device.type == 'cuda':
            return x.detach().cpu().numpy()
        else:
            return x.detach().numpy()


class NaNLoss(nn.Module):
    def __init__(self, mode: str = 'L2', reduction: str = 'element'):
        """L1 or L2 loss that ignores NaN in target.

        Shapes:
            input:
                    (batch, ...)
            target:
                    (batch, ...)
            weights, optional:
                if reduction == element:
                    (batch, ...)
                if reduction == batch:
                    (batch, )

            *Note: ... is arbitrary but the consistent between Tensors.


        Args:
            mode (ste):
                Either `L1` for the L1 loss (MAE) or `L2` for the L2 loss (MSE). Default
                is `L2`.
            reduction (str):
                Either `element` for equal weighting of each elements or `batch` for
                equal weighting of each batch element. Default is `element`.

        Returns:
            Tensor: the loss.
        """
        super().__init__()

        mode = mode.upper()
        if mode not in ['L1', 'L2']:
            raise ValueError(f'`mode` must be on of (`L1` | `L2`), is `{mode}`.')

        reduction = reduction.lower()

        self.mode = mode
        self.reduction = reduction

        if reduction == 'element':
            self.loss_fn = self.nan_loss
        elif reduction == 'batch':
            self.loss_fn = self.nan_batch_loss
        else:
            if reduction not in ['element', 'batch']:
                raise ValueError(f'`reduction` must be on of (`element` | `batch`), is `{reduction}`.')

    def forward(self, input: Tensor, target: Tensor, weights: Optional[Tensor] = None):
        """Returns the loss."""

        return self.loss_fn(input, target, weights)

    def nan_loss(self, input: Tensor, target: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        """L1 or L2 loss that ignores NaN in target.

        Args:
            input (Tensor):
                The input tensor.
            target (Tensor):
                The target tensor, same shape as input.
            weights (Tensor, optional):
                Optional weights for each batch element, must have length of first
                dimension of input and target. The tensor will be used to weight
                batch elements (larger=more weight).

        Returns:
            Tensor: the loss.
        """

        mask = torch.isnan(target)

        if self.mode == 'L1':
            err = (input[~mask] - target[~mask]).abs()
        else:
            err = (input[~mask] - target[~mask]) ** 2

        if weights is not None:
            weights = weights[(..., ) + (None, ) * (target.ndim - 1)].expand(target.shape)
            weights = weights[~mask]
            err = err * weights
            loss = err.sum() / weights.sum()
        else:
            loss = err.mean()

        return loss

    def nan_batch_loss(self, input: Tensor, target: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        """L1 or L2 loss that ignores NaN in target. Batch elements are weighted equally.

        Args:
            input (Tensor):
                The input tensor.
            target (Tensor):
                The target tensor, same shape as input.
            weights (Tensor, optional):
                Optional weights for each batch element, must have length of first
                dimension of input and target. The tensor will be used to weight
                batch elements (larger=more weight).

        Returns:
            Tensor: the loss.
        """

        mask = torch.isfinite(target)
        target = torch.where(mask, target, input)
        if self.mode == 'L1':
            err = (target - input).abs()
        else:
            err = (target - input) ** 2.0
        batch_arr = err.sum(tuple(range(1, err.ndim)))
        batch_numel = (mask * 1.0).sum(tuple(range(1, err.ndim)))
        err = batch_arr / batch_numel
        if weights is not None:
            weights /= weights.sum()
            err = err * weights
            loss = err.sum()
        else:
            loss = err.mean()

        return loss

    def __repr__(self):
        s = f'NaNLoss(mode=`{self.mode}`, reduction=`{self.reduction}`)'
        return s


def nan_mse(input: Tensor, target: Tensor) -> Tensor:
    """Mean squared error (MSE) that ignores NaN in target.

    Args:
        input (Tensor):
            The input tensor.
        target (Tensor):
            The target tensor, same shape as input.

    Returns:
        Tensor: the MSE loss.
    """

    mask = torch.isnan(target)

    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()

    return loss


def nan_mae(input: Tensor, target: Tensor) -> Tensor:
    """Mean absolute error (MAE) that ignores NaN in target.

    Args:
        input (Tensor):
            The input tensor.
        target (Tensor):
            The target tensor, same shape as input.

    Returns:
        Tensor: the MAE loss.
    """
    mask = torch.isnan(target)

    out = (input[~mask] - target[~mask]).abs()
    loss = out.mean()

    return loss


class NoFreeGPUException(Exception):
    """Exception raised when no GPU can be allocated.

    """

    def __init__(self, message="No GPU available."):
        super().__init__(message)
        self.message = message


def get_next_free_gpu(device_list: Optional[list[int]] = None) -> int:
    """Get the next free GPU out of GPUs in `device_list`. Ugly but works :)"""

    if device_list is None:
        device_list = list(range(cuda.device_count()))

    for d_id in device_list:
        if 'no processes are running' not in cuda.list_gpu_processes(d_id):
            continue
        else:
            device = torch.device(f"cuda:{d_id}")
            try:
                torch.ones(1).to(device)
            except RuntimeError as e:
                raise e
            return d_id

    raise NoFreeGPUException


def get_device_list() -> list[int]:
    return [i for i in range(cuda.device_count())]


class AvgPoolTime(nn.Module):
    """Average pool along time dimension.

    Args:
        kernel_size (int):
            The kernel size, i.e., the reduction factor.
        is_seq_last (bool):
            Whether the last dimension are the features (if `False`, default), or the sequence dimension (`True`).

    Shapes:
        x:
            - if not is_seq_last: (batch_size, sequence_length, features)
            - if is_seq_last: (batch_size, features, sequence_length)
        output:
            - if not is_seq_last: (batch_size, sequence_length / kernel_size, features)
            - if is_seq_last: (batch_size, features, sequence_length / kernel_size)
    """
    def __init__(
            self,
            kernel_size: int,
            is_seq_last: bool = False):

        super().__init__()

        self.kernel_size = kernel_size
        self.is_seq_last = is_seq_last
        self.avg_pool_layer = nn.AvgPool1d(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        """Average pool along time dimension.

        Args:
            x (Tensor): the sequencial tensor with shape (batch_size, features, sequence_length) if `is_seq_last`
                else (batch_size, sequence_length, features).

        Returns:
            Tensor: the pooled tensor with shape (batch_size, features, sequence_length / kernel_size) if `is_seq_last`
                else (batch_size, sequence_length / kernel_size, features).
        """
        if not self.is_seq_last:
            x = x.permute(0, 2, 1)

        x = self.avg_pool_layer(x)

        if not self.is_seq_last:
            x = x.permute(0, 2, 1)

        return x

    def __repr__(self):
        return f'AvgPoolTime(kernel_size={self.kernel_size}, is_seq_last={self.is_seq_last})'


class RepeatTime(nn.Module):
    """Repeat elements along time dimensions num_repeat times.

    Args:
        num_repeat (int):
            The number of repetitions per element (if 2, [0, 1] -> [0, 0, 1, 1]).
        is_seq_last (bool):
            Whether the last dimension are the features (if `False`, default), or the sequence dimension (`True`).

    Shapes:
        x:
            - if not is_seq_last: (batch_size, sequence_length, features)
            - if is_seq_last: (batch_size, features, sequence_length)
        output:
            - if not is_seq_last: (batch_size, sequence_length * num_repeat, features)
            - if is_seq_last: (batch_size, features, sequence_length * num_repeat)
    """
    def __init__(
            self,
            num_repeat: int,
            is_seq_last: bool = False):

        super().__init__()

        self.num_repeat = num_repeat
        self.is_seq_last = is_seq_last

    def forward(self, x: Tensor) -> Tensor:
        """Average pool along time dimension.

        Args:
            x (Tensor): the sequencial tensor with shape (batch_size, features, sequence_length) if `is_seq_last`
                else (batch_size, sequence_length, features).

        Returns:
            Tensor: the pooled tensor with shape (batch_size, features, sequence_length / kernel_size) if `is_seq_last`
                else (batch_size, sequence_length / kernel_size, features).
        """

        return torch.repeat_interleave(x, repeats=self.num_repeat, dim=2 if self.is_seq_last else 1)

    def __repr__(self):
        return f'RepeatTime(kernel_size={self.num_repeat}, is_seq_last={self.is_seq_last})'


class GradientReversalFunction(Function):
    """Gradient reversal function.

    From:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)

    Code from: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    """

    @staticmethod
    def forward(ctx, x, lmbda):
        ctx.lmbda = lmbda
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lmbda = ctx.lmbda
        lmbda = grads.new_tensor(lmbda)
        dx = -lmbda * grads
        return dx, None


class GradientReversal(nn.Module):
    """Gradient reversal layer.

    Args:
        lmbda (float): Weight for gradients.

    Returns:
        A tensor with reverted gradients.

    Adaption from: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    """
    def __init__(self, lmbda: float = 1.0):
        super(GradientReversal, self).__init__()
        self.lmbda = lmbda

    def forward(self, x: Tensor):
        return GradientReversalFunction.apply(x, self.lmbda)

    def set_lambda(self, lmbda: float):
        """Set lambda, must be >= 0."""
        self.lmbda = lmbda

    @property
    def lmbda(self):
        return self._lmbda

    @lmbda.setter
    def lmbda(self, value):
        if value < 0:
            raise ValueError(f'lambda cannot be < 0, is {value}.')
        self._lmbda = value


class DropVal(nn.Module):
    """Drop specific values from input tensor.

    The specified value `dropval` is masked and the remaining values are scaled accordingly, as in
    standard dropout.

    Shapes:
        x: any.
        out: same as x.

    Args:
        dropval (float): the value to drop from the input.
        validate_mask (bool): if True (default), a ZeroDivisionError is thrown if all values are dropped, which
            causes division by zero and as a consequence, NaN in the returned tensor.
    """
    def __init__(self, dropval: float, validate_mask: bool = True):
        super().__init__()

        self.dropval = dropval
        self.validate_mask = validate_mask

    def forward(self, x: Tensor) -> Tensor:
        mask = x != self.dropval
        mask_sum = mask.sum()
        if self.validate_mask:
            if mask_sum == 0:
                raise ZeroDivisionError(
                    'all values are dropped, causing division by zero.'
                )

        out = mask * x * (mask.numel() / mask_sum)

        return out


def maybe_detach(x: Any):
    """Detaches tensor if necessary, returns value.

    Args:
        x (Any): Any value, will be detached if it is a tensor with gradients.

    Returns:
        x, same as input but detached.
    """
    if hasattr(x, 'requires_grad'):
        if x.requires_grad:
            x = x.detach()

    return x


def count_parameters(model):
    """Count and print model parameters."""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def nan_mse(input: Tensor, target: Tensor) -> Tensor | None:
    """Mean squared error (MSE) that ignores NaN in target.
    Args:
        input:
            The input tensor.
        target:
            The target tensor, same shape as input.
    Returns:
        The MSE loss. None if `n` is zero.
    """

    mask = torch.isfinite(target)

    se = (input[mask] - target[mask]) ** 2
    sse = se.sum()
    n = mask.sum()

    if n == 0:
        return None
    else:
        return sse / n


def get_worker_id():
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return None
    else:
        return worker_info.id
