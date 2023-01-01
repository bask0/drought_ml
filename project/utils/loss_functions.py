

import torch
from torch import nn
from torch import Tensor


class BetaNLLLoss(nn.Module):
    r"""Creates beta-NLL criterion

    The beta-NLL criterion extends the standard Gaussian NLL criterion. The beta
    parameter specifies a weight fo the errors from 0 to 1, 0 yielding standard NLL
    criterion, and 1 weighing as in MSE (but still with uncertainty). This solves a
    problem with standard NLL, which tends to emphasize regions with low variance
    during trainging. See https://arxiv.org/pdf/2203.09168.pdf

    Notes:
        When beta is set to 0, this loss is equivalent to the NLL loss. Beta of 1 corresponds
        to equal weighting of low and high error points.

    Args:
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        beta: Specifies the weighting of errors. 0 means standard NLL is used, while 1 means that
            they are weighted equally as in MSE. The paper suggests a value of 0.5 (default).

    Shape:
        - mean: (batch_size, ...).
        - variance: (batch_size, ...), same shape as the mean.
        - target: (batch_size, ...), same shape as the mean.
        - output: scalar, or, if `reduction` is ``'none'``, then (batch_size, ...), same shape as the input.
    """

    def __init__(self, reduction: str = 'mean', beta: float = 0.5) -> None:
        super().__init__()

        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(
                f'argument `reduction` must be one of (\'none\', \'mean\', \'sum\'), is {reduction}.'
            )

        if not (0 <= beta <= 1):
            raise ValueError(
                f'`beta` must be a value in the range (0, 1), is {beta}.'
            )

        self.reduction = reduction
        self.beta = beta

    def forward(self, mean: Tensor, variance: Tensor, target: Tensor) -> Tensor:
        """Compute beta-NLL loss (https://arxiv.org/pdf/2203.09168.pdf)

        Args:
            mean: Predicted mean of shape B x D
            variance: Predicted variance of shape B x D
            target: Target of shape B x D
            beta: Parameter from range [0, 1] controlling relative weighting between data points,
                where `0` corresponds to high weight on low error points and `1` to an equal weighting.

        Returns:
            Loss per batch element of shape B
        """

        loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

        if self.beta > 0.0:
            loss = loss * variance.detach() ** self.beta

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class RegressionLoss(nn.Module):
    """Loss functions that ignore NaN in target.
    The loss funtion allows for having missing / non-finite values in the target.

    Example
    -------
    >>> import torch
    >>> mse_loss = RegressionLoss(sample_wise=True)
    >>> input = torch.ones(2, 2, requires_grad=True)
    >>> target = torch.ones(2, 2, requires_grad=False) + 1.
    >>> target[0, 0] = float('NaN')
    >>> loss = mse_loss(input, target)
    >>> loss.backward()
    >>> print('input:', input)
    input:
     tensor([[1., 1.],
             [1., 1.]], requires_grad=True)
    >>> print('target:', target)
    target:
     tensor([[nan, 2.],
             [2., 2.]])
    >>> print('mse:', loss)
    mse:
     tensor(1., grad_fn=<MeanBackward0>)
    >>> print('gradients:', input.grad)
    gradients:
     tensor([[ 0.0000, -1.0000],
             [-0.5000, -0.5000]])

    Shape
    -----
    * input: (N, *), where * means, any number of additional dimensions
    * target: (N, *), same shape as the input
    """
    def __init__(self, criterion: str = 'l1', sample_wise: bool = True, beta: float = None) -> None:
        """Initialize RegressionLoss.
        
        Args:
            criterion : str (``'l1'`` | ``'l2'`` | ``'huber'`` | ``'betanll'``)
                ``'l1'`` for Mean Absolute Error (MAE),
                ``'l2'`` for Mean Squared Error (MSE),
                ``'huber'`` for Huber loss (mix of l1 and l2),
                ``'betanll'`` beta-weighted negative log likelihood (NLL).
            sample_wise : bool
                Whether to calculate sample-wise loss first and average then (`True`, default) or to
                calculate the loss across all elements. The former weights each batch element equally,
                the latter weights each observation equally. This is relevant especially with many NaN
                in the target tensor, while there is no difference without NaN.
            beta: Parameter from range [0, 1] controlling relative weighting between data points with
                the ``'betanll'`` criterion, where `0` corresponds to high weight on low error points
                and `1` to an equal weighting. A value of 0.5 has been reported to work best in general.
                This parameter must be set with `criterion`=``'betanll'``, but cannot be set with other modes.

        """

        super(RegressionLoss, self).__init__()

        criterion = criterion.lower()

        if criterion not in ('l1', 'l2', 'huber', 'betanll'):
            raise ValueError(
                f'argument `criterion` must be one of (\'l1\' | \'l2\', | \'huber\' | \'betanll\'), is {criterion}.'
            )

        if criterion == 'betanll':
            if beta is None:
                raise ValueError(
                    'parameter `beta` required with `criterion`=\'betanll\'.'
                )
        else:
            if beta is not None:
                raise ValueError(
                    f'parameter `beta` cannot be passed with `criterion`=\'{criterion}\'.'
                )

        self.criterion = criterion
        self.sample_wise = sample_wise
        self.loss_fn = {
            'l1': nn.L1Loss,
            'l2': nn.MSELoss,
            'huber': nn.HuberLoss,
            'betanll': BetaNLLLoss,
        }[self.criterion](reduction='none')

    def forward(self, input: Tensor, target: Tensor, variance: Tensor | None = None) -> Tensor:
        """Forward call, calculate loss from input and target, must have same shape.

        Args:
            input: Predicted mean of shape (B x ...)
            target: Target of shape (B x ...)
            variance: Predicted variance of shape (B x ...). Only needed / allowed
                with `criterion`=``'betanll'``

        Returns:
            The loss, a scalar.
        """

        if self.criterion == 'betanll':
            if variance is None:
                raise ValueError(
                    'argument `variance` required with `criterion`=\'betanll\'.'
                )
        else:
            if variance is not None:
                raise ValueError(
                    f'argument `variance` not allowed with `criterion`=\'{self.criterion}\'.'
                )
        mask = target.isfinite()
        # By setting target to input for NaNs, we set gradients to zero.
        target = target.where(mask, input)

        if self.criterion == 'betanll':
            # By setting variance to a large value for NaNs, we set gradients to (almost) zero.
            variance = variance.where(mask, torch.tensor(1e30, dtype=variance.dtype, device=variance.device))
            element_error = self.loss_fn(mean=input, variance=variance, target=target)
        else:
            element_error = self.loss_fn(input, target)

        if self.sample_wise:
            red_dims = tuple(range(1, element_error.ndim))
            batch_error = element_error.sum(red_dims) / mask.sum(red_dims)
            err = batch_error.mean()

        else:
            err = element_error.sum() / mask.sum()

        return err
