"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch
import torch.nn.functional as F

from .vector_ops import *
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test
from .norm_utils import _norm_forward, _norm_backward

f_batch_norm = F.batch_norm

class BatchNormFunction(torch.autograd.Function):
    """ Batch Normalization applied over N-dimensional input.

            y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

        If is_training==True:
            mean(x) and var(x) are calculated over the minibatch dim
            (outermost dim). Exponential moving averages of mean(x)
            and var(x) are tracked via running_mean and running_var.

        If is_training==False:
            Use running_mean and running_var for mean(x) and var(x).

        Running stats are updated as:

            running_x = (1 - momentum)*running_x + momentum*x

        Args:
            x:              input tensor (batch, channels, ...)
            running_mean:   tensor shaped (channels), can be None
            running_var:    tensor shaped (channels), can be None
            is_training:    If True, calculate the per-batch stats
                            and update running_mean and running_var if
                            they are not None.
                            If False, use running_mean and running_var
                            (they can't be None in this case).

        We currently assume that H is the second outermost
        dim. I.e., the memory layout is NCHW.
    """
    @staticmethod
    def forward(ctx, x, running_mean, running_var,
                weight, bias, is_training=False,
                momentum=0.1, eps=1e-5,
                mx_specs=None):

        # For training, BN uses per-batch statistics
        # For inference, BN uses running mean and var
        if not is_training:
            assert(running_mean is not None)
            assert(running_var is not None)

        ctx.is_training = is_training
        ctx.momentum = momentum
        ctx.eps = eps

        x = vec_quantize(x, mx_specs=mx_specs)
        bf_weight = vec_quantize(weight, mx_specs=mx_specs)
        bf_bias = vec_quantize(bias, mx_specs=mx_specs)

        H = x.shape[1]
        sum_axes = [0] + list(range(2, x.ndim))

        output, x_shift, x_norm, x_std_inv, x_mean, x_var = \
                _norm_forward(
                        x, sum_axes, bf_weight, bf_bias, eps,
                        mx_specs,
                        weight_axis=1,
                        use_running_stats=not is_training,
                        running_mean=running_mean,
                        running_var=running_var)

        # Update running statistics, in-place
        if is_training and running_mean is not None:
            t1 = vec_mul((1 - momentum), running_mean,
                         mx_specs=mx_specs)
            t2 = vec_mul(momentum, x_mean.view(H),
                         mx_specs=mx_specs)
            t3 = vec_add(t1, t2, mx_specs=mx_specs)
            running_mean.copy_(t3)
        if is_training and running_var is not None:
            t1 = vec_mul((1 - momentum), running_var,
                         mx_specs=mx_specs)
            t2 = vec_mul(momentum, x_var.view(H),
                         mx_specs=mx_specs)
            t3 = vec_add(t1, t2, mx_specs=mx_specs)
            running_var.copy_(t3)

        # Stash for backprop
        if mx_specs['quantize_backprop']:
            ctx.save_for_backward(x_shift, x_norm, x_std_inv, bf_weight)
        else:
            ctx.save_for_backward(x_shift, x_norm, x_std_inv, weight)

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        H = grad_output.shape[1]
        sum_axes = [0] +  list(range(2, grad_output.ndim))

        x_shift, x_norm, x_std_inv, weight = ctx.saved_tensors

        grad_output = vec_quantize(grad_output, mx_specs=ctx.mx_specs)

        # grad_bias, sum over all axis except H
        grad_bias = vec_reduce_sum(grad_output, sum_axes,
                                   mx_specs=ctx.mx_specs)

        # grad_weight, sum over all axis except H
        grad_weight = vec_mul(grad_output, x_norm, mx_specs=ctx.mx_specs)

        grad_weight = vec_reduce_sum(grad_weight, sum_axes,
                                     mx_specs=ctx.mx_specs)

        grad_input = _norm_backward(
                grad_output, sum_axes, weight, x_shift,
                x_std_inv, ctx.mx_specs, weight_axis=1)

        return (grad_input, None, None,
                grad_weight, grad_bias,
                None, None, None, None)


def batch_norm(x, running_mean, running_var,
               weight, bias, is_training=False,
               momentum=0.1, eps=1e-5,
               mx_specs=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_batch_norm(
                x, running_mean, running_var, weight, bias,
                is_training, momentum, eps)

    mx_specs = apply_mx_specs(mx_specs)
    return BatchNormFunction.apply(
            x, running_mean, running_var, weight, bias,
            is_training, momentum, eps, mx_specs)


class _BatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, mx_specs=None,
                 name=None, **kwargs):

        # Early pytorch did not have device/dtype args
        try:
            super(_BatchNorm, self).__init__(
                num_features, **kwargs)
        except TypeError:
            device = kwargs.pop('device')
            dtype = kwargs.pop('dtype')

            super(_BatchNorm, self).__init__(
                num_features, **kwargs)

            self.to(device)
            self.to(dtype)

        mx_assert_test(mx_specs)
        self.mx_none = (mx_specs is None)

        self.mx_specs = apply_mx_specs(mx_specs)
        self.name = name

    def forward(self, input):
        if self.mx_none:
            return super().forward(input)

        self._check_input_dim(input)

        if self.momentum is None:
             exponential_average_factor = 0.0
        else:
             exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1

                if self.momentum is None:
                    # use cumulative moving avg
                    exponential_average_factor = \
                            1.0 / float(self.num_batches_tracked)
                else:
                    # use exponential moving average
                    exponential_average_factor = self.momentum

        # If training, use mini-batch stats
        # If eval use mini-batch stats if buffers are None
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and \
                          (self.running_var is None)

        output = batch_norm(
            input,
            self.running_mean \
                    if not self.training or self.track_running_stats \
                    else None,
            self.running_var \
                    if not self.training or self.track_running_stats \
                    else None,
            self.weight, self.bias,
            is_training=bn_training,
            momentum=exponential_average_factor,
            eps=self.eps,
            mx_specs=self.mx_specs)
        return output


class BatchNorm1d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class BatchNorm3d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
