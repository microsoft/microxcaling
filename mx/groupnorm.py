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

f_group_norm = F.group_norm

class GroupNormFunction(torch.autograd.Function):
    """ input is (N, C, ...) where C is channels
        Split channels into num_groups and normalize each group
        separately. Average over channels and spatial dims per group.
        output is (N, C, ...)
    """
    @staticmethod
    def forward(ctx, x, num_groups, weight, bias, eps,
                mx_specs=None, name=None):
        ctx.num_groups = num_groups
        ctx.eps = eps
        ctx.name = name

        x = vec_quantize(x, mx_specs=mx_specs)
        bf_weight = vec_quantize(weight, mx_specs=mx_specs)
        bf_bias = vec_quantize(bias, mx_specs=mx_specs)

        sum_axes = list(range(1, x.ndim))

        output, x_shift, x_norm, x_std_inv, _, _ = \
                _norm_forward(
                        x, sum_axes, bf_weight, bf_bias, eps,
                        mx_specs,
                        groups=num_groups,
                        weight_axis=1)

        # stash for backprop
        if mx_specs['quantize_backprop']:
            ctx.save_for_backward(x_shift, x_norm, x_std_inv, bf_weight)
        else:
            ctx.save_for_backward(x_shift, x_norm, x_std_inv, weight)

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        sum_axes = [0] + list(range(2, grad_output.ndim))
        # get stashed intermediate
        x_shift, x_norm, x_std_inv, weight = ctx.saved_tensors

        grad_output = vec_quantize(grad_output, mx_specs=ctx.mx_specs)

        # grad_bias
        grad_bias = vec_reduce_sum(grad_output, sum_axes,
                                   mx_specs=ctx.mx_specs)

        # grad_weight
        grad_weight = vec_mul(grad_output, x_norm, mx_specs=ctx.mx_specs)

        grad_weight = vec_reduce_sum(grad_weight, sum_axes,
                                     mx_specs=ctx.mx_specs)

        grad_input = _norm_backward(
                grad_output, list(range(1, grad_output.ndim)),
                weight, x_shift,
                x_std_inv, ctx.mx_specs,
                groups=ctx.num_groups,
                weight_axis=1)

        return (grad_input, None, grad_weight, grad_bias,
                None, None, None)


def group_norm(x, num_groups, weight, bias, eps=1e-5,
               mx_specs=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_group_norm(
                x, num_groups, weight=weight, bias=bias, eps=eps)

    mx_specs = apply_mx_specs(mx_specs)
    return GroupNormFunction.apply(
            x, num_groups, weight, bias, eps, mx_specs)


class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_groups, num_channels, 
                 mx_specs=None, name=None, **kwargs):
        try:
            super().__init__(num_groups, num_channels, **kwargs)
        except TypeError:
            device = kwargs.pop('device')
            dtype = kwargs.pop('dtype')

            super().__init__(num_groups, num_channels, **kwargs)

            self.to(device)
            self.to(dtype)

        mx_assert_test(mx_specs)
        self.mx_none = (mx_specs is None)

        self.name = name
        self.mx_specs = apply_mx_specs(mx_specs)

    def forward(self, input):
        if self.mx_none:
            return super().forward(input)

        return group_norm(input, self.num_groups,
                          self.weight, self.bias, self.eps,
                          self.mx_specs)

