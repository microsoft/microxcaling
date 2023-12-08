"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vector_ops import *
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test

f_softmax = F.softmax
LN_2_BF16 = 0.69140625   # ln(2) in bfloat16 precision


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=None, mx_specs=None, name=None):
        dim = dim + len(input.shape) if dim < 0 else dim
        ctx.dim = dim

        ctx.softmax_exp2 = mx_specs.get('softmax_exp2', False)

        input = vec_quantize(input, mx_specs=mx_specs)

        # compute max
        max_data, _ = input.max(dim, keepdim=True)

        # subtraction
        input = vec_sub(input, max_data, mx_specs=mx_specs,
                        round=mx_specs['round'])
        # exponentiation
        if mx_specs.get('softmax_exp2', False):
            output = vec_exp2(input, mx_specs=mx_specs,
                              round=mx_specs['round'])
        else:
            output = vec_exp(input, mx_specs=mx_specs,
                             round=mx_specs['round'])
        # sum
        output_sum = vec_reduce_sum(output, dim, keepdim=True,
                                    mx_specs=mx_specs,
                                    round=mx_specs['round'])
        # divide
        output = vec_div(output, output_sum,
                         mx_specs=mx_specs,
                         round=mx_specs['round'])

        # save context after quantize
        ctx.save_for_backward(output)
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # load context
        output, = ctx.saved_tensors

        grad_output = vec_quantize(grad_output,
                                   mx_specs=ctx.mx_specs,
                                   round=ctx.mx_specs['round'])

        # dot product calculation
        grad_input = vec_mul(grad_output, output,
                             mx_specs=ctx.mx_specs,
                             round=ctx.mx_specs['round'])
        # sum
        grad_input = vec_reduce_sum(grad_input, ctx.dim, keepdim=True,
                                    mx_specs=ctx.mx_specs,
                                    round=ctx.mx_specs['round'])
        # subtraction (dim is broadcasted)
        grad_input = vec_sub(grad_output, grad_input,
                             mx_specs=ctx.mx_specs,
                             round=ctx.mx_specs['round'])

        # elementwise multiplication
        grad_input = vec_mul(output, grad_input,
                             mx_specs=ctx.mx_specs,
                             round=ctx.mx_specs['round'])

        # Adjust for exp2 constant
        if ctx.mx_specs.get('softmax_exp2', False):
            grad_input = vec_mul(grad_input, LN_2_BF16,
                                 mx_specs=ctx.mx_specs,
                                 round=ctx.mx_specs['round'])

        return (grad_input, None, None, None)


def softmax(input, dim=-1, mx_specs=None, name=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_softmax(input, dim=dim)

    mx_specs = apply_mx_specs(mx_specs)
    return SoftmaxFunction.apply(
            input, dim, mx_specs, name)


class Softmax(nn.Softmax):
    def __init__(self, dim=None, mx_specs=None, name=None):
        mx_assert_test(mx_specs)
        self.mx_none = (mx_specs is None)

        self.name = name
        self.mx_specs = apply_mx_specs(mx_specs)
        super(Softmax, self).__init__(dim)

    def apply_mx_specs(self, mx_specs):
        self.mx_specs = mx_specs
        self.mx_none = (mx_specs is None)
        self.mx_specs = apply_mx_specs(mx_specs)

    def append_name(self, postfix):
        self.name += postfix

    def forward(self, input):
        if self.mx_none:
            return super().forward(input)

        return SoftmaxFunction.apply(
                input, self.dim, self.mx_specs, self.name)
