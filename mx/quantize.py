"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch
import numpy as np

from .elemwise_ops import quantize_elemwise_op
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test


def quantize_bfloat(x, mx_specs, round=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return x

    mx_specs = apply_mx_specs(mx_specs)

    if round is None:
        round = mx_specs["round"]

    return QuantizeBfloatFunction.apply(x, mx_specs, round)


class QuantizeBfloatFunction(torch.autograd.Function):
    """Forward pass: quantize to bfloat
    Backward pass: quantize to bfloat
    """

    @staticmethod
    def forward(ctx, x, mx_specs, round=None):
        if round is None:
            round = mx_specs["round"]

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        ctx.round = round

        return quantize_elemwise_op(x, mx_specs=mx_specs, round=round)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = quantize_elemwise_op(
            grad_output, mx_specs=ctx.mx_specs, round=ctx.round,
        )

        return (grad_input, None, None)
