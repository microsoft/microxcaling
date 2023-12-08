"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Non-differentiable vector operations for use within the MSFP lib.
These should not be exposed to a library user.

Methods:
    vec_quantize    y = q(x)
    vec_add         y = x1 + x2
    vec_sub         y = x1 - x2
    vec_mul         y = x1*x2
    vec_div         y = x1/x2
    vec_exp         y = e^x
    vec_exp2        y = 2^x
    vec_recip       y = 1/x
    vec_sqrt        y = sqrt(x)
    vec_tanh        y = tanh(x)
    vec_reduce_sum      y = x.sum()
    vec_reduce_mean     y = x.mean()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from .elemwise_ops import quantize_elemwise_op

torch_exp = torch.exp
torch_exp2 = torch.exp2
torch_sqrt = torch.sqrt
torch_tanh = torch.tanh
LN_2_EXACT = 0.69314718056
LOG2_E_BF16 = 1.4453125  # 1 + 2**-2 + 2**-3 + 2**-4 + 2**-7


def vec_quantize(input, mx_specs=None, round=None):
    return quantize_elemwise_op(input, mx_specs=mx_specs,
                                round=round)


#-------------------------------------------------------------------------
# Vec regular ops
#-------------------------------------------------------------------------
def vec_add(a, b, mx_specs=None, round=None):
    return quantize_elemwise_op(a + b, mx_specs=mx_specs,
                                round=round)


def vec_sub(a, b, mx_specs=None, round=None):
    return quantize_elemwise_op(a - b, mx_specs=mx_specs,
                                round=round)


def vec_mul(a, b, mx_specs=None, round=None):
    return quantize_elemwise_op(a * b, mx_specs=mx_specs,
                                round=round)


def vec_div(a, b, mx_specs=None, round=None):
    if mx_specs and mx_specs['vec_use_recip']:
        recip_b = vec_recip(b, mx_specs=mx_specs, round=round)
        return vec_mul(a, recip_b, mx_specs=mx_specs, round=round)
    else:
        return quantize_elemwise_op(a / b, mx_specs=mx_specs,
                                    round=round)


#-------------------------------------------------------------------------
# Vec special ops
#-------------------------------------------------------------------------
def vec_exp(input, mx_specs=None, round=None):
    if mx_specs and mx_specs['vec_use_exp2']:
        phi = quantize_elemwise_op(LOG2_E_BF16 * input,
                                   mx_specs=mx_specs, round=round)
        phi = vec_exp2(phi, mx_specs=mx_specs, round=round)
    else:
        phi = quantize_elemwise_op(torch_exp(input),
                                   mx_specs=mx_specs, round=round)
    return phi


def vec_exp2(input, mx_specs=None, round=None):
    # Pytorch 1.2 does not have exp2
    if hasattr(torch, 'exp2'):
        phi = quantize_elemwise_op(torch_exp2(input),
                                   mx_specs=mx_specs, round=round)
    else:
        # Here we're trying to emulate torch.exp2 with torch.exp,
        # so the constant is exact
        phi = quantize_elemwise_op(torch_exp(input * LN_2_EXACT),
                                   mx_specs=mx_specs, round=round)
    return phi


def vec_recip(input, mx_specs=None, round=None):
    return quantize_elemwise_op(1. / input, mx_specs=mx_specs,
                                round=round)


def vec_sqrt(input, mx_specs=None, round=None):
    return quantize_elemwise_op(torch_sqrt(input), mx_specs=mx_specs,
                                round=round)


def vec_tanh(input, mx_specs=None, round=None):
    return quantize_elemwise_op(torch_tanh(input), mx_specs=mx_specs,
                                round=round)


#-------------------------------------------------------------------------
# Vector reduce ops
#-------------------------------------------------------------------------
def vec_reduce_sum(input, dim, keepdim=False, mx_specs=None,
                   round=None):
    return quantize_elemwise_op(input.sum(dim, keepdim=keepdim),
                                mx_specs=mx_specs, round=round)


def vec_reduce_mean(input, dim, keepdim=False, mx_specs=None,
                    round=None):
    # np.prod returns 1.0 for empty list
    dim = dim if type(dim) is list else [dim]
    denom = np.prod([input.shape[i] for i in dim])

    s = vec_reduce_sum(input, dim, keepdim=keepdim,
                       mx_specs=mx_specs, round=round)
    s = vec_div(s, denom, mx_specs=mx_specs, round=round)
    return s
