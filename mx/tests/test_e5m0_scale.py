"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import pytest
import torch
import numpy as np
import sys

from .common_lib import check_diff_quantize

from mx.mx_ops import _quantize_mx as mx

inf = float("Inf")

# Test vectors
# Shared exp overflows (> 2**15) become vector of NaNs
# Shared exp underflows (< 2**-15) are flushed to vector of 0
#
# _x is a 2D matrix, the first element of each row is non-zero
# and determines the shared exp.
_x = [2**16, 2**15, 2**(-16), 2**(-17), 2**(-15), 2**(-14), 1, 0]
_x = [[e]+7*[0] for e in _x]

_t = [inf,   2**15, 2**(-15), 0,        2**(-15), 2**(-14), 1, 0]
_t = [[e]+7*[inf if e==inf else 0] for e in _t]

def test_e5m0_scale_pytorch():
    x = torch.tensor(_x, dtype=torch.float32)
    t = torch.tensor(_t, dtype=torch.float32)

    y  = mx( x, 5, elem_format="int2", axes=[-1],
            block_size=x.shape[-1])
    ny = mx(-x, 5, elem_format="int2", axes=[-1],
            block_size=x.shape[-1])
    check_diff_quantize(x, t, y, tol=0, handle_infs=True)
    check_diff_quantize(x, -t, ny, tol=0, handle_infs=True)


def test_e5m0_scale_cpp():
    x = torch.tensor(_x, dtype=torch.float32)
    t = torch.tensor(_t, dtype=torch.float32)

    y  = mx( x, 5, elem_format="int2", axes=[-1],
            block_size=x.shape[-1], custom_cuda=True)
    ny = mx(-x, 5, elem_format="int2", axes=[-1],
            block_size=x.shape[-1], custom_cuda=True)
    check_diff_quantize(x, t, y, tol=0, handle_infs=True)
    check_diff_quantize(x, -t, ny, tol=0, handle_infs=True)


def test_e5m0_scale_innermost_cuda():
    x = torch.tensor(_x, dtype=torch.float32)
    t = torch.tensor(_t, dtype=torch.float32)

    x = x.cuda()
    y  = mx( x, 5, elem_format="int2", axes=[-1],
            block_size=x.shape[-1], custom_cuda=True)
    ny = mx(-x, 5, elem_format="int2", axes=[-1],
            block_size=x.shape[-1], custom_cuda=True)
    check_diff_quantize(x, t, y, tol=0, handle_infs=True)
    check_diff_quantize(x, -t, ny, tol=0, handle_infs=True)


def test_e5m0_scale_by_tile_cuda():
    """ Call quantize_by_tile_cuda by making the shared axis 0 """
    x = torch.tensor(_x, dtype=torch.float32).transpose(0,1)
    t = torch.tensor(_t, dtype=torch.float32).transpose(0,1)

    x = x.cuda()
    y  = mx( x, 5, elem_format="int2", axes=[0],
            block_size=x.shape[-1], custom_cuda=True)
    ny = mx(-x, 5, elem_format="int2", axes=[0],
            block_size=x.shape[-1], custom_cuda=True)
    check_diff_quantize(x, t, y, tol=0, handle_infs=True)
    check_diff_quantize(x, -t, ny, tol=0, handle_infs=True)

def test_e5m0_scale_func_cuda():
    """ Call quantize_func_cuda by making a block size 64 """
    _xn = [e+56*[e[-1]] for e in _x]
    _tn = [e+56*[e[-1]] for e in _t]

    x = torch.tensor(_xn, dtype=torch.float32)
    t = torch.tensor(_tn, dtype=torch.float32)

    x = x.cuda()
    y  = mx( x, 5, elem_format="int2", axes=[-1],
            block_size=x.shape[-1], custom_cuda=True)
    ny = mx(-x, 5, elem_format="int2", axes=[-1],
            block_size=x.shape[-1], custom_cuda=True)
    check_diff_quantize(x, t, y, tol=0, handle_infs=True)
    check_diff_quantize(x, -t, ny, tol=0, handle_infs=True)

