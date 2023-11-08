"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Test that the FP8_e4m3 format's NaN implementation is correct.
"""

import pytest
import torch
import numpy as np
import sys

from .common_lib import check_diff_quantize

from mx.mx_ops import _quantize_mx as mx

_x = [485, 475, 400, -485, -475, -450, 2**-6, -(2**-6)]
_t = [448, 448, 416, -448, -448, -448, 2**-6, -(2**-6)]
sbs = list(range(1,15))


def test_fp8_e4m3_fix_pytorch():
    """ test pytorch """
    x = torch.tensor(_x, dtype=torch.float32)
    t = torch.tensor(_t, dtype=torch.float32)

    y = mx(x, 8, elem_format="fp8_e4m3",
           block_size=x.shape[-1], axes=[-1])

    check_diff_quantize(x, t, y, tol=0, handle_infs=True)


def test_fp8_e4m3_fix_cpp():
    """ test quantize_mx_func_cpp """
    x = torch.tensor(_x, dtype=torch.float32)
    t = torch.tensor(_t, dtype=torch.float32)
    
    y = mx(x, 8, elem_format="fp8_e4m3",
           block_size=x.shape[-1], axes=[-1],
           custom_cuda=True)

    check_diff_quantize(x, t, y, tol=0, handle_infs=True)


def test_fp8_e4m3_fix_innermost_cuda():
    """ test quantize_mx_innermost_cuda """
    x = torch.tensor(_x, dtype=torch.float32)
    t = torch.tensor(_t, dtype=torch.float32)

    x = x.cuda()
    y = mx(x, 8, elem_format="fp8_e4m3",
           block_size=x.shape[-1], axes=[-1],
           custom_cuda=True)

    check_diff_quantize(x, t, y, tol=0, handle_infs=True)


def test_fp8_e4m3_fix_by_tile_cuda():
    """ test quantize_mx_by_tile_func_cuda """
    _xn = _x + [0]
    _tn = _t + [0]
    x = torch.tensor(_xn, dtype=torch.float32)
    t = torch.tensor(_tn, dtype=torch.float32)

    x = x.cuda()
    y = mx(x, 8, elem_format="fp8_e4m3",
           block_size=x.shape[-1], axes=[-1],
           custom_cuda=True)

    check_diff_quantize(x, t, y, tol=0, handle_infs=True)


def test_fp8_e4m3_fix_func_cuda():
    """ test quantize_mx_func_cuda,
        this is supposed to throw an exception """
    _xn = _x + 56*[0]  # length 64
    _tn = _t + 56*[0]
    x = torch.tensor(_xn, dtype=torch.float32)
    t = torch.tensor(_tn, dtype=torch.float32)

    x = x.cuda()
    y = mx(x, 8, elem_format="fp8_e4m3",
           block_size=x.shape[-1], axes=[-1],
           custom_cuda=True)

    check_diff_quantize(x, t, y, tol=0, handle_infs=True)


@pytest.mark.parametrize("round", ('nearest', 'even'))
@pytest.mark.parametrize("custom_cuda", (False, True))
def test_mxfp8_e4m3_round(round, custom_cuda):
    x_ = [-0.582733273506, -0.256973713636,  0.506033003330,  0.400039970875,
          -0.437093466520,  1.275019764900, -2.123294353485,  1.514909625053,
          -2.660086154938, -0.200791791081, -0.060985822231, -0.209203109145,
           2.385987281799,  0.062245476991,  0.217003762722, -0.857734560966,
           0.507945835590,  0.896152675152, -0.751049160957, -0.488164335489,
           0.805381953716, -0.172028362751,  0.271137863398, -0.503807783127,
           1.879478693008, -0.294227510691,  0.968807995319,  0.670037031174,
          -0.871595799923,  0.304561793804, -0.567594051361, -1.265962004662]

    t_ = [-0.562500000000, -0.250000000000,  0.500000000000,  0.406250000000,
          -0.437500000000,  1.250000000000, -2.000000000000,  1.500000000000,
          -2.750000000000, -0.203125000000, -0.062500000000, -0.203125000000,
           2.500000000000,  0.062500000000,  0.218750000000, -0.875000000000,
           0.500000000000,  0.875000000000, -0.750000000000, -0.500000000000,
           0.812500000000, -0.171875000000,  0.281250000000, -0.500000000000,
           1.875000000000, -0.281250000000,  1.000000000000,  0.687500000000,
          -0.875000000000,  0.312500000000, -0.562500000000, -1.250000000000]


    x = torch.as_tensor(x_, dtype=torch.float32, device=torch.device("cpu"))
    t = torch.as_tensor(t_, dtype=torch.float32, device=torch.device("cpu"))
    x_cuda = x.clone().to('cuda')

    y = mx(x, 8, elem_format="fp8_e4m3",
           block_size=32, axes=0,
           round=round, custom_cuda=custom_cuda)

    check_diff_quantize(x, t, y, tol=0, handle_infs=True)
