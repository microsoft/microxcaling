"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Test MX corner cases (Infs, NaNs, Denorms, etc).
"""

import pytest
import torch
import numpy as np
import sys

from .common_lib import check_diff_quantize

from mx.mx_ops import _quantize_mx

np.random.seed(0xd10)

DEVICE__CUSTOM_CUDA = [
    ("cpu",  False),
    ("cuda", False),
    ("cuda", True),
]


# This tests that MX preserves NaN/Inf
@pytest.mark.parametrize("val", [
    float("NaN"),
    float("Inf"),
    float("-Inf")
])
@pytest.mark.parametrize("elem_format", [
    ("fp8_e4m3"),
    ("fp4_e2m1"),
    ("int4"),
])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_mx_nans(val, elem_format, device, custom_cuda):
    """ MX perserves NaNs/Infs, but NaNs/Infs may be converted to each other.
        All other values in a shared vector with a NaN/Inf become NaNs/Infs.
    """
    nan = float("NaN")
    _x = np.array([val, 0,   2**127,  0], dtype=np.float32)
    _t = np.array([nan, nan, nan,   nan], dtype=np.float32)

    x = torch.as_tensor(_x, dtype=torch.float32, device=torch.device(device))
    t = torch.as_tensor(_t, dtype=torch.float32, device=torch.device(device))

    y = _quantize_mx(x, 8,
                     elem_format=elem_format,
                     block_size=x.shape[-1],
                     axes=-1,
                     round="floor",
                     custom_cuda=custom_cuda)

    check_diff_quantize(x, t, y, handle_infs=True)


# Tests that the 2's complement 1000 representation is not used
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_mx_rounding(device, custom_cuda):
    x = np.array([[-1.0,  -1.75, -1.99,  1.99],  # -2.0 is not used
                  [ 1.0,  -1.75, -1.99,  1.99],])

    # Set vectors with small max exps to 0
    y = np.array([[-1.0, -1.75, -1.75,  1.75],
                  [ 1.0, -1.75, -1.75,  1.75]])

    y_torch = torch.as_tensor(y, dtype=torch.float32, device=torch.device(device))

    x_cuda = torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))
    y_cuda = _quantize_mx(x_cuda, 8,
                          elem_format="int4",
                          block_size=0,
                          axes=1,
                          round='nearest',
                          custom_cuda=custom_cuda)

    check_diff_quantize(x, y_torch, y_cuda, handle_infs=True)


@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_mx_hw_test(device, custom_cuda):
    x = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        # rounding for positive float
        [1.015625, 1.0234375, 1.03125, 1.0390625, 1.25,
         1.2578125, 1.9375, 1.9453125, 1.984375, 1.9921875],
        # rounding for negative float
        [-1.984375, -1.9765625, -1.96875, -1.9609375, -1.9375,
         -1.9296875, -1.75, -1.7421875, -1.0, -1.9921875],
        # out of bound checks
        [1.99609375, 1.98828125, 0.0, 0.00390625, 0.0078125,
         0.01171875, -0.015625, -0.01171875, -0.0078125, -0.00390625]])

    # Set vectors with small max exps to 0
    y = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        # rounding for positive float
        [1.015625, 1.03125, 1.03125, 1.046875, 1.25,
         1.265625, 1.9375, 1.953125, 1.984375, 1.984375],
        # rounding for negative float
        [-1.984375, -1.984375, -1.96875, -1.96875, -1.9375,
         -1.9375, -1.75, -1.75, -1.0, -1.984375],
        [1.984375, 1.984375, 0.0, 0.0, 0.015625,
         0.015625, -0.015625, -0.015625, -0.015625, 0.0]])

    y_torch = torch.as_tensor(y, dtype=torch.float32, device=torch.device(device))

    x_cuda = torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))
    y_cuda = _quantize_mx(x_cuda, 8,
                          elem_format="int8",
                          block_size=10,
                          axes=1,
                          round='nearest',
                          custom_cuda=custom_cuda)

    check_diff_quantize(x, y_torch, y_cuda, handle_infs=True)



