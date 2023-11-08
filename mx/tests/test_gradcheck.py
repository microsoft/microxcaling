"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import pytest
import sys
import numpy as np
import torch

from .common_lib import check_diff

from mx import (
        linear, matmul, bmm, conv1d, conv2d, conv3d,
        sigmoid, tanh, relu, leaky_relu, silu, gelu,
        softmax,
        simd_add, simd_sub, simd_mul, simd_div, simd_split,
        simd_exp, simd_log, simd_sqrt, simd_square,
        simd_norm
)

np.random.seed(0xdeadbeef)
torch.manual_seed(0xdeadbeef)


@pytest.mark.parametrize("func", [
    sigmoid, tanh, relu, leaky_relu, silu, gelu,
    softmax,
    simd_sqrt, simd_square, simd_norm, simd_split,
    simd_exp, simd_log
])
@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_finite_diff_1input(func, device):
    # gradcheck checks usually fail without float64
    x = torch.randn(64, dtype=torch.float64, device=device,
                    requires_grad=True)

    if func in (simd_sqrt, simd_log):
        x = torch.abs(x)

    torch.autograd.gradcheck(func, x)


@pytest.mark.parametrize("func", [
    linear, matmul, bmm,
    simd_add, simd_sub, simd_mul, simd_div
])
@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_finite_diff_2input(func, device):
    S1 = (3, 8, 8)
    S2 = (8, 8)

    x = torch.randn(*S1, dtype=torch.float64,
                    device=device, requires_grad=True)

    if func in (linear, matmul):
        y = torch.randn(*S2, dtype=torch.float64,
                        device=device, requires_grad=True)
    else:
        y = torch.randn(*S1, dtype=torch.float64,
                        device=device, requires_grad=True)

    torch.autograd.gradcheck(func, (x, y))


@pytest.mark.parametrize("func", [
    conv1d, conv2d, conv3d,
])
@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_finite_diff_conv(func, device):
    """
    Older pytorch versions may see this error when running
    many unit tests at once:
        RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.
    This is a known bug in pytorch:
    https://pytorch.org/docs/stable/notes/broadcasting.html?highlight=broadcasting
    """
    if func is conv1d:
        spatial_dims = 1
    elif func is conv2d:
        spatial_dims = 2
    elif func is conv3d:
        spatial_dims = 3

    S1 = [2, 8] + spatial_dims*[3]
    S2 = [2, 8] + spatial_dims*[1]

    x = torch.randn(*S1, dtype=torch.float64,
                    device=device, requires_grad=True)
    w = torch.randn(*S2, dtype=torch.float64,
                    device=device, requires_grad=True)

    torch.autograd.gradcheck(func, (x, w))
