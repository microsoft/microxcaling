"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import pytest
import sys
import numpy as np
import torch

from .common_lib import check_diff

from mx.specs import finalize_mx_specs
from mx import (
    Conv1d, Conv2d, Conv3d,
    ConvTranspose2d, conv2d
)

np.random.seed(0xdeadbeef)
torch.manual_seed(0xdeadbeef)

DEVICE__CUSTOM_CUDA = [
    ('cpu', False),
    ('cuda', False),
    ('cuda', True)
]

@pytest.mark.parametrize("C1, C2", [
    (torch.nn.Conv1d, Conv1d),
    (torch.nn.Conv2d, Conv2d),
    (torch.nn.Conv3d, Conv3d),
    (torch.nn.ConvTranspose2d, ConvTranspose2d)
])
@pytest.mark.parametrize("in_channels, out_channels, groups", [
    (32, 33, 1),
    (8,  6,  2),
    (15, 12, 3),
])
@pytest.mark.parametrize("filter_size", [1, 3, 5])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1, 2])
@pytest.mark.parametrize("quantize_backprop", [False, True])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_conv(C1, C2, in_channels, out_channels, groups, filter_size,
              stride, padding, quantize_backprop, device, custom_cuda):

    torch.backends.cudnn.deterministic = True

    # mx specs. Use a large bitwidth since we're testing
    # algorithmic correctness, not precision
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = quantize_backprop
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    if C2 is Conv1d:
        num_spatial_dims = 1
    elif C2 in (Conv2d, ConvTranspose2d):
        num_spatial_dims = 2
    elif C2 is Conv3d:
        num_spatial_dims = 3

    batch_size = 7
    spatial_dims = [16] * num_spatial_dims
    dilation = 1

    # Create shared input and weights for the two networks
    # input shape  (batch, in_channels, H, W)
    # weight shape (out_channels, in_channels/groups, kH, kW)
    input_shape = [batch_size, in_channels] + spatial_dims
    m_ = np.random.randn(*input_shape)
    m1 = torch.tensor(m_, dtype=torch.float32, device=device,
                      requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device,
                      requires_grad=True)

    # Baseline
    if C1 == torch.nn.Conv2d:
        c1 = C1(in_channels, out_channels, filter_size, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups)
        c2 = C2(in_channels, out_channels, filter_size, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, mx_specs=mx_specs)
    else:
        c1 = C1(in_channels, out_channels, filter_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups)
        c2 = C2(in_channels, out_channels, filter_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups, mx_specs=mx_specs)

    with torch.no_grad():
        assert(c1.weight.shape == c2.weight.shape)
        c2.weight.copy_(c1.weight)
        assert(c1.bias.shape == c2.bias.shape)
        c2.bias.copy_(c1.bias)

    if device == 'cuda':
        c1.cuda()
        c2.cuda()

    q1 = c1(m1)
    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    torch.cuda.synchronize()

    q2 = c2(m2)
    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=1e-7)
    check_diff(m1.grad, m2.grad, tol=1e-6)
    check_diff(c1.weight.grad, c2.weight.grad, tol=1e-6)
    check_diff(c1.bias.grad, c2.bias.grad, tol=1e-5)
