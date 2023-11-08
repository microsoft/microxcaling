"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import pytest
import sys
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_lib import check_diff

from mx.specs import finalize_mx_specs
from mx import adaptive_avg_pool2d
from mx import AdaptiveAvgPool2d 

np.random.seed(0xdeadbeef)
torch.manual_seed(0xdeadbeef)



DEVICE__CUSTOM_CUDA = [
    ('cpu', False),
    ('cuda', False),
    ('cuda', True)
]

@pytest.mark.parametrize("f1, f2", [
    (F.adaptive_avg_pool2d, adaptive_avg_pool2d)
])
@pytest.mark.parametrize("input_size_4D, out_size", [
    ((1, 64, 8, 9), (5,7)),
    ((1, 64, 8, 9), 7),
    ((1, 64, 10, 9), (None, 7)),
    ((3, 2, 8, 9), (5,7)),
    ((3, 2, 8, 9), 7),
    ((3, 64, 10, 9), (None, 7)),
])
@pytest.mark.parametrize("quantize_backprop", [False, True])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_pooling(f1, f2, input_size_4D, out_size, quantize_backprop, device, custom_cuda):
    mx_specs = {}
    mx_specs['bfloat'] = 30
    mx_specs['quantize_backprop'] = quantize_backprop
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    # Create shared input for two networks
    m_ = np.random.randn(input_size_4D[0], input_size_4D[1], input_size_4D[2], input_size_4D[3])

    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)


    q1 = f1(m1, out_size)
    q2 = f2(m2, out_size, mx_specs=mx_specs)

    loss1 = (q1**2).sum()
    loss1.backward()
    torch.cuda.synchronize()
    loss2 = (q2**2).sum()
    loss2.backward()
    torch.cuda.synchronize()

    # Order of calculations are slightly different, so tol is low
    check_diff(q1, q2, tol=1e-3)
    check_diff(m1.grad, m2.grad, tol=1e-3)


@pytest.mark.parametrize("c1, c2", [
    (nn.AdaptiveAvgPool2d,    AdaptiveAvgPool2d)
])
@pytest.mark.parametrize("input_size_4D, out_size", [
    ((1, 64, 8, 9), (5,7)),
    ((1, 64, 8, 9), 7),
    ((1, 64, 10, 9), (None, 7)),
    ((3, 64, 8, 9), (5,7)),
    ((3, 64, 8, 9), 7),
    ((3, 64, 10, 9), (None, 7)),
])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_pooling_class(c1, c2, input_size_4D, out_size,  device, custom_cuda):
    # mx specs. Use a large bitwidth since we're testing
    # algorithmic correctness, not precision
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = True
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    # Create shared input for two networks
    m_ = np.random.randn(input_size_4D[0], input_size_4D[1], input_size_4D[2], input_size_4D[3])

    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)

    q1 = c1(out_size)(m1)
    loss1 = (q1**2).sum()
    loss1.backward()
    torch.cuda.synchronize()

    q2 = c2(out_size, mx_specs=mx_specs)(m2)
    loss2 = (q2**2).sum()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=1e-6)
    check_diff(m1.grad, m2.grad, tol=1e-6)    


@pytest.mark.parametrize("c1, c2", [
    (nn.AdaptiveAvgPool2d,    AdaptiveAvgPool2d)
])
@pytest.mark.parametrize("input_size_2D, out_size", [
    ((8, 9), (5,7)),
    ((8, 9), 7),
    ((10, 9), (None, 7)),

])
@pytest.mark.parametrize("quantize_backprop", [False, True])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_network(c1, c2, input_size_2D, out_size,  quantize_backprop, device, custom_cuda):
    # mx specs. Use a large bitwidth since we're testing
    # algorithmic correctness, not precision
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = quantize_backprop
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    channels_1 = 3
    channels_2 = 7

    # Create shared input for two networks
    # (batch, in_channels, height, width)
    m_ = np.random.randn(10, channels_1, input_size_2D[0], input_size_2D[1])

    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
 
    # Copy the conv layer so weights are identical between the two networks
    conv1 = nn.Conv2d(channels_1, channels_2, 1)


    # Baseline network
    net1 = nn.Sequential(deepcopy(conv1),
                         c1(out_size)).to(device)
    q1 = net1(m1)
    loss1 = (q1**2).sum()
    loss1.backward()
    torch.cuda.synchronize()

    # Quantized network
    net2 = nn.Sequential(deepcopy(conv1),
                         c2(out_size, mx_specs=mx_specs)).to(device)
    q2 = net2(m2)
    loss2 = (q2**2).sum()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=1e-6)
    check_diff(m1.grad, m2.grad, tol=1e-6)   
