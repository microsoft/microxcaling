"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Backpropagation functional tests of layernorm.
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
from mx import LayerNorm, RMSNorm

np.random.seed(0xdeadbeef)
torch.manual_seed(0xdeadbeef)

# MXFP LayerNorm is like torch LayerNorm but you can only normalize
# the innermost dim. All previous dims are averaged.

BATCH_SIZE = 32
SIZE = [17, 513]
DEVICE__CUSTOM_CUDA = [
#    ('cpu', False),
    ('cuda', False),
    ('cuda', True)
]

class BaseRMSNorm(torch.nn.Module):
    """ RMS norm baseline for testing purposes """
    def __init__(self, hidden_size, eps=1e-12):
        super(BaseRMSNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        """ y = x/rms(x) * w + b """
        H = x.shape[-1]
        x_rms = torch.sqrt(torch.mean(x*x, dim=-1, keepdim=True) + self.eps)
        x_norm = x / x_rms
        return x_norm * self.weight + self.bias


@pytest.mark.parametrize("c1, c2", [
    (nn.LayerNorm,  LayerNorm),
    (BaseRMSNorm,   RMSNorm),
])
@pytest.mark.parametrize("size", SIZE)
@pytest.mark.parametrize("eps", [1e-6, 1e-9])
@pytest.mark.parametrize("quantize_backprop", [False, True])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_class(c1, c2, size, eps, quantize_backprop, device, custom_cuda):
    # No-quantization mx specs
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = quantize_backprop
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    # Create shared input for two networks
    m_ = np.random.randn(10, size)

    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)

    C1 = c1(size, eps=eps).to(device)
    C2 = c2(size, eps=eps, mx_specs=mx_specs).to(device)
    with torch.no_grad():
        C2.weight.copy_(C1.weight)
        C2.bias.copy_(C1.bias)

    q1 = C1.forward(m1)
    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    torch.cuda.synchronize()

    q2 = C2.forward(m2)
    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=1e-6)
    check_diff(m1.grad, m2.grad, tol=1e-6)
    check_diff(C1.weight.grad, C2.weight.grad, tol=1e-6)
    check_diff(C1.bias.grad, C2.bias.grad, tol=1e-6)


@pytest.mark.parametrize("c1, c2", [
    (nn.LayerNorm,  LayerNorm),
    (BaseRMSNorm,   RMSNorm)
])
@pytest.mark.parametrize("eps", [1e-6, 1e-9])
@pytest.mark.parametrize("quantize_backprop", [False, True])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_network(c1, c2, eps, quantize_backprop, device, custom_cuda):
    # No-quantization mx specs
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = quantize_backprop
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    channels_1 = 3
    channels_2 = 7
    S = 16

    # Create shared input for two networks
    # (batch, in_channels, height, width)
    m_ = np.random.randn(BATCH_SIZE, channels_1, S, S)

    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)

    # Copy the conv layer so weights are identical between the two networks
    conv1 = nn.Conv2d(channels_1, channels_2, 1)
    C1 = c1(S, eps=eps)
    C2 = c2(S, eps=eps, mx_specs=mx_specs)
    with torch.no_grad():
        C2.weight.copy_(C1.weight)
        C2.bias.copy_(C2.bias)

    # Baseline network
    net1 = nn.Sequential(deepcopy(conv1), C1).to(device)
    q1 = net1(m1)
    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    torch.cuda.synchronize()

    # Quantized network
    net2 = nn.Sequential(deepcopy(conv1), C2).to(device)
    q2 = net2(m2)
    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=1e-5)
    check_diff(m1.grad, m2.grad, tol=1e-5)
    check_diff(C1.weight.grad, C2.weight.grad, tol=1e-5)
    check_diff(C1.bias.grad, C2.bias.grad, tol=1e-5)
