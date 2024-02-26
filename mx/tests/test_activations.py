"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Backpropagation functional tests of softmax.
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
from mx import sigmoid, tanh, relu, relu6, leaky_relu, silu, gelu
from mx import Sigmoid, Tanh, ReLU, ReLU6, LeakyReLU, SiLU, GELU

np.random.seed(0xdeadbeef)
torch.manual_seed(0xdeadbeef)

SIZE = [10, 10000]
DEVICE__CUSTOM_CUDA = [
    ('cpu', False),
    ('cuda', False),
    ('cuda', True)
]

def torch_gelu(x):
    return 0.5 * x \
               * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def torch_quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


@pytest.mark.parametrize("first_order", [False, True])
@pytest.mark.parametrize("size", SIZE)
@pytest.mark.parametrize("quantize_backprop", [False, True])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_gelu(first_order, size, quantize_backprop, device, custom_cuda):
    mx_specs = {}
    mx_specs['bfloat'] = 30
    mx_specs['quantize_backprop'] = quantize_backprop
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    # Create shared input for two networks
    m_ = np.random.randn(size)

    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)

    if first_order:
        q1 = torch_quick_gelu(m1)
        q2 = gelu(m2, mx_specs=mx_specs, first_order_gelu=True)
    else:
        q1 = torch_gelu(m1)
        q2 = gelu(m2, mx_specs=mx_specs, first_order_gelu=False,
                  approximate='tanh')

    loss1 = (q1**2).sum()
    loss1.backward()
    torch.cuda.synchronize()
    loss2 = (q2**2).sum()
    loss2.backward()
    torch.cuda.synchronize()

    # Order of calculations are slightly different, so tol is low
    check_diff(q1, q2, tol=1e-3)
    check_diff(m1.grad, m2.grad, tol=1e-3)


@pytest.mark.parametrize("f1, f2", [
    (torch.sigmoid, sigmoid),
    (torch.tanh,    tanh),
    (F.relu,        relu),
    (F.relu6,       relu6),
    (F.leaky_relu,  leaky_relu),
    (F.silu,        silu),
])
@pytest.mark.parametrize("size", SIZE)
@pytest.mark.parametrize("quantize_backprop", [True, False])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_activation(f1, f2, size, quantize_backprop, device, custom_cuda):
    # mx specs. Use a large bitwidth since we're testing
    # algorithmic correctness, not precision
    mx_specs = {}
    mx_specs['bfloat'] = 30
    mx_specs['quantize_backprop'] = quantize_backprop
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)
    kwargs = {'negative_slope': 0.4} if f2 is leaky_relu else {}

    # Create shared input for two networks
    m_ = np.random.randn(size)

    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)

    q1 = f1(m1, **kwargs)
    loss1 = (q1**2).sum()
    loss1.backward()
    torch.cuda.synchronize()

    q2 = f2(m2, mx_specs=mx_specs, **kwargs)
    loss2 = (q2**2).sum()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=5e-6)
    check_diff(m1.grad, m2.grad, tol=5e-6)


@pytest.mark.parametrize("c1, c2", [
    (nn.Sigmoid,    Sigmoid),
    (nn.Tanh,       Tanh),
    (nn.ReLU,       ReLU),
    (nn.ReLU6,      ReLU6),
    (nn.LeakyReLU,  LeakyReLU),
    (nn.SiLU,       SiLU),
])
@pytest.mark.parametrize("size", SIZE)
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_activation_class(c1, c2, size, device, custom_cuda):
    # mx specs. Use a large bitwidth since we're testing
    # algorithmic correctness, not precision
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = True
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    # Create shared input for two networks
    m_ = np.random.randn(size)

    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)

    q1 = c1()(m1)
    loss1 = (q1**2).sum()
    loss1.backward()
    torch.cuda.synchronize()

    q2 = c2(mx_specs=mx_specs)(m2)
    loss2 = (q2**2).sum()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=1e-6)
    check_diff(m1.grad, m2.grad, tol=1e-6)


@pytest.mark.parametrize("c1, c2", [
    (nn.ReLU,       ReLU),
    (nn.ReLU6,      ReLU6),
    (nn.LeakyReLU,  LeakyReLU),
    (nn.SiLU,       SiLU)
])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("quantize_backprop", [False, True])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_network(c1, c2, inplace, quantize_backprop, device, custom_cuda):
    # mx specs. Use a large bitwidth since we're testing
    # algorithmic correctness, not precision
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = quantize_backprop
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    channels_1 = 3
    channels_2 = 7
    channels_3 = 5

    # Create shared input for two networks
    # (batch, in_channels, height, width)
    m_ = np.random.randn(10, channels_1, 32, 32)

    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)

    # Copy the conv layer so weights are identical between the two networks
    conv1 = nn.Conv2d(channels_1, channels_2, 1)
    conv2 = nn.Conv2d(channels_2, channels_3, 1)

    # Baseline network
    net1 = nn.Sequential(deepcopy(conv1),
                         c1(inplace=inplace),
                         deepcopy(conv2),
                         c1(inplace=inplace)).to(device)
    q1 = net1(m1)
    loss1 = (q1**2).sum()
    loss1.backward()
    torch.cuda.synchronize()

    # Quantized network
    net2 = nn.Sequential(deepcopy(conv1),
                         c2(mx_specs=mx_specs, inplace=inplace),
                         deepcopy(conv2),
                         c2(mx_specs=mx_specs, inplace=inplace)).to(device)
    q2 = net2(m2)
    loss2 = (q2**2).sum()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=1e-6)
    check_diff(m1.grad, m2.grad, tol=1e-6)


@pytest.mark.parametrize("func", [relu]) #, relu6, leaky_relu, silu])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_inplace(func, inplace, device, custom_cuda):
    mx_specs = {}
    mx_specs['bfloat'] = 10
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    # Create shared input for two networks
    # (batch, in_channels, height, width)
    m_ = np.random.randn(10, 32)

    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)

    # Leaf vars cannot be used inplace
    n1, n2 = m1*2, m2*2

    q1 = func(n1, mx_specs=mx_specs, inplace=False)
    loss1 = (q1**2).sum()
    loss1.backward()
    torch.cuda.synchronize()

    q2 = func(n2, mx_specs=mx_specs, inplace=True)
    loss2 = (q2**2).sum()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, n2, tol=1e-6)
    check_diff(q1, q2, tol=1e-6)
    check_diff(m1.grad, m2.grad, tol=1e-6)
