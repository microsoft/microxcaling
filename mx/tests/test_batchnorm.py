"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Forward and backward tests of batchnorm class.
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
from mx import BatchNorm1d, BatchNorm2d, BatchNorm3d

np.random.seed(0xdeadbeef)
torch.manual_seed(0xdeadbeef)

# MXFP LayerNorm is like torch LayerNorm but you can only normalize
# the innermost dim. All previous dims are averaged.

BATCH_SIZE = 16
HIDDEN_SIZE = [128]
DEVICE__CUSTOM_CUDA = [
    ('cpu', False),
    ('cuda', False),
    ('cuda', True)
]


def create_input(test_class, H, l1=15, l2=14, l3=13):
    """ Input size depends on the class under test """
    if test_class is BatchNorm1d:
        m_ = np.random.random_sample((BATCH_SIZE, H, l1))
    elif test_class is BatchNorm2d:
        m_ = np.random.random_sample((BATCH_SIZE, H, l1,l2))
    elif test_class is BatchNorm3d:
        m_ = np.random.random_sample((BATCH_SIZE, H, l1,l2,l3))
    return m_


@pytest.mark.parametrize("c1, c2", [
    (nn.BatchNorm1d,  BatchNorm1d),
    (nn.BatchNorm2d,  BatchNorm2d),
    (nn.BatchNorm3d,  BatchNorm3d),
])
@pytest.mark.parametrize("H", HIDDEN_SIZE)
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
@pytest.mark.parametrize("momentum", [0.1, 0.6])
@pytest.mark.parametrize("is_training, track_running_stats", [
        (False, False),
        (True,  False),
        (True,  True)
])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_class(c1, c2, H, eps, momentum, is_training,
               track_running_stats, device, custom_cuda):
    # No-quantization mx specs
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = True
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    m_ = create_input(c2, H)
    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)

    # BatchNorm in early pytorch don't have device argument
    try:
        C1 = c1(H, eps=eps, momentum=momentum, device=device,
                track_running_stats=track_running_stats)
    except TypeError:
        C1 = c1(H, eps=eps, momentum=momentum,
                track_running_stats=track_running_stats)
        C1.to(device)

    C2 = c2(H, eps=eps, momentum=momentum, device=device,
            track_running_stats=track_running_stats,
            mx_specs=mx_specs)

    with torch.no_grad():
        C2.weight.copy_(C1.weight)
        C2.bias.copy_(C1.bias)

    if is_training:
        C1.train()
        C2.train()
    else:
        C1.eval()
        C2.eval()

    for _ in range(5):
        q1 = C1.forward(m1)
        loss1 = (q1**2).mean().sqrt()
        loss1.backward()
        torch.cuda.synchronize()

        q2 = C2.forward(m2)
        loss2 = (q2**2).mean().sqrt()
        loss2.backward()
        torch.cuda.synchronize()

    check_diff(q1, q2, tol=1e-5)
    check_diff(m1.grad, m2.grad, tol=1e-5)
    check_diff(C1.weight.grad, C2.weight.grad, tol=1e-5)
    check_diff(C1.bias.grad, C2.bias.grad, tol=1e-5)

    if track_running_stats:
        check_diff(C1.running_mean, C2.running_mean, tol=1e-3)
        check_diff(C1.running_var, C2.running_var, tol=1e-3)


@pytest.mark.parametrize("c1, c2", [
    (nn.BatchNorm2d,  BatchNorm2d),
])
@pytest.mark.parametrize("H", HIDDEN_SIZE)
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
@pytest.mark.parametrize("momentum", [0.1, 0.6])
@pytest.mark.parametrize("is_training, track_running_stats", [
        (False, False),
        (True,  False),
        (True,  True)
])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_network(c1, c2, H, eps, momentum, is_training,
                 track_running_stats, device, custom_cuda):
    # No-quantization mx specs
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = True
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    m_ = create_input(c2, H)
    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)

    # BatchNorm in early pytorch don't have device argument
    try:
        C1 = c1(H, eps=eps, momentum=momentum, device=device,
                track_running_stats=track_running_stats)
    except TypeError:
        C1 = c1(H, eps=eps, momentum=momentum,
                track_running_stats=track_running_stats)
        C1.to(device)

    C2 = c2(H, eps=eps, momentum=momentum, device=device,
            track_running_stats=track_running_stats,
            mx_specs=mx_specs)

    # Copy the conv layer so weights are identical between the two networks
    with torch.no_grad():
        C2.weight.copy_(C1.weight)
        C2.bias.copy_(C1.bias)

    conv1 = nn.Conv2d(H, H, 1)
    net1 = nn.Sequential(deepcopy(conv1), C1).to(device)
    net2 = nn.Sequential(deepcopy(conv1), C2).to(device)

    if is_training:
        net1.train()
        net2.train()
    else:
        net1.eval()
        net2.eval()

    # Baseline network
    q1 = net1(m1)
    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    torch.cuda.synchronize()

    # Quantized network
    q2 = net2(m2)
    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=1e-5)
    check_diff(m1.grad, m2.grad, tol=1e-5)
    check_diff(C1.weight.grad, C2.weight.grad, tol=1e-5)
    check_diff(C1.bias.grad, C2.bias.grad, tol=1e-5)
    if track_running_stats:
        check_diff(C1.running_mean, C2.running_mean, tol=1e-3)
        check_diff(C1.running_var, C2.running_var, tol=1e-3)
