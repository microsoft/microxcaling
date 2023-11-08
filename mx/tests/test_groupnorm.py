"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Forward and backward tests of groupnorm class.
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
from mx import GroupNorm

np.random.seed(0xdeadbeef)
torch.manual_seed(0xdeadbeef)

# MXFP LayerNorm is like torch LayerNorm but you can only normalize
# the innermost dim. All previous dims are averaged.

BATCH_SIZE = 16
DEVICE__CUSTOM_CUDA = [
    ('cpu', False),
    ('cuda', False),
    ('cuda', True)
]


@pytest.mark.parametrize("c1, c2", [
    (nn.GroupNorm,  GroupNorm),
])
@pytest.mark.parametrize("shape, num_groups", [
    ([32], 2),
    ([32, 8], 8),
    ([32, 8, 8], 16),
])
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_class(c1, c2, shape, num_groups, eps, device, custom_cuda):
    # No-quantization mx specs
    mx_specs = {}
    mx_specs['bfloat'] = 32
    mx_specs['quantize_backprop'] = True
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    # Groupnorm is hardcoded to treat dim=1 as the hidden dim
    H = shape[0]
    m_ = np.random.random_sample([BATCH_SIZE] + shape)
    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)

    # BatchNorm in early pytorch don't have device argument
    try:
        C1 = c1(num_groups, H, eps=eps, device=device)
    except TypeError:
        C1 = c1(num_groups, H, eps=eps)
        C1.to(device)

    C2 = c2(num_groups, H, eps=eps, device=device,
            mx_specs=mx_specs)

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

    check_diff(q1, q2, tol=1e-5)
    check_diff(m1.grad, m2.grad, tol=1e-5)
    check_diff(C1.weight.grad, C2.weight.grad, tol=1e-5)
    check_diff(C1.bias.grad, C2.bias.grad, tol=1e-5)

