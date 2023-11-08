"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import pytest
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_lib import check_diff

from mx.specs import finalize_mx_specs
from mx import linear, matmul, bmm

np.random.seed(0xdeadbeef)
torch.manual_seed(0xdeadbeef)

DEVICE__CUSTOM_CUDA = [
    ('cpu',  False),
    ('cuda', False),
    ('cuda', True)
]

def t_linear_core(f1, f2, shape, use_bias, device, mx_specs,
                  tolf=1e-6, tolb=1e-5):
    # Shape is (batch, in_features, out_features, inner_dim_size)
    m_ = np.random.randn(shape[0], 2, shape[1], shape[-1])
    w_ = np.random.randn(shape[2], shape[-1])
    b_ = np.random.randn(shape[2])

    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    w1 = torch.tensor(w_, dtype=torch.float32, device=device, requires_grad=True)
    w2 = torch.tensor(w_, dtype=torch.float32, device=device, requires_grad=True)
    if use_bias:
        b1 = torch.tensor(b_, dtype=torch.float32, device=device, requires_grad=True)
        b2 = torch.tensor(b_, dtype=torch.float32, device=device, requires_grad=True)
    else:
        b1 = None
        b2 = None

    # Baseline
    q1 = f1(m1, w1, b1)
    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    torch.cuda.synchronize()

    # MX library
    q2 = f2(m2, w2, b2, mx_specs)
    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=tolf)
    check_diff(m1.grad, m2.grad, tol=tolb)
    check_diff(w1.grad, w2.grad, tol=tolb)
    if use_bias:
        check_diff(b1.grad, b2.grad, tol=tolb)


@pytest.mark.parametrize("f1, f2", [
    (F.linear,  linear)
])
@pytest.mark.parametrize("shape", [
    # batch, in_features, out_features, inner_dim_size
    (1, 32, 5, 7),
    (8, 1,  5, 7),
    (8, 13, 4, 1),
    (8, 13, 1, 4),
    (8, 13, 491, 511),
])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_linear(f1, f2, shape, use_bias, device, custom_cuda):
    torch.backends.cudnn.deterministic = True

    # No mx quantization since we're testing correctness not precision
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = True
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    if use_bias:
        tolf = 5e-5
        tolb = 5e-5
    else:
        tolf = 1e-6
        tolb = 1e-5

    t_linear_core(f1, f2, shape, use_bias, device, mx_specs,
                  tolf=tolf, tolb=tolb)
