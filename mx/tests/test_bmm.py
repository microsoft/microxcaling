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
    ('cpu', False),
    ('cuda', False),
    ('cuda', True)
]


def t_bmm_core(f1, f2, shape, device, mx_specs, tolf=1e-6, tolb=1e-5):
    # Shape is (batch, in_rows, inner_dim, out_cols)
    m_ = np.random.randn(shape[0], shape[1], shape[2])
    w_ = np.random.randn(shape[0], shape[2], shape[3])

    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    w1 = torch.tensor(w_, dtype=torch.float32, device=device, requires_grad=True)
    w2 = torch.tensor(w_, dtype=torch.float32, device=device, requires_grad=True)

    # Baseline    
    q1 = f1(m1, w1)
    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    torch.cuda.synchronize()

    # MX library
    q2 = f2(m2, w2, mx_specs=mx_specs)
    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=tolf)
    check_diff(m1.grad, m2.grad, tol=tolb)
    check_diff(w1.grad, w2.grad, tol=tolb)


@pytest.mark.parametrize("f1, f2", [
    (torch.bmm,  bmm)
])
@pytest.mark.parametrize("shape", [
    # Shape is (batch, in_rows, inner_dim, out_cols)
    (8, 1,  32,  4),
    (7, 17, 19,  89),
    (4, 12, 490, 513)
])
@pytest.mark.parametrize("quantize_backprop", [False, True])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_bmm(f1, f2, shape, quantize_backprop, device, custom_cuda):
    torch.backends.cudnn.deterministic = True

    # No mx quantization since we're testing correctness not precision
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = quantize_backprop
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    t_bmm_core(f1, f2, shape, device, mx_specs, tolf=1e-6, tolb=1e-5)
