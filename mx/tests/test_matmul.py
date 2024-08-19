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


def t_matmul_core(f1, f2, shape, device, mx_specs, has_bias=False,
                  tolf=1e-6, tolb=1e-5):
    # Shape is (..., inner_dim, out_cols)
    m_ = np.random.standard_normal(shape[:-1])
    w_ = np.random.standard_normal(shape[-2:])
    b_ = np.random.standard_normal(shape[-1])

    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    w1 = torch.tensor(w_, dtype=torch.float32, device=device, requires_grad=True)
    w2 = torch.tensor(w_, dtype=torch.float32, device=device, requires_grad=True)

    if has_bias:
        b1 = torch.tensor(b_, dtype=torch.float32, device=device,
                          requires_grad=True)
        b2 = torch.tensor(b_, dtype=torch.float32, device=device,
                          requires_grad=True)
    else:
        b1 = None
        b2 = None

    # Baseline    
    q1 = f1(m1, w1)
    if has_bias:
        q1 = q1 + b1
    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    torch.cuda.synchronize()

    # MX library
    q2 = f2(m2, w2, bias=b2, mx_specs=mx_specs)
    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=tolf)
    check_diff(m1.grad, m2.grad, tol=tolb)
    check_diff(w1.grad, w2.grad, tol=tolb)
    if has_bias:
        check_diff(b1.grad, b2.grad, tol=tolb)

@pytest.mark.parametrize("f1, f2", [
    (torch.matmul,  matmul)
])
@pytest.mark.parametrize("shape", [
    # Shape is (batch, in_rows, inner_dim, out_cols)
    (8, 5,  32),
    (8, 5,   7,  32),
    (7, 17, 19,  89,  13),
    (4, 12, 490, 513)
])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("quantize_backprop", [False, True])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_matmul(f1, f2, shape, bias, quantize_backprop,
                device, custom_cuda):
    torch.backends.cudnn.deterministic = True

    # No mx quantization since we're testing correctness not precision
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = quantize_backprop
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    t_matmul_core(f1, f2, shape, device, mx_specs,
                  has_bias=bias, tolf=1e-6, tolb=1e-5)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mx_none", [False, True])
@pytest.mark.parametrize("shape", [
    (8, 12, 4),
    (7, 43, 89, 17),
    (7, 43, 89, 17, 9),
    (490, 119, 513)
])
@pytest.mark.parametrize("quantize_backprop", [False, True])
@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_mm(bias, mx_none, shape, quantize_backprop, device):
    """ Test against torch.mm and torch.addmm """
    torch.backends.cudnn.deterministic = True

    if mx_none:
        mx_specs = None
    else:
        # No mx quantization since we're testing correctness not precisio
        mx_specs = {}
        mx_specs['bfloat'] = 0
        mx_specs['quantize_backprop'] = quantize_backprop
        mx_specs['custom_cuda'] = True
        mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    m_ = np.random.randn(shape[0], shape[1])
    w_ = np.random.randn(shape[1], shape[2])
    b_ = np.random.randn(shape[2])

    m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
    w1 = torch.tensor(w_, dtype=torch.float32, device=device, requires_grad=True)
    w2 = torch.tensor(w_, dtype=torch.float32, device=device, requires_grad=True)
    b1 = torch.tensor(b_, dtype=torch.float32, device=device, requires_grad=True)
    b2 = torch.tensor(b_, dtype=torch.float32, device=device, requires_grad=True)

    # Baseline
    if not bias:
        q1 = torch.mm(m1, w1)
        q2 = matmul(m2, w2, bias=None, mx_specs=mx_specs)
    else:
        q1 = torch.addmm(b1, m1, w1)
        q2 = matmul(m2, w2, bias=b2, mx_specs=mx_specs)

    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    torch.cuda.synchronize()

    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=1e-5)
    check_diff(m1.grad, m2.grad, tol=1e-5)
    check_diff(w1.grad, w2.grad, tol=1e-5)
    if bias:
        check_diff(b1.grad, b2.grad, tol=1e-5)
