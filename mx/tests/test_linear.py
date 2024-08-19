"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_lib import check_diff

from mx.specs import finalize_mx_specs
from mx import linear, Linear
from mx.elemwise_ops import _quantize_bfloat

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

@pytest.mark.parametrize("device, custom_cuda", [
    ("cpu", True),
    ("cuda", True),
])
def test_linear_prequantized(device, custom_cuda):
    S = 1000

    _x = np.random.standard_normal(S)
    _w = np.random.standard_normal((S,S))
    _b = np.random.standard_normal(S)
    x = torch.tensor(_x, dtype=torch.float32, device=device)
    w = torch.tensor(_w, dtype=torch.float32, device=device)
    b = torch.tensor(_b, dtype=torch.float32, device=device)

    mx_specs = {}
    mx_specs['bfloat'] = 16
    mx_specs['round'] = 'even'
    mx_specs['bfloat_subnorms'] = True
    mx_specs['w_elem_format'] = 'fp8_e4m3'
    mx_specs['a_elem_format'] = 'fp8_e4m3'
    mx_specs['block_size'] = 32
    mx_specs['quantize_backprop'] = True
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs)

    # Baseline
    # weight and biases are quantized to bf16 yet emulated as fp32
    # input is thus emulated in fp32.
    y1 = linear(x, w, bias=b, mx_specs=mx_specs)

    # Prequantized
    # Everything is in bf16.
    w = _quantize_bfloat(w, 16, round=mx_specs["round_weight"]
                        ).to(torch.bfloat16)
    b = _quantize_bfloat(b, 16, round=mx_specs["round_weight"]
                        ).to(torch.bfloat16)
    y2 = linear(x, w, bias=b, mx_specs=mx_specs,
                prequantized_weights=True)

    check_diff(y1.to(torch.float32), y2.to(torch.float32), tol=0)

@pytest.mark.parametrize("device, custom_cuda", [
    ("cpu", True),
    ("cuda", True),
])
def test_linear_layer_prequantized(device, custom_cuda):
    S = 1000

    _x = np.random.standard_normal(S)
    _w = np.random.standard_normal((S,S))
    _b = np.random.standard_normal(S)
    x = torch.tensor(_x, dtype=torch.float32, device=device)
    w = torch.tensor(_w, dtype=torch.float32, device=device)
    b = torch.tensor(_b, dtype=torch.float32, device=device)

    mx_specs = {}
    mx_specs['bfloat'] = 16
    mx_specs['round'] = 'even'
    mx_specs['bfloat_subnorms'] = True
    mx_specs['w_elem_format'] = 'fp8_e4m3'
    mx_specs['a_elem_format'] = 'fp8_e4m3'
    mx_specs['block_size'] = 16
    mx_specs['quantize_backprop'] = True
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs)

    L1 = Linear(S, S, bias=True, mx_specs=mx_specs)
    L2 = Linear(S, S, bias=True, mx_specs=mx_specs)

    with torch.no_grad():
        L2.weight.copy_(L1.weight)
        L2.bias.copy_(L1.bias)

    L1.to(device)
    L2.to(device)
    L1.eval()
    L2.eval()

    L2.prequantize_weights()

    y1 = L1(x)
    y2 = L2(x)

    check_diff(y1.to(torch.float32), y2.to(torch.float32), tol=0)