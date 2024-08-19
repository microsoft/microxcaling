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
from mx.simd_ops import *

np.random.seed(0xdeadbeef)
torch.manual_seed(0xdeadbeef)

SIZE = (16, 32, 32)
DEVICE__CUSTOM_CUDA = [
    ('cpu', False),
    ('cuda', False),
    ('cuda', True)
]


@pytest.mark.parametrize("f1, f2, nargs", [
    (torch.add,         simd_add,       2),
    (torch.sub,         simd_sub,       2),
    (torch.mul,         simd_mul,       2),
    (torch.div,         simd_div,       2),
    (torch.sqrt,        simd_sqrt,      1),
    (torch.exp,         simd_exp,       1),
    (torch.log,         simd_log,       1),
    (torch.square,      simd_square,    1),
    (torch.linalg.norm, simd_norm,      1),
])
@pytest.mark.parametrize("quantize_backprop", [False, True])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_simd1(f1, f2, nargs, quantize_backprop,
               device, custom_cuda):
    torch.backends.cudnn.deterministic = True
    tol = 5e-6 if f2==simd_norm else 1e-7

    # mx specs. Use a large bitwidth since we're testing
    # algorithmic correctness, not precision
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = quantize_backprop
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    _x = np.random.randn(*SIZE)
    _y = np.random.randn(*SIZE)
    if f2 in (simd_sqrt, simd_log):
        _x = np.abs(_x)
        _y = np.abs(_y)

    x1 = torch.tensor(_x, dtype=torch.float32, device=device,
                      requires_grad=True)
    x2 = torch.tensor(_x, dtype=torch.float32, device=device,
                      requires_grad=True)
    y1 = torch.tensor(_y, dtype=torch.float32, device=device,
                      requires_grad=True)
    y2 = torch.tensor(_y, dtype=torch.float32, device=device,
                      requires_grad=True)

    if nargs == 1:
        q1 = f1(x1)
        q2 = f2(x2, mx_specs=mx_specs)
    elif nargs == 2:
        q1 = f1(x1, y1)
        q2 = f2(x2, y2, mx_specs=mx_specs)
    else:
        raise ValueError('nargs can only be 1 or 2')

    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    torch.cuda.synchronize()

    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=tol)
    check_diff(x1.grad, x2.grad, tol=tol)
    if nargs == 2:
        check_diff(y1.grad, y2.grad, tol=tol)


@pytest.mark.parametrize("f1, f2", [
    (torch.add,         simd_add),
    (torch.sub,         simd_sub),
    (torch.mul,         simd_mul),
    (torch.div,         simd_div),
])
@pytest.mark.parametrize("quantize_backprop", [False, True])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_const(f1, f2, quantize_backprop, device, custom_cuda):
    torch.backends.cudnn.deterministic = True

    # mx specs. Use a large bitwidth since we're testing
    # algorithmic correctness, not precision
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = quantize_backprop
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    _x = np.random.randn(*SIZE)
    x1 = torch.tensor(_x, dtype=torch.float32, device=device,
                      requires_grad=True)
    x2 = torch.tensor(_x, dtype=torch.float32, device=device,
                      requires_grad=True)

    q1 = f1(x1, 5.129)
    q2 = f2(x2, 5.129, mx_specs=mx_specs)

    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    torch.cuda.synchronize()

    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=0)
    check_diff(x1.grad, x2.grad, tol=0)


@pytest.mark.parametrize("f1, f2", [
    (torch.sum,     simd_reduce_sum),
    (torch.mean,    simd_reduce_mean),
])
@pytest.mark.parametrize("dim", [None, [0], [1], [-1], [0,-1], [0,1,2]])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("quantize_backprop", [False, True])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_simd_reduce(f1, f2, dim, keepdim, quantize_backprop,
                     device, custom_cuda):
    torch.backends.cudnn.deterministic = True

    # mx specs. Use a large bitwidth since we're testing
    # algorithmic correctness, not precision
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['quantize_backprop'] = quantize_backprop
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    _x = np.random.randn(*SIZE)
    x1 = torch.tensor(_x, dtype=torch.float32, device=device,
                      requires_grad=True)
    x2 = torch.tensor(_x, dtype=torch.float32, device=device,
                      requires_grad=True)

    if dim is None and keepdim == False:
        q1 = f1(x1)
    else:
        q1 = f1(x1, dim=dim, keepdim=keepdim)
    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    torch.cuda.synchronize()

    if dim is None and keepdim == False:
        q2 = f2(x2, mx_specs=mx_specs)
    else:
        q2 = f2(x2, dim=dim, keepdim=keepdim, mx_specs=mx_specs)
    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=1e-7)
    check_diff(x1.grad, x2.grad, tol=1e-7)


@pytest.mark.parametrize("f1, f2", [
    (torch.add,         simd_add),
    (torch.sub,         simd_sub),
    (torch.mul,         simd_mul),
    (torch.div,         simd_div),
])
@pytest.mark.parametrize("size1, size2", [
    ((4, 4, 4), (4, 4, 1)),
    ((4, 4, 4), (4, 1, 1)),
    ((4, 4, 4), (1, 1, 1)),
    ((4, 1, 4), (4, 4, 4)),
    ((1, 4, 1), (4, 4, 4)),
    ((4, 4, 4), (4, 1)),
    ((4, 4, 4), (1, 4)),
    ((4, 4, 4), [4]),
    ([1],       (4, 4, 4)),
])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_simd_broadcast(f1, f2, size1, size2, device, custom_cuda):
    torch.backends.cudnn.deterministic = True

    # mx specs. Use a large bitwidth since we're testing
    # algorithmic correctness, not precision
    mx_specs = {}
    mx_specs['bfloat'] = 0
    mx_specs['custom_cuda'] = custom_cuda
    mx_specs = finalize_mx_specs(mx_specs, early_exit=False)

    _x = np.random.randn(*size1)
    _y = np.random.randn(*size2)

    x1 = torch.tensor(_x, dtype=torch.float32, device=device,
                      requires_grad=True)
    x2 = torch.tensor(_x, dtype=torch.float32, device=device,
                      requires_grad=True)
    y1 = torch.tensor(_y, dtype=torch.float32, device=device,
                      requires_grad=True)
    y2 = torch.tensor(_y, dtype=torch.float32, device=device,
                      requires_grad=True)

    q1 = f1(x1, y1)
    q2 = f2(x2, y2, mx_specs=mx_specs)

    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    torch.cuda.synchronize()

    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=0)
    check_diff(x1.grad, x2.grad, tol=0)
    check_diff(y1.grad, y2.grad, tol=0)
