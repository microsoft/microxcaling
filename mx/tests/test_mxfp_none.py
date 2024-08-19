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

from .common_lib import check_diff, torch_version_ge

from mx import *

np.random.seed(0xdeadbeef)
torch.manual_seed(0xdeadbeef)


def _issubclass(c1, c2):
    """ returns False if c1 is not a class """
    try:
        return issubclass(c1,c2)
    except TypeError:
        return False


@pytest.mark.parametrize("f1, f2, nargs", [
    (torch.add,             simd_add,            2),
    (torch.sub,             simd_sub,            2),
    (torch.mul,             simd_mul,            2),
    (torch.div,             simd_div,            2),
    (torch.sqrt,            simd_sqrt,           1),
    (torch.square,          simd_square,         1),
    (torch.exp,             simd_exp,            1),
    (torch.log,             simd_log,            1),
    (torch.linalg.norm,     simd_norm,           1),
    (torch.sum,             simd_reduce_sum,     1),
    (torch.mean,            simd_reduce_mean,    1),
    (lambda x:x,            quantize_bfloat,     1),
    
    (torch.sigmoid,         sigmoid,             1),
    (torch.tanh,            tanh,                1),
    (F.relu,                relu,                1),
    (F.relu6,               relu6,               1),
    (F.leaky_relu,          leaky_relu,          1),
    (F.silu,                silu,                1),
    (F.gelu,                gelu,                1),

    (F.adaptive_avg_pool2d, adaptive_avg_pool2d, 1),

    (nn.Sigmoid,            Sigmoid,             1),
    (nn.Tanh,               Tanh,                1),
    (nn.ReLU,               ReLU,                1),
    (nn.ReLU6,              ReLU6,               1),
    (nn.SiLU,               SiLU,                1),
    (nn.GELU,               GELU,                1),

    (nn.AdaptiveAvgPool2d,  AdaptiveAvgPool2d,   1),
   
    (F.softmax,             softmax,             1),
    (nn.Softmax,            Softmax,             1),
])
def test_none1(f1, f2, nargs):
    torch.backends.cudnn.deterministic = True
    device = 'cuda'

    if f1 is F.leaky_relu:
        kwargs = {'negative_slope': 0.4}
    elif f1 in (torch.sum, torch.mean, F.softmax, nn.Softmax):
        kwargs = {'dim': -1}
    elif f1 in (F.adaptive_avg_pool2d, adaptive_avg_pool2d, nn.AdaptiveAvgPool2d,  AdaptiveAvgPool2d):
        input_size_4D = (1, 64, 8, 9)
        out_size = (5, 7)
        kwargs = {}
    else:
        kwargs = {}

    kwargs1 = {}
    if torch_version_ge("1.12"):
        if f1 in (nn.GELU, F.gelu):
            kwargs1 = {'approximate': 'tanh'}
            pytest.xfail('GELU has slight numerical differences')
    if f1 in (F.adaptive_avg_pool2d, adaptive_avg_pool2d, nn.AdaptiveAvgPool2d,  AdaptiveAvgPool2d):
        _x = np.random.randn(input_size_4D[0], input_size_4D[1], input_size_4D[2], input_size_4D[3])
        _y = np.random.randn(input_size_4D[0], input_size_4D[1], input_size_4D[2], input_size_4D[3])
    else:
        _x = np.random.randn(16)
        _y = np.random.randn(16)

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
        if _issubclass(f1, nn.Module):
            if f1 in (nn.AdaptiveAvgPool2d,  AdaptiveAvgPool2d):
                q1 = f1(out_size, **kwargs, **kwargs1)(x1)
                q2 = f2(out_size, mx_specs=None, **kwargs)(x2)
            else:
                q1 = f1(**kwargs, **kwargs1)(x1)
                q2 = f2(mx_specs=None, **kwargs)(x2)
        else:
            if f1 in (F.adaptive_avg_pool2d, adaptive_avg_pool2d):
                q1 = f1(x1, out_size, **kwargs, **kwargs1)
                q2 = f2(x2, out_size, mx_specs=None, **kwargs)
            else:
                q1 = f1(x1, **kwargs, **kwargs1)
                q2 = f2(x2, mx_specs=None, **kwargs)
    elif nargs == 2:
        if _issubclass(f1, nn.Module):
            q1 = f1(**kwargs)(x1, y1)
            q2 = f2(mx_specs=None, **kwargs)(x2, y2)
        else:
            q1 = f1(x1, y1, **kwargs)
            q2 = f2(x2, y2, mx_specs=None, **kwargs)
    else:
        raise ValueError('nargs can only be 1 or 2')

    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    torch.cuda.synchronize()

    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    torch.cuda.synchronize()

    print(f'{f1=} {f2=} {nargs=}')
    check_diff(q1, q2, tol=0)
    check_diff(x1.grad, x2.grad, tol=0)
    if nargs == 2:
        check_diff(y1.grad, y2.grad, tol=0)


@pytest.mark.parametrize("f1, f2", [
    (torch.matmul,  matmul),
    (torch.bmm,     bmm),
    (F.linear,      linear),
    (F.conv1d,      conv1d),
    (F.conv2d,      conv2d),
    (F.conv3d,      conv3d),
])
def test_none2(f1, f2):
    torch.backends.cudnn.deterministic = True
    device = 'cuda'

    if f1 is F.linear:
        kwargs = {'bias': None}
    else:
        kwargs = {}

    if f1 is F.conv1d:
        s1 = (16, 16, 16)
        s2 = (4, 16, 1)
    elif f1 is F.conv2d:
        s1 = (16, 16, 16, 16)
        s2 = (4, 16, 1, 1)
    elif f1 is F.conv3d:
        s1 = (16, 16, 16, 16, 16)
        s2 = (4, 16, 1, 1, 1)
    elif f1 is torch.bmm:
        s1 = (16, 16, 16)
        s2 = (16, 16, 16)
    else:
        s1 = (16, 16)
        s2 = (16, 16)

    _x = np.random.randn(*s1)
    _y = np.random.randn(*s2)

    x1 = torch.tensor(_x, dtype=torch.float32, device=device,
                      requires_grad=True)
    x2 = torch.tensor(_x, dtype=torch.float32, device=device,
                      requires_grad=True)
    y1 = torch.tensor(_y, dtype=torch.float32, device=device,
                      requires_grad=True)
    y2 = torch.tensor(_y, dtype=torch.float32, device=device,
                      requires_grad=True)

    q1 = f1(x1, y1, **kwargs)
    q2 = f2(x2, y2, mx_specs=None, **kwargs)

    loss1 = (q1**2).mean().sqrt()
    loss1.backward()
    torch.cuda.synchronize()

    loss2 = (q2**2).mean().sqrt()
    loss2.backward()
    torch.cuda.synchronize()

    check_diff(q1, q2, tol=0)
    check_diff(x1.grad, x2.grad, tol=0)
    check_diff(y1.grad, y2.grad, tol=0)


