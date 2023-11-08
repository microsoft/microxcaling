"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import sys
import copy
import pytest
import torch
import numpy as np

from .common_lib import check_diff

from mx.specs import finalize_mx_specs
from mx import Softmax


class SoftmaxExp2Base(torch.nn.Module):
    """ Baseline for Softmax with exp2 """
    def __init__(self, dim=None):
        self.dim = dim
        super(SoftmaxExp2Base, self).__init__()

    def forward(self, x):
        max_x, _ = x.max(self.dim, keepdim=True)
        x = x - max_x
        exp2_x = torch.exp(x * 0.69314718056)    # Pytorch 1.2 has no exp2
        sum_exp2_x = exp2_x.sum(self.dim, keepdim=True)
        return exp2_x / sum_exp2_x

#------------------------------------------------------------------------
# Test softmax against torch.nn.Softmax without quantization
#------------------------------------------------------------------------
@pytest.mark.parametrize("size", [
    (5, 32),       # uses 1 block of 1024 threads
    (35, 32),      # uses multiple blocks
    (11, 4096),
])
@pytest.mark.parametrize("device, custom_cuda", [
    (torch.device("cpu"), False),
    (torch.device("cuda"), False),
    (torch.device("cuda"), True)
])
def test_softmax(size, device, custom_cuda):
    mx_specs = finalize_mx_specs({}, early_exit=False)
    iterations = 5

    for _ in range(iterations):
        m_ = np.random.randn(*size)
        m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
        m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)

        q1 = torch.nn.Softmax(dim=-1)(m1)
        loss1 = (q1**2).mean().sqrt()
        loss1.backward()
        torch.cuda.synchronize()

        m2 = m2.contiguous()
        q2 = Softmax(dim=-1, mx_specs=mx_specs)(m2)
        loss2 = (q2**2).mean().sqrt()
        loss2.backward()
        torch.cuda.synchronize()

        check_diff(q1, q2, tol=1e-7)
        check_diff(m1.grad, m2.grad, tol=1e-7)


#------------------------------------------------------------------------
# Test softmax using 2^x against torch.nn.Softmax without quantization
#------------------------------------------------------------------------
@pytest.mark.parametrize("size", [
    (35, 32),
    (11, 4096),
])
@pytest.mark.parametrize("quantize_backprop", [False, True])
@pytest.mark.parametrize("device, custom_cuda", [
    (torch.device("cpu"), False),
    (torch.device("cuda"), False),
    (torch.device("cuda"), True)
])
def test_softmax_exp2(size, quantize_backprop, device, custom_cuda):
    specs2 = {
             'w_elem_format': 'fp8_e4m3',
             'bfloat': 0,
             'round': 'nearest',
             'fp': 0,
             'softmax_exp2': True,
             'quantize_backprop': quantize_backprop,
             'custom_cuda': custom_cuda}
    specs2 = finalize_mx_specs(specs2, early_exit=False)

    iterations = 5

    for _ in range(iterations):
        m_ = np.random.randn(*size)
        m1 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)
        m2 = torch.tensor(m_, dtype=torch.float32, device=device, requires_grad=True)

        q1 = SoftmaxExp2Base(dim=-1)(m1)
        loss1 = (q1**2).mean().sqrt()
        loss1.backward()
        torch.cuda.synchronize()

        m2 = m2.contiguous()
        q2 = Softmax(dim=-1, mx_specs=specs2)(m2)
        loss2 = (q2**2).mean().sqrt()
        loss2.backward()
        torch.cuda.synchronize()

        check_diff(q1, q2, tol=1e-6)
        check_diff(m1.grad, m2.grad, tol=1e-5)


