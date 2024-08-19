"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import pytest
import torch
import numpy as np
import sys

from .common_lib import (
        check_diff,
        check_diff_quantize,
        all_encodings
)

from mx.formats import _get_format_params
from mx.mx_ops import _quantize_mx

np.random.seed(0xd10)

DEVICE__CUSTOM_CUDA = [
    ("cpu",  False),
    ("cpu",  True),
    ("cuda", True),
]

ELEM_FMTS = [
    ("fp8_e5m2"),
    ("fp8_e4m3"),
    ("fp6_e3m2"),
    ("fp6_e2m3"),
    ("fp4_e2m1"),
    ("int4"),
]

@pytest.mark.parametrize("elem_format", ELEM_FMTS)
def test_empty_quantization(elem_format):
    x_cuda = torch.as_tensor([], dtype=torch.float32, device=torch.device('cuda'))
    _ = _quantize_mx(x_cuda, 8,
                          elem_format=elem_format,
                          block_size=32,
                          axes=-1,
                          round='nearest',
                          custom_cuda=True)

@pytest.mark.parametrize("scale_bits", (8,5))
@pytest.mark.parametrize("elem_format", ELEM_FMTS)
@pytest.mark.parametrize("block_size", (8, 9, 64))
@pytest.mark.parametrize("round", ('nearest', 'floor', 'even'))
@pytest.mark.parametrize("flush_fp32_subnorms", (False,True))
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_mx_encoding(scale_bits, elem_format, block_size, round,
                     flush_fp32_subnorms, device, custom_cuda):

    x1 = all_encodings(8, 9, device="cpu")
    x2 = x1.clone().detach().to(device)

    y1 = _quantize_mx(x1, scale_bits, elem_format,
                      block_size=block_size,
                      axes=[-1],
                      round=round,
                      flush_fp32_subnorms=flush_fp32_subnorms,
                      custom_cuda=False)


    y2 = _quantize_mx(x2, scale_bits, elem_format,
                      block_size=block_size,
                      axes=[-1],
                      round=round,
                      flush_fp32_subnorms=flush_fp32_subnorms,
                      custom_cuda=custom_cuda)

    check_diff_quantize(x1, y1, y2)

@pytest.mark.parametrize("scale_bits", (8,5))
@pytest.mark.parametrize("elem_format", ELEM_FMTS)
@pytest.mark.parametrize("block_size", (8, 9, 64))
@pytest.mark.parametrize("round", ('nearest', 'floor', 'even'))
@pytest.mark.parametrize("flush_fp32_subnorms", (False,True))
def test_mx_encoding_cpu_cuda(scale_bits, elem_format, block_size, round,
                     flush_fp32_subnorms):

    x1 = all_encodings(8, 9, device="cpu")
    x2 = x1.clone().detach().to('cuda')

    y1 = _quantize_mx(x1, scale_bits, elem_format,
                      block_size=block_size,
                      axes=[-1],
                      round=round,
                      flush_fp32_subnorms=flush_fp32_subnorms,
                      custom_cuda=False)


    y2 = _quantize_mx(x2, scale_bits, elem_format,
                      block_size=block_size,
                      axes=[-1],
                      round=round,
                      flush_fp32_subnorms=flush_fp32_subnorms,
                      custom_cuda=True)

    check_diff_quantize(x1, y1, y2)