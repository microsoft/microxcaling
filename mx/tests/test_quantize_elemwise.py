"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Test PyTorch and custom C++/CUDA implementations of elemwise quantize
functions against each other.
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

from mx.specs import get_default_mx_specs
from mx.formats import _get_format_params
from mx.elemwise_ops import (
        _quantize_fp,
        _quantize_bfloat,
        _quantize_elemwise
)

np.random.seed(0xd10)

ITERATIONS = 10
BITS__EXP_BITS = [
    (7, 8),     # bfloat16
    (10, 5),    # fp16
]
DEVICE__CUSTOM_CUDA__HALF = [
    ("cpu",  False, False),
    ("cpu",  True,  False),
    ("cuda", False, False),
    ("cuda", True,  False),
    #("cuda", True,  True)
]

def test_empty_quantization():
    x_cuda = torch.as_tensor([], dtype=torch.float32, device=torch.device('cuda'))
    _ = _quantize_bfloat(x_cuda, bfloat=16, round='even', custom_cuda=True,
                         allow_denorm=True)
    
@pytest.mark.parametrize("allow_denorm", [True, False])
@pytest.mark.parametrize("device, custom_cuda", [
    ("cpu", False),
    ("cpu", True),
    ("cuda", True),
])
def test_bfloat16_RNE(allow_denorm, device, custom_cuda):
    x_ = np.array([65535., -65535., -1.9985847, 0.4999000132083893, -0.49968934059143066,
                   -0.0018959, 0.0018959, -2.52075195e-02, 0.0156078, 0.00780624,
                   -0.00048747, -0.01561374, -0.00780595, -0.0039039, 0.01557725,
                   -0.0077924, 0.00389144,  -0.00779678, -0.01556631, -0.00194835])
    t_ = np.array([65536., -65536., -2.0, 0.5, -0.5,
                   -0.00189209, 0.00189209, -2.51464844e-02, 0.015625, 0.0078125,
                   -0.00048828, -0.015625, -0.0078125, -0.00390625, 0.01556396,
                   -0.00778198, 0.00389099, -0.00778198, -0.01556396, -0.0019455])

    x = torch.as_tensor(x_, dtype=torch.float32, device=device)
    t = torch.as_tensor(t_, dtype=torch.float32, device=device)

    y = _quantize_bfloat(x, bfloat=16, round='even', custom_cuda=custom_cuda,
                         allow_denorm=allow_denorm)
    check_diff_quantize(x, y, t, tol=1e-8, handle_infs=False)


@pytest.mark.parametrize("bits, exp_bits", BITS__EXP_BITS)
@pytest.mark.parametrize("round", ['nearest', 'floor', 'even'])
@pytest.mark.parametrize("allow_denorm", [True, False])
@pytest.mark.parametrize("device, custom_cuda, half", DEVICE__CUSTOM_CUDA__HALF)
def test_exponents(bits, exp_bits, round, allow_denorm, device, custom_cuda, half):

    if exp_bits==8 and device=='cuda' and not custom_cuda and not half:
        pytest.xfail('Expected fail, not using custom CUDA, '
                     'likely due to Pytorch rounding inaccuracy')

    x = np.array([2**(n) for n in range(-20, 20)])

    if half:
    # Make sure inputs are representable in fp16
        x = x.astype(np.float16).astype(np.float32)
    x_torch = torch.as_tensor(x, dtype=torch.float32, device=torch.device("cpu"))
    y_torch = _quantize_fp(x_torch,
                           mantissa_bits=bits,
                           exp_bits=exp_bits,
                           round=round,
                           allow_denorm=allow_denorm,
                           custom_cuda=False)
    x_cuda = torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))
    x_cuda = x_cuda.half() if half else x_cuda
    y_cuda = _quantize_fp(x_cuda,
                          mantissa_bits=bits,
                          exp_bits=exp_bits,
                          round=round,
                          allow_denorm=allow_denorm,
                          custom_cuda=custom_cuda)
    y_cuda = y_cuda.float()
    check_diff_quantize(x, y_torch, y_cuda, handle_infs=True)


@pytest.mark.parametrize("device, custom_cuda", [
    ("cpu", False),
    ("cpu", True),
    ("cuda", True),
])
def test_torch_bfloat16(device, custom_cuda):
    """ There is a single discrepancy between our BF16 and torch.bfloat16:
        - For us, any values that exceed max_norm -> Inf.
        - For torch, values are rounded first, and values that exceed max_norm
          can round *down* to max_norm.
    """
    x_ = all_encodings(8, 9, device=device)
    x = torch.as_tensor(x_, dtype=torch.float32, device=device)

    # Cast to torch.bfloat16 and back
    t = x.clone().detach().to(torch.bfloat16).to(torch.float32)

    y = _quantize_bfloat(x, bfloat=16, round='even', custom_cuda=custom_cuda,
                         allow_denorm=True)
    check_diff_quantize(x, y, t, tol=0, handle_infs=True)


@pytest.mark.parametrize("input_size", [(100)])
@pytest.mark.parametrize("bits, exp_bits", BITS__EXP_BITS)
@pytest.mark.parametrize("round", ['nearest', 'floor', 'even'])
@pytest.mark.parametrize("allow_denorm", [True, False])
@pytest.mark.parametrize("device, custom_cuda, half", DEVICE__CUSTOM_CUDA__HALF)
def test_bfloat_random(input_size, bits, exp_bits, round, allow_denorm,
                         device, custom_cuda, half):
    for _ in range(ITERATIONS):
        x = np.random.normal(size=input_size)
        if half:
            # Make sure inputs are representable in fp16
            x = x.astype(np.float16).astype(np.float32)
        x_torch = torch.as_tensor(x, dtype=torch.float32, device=torch.device("cpu"))
        y_torch = _quantize_fp(x_torch,
                               mantissa_bits=bits,
                               exp_bits=exp_bits,
                               round=round,
                               allow_denorm=allow_denorm,
                               custom_cuda=False)
        x_cuda = torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))
        x_cuda = x_cuda.half() if half else x_cuda
        y_cuda = _quantize_fp(x_cuda,
                              mantissa_bits=bits,
                              exp_bits=exp_bits,
                              round=round,
                              allow_denorm=allow_denorm,
                              custom_cuda=custom_cuda)
        y_cuda = y_cuda.float()
        check_diff_quantize(x, y_torch, y_cuda)


@pytest.mark.parametrize("input_size", [(100)])
@pytest.mark.parametrize("bfloat", [16, 12])
@pytest.mark.parametrize("round", ['nearest', 'floor', 'even'])
@pytest.mark.parametrize("allow_denorm", [True, False])
@pytest.mark.parametrize("device, custom_cuda, half", DEVICE__CUSTOM_CUDA__HALF)
def test_fp_random(input_size, bfloat, round, allow_denorm,
                         device, custom_cuda, half):
    if bfloat <= 9:
        pytest.xfail('Expected fail. bfloat<=9')

    mx_specs = get_default_mx_specs()
    mx_specs['bfloat'] = bfloat
    mx_specs['bfloat16_subnorm'] = allow_denorm
    mx_specs['round'] = round
    mx_specs['custom_cuda'] = custom_cuda

    for _ in range(ITERATIONS):
        x = np.random.normal(size=input_size)
        if half:
            # Make sure inputs are representable in fp16
            x = x.astype(np.float16).astype(np.float32)
        x_torch = torch.as_tensor(x, dtype=torch.float32, device=torch.device("cpu"))
        y_torch = _quantize_bfloat(x_torch, bfloat=bfloat, round=round,
                                  custom_cuda=custom_cuda, allow_denorm=allow_denorm)
        x_cuda = torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))
        x_cuda = x_cuda.half() if half else x_cuda
        y_cuda = _quantize_bfloat(x_cuda, bfloat=bfloat, round=round,
                                  custom_cuda=custom_cuda, allow_denorm=allow_denorm)
        y_cuda = y_cuda.float()
        check_diff_quantize(x, y_torch, y_cuda)


@pytest.mark.parametrize("elem_format, encodes_infs", [
    ("int8", False),
    ("fp8_e4m3", False),
    ("fp8_e5m2", True),
    ("fp6_e3m2", False),
    ("fp6_e2m3", False),
    ("fp4_e2m1", False),
    ("bf16", True),
    ("fp16", True),
])
@pytest.mark.parametrize("round", ['nearest', 'floor', 'even'])
@pytest.mark.parametrize("device, custom_cuda", [
    ("cpu",  False),
    ("cpu",  True),
    ("cuda", True)
])
def test_elem_formats_exact(elem_format, encodes_infs, round,
                            device, custom_cuda):

    # generate inputs with exact number of mbits
    ebits, mbits, _, _, _ = _get_format_params(elem_format)
    if ebits == 0:
        mbits = mbits - 1   # remove sign bit
    else:
        mbits = mbits - 2   # remove sign and implicit bits

    x = all_encodings(ebits, mbits, encodes_infs,
                                device=device)

    y = _quantize_elemwise(x, elem_format, round=round,
                           allow_denorm=True,
                           custom_cuda=custom_cuda)

    check_diff(x, y)


@pytest.mark.parametrize("elem_format, encodes_infs", [
    ("int8", False),
    ("fp8_e4m3", False),
    ("fp8_e5m2", True),
    ("fp6_e3m2", False),
    ("fp6_e2m3", False),
    ("fp4_e2m1", False),
    ("bf16", True),
    ("fp16", True),
])
@pytest.mark.parametrize("round", ['nearest', 'floor', 'even'])
@pytest.mark.parametrize("saturate_normals", [True, False])
@pytest.mark.parametrize("allow_denorm", [True, False])
@pytest.mark.parametrize("device, custom_cuda", [
    ("cpu",  False),
    ("cpu",  True),
    ("cuda", True)
])
def test_elem_formats_round(elem_format, encodes_infs, round,
                            saturate_normals, allow_denorm,
                            device, custom_cuda):

    # generate inputs with 2 additional mbits, so rounding for 0.25, 0.5, 0.75
    ebits, mbits, _, _, _ = _get_format_params(elem_format)
    if ebits == 0:
        mbits = mbits + 1

    x1 = all_encodings(ebits, mbits, encodes_infs,
                       device='cpu')
    x2 = x1.clone().detach().to(device)

    y1 = _quantize_elemwise(x1, elem_format, round=round,
                            saturate_normals=saturate_normals,
                            allow_denorm=allow_denorm,
                            custom_cuda=False)
    y2 = _quantize_elemwise(x2, elem_format, round=round,
                            saturate_normals=saturate_normals,
                            allow_denorm=allow_denorm,
                            custom_cuda=custom_cuda)

    check_diff_quantize(x1, y1, y2)

@pytest.mark.parametrize("input_size", [(1000)])
@pytest.mark.parametrize("bfloat", [16, 12])
@pytest.mark.parametrize("round", ['nearest', 'floor', 'even'])
@pytest.mark.parametrize("allow_denorm", [True, False])
def test_cpu_cuda_random(input_size, bfloat, round, allow_denorm):
    if bfloat <= 9:
        pytest.xfail('Expected fail. bfloat<=9')

    mx_specs = get_default_mx_specs()
    mx_specs['bfloat'] = bfloat
    mx_specs['bfloat16_subnorm'] = allow_denorm
    mx_specs['round'] = round
    mx_specs['custom_cuda'] = True

    for _ in range(ITERATIONS):
        x = np.random.normal(size=input_size)
        x_torch = torch.as_tensor(x, dtype=torch.float32, device=torch.device("cpu"))
        y_torch = _quantize_bfloat(x_torch, bfloat=bfloat, round=round,
                                  custom_cuda=True, allow_denorm=allow_denorm)
        x_cuda = torch.as_tensor(x, dtype=torch.float32, device=torch.device('cuda'))
        y_cuda = _quantize_bfloat(x_cuda, bfloat=bfloat, round=round,
                                  custom_cuda=True, allow_denorm=allow_denorm)
        check_diff_quantize(x, y_torch, y_cuda)