"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Test elemwise corner cases (Infs, NaNs, Denorms, etc).
"""

import pytest
import torch
import numpy as np
import sys

from .common_lib import check_diff_quantize

from mx.elemwise_ops import _quantize_bfloat, _quantize_fp

np.random.seed(0xd10)

DEVICE__CUSTOM_CUDA = [
    ('cpu', False),
    ('cuda', False),
    ('cuda', True)
]


@pytest.mark.parametrize("round", ['nearest'])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_bf16_max(round, device, custom_cuda):
    """ quantize_bfloat should return Inf/NaN for out-of-bounds values """
    x = np.array([ 2**127 * (2**8 - 1) / 2**7,   # max_norm
                  -2**127 * (2**8 - 1) / 2**7,
                   2**127 * (2**10 - 3) / 2**9,  # rounds to max_norm
                  -2**127 * (2**10 - 3) / 2**9,
                   2**127 * (2**10 - 1) / 2**9,  # rounds to Inf
                  -2**127 * (2**10 - 1) / 2**9])

    x_torch = torch.as_tensor(x, dtype=torch.float32, device=torch.device("cpu"))
    y_torch = _quantize_bfloat(x_torch,
                               bfloat=16,
                               round=round,
                               custom_cuda=False)
    x_cuda = torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))
    y_cuda = _quantize_bfloat(x_cuda,
                              bfloat=16,
                              round=round,
                              custom_cuda=custom_cuda)

    isfinite = torch.isfinite(y_torch).numpy()
    assert(np.allclose(isfinite, [True, True, True, True, False, False]))

    check_diff_quantize(x, y_torch, y_cuda, tol=0, handle_infs=True)


@pytest.mark.parametrize("bits, exp_bits", [(10,5)])
@pytest.mark.parametrize("round", ['nearest'])
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_fp16_max(bits, exp_bits, round, device, custom_cuda):
    """ quantize_fp should return Inf/NaN for out-of-bounds values """
    x = np.array([65504., -65504.,  # max_norm
                  65519., -65519.,  # rounds to max_norm
                  65520., -65520.]) # rounds to Inf

    x_torch = torch.as_tensor(x, dtype=torch.float32, device=torch.device("cpu"))
    y_torch = _quantize_fp(x_torch,
                           mantissa_bits=bits,
                           exp_bits=exp_bits,
                           round=round,
                           custom_cuda=False)
    x_cuda = torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))
    y_cuda = _quantize_fp(x_cuda,
                          mantissa_bits=bits,
                          exp_bits=exp_bits,
                          round=round,
                          custom_cuda=custom_cuda)

    isfinite = torch.isfinite(y_torch).numpy()
    assert(np.allclose(isfinite, [True, True, True, True, False, False]))

    check_diff_quantize(x, y_torch, y_cuda, tol=0, handle_infs=True)


@pytest.mark.parametrize("func", [
    '_quantize_bfloat',
    '_quantize_fp'
])
@pytest.mark.parametrize("bfloat", [16]) # bfloat16
@pytest.mark.parametrize("round", ['nearest', 'floor', 'even']) # bfloat16
@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_bfloat_nans(func, bfloat, round, device, custom_cuda):
    """ Bfloat should preserve NaNs and Infs. """
    x = np.array([0., np.nan, np.inf, 0.])

    x_cuda = torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))

    if func == '_quantize_bfloat':
        y_cuda = _quantize_bfloat(x_cuda,
                                  bfloat=bfloat,
                                  round=round,
                                  custom_cuda=custom_cuda)
    else:
        y_cuda = _quantize_fp(x_cuda,
                              mantissa_bits=bfloat-7,
                              exp_bits=8,
                              round=round,
                              custom_cuda=custom_cuda)

    y_cuda = y_cuda.float()

    check_diff_quantize(x, x_cuda, y_cuda, handle_infs=True)


# a float x by another float e (x = x * 2**e) is inaccurate when e >= 16
# on GPU. CPU and custom_cuda works fine.
@pytest.mark.parametrize("bits, exp_bits", [(7, 8)])
@pytest.mark.parametrize("device, custom_cuda", [
    ("cpu",  False),
    #("cuda", False),
    ("cuda", True)
])
def test_custom_shift(bits, exp_bits, device, custom_cuda):
    """ Test to catch possible issues with 2**e in the code. """
    x = np.array([65535., 65536., 131072., 262144., 524288.])
    x_torch = torch.as_tensor(x, dtype=torch.float32, device='cpu')
    x_cuda = torch.as_tensor(x, dtype=torch.float32, device=torch.device(device))
    y_torch = _quantize_fp(x_torch,
                           mantissa_bits=bits,
                           exp_bits=exp_bits,
                           custom_cuda=False)
    y_cuda = _quantize_fp(x_cuda,
                          mantissa_bits=bits,
                          exp_bits=exp_bits,
                          custom_cuda=custom_cuda)
    check_diff_quantize(x, y_torch, y_cuda, handle_infs=True)


@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_bfloat16_round(device, custom_cuda):
    x_ = np.array([65535., -65535., -1.9985847, 0.4999000132083893, -0.49968934059143066])
    t_ = np.array([65536., -65536., -2.0, 0.5, -0.5])
    x = torch.as_tensor(x_, dtype=torch.float32, device=device)
    t = torch.as_tensor(t_, dtype=torch.float32, device=device)

    y = _quantize_bfloat(x, bfloat=16, round='nearest', custom_cuda=custom_cuda)
    check_diff_quantize(x, y, t, handle_infs=False)


@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
@pytest.mark.parametrize("allow_denorm", [False, True])
def test_float16_subnorms(device, custom_cuda, allow_denorm):
    x_ = np.array([2**(-14), 2**(-14)*0.5, 2**(-14)*(2**-10), 2**(-14)*(2**-11),
                   2**(-14)*(2**-12),
                   2**(-14)*(1023/1024),  # Largest subnorm
                   2**(-14)*(2047/2048)]) # Rounds up to smallest normal

    if allow_denorm:
        t_ = np.array([2**(-14), 2**(-14)*0.5, 2**(-14)*(2**-10), 2**(-14)*(2**-10),
                       0,
                       2**(-14)*(1023/1024),
                       2**(-14)])
    else:
        t_ = np.array([2**(-14), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    x = torch.as_tensor(x_, dtype=torch.float32, device=device)
    t = torch.as_tensor(t_, dtype=torch.float32, device=device)

    y = _quantize_fp(x, mantissa_bits=10, exp_bits=5,
                     round='nearest', allow_denorm=allow_denorm,
                     custom_cuda=custom_cuda)
    check_diff_quantize(x, y, t, handle_infs=True)


@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_bfloat16_limits(device, custom_cuda):
    # 1.9921875 is the max bfloat16 mantissa 1.1111111
    # 1.99609375 is the next mantissa 1.11111111
    x_ = np.array([2**127 * 1.9921875, 2**127 * 1.9921874, -2**127 * 1.9921875,
                   2**127 * 1.99609375, -2**127 * 1.99609375])
    t_ = np.array([2**127 * 1.9921875, 2**127 * 1.9921875, -2**127 * 1.9921875,
                   np.inf, -np.inf])
    x = torch.as_tensor(x_, dtype=torch.float32, device=device)
    t = torch.as_tensor(t_, dtype=torch.float32, device=device)

    y = _quantize_fp(x, mantissa_bits=7, exp_bits=8,
                     round='nearest', custom_cuda=custom_cuda)
    check_diff_quantize(x, y, t, handle_infs=True)


@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
@pytest.mark.parametrize("allow_denorm", [False, True])
def test_bfloat16_subnorms(device, custom_cuda, allow_denorm):
    x_ = np.array([2**(-126), 2**(-126)*0.5, 2**(-126)*(2**-7), 2**(-126)*(2**-8),
                   2**(-126)*(2**-9),
                   2**(-126)*(127/128),     # Largest subnorm
                   2**(-126)*(255/256)])    # Rounds up to smallest normal

    if allow_denorm:
        t_ = np.array([2**(-126), 2**(-126)*0.5, 2**(-126)*(2**-7), 2**(-126)*(2**-7),
                       0,
                       2**(-126)*(127/128),
                       2**(-126)])
    else:
        t_ = np.array([2**(-126), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    x = torch.as_tensor(x_, dtype=torch.float32, device=device)
    t = torch.as_tensor(t_, dtype=torch.float32, device=device)

    y = _quantize_fp(x, mantissa_bits=7, exp_bits=8,
                     round='nearest', allow_denorm=allow_denorm,
                     custom_cuda=custom_cuda)
    check_diff_quantize(x, y, t, handle_infs=True)


@pytest.mark.parametrize("device, custom_cuda", DEVICE__CUSTOM_CUDA)
def test_subnorm_rne(device, custom_cuda):
    #                         round down           round up
    x_ = np.array([2**(-126)*(253/256), 2**(-126)*(251/256)])
    t_ = np.array([2**(-126)*(126/128), 2**(-126)*(126/128)])

    x = torch.as_tensor(x_, dtype=torch.float32, device=device)
    t = torch.as_tensor(t_, dtype=torch.float32, device=device)

    y = _quantize_fp(x, mantissa_bits=7, exp_bits=8,
                     round='even', allow_denorm=True,
                     custom_cuda=custom_cuda)
    check_diff_quantize(x, y, t, handle_infs=True)

