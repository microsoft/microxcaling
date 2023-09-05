"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import pytest
import sys

from mx.formats import _get_format_params


def test_format_parameters():
    ebits, mbits, emax, max_norm, min_norm = _get_format_params("float16")
    assert(ebits == 5)
    assert(emax == 15)
    assert(max_norm == 2**15 * (1 + 1023/1024))
    assert(min_norm == 2**(-14))
    print("Float16 Pass!")
    ebits, mbits, emax, max_norm, min_norm = _get_format_params("fp16")
    assert(ebits == 5)
    assert(emax == 15)
    assert(max_norm == 2**15 * (1 + 1023/1024))
    assert(min_norm == 2**(-14))
    print("FP16 Pass!")

    ebits, mbits, emax, max_norm, min_norm = _get_format_params("bfloat16")
    assert(ebits == 8)
    assert(emax == 127)
    assert(max_norm == 2**127 * (1 + 127/128))
    assert(min_norm == 2**(-126))
    print("Bfloat16 Pass!")
    ebits, mbits, emax, max_norm, min_norm = _get_format_params("bf16")
    assert(ebits == 8)
    assert(emax == 127)
    assert(max_norm == 2**127 * (1 + 127/128))
    assert(min_norm == 2**(-126))
    print("BF16 Pass!")

    ebits, mbits, emax, max_norm, min_norm = _get_format_params("fp8_e5m2")
    assert(ebits == 5)
    assert(emax == 15)
    assert(max_norm == 2**15 * 1.75)
    assert(min_norm == 2**(-14))
    print("FP8_e5m2 Pass!")

    ebits, mbits, emax, max_norm, min_norm = _get_format_params("fp8_e4m3")
    assert(ebits == 4)
    assert(emax == 8)
    assert(max_norm == 2**8 * 1.75)
    assert(min_norm == 2**(-6))
    print("FP8_e4m3 Pass!")

    ebits, mbits, emax, max_norm, min_norm = _get_format_params("fp6_e3m2")
    assert(ebits == 3)
    assert(emax == 4)
    assert(max_norm == 2**4 * 1.75)
    assert(min_norm == 2**(-2))
    print("FP6_e3m2 Pass!")

    ebits, mbits, emax, max_norm, min_norm = _get_format_params("fp6_e2m3")
    assert(ebits == 2)
    assert(emax == 2)
    assert(max_norm == 2**2 * 1.875)
    assert(min_norm == 2**(0))
    print("FP6_e2m3 Pass!")

    ebits, mbits, emax, max_norm, min_norm = _get_format_params("fp4_e2m1")
    assert(ebits == 2)
    assert(emax == 2)
    assert(max_norm == 2**2 * 1.5)
    assert(min_norm == 2**(0))
    print("FP4_e2m1 Pass!")

    ebits, mbits, emax, max_norm, min_norm = _get_format_params("fp4")
    assert(ebits == 2)
    assert(emax == 2)
    assert(max_norm == 2**2 * 1.5)
    assert(min_norm == 2**(0))
    print("FP4_e2m1 Pass!")

    ebits, mbits, emax, max_norm, min_norm = _get_format_params("int8")
    assert(ebits == 0)
    assert(mbits == 8)
    assert(emax == 0)
    assert(max_norm == 1 + 63/64)
    assert(min_norm == 0)
    print("INT8 Pass!")

    ebits, mbits, emax, max_norm, min_norm = _get_format_params("int4")
    assert(ebits == 0)
    assert(mbits == 4)
    assert(emax == 0)
    assert(max_norm == 1.75)
    assert(min_norm == 0)
    print("INT4 Pass!")
