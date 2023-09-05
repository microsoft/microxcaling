"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Functions and classes used by multiple test modules

Functions:
  get_cuda_major_version
  check_diff
  float16
  decode_float16
  truncate_float16

Classes:
  Float16Iterator
"""

import json
import torch
import struct
import numpy as np
from packaging import version

def torch_version_ge(v):
    # Return True if torch major version >= v
    if type(v) is str:
        v = version.parse(v)
    return version.parse(torch.__version__) >= v


def get_s_e_m(value_in_float):
    def float_to_bits(value_in_float):
            s = struct.pack('@f', value_in_float)
            return struct.unpack('@I', s)[0]
    bits = float_to_bits(value_in_float) # bits in form of uint32
    sign = (bits & 0x80000000) >> 31 # sign bit
    exp = ((bits & 0x7F800000) >> 23) - 127 # exponent
    mant = bits & 0x007FFFFF
    return sign, exp, mant


def check_diff(y1, y2, tol=0, ntol=0, handle_infs=False):
    """ In floating-point x==y with inf on both sides returns NaN.
        If handle_infs is True, then we allow inf==inf to pass.
    """
    __tracebackhide__ = True

    if y1 == None and y2 == None:
        return
    elif y1 == None:
        raise ValueError("y1 == None, but y2 exists")
    elif y2 == None:
        raise ValueError("y1 exists, but y2 == None")

    # Check shapes
    if y1.size() != y2.size():
        raise ValueError("Size mismatch: ",
                list(y1.size()), '!=', list(y2.size()))

    # Convert to numpy
    y1 = y1.detach().cpu().numpy()
    y2 = y2.detach().cpu().numpy()

    # Handle infs
    if handle_infs:
        y1_infs = np.isinf(y1)
        y2_infs = np.isinf(y2)
        y1 = np.where(y1_infs, 0., y1)
        y2 = np.where(y2_infs, 0., y2)
    else:
        y1_infs = None
        y2_infs = None

    # Check for differences
    diff = abs(y1 - y2)
    max_diff = np.max(diff)                     # max error
    tol = tol + tol*np.sqrt(np.mean(y1.astype(np.float64)**2))     # error tolerance
    ndiff = np.sum(diff > tol)                  # num of violations
    if (max_diff > tol and ndiff > ntol) or not np.all(np.isfinite(diff)) or np.any(y1_infs != y2_infs):
        where_diff = diff > tol
        print("%d/%d mismatches" % (np.count_nonzero(where_diff), where_diff.size))
        print("y1:")
        print(y1[where_diff])
        print("y2:")
        print(y2[where_diff])
        print("Diff:")
        print(diff[where_diff])
        raise ValueError


def check_diff_quantize(x, y1, y2, tol=0, ntol=0, handle_infs=False):
    """ In floating-point x==y with inf on both sides returns NaN.
        If handle_infs is True, then we allow inf==inf to pass.
    """
    __tracebackhide__ = True

    # Check shapes
    if y1.size() != y2.size():
        print("Size mismatch: ", list(y1.size()), '!=', list(y2.size()))
        raise IndexError

    # Convert to numpy
    x = np.array(x) if type(x) is list else x
    x = x.cpu().numpy() if type(x) is torch.Tensor else x
    y1 = y1.detach().cpu().numpy()
    y2 = y2.detach().cpu().numpy()

    torch_infs = np.isinf(y1) | np.isnan(y1)
    cuda_infs = np.isinf(y2)| np.isnan(y2)
    y1_ = np.where(torch_infs, 0., y1)
    y2_ = np.where(cuda_infs, 0., y2)
    diff = abs(y1_ - y2_)

    # Don't compare infs if requested
    if not handle_infs:
        torch_infs = None
        cuda_infs = None

    # Check for differences
    max_diff = np.max(diff)
    ndiff = np.sum(diff > tol)                # num of violations
    if (max_diff > tol and ndiff > ntol) or np.any(torch_infs != cuda_infs):
        where_diff = (diff != 0) | (torch_infs != cuda_infs)
        print("%d/%d mismatches" % (np.count_nonzero(where_diff), where_diff.size))
        print("First and last mismatch:")
        print("Orig:", x[where_diff][0], get_s_e_m(x[where_diff][0]))
        print("y1:  ", y1[where_diff][0], get_s_e_m(y1[where_diff][0]))
        print("y2:  ", y2[where_diff][0], get_s_e_m(y2[where_diff][0]))
        if (np.count_nonzero(where_diff) > 1):
            print("--------------------")
            print("Orig:", x[where_diff][-1], get_s_e_m(x[where_diff][-1]))
            print("y1:  ", y1[where_diff][-1], get_s_e_m(y1[where_diff][-1]))
            print("y2:  ", y2[where_diff][-1], get_s_e_m(y2[where_diff][-1]))
        raise ValueError


#--------------------------------------------------------------------
# This function generates the set of all quantization points for eXmY
# floating-point
#--------------------------------------------------------------------
_CACHE = {}
def all_encodings(_e, _m, encodes_infs=True, device="cpu"):
    if (_e, _m, encodes_infs) in _CACHE:
        x = _CACHE[(_e, _m, encodes_infs)]
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    # Holds all positive and negative
    x = np.zeros((2**(_e+_m+1)), dtype=np.float32)

    for _i in range(2**(_e+_m)):
        if _e > 0:
            _exp = _i>>_m
            # Skip exp == all ones
            if encodes_infs and _exp == 2**_e - 1:
                continue
            # Normal or subnormal encoding
            if _exp == 0:
                _exp = 1 - (2**(_e-1) - 1)
                _explicit = 0.0
            else:
                _exp -= 2**(_e-1) - 1
                _explicit = 1.0
            # Obtain mantissa value
            _mant = _i&((2**_m)-1)
            _mmant = _mant / (2**_m)

            # FP8 e4m3 hack
            if _e == 4 and _m == 3 and _exp == 8 and _mmant == 0.875:
                _value = 0
            else:
                _value = 2**(_exp) * (_explicit + _mmant)
        else:
            _value = _i / (2**(_m-1))

        x[_i] = _value
        x[_i + 2**(_e+_m)] = -_value

    _CACHE[(_e, _m, encodes_infs)] = x

    return torch.as_tensor(x, dtype=torch.float32, device=device)
