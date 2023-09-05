"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Name:    mx_ops.py

Pytorch methods for MX quantization.

Usage Notes:
 - Use the "Exposed Methods" below to implement autograd functions
 - Use autograd functions to then implement torch.nn.Module(s)
 - Do *not* use methods in this file in Modules, they have no defined
   backwards pass and will block gradient computation.
 - Avoid importing internal function if at all possible.

Exposed Methods:
    quantize_mx_op - quantizes a tensor to MX format.

Internal Methods:
    _safe_lshift, _safe_rshift - fp16 compatible shifts
    _shared_exponents - Returns MX shared exponent for the passed tensor
    _reshape_to_blocks - tiles a tensor by splitting one dim into two
    _undo_reshape_to_blocks - undos the above reshaping
    _quantize_mx - quantizes a tensor to MX format
"""

import os
import torch
import numpy as np

from .specs import mx_assert_test
from .formats import (
        RoundingMode,
        ElemFormat,
        FP32_EXPONENT_BIAS,
        FP32_MIN_NORMAL,
        _get_format_params
)
from .elemwise_ops import (
        _safe_lshift, _safe_rshift,
        _round_mantissa,
        _quantize_elemwise_core
)


# -------------------------------------------------------------------------
# Helper funcs
# -------------------------------------------------------------------------
def _shared_exponents(A, method="max", axes=None, ebits=0):
    """
    Get shared exponents for the passed matrix A.
    Args:
      A      {PyTorch tensor} -- Input tensor
      method {str}            -- Exponent selection method.
                                 "max" uses the max absolute value
                                 "none" uses an exponent for each value (i.e., no sharing)
      axes   {list(int)}      -- List of integers which specifies the axes across which
                                 shared exponents are calculated.
    Returns:
      shared_exp {PyTorch tensor} -- Tensor of shared exponents
    """

    if method == "max":
        if axes is None:
            shared_exp = torch.max(torch.abs(A))
        else:
            shared_exp = A
            for axis in axes:
                shared_exp, _ = torch.max(torch.abs(shared_exp), dim=axis, keepdim=True)
    elif method == "none":
        shared_exp = torch.abs(A)
    else:
        raise Exception("Unrecognized shared exponent selection method %s" % (method))

    # log2(shared_exp) and truncate to integer
    shared_exp = torch.floor(
        torch.log2(
            shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
        )
    )

    # Restrict to [-emax, emax] range
    if ebits > 0:
        emax = 2**(ebits-1) - 1
        #shared_exp = torch.clamp(shared_exp, -emax, emax)
        # Overflow to Inf
        shared_exp[shared_exp > emax] = float("NaN")
        # Underflows are set to -127 which causes them to be
        # flushed to 0 later
        shared_exp[shared_exp < -emax] = -emax

    return shared_exp


def _reshape_to_blocks(A, axes, block_size):
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension toapply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i  # Shift axes due to added dimensions
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        # Don't pad if the axis is short enough to fit inside one tile
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)

    A = A.view(reshape)
    return A, axes, orig_shape, padded_shape


def _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes):
    # Undo tile reshaping
    A = A.view(padded_shape)
    # Undo padding
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        # Remove extra dimension
        A = torch.squeeze(A, dim=axis + 1)
    return A


# -------------------------------------------------------------------------
# Main funcs
# -------------------------------------------------------------------------
def _quantize_mx(
    A,
    scale_bits,
    elem_format,    # can be None for no quantization
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
    custom_cuda=False,
):
    """Function used for MX* quantization
    """
    # Shortcut for no quantization
    if elem_format == None:
        return A

    assert(scale_bits > 0)

    # Make sure axes is a list of non-negative numbers
    axes = [axes] if type(axes) == int else axes
    axes = [x + A.ndim if x < 0 else x for x in axes]

    # Custom CUDA only supports limited rounding modes
    custom_cuda = custom_cuda and round in RoundingMode.string_enums()

    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)

    # Use quantize_mx_by_tile when there is only a single shared axis and
    # - The block size is small, OR
    # - The shared axis is not the innermost
    if A.device.type == "cuda" and custom_cuda and len(axes) == 1:
        axis = axes[0]
        if block_size == 0:
            block_size = A.shape[axis]

        if axis != len(A.shape) - 1 or block_size <= 32:
            A = A.contiguous()

            from . import custom_extensions as ce
            A = ce.funcs.quantize_mx_by_tile_func_cuda(
                A,
                scale_bits,
                ebits,
                mbits,
                max_norm,
                block_size,
                axis,
                flush_fp32_subnorms,
                RoundingMode[round],
            )
            return A


    # Perform tiling to the hardware vector size
    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(
            A, axes, block_size
        )

    ####################
    # Quantize
    ####################
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    if custom_cuda:
        # Custom CUDA code only supports a single axis
        if shared_exp_axes is None:
            axis = 0
        else:
            assert len(shared_exp_axes) == 1
            axis = shared_exp_axes[0]

        assert(shared_exp_method == "max")
        max_values = A.abs().max(dim=axis, keepdim=True).values

        A = A.contiguous()

        if A.device.type == "cuda":
            from . import custom_extensions as ce
            A = ce.funcs.quantize_mx_func_cuda(
                A, scale_bits, ebits, mbits, max_norm,
                max_values, axis,
                flush_fp32_subnorms, RoundingMode[round]);

        elif A.device.type == "cpu":
            from . import custom_extensions as ce
            A = ce.funcs.quantize_mx_func_cpp(
                A, scale_bits, ebits, mbits, max_norm,
                max_values, axis,
                flush_fp32_subnorms, RoundingMode[round]);

        else:
            raise ValueError("Unrecognized device type %s" % A.device.type)
    else:
        # Get shared exponents
        shared_exp = _shared_exponents(
            A, method=shared_exp_method, axes=shared_exp_axes, ebits=0,
        )

        # Flush subnormal FP32 inputs to zero
        if flush_fp32_subnorms:
            A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

        # Offset the max exponent by the largest representable exponent
        # in the element data format
        shared_exp = shared_exp - emax

        scale_emax = 2**(scale_bits-1) - 1
        shared_exp[shared_exp > scale_emax] = float("NaN")
        shared_exp[shared_exp < -scale_emax] = -scale_emax

        A = A / (2**shared_exp)

        A = _quantize_elemwise_core(
                A, mbits, ebits, max_norm, round=round,
                allow_denorm=True, saturate_normals=True,
                custom_cuda=custom_cuda)

        A = A * (2**shared_exp)

    # Undo tile reshaping
    if block_size:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A


def quantize_mx_op(
    A,
    mx_specs: dict,
    elem_format=None,
    block_size=None,
    axes=None,
    round="nearest",
    expand_and_reshape=False,
):
    mx_assert_test(mx_specs)

    if elem_format == None:
        return A
    elif type(elem_format) is str:
        elem_format = ElemFormat.from_str(elem_format)

    if block_size == None:
        block_size = mx_specs["block_size"]

    if mx_specs["scale_bits"] == 0:
        scale_bits = 8
    else:
        scale_bits = mx_specs["scale_bits"]

    return _quantize_mx(
            A, scale_bits,
            elem_format, block_size=block_size,
            axes=axes, round=round,
            shared_exp_method=mx_specs["shared_exp_method"],
            flush_fp32_subnorms=mx_specs["mx_flush_fp32_subnorms"],
            custom_cuda=mx_specs["custom_cuda"])
