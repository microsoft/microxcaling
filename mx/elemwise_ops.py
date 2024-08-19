"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Name:    elemwise_ops.py

Pytorch functions for elementwise (i.e. bfloat) quantization.

Usage Notes:
 - Use the "Exposed Methods" below to implement autograd functions
 - Use autograd functions to then implement torch.nn.Module(s)
 - Do *not* use methods in this file in Modules, they have no defined
   backwards pass and will block gradient computation.
 - Avoid importing internal function if at all possible.

Exposed Methods:
    quantize_elemwise_op - quantizes a tensor to bfloat or other
                           custom float format
"""
import torch

from .formats import RoundingMode, _get_format_params
from .formats import _get_min_norm, _get_max_norm


# -------------------------------------------------------------------------
# Helper funcs
# -------------------------------------------------------------------------
# Never explicitly compute 2**(-exp) since subnorm numbers have
# exponents smaller than -126
def _safe_lshift(x, bits, exp):
    if exp is None:
        return x * (2**bits)
    else:
        return x / (2 ** exp) * (2**bits)


def _safe_rshift(x, bits, exp):
    if exp is None:
        return x / (2**bits)
    else:
        return x / (2**bits) * (2 ** exp)


def _round_mantissa(A, bits, round, clamp=False):
    """
    Rounds mantissa to nearest bits depending on the rounding method 'round'
    Args:
      A     {PyTorch tensor} -- Input tensor
      round {str}            --  Rounding method
                                 "floor" rounds to the floor
                                 "nearest" rounds to ceil or floor, whichever is nearest
    Returns:
      A {PyTorch tensor} -- Tensor with mantissas rounded
    """

    if round == "dither":
        rand_A = torch.rand_like(A, requires_grad=False)
        A = torch.sign(A) * torch.floor(torch.abs(A) + rand_A)
    elif round == "floor":
        A = torch.sign(A) * torch.floor(torch.abs(A))
    elif round == "nearest":
        A = torch.sign(A) * torch.floor(torch.abs(A) + 0.5)
    elif round == "even":
        absA = torch.abs(A)
        # find 0.5, 2.5, 4.5 ...
        maskA = ((absA - 0.5) % 2 == torch.zeros_like(A)).type(A.dtype)
        A = torch.sign(A) * (torch.floor(absA + 0.5) - maskA)
    else:
        raise Exception("Unrecognized round method %s" % (round))

    # Clip values that cannot be expressed by the specified number of bits
    if clamp:
        max_mantissa = 2 ** (bits - 1) - 1
        A = torch.clamp(A, -max_mantissa, max_mantissa)
    return A


# -------------------------------------------------------------------------
# Main funcs
# -------------------------------------------------------------------------
def _quantize_elemwise_core(A, bits, exp_bits, max_norm, round='nearest',
                            saturate_normals=False, allow_denorm=True,
                            custom_cuda=False):
    """ Core function used for element-wise quantization
    Arguments:
      A         {PyTorch tensor} -- A tensor to be quantized
      bits      {int}            -- Number of mantissa bits. Includes
                                    sign bit and implicit one for floats
      exp_bits  {int}            -- Number of exponent bits, 0 for ints
      max_norm  {float}          -- Largest representable normal number
      round     {str}            -- Rounding mode: (floor, nearest, even)
      saturate_normals {bool}    -- If True, normal numbers (i.e., not NaN/Inf)
                                    that exceed max norm are clamped.
                                    Must be True for correct MX conversion.
      allow_denorm     {bool}    -- If False, flush denorm numbers in the
                                    elem_format to zero.
      custom_cuda      {str}     -- If True, use custom CUDA kernels
    Returns:
      quantized tensor {PyTorch tensor} -- A tensor that has been quantized
    """
    A_is_sparse = A.is_sparse
    if A_is_sparse:
        if A.layout != torch.sparse_coo:
            raise NotImplementedError("Only COO layout sparse tensors are currently supported.")

        sparse_A = A.coalesce()
        A = sparse_A.values().clone()

    # custom cuda only support floor and nearest rounding modes
    custom_cuda = custom_cuda and round in RoundingMode.string_enums()

    if custom_cuda:
        A = A.contiguous()

        from . import custom_extensions
        if A.device.type == "cuda":
            A = custom_extensions.funcs.quantize_elemwise_func_cuda(
                A, bits, exp_bits, max_norm, RoundingMode[round],
                saturate_normals, allow_denorm)
        elif A.device.type == "cpu":
            A = custom_extensions.funcs.quantize_elemwise_func_cpp(
                A, bits, exp_bits, max_norm, RoundingMode[round],
                saturate_normals, allow_denorm)
        return A

    # Flush values < min_norm to zero if denorms are not allowed
    if not allow_denorm and exp_bits > 0:
        min_norm = _get_min_norm(exp_bits)
        out = (torch.abs(A) >= min_norm).type(A.dtype) * A
    else:
        out = A

    if exp_bits != 0:
        private_exp = torch.floor(torch.log2(
            torch.abs(A) + (A == 0).type(A.dtype)))

        # The minimum representable exponent for 8 exp bits is -126
        min_exp = -(2**(exp_bits-1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    # Scale up so appropriate number of bits are in the integer portion of the number
    out = _safe_lshift(out, bits - 2, private_exp)

    out = _round_mantissa(out, bits, round, clamp=False)

    # Undo scaling
    out = _safe_rshift(out, bits - 2, private_exp)

    # Set values > max_norm to Inf if desired, else clamp them
    if saturate_normals or exp_bits == 0:
        out = torch.clamp(out, min=-max_norm, max=max_norm)
    else:
        out = torch.where((torch.abs(out) > max_norm),
                           torch.sign(out) * float("Inf"), out)

    # handle Inf/NaN
    if not custom_cuda:
        out[A == float("Inf")] = float("Inf")
        out[A == -float("Inf")] = -float("Inf")
        out[A == float("NaN")] = float("NaN")

    if A_is_sparse:
        output = torch.sparse_coo_tensor(sparse_A.indices(), output,
                sparse_A.size(), dtype=sparse_A.dtype, device=sparse_A.device,
                requires_grad=sparse_A.requires_grad)

    return out


def _quantize_elemwise(A, elem_format, round='nearest', custom_cuda=False,
                       saturate_normals=False, allow_denorm=True):
    """ Quantize values to a defined format. See _quantize_elemwise_core()
    """
    if elem_format == None:
        return A

    ebits, mbits, _, max_norm, _ = _get_format_params(elem_format)

    output = _quantize_elemwise_core(
            A, mbits, ebits, max_norm,
            round=round, allow_denorm=allow_denorm,
            saturate_normals=saturate_normals,
            custom_cuda=custom_cuda)

    return output


def _quantize_bfloat(A, bfloat, round='nearest', custom_cuda=False, allow_denorm=True):
    """ Quantize values to bfloatX format
    Arguments:
      bfloat      {int}       -- Total number of bits for bfloatX format,
                                 Includes 1 sign, 8 exp bits, and variable
                                 mantissa bits. Must be >= 9.
    """
    # Shortcut for no quantization
    if bfloat == 0 or bfloat == 32:
        return A

    max_norm = _get_max_norm(8, bfloat-7)

    return _quantize_elemwise_core(
            A, bits=bfloat-7, exp_bits=8, max_norm=max_norm, round=round,
            allow_denorm=allow_denorm, custom_cuda=custom_cuda)


def _quantize_fp(A, exp_bits=None, mantissa_bits=None,
                 round='nearest', custom_cuda=False, allow_denorm=True):
    """ Quantize values to IEEE fpX format. The format defines NaN/Inf
        and subnorm numbers in the same way as FP32 and FP16.
    Arguments:
        exp_bits        {int} -- number of bits used to store exponent
        mantissa_bits   {int} -- number of bits used to store mantissa, not
                                 including sign or implicit 1
        round           {str} -- Rounding mode, (floor, nearest, even)
    """
    # Shortcut for no quantization
    if exp_bits is None or mantissa_bits is None:
        return A

    max_norm = _get_max_norm(exp_bits, mantissa_bits+2)

    output = _quantize_elemwise_core(
            A, bits=mantissa_bits + 2, exp_bits=exp_bits,
            max_norm=max_norm, round=round, allow_denorm=allow_denorm,
            custom_cuda=custom_cuda)

    return output


def quantize_elemwise_op(A, mx_specs, round=None):
    """A function used for element-wise quantization with mx_specs
    Arguments:
      A          {PyTorch tensor} -- a tensor that needs to be quantized
      mx_specs {dictionary}     -- dictionary to specify mx_specs
      round      {str}            -- Rounding mode, choose from (floor, nearest, even)
                                     (default: "nearest")
    Returns:
      quantized value {PyTorch tensor} -- a tensor that has been quantized
    """
    if mx_specs is None:
        return A
    elif round is None:
        round = mx_specs['round']

    if mx_specs['bfloat'] == 16 and round == 'even'\
        and torch.cuda.is_bf16_supported() \
        and mx_specs['bfloat_subnorms'] == True:
        return A.to(torch.bfloat16)

    if mx_specs['bfloat'] > 0 and mx_specs['fp'] > 0:
        raise ValueError("Cannot set both [bfloat] and [fp] in mx_specs.")
    elif mx_specs['bfloat'] > 9:
        A = _quantize_bfloat(A, bfloat=mx_specs['bfloat'], round=round,
                             custom_cuda=mx_specs['custom_cuda'],
                             allow_denorm=mx_specs['bfloat_subnorms'])
    elif mx_specs['bfloat'] > 0 and mx_specs['bfloat'] <= 9:
        raise ValueError("Cannot set [bfloat] <= 9 in mx_specs.")
    elif mx_specs['fp'] > 6:
        A = _quantize_fp(A, exp_bits=5, mantissa_bits=mx_specs['fp'] - 6,
                         round=round, custom_cuda=mx_specs['custom_cuda'],
                         allow_denorm=mx_specs['bfloat_subnorms'])
    elif mx_specs['fp'] > 0 and mx_specs['fp'] <= 6:
        raise ValueError("Cannot set [fp] <= 6 in mx_specs.")
    return A
