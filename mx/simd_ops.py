"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

User-facing differentiable SIMD operations

Exposed Methods:
    simd_add            y = x1 + x2
    simd_sub            y = x1 - x2
    simd_mul            y = x1 * x2
    simd_div            y = x1 / x2
    simd_split          x1=x, x2=x
    simd_square         y = x**2
    simd_sqrt           y = sqrt(x)
    simd_exp            y = e^x
    simd_log            y = log(x)
    simd_reduce_sum     y = x.sum(dim)
    simd_reduce_mean    y = x.mean(dim)
    simd_norm           y = (x**2).sum().sqrt()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from .vector_ops import *
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test

torch_sum = torch.sum
torch_mean = torch.mean
torch_sqrt = torch.sqrt
torch_exp = torch.exp
torch_log = torch.log
torch_square = torch.square


def _broadcast_gradient(grad_out, in_shape, mx_specs):
    """ Computes the gradient of y = broadcast_shape(x)
        Pytorch broadcasting rules:
        https://pytorch.org/docs/stable/notes/broadcasting.html?highlight=broadcasting
        - Match dims starting from innermost
        - Broadcast dims if dim sizes are equal, one of the two is 1,
          or one of the two does not exist
    """
    if list(grad_out.shape) == in_shape:
        return grad_out

    assert(grad_out.ndim >= len(in_shape))

    # Iterate each dim starting from -1 backwards
    # add any broadcasted dims to reduce_dims
    reduce_dims = []
    for i in range(grad_out.ndim):
        if i+1 > len(in_shape): # dim does not exist
            reduce_dims.append(-1 - i)
            continue

        dout = grad_out.shape[-1 - i]
        din  = in_shape[-1 - i]

        if dout == din: # dims match
            pass
        elif din == 1:  # one of the dims is 1
            reduce_dims.append(-1 - i)
        else:
            raise ValueError(
                    "simd_add _gradient shape error. grad_out is" + \
                    str(grad_out.shape) + "and input is" + str(in_shape))

    # Quantize to bfloat if reduction is needed
    if len(reduce_dims) > 0:
        grad_out = vec_quantize(grad_out, mx_specs=mx_specs)
        grad_in = torch_sum(grad_out, dim=reduce_dims)
        grad_in = vec_quantize(grad_in, mx_specs=mx_specs)
        return grad_in.view(in_shape)
    else:
        return grad_out.view(in_shape)


#----------------------------------------------------
# Autograd functions
#----------------------------------------------------
class SIMDAdd(torch.autograd.Function):
    """ Fwd: y = x1 + x2
        Bwd: dy/dx1 = dy/dx2 = 1
        Shape broadcasting is fully supported
    """
    @staticmethod
    def forward(ctx, in1, in2, mx_specs=None):
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        assert(isinstance(in1, torch.Tensor))
    
        qin1 = vec_quantize(in1, mx_specs=mx_specs)

        if isinstance(in2, torch.Tensor):
            ctx.in2_const = False
            ctx.in1_shape = list(in1.shape)
            ctx.in2_shape = list(in2.shape)
            qin2 = vec_quantize(in2, mx_specs=mx_specs)
        else:
            ctx.in2_const = True
            qin2 = in2

        return vec_add(qin1, qin2, mx_specs=mx_specs)

    @staticmethod
    def backward(ctx, g):
        g = vec_quantize(g, mx_specs=ctx.mx_specs)

        if not ctx.in2_const:
            g1 = _broadcast_gradient(g, ctx.in1_shape, ctx.mx_specs)
            g2 = _broadcast_gradient(g, ctx.in2_shape, ctx.mx_specs)
            return (g1, g2, None)
        else:
            return (g, None, None)


class SIMDSub(torch.autograd.Function):
    """ Fwd: y = x1 - x2
        Bwd: dy/dx1 = 1, dy/dx2 = -1
        Shape broadcasting is fully supported
    """
    @staticmethod
    def forward(ctx, in1, in2, mx_specs=None):
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        assert(isinstance(in1, torch.Tensor))
    
        qin1 = vec_quantize(in1, mx_specs=mx_specs)

        if isinstance(in2, torch.Tensor):
            ctx.in2_const = False
            ctx.in1_shape = list(in1.shape)
            ctx.in2_shape = list(in2.shape)
            qin2 = vec_quantize(in2, mx_specs=mx_specs)
        else:
            ctx.in2_const = True
            qin2 = in2

        return vec_sub(qin1, qin2, mx_specs=mx_specs)

    @staticmethod
    def backward(ctx, g):
        if not ctx.in2_const:
            n_g = vec_quantize(-g, mx_specs=ctx.mx_specs)
            g1 = _broadcast_gradient(g, ctx.in1_shape, ctx.mx_specs)
            g2 = _broadcast_gradient(n_g, ctx.in2_shape, ctx.mx_specs)
            return (g1, g2, None)
        else:
            return (g, None, None)


class SIMDMul(torch.autograd.Function):
    """ Supports Tensor*Tensor or Tensor*Const
        Fwd: y = x1 * x2
        Bwd: dy/dx1 = x2, dy/dx2 = x1
    """
    @staticmethod
    def forward(ctx, in1, in2, mx_specs=None):
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        assert(isinstance(in1, torch.Tensor))

        qin1 = vec_quantize(in1, mx_specs=mx_specs)

        if isinstance(in2, torch.Tensor):
            ctx.in2_const = False
            ctx.in1_shape = list(in1.shape)
            ctx.in2_shape = list(in2.shape)

            qin2 = vec_quantize(in2, mx_specs=mx_specs)

            if mx_specs['quantize_backprop']:
                ctx.save_for_backward(qin1, qin2)
            else:
                ctx.save_for_backward(in1, in2)
        else:
            ctx.in2_const = True
            ctx.in2 = in2
            qin2 = in2

            if mx_specs['quantize_backprop']:
                ctx.save_for_backward(qin1)
            else:
                ctx.save_for_backward(in1)

        return vec_mul(qin1, qin2, mx_specs=mx_specs)

    @staticmethod
    def backward(ctx, g):
        g = vec_quantize(g, mx_specs=ctx.mx_specs)

        if not ctx.in2_const:
            in1, in2 = ctx.saved_tensors
            g1 = vec_mul(g, in2, mx_specs=ctx.mx_specs)
            g2 = vec_mul(g, in1, mx_specs=ctx.mx_specs)
            g1 = _broadcast_gradient(g1, ctx.in1_shape, ctx.mx_specs)
            g2 = _broadcast_gradient(g2, ctx.in2_shape, ctx.mx_specs)
            return (g1, g2, None)
        else:
            in1, = ctx.saved_tensors
            g1 = vec_mul(g, ctx.in2, mx_specs=ctx.mx_specs)
            return (g1, None, None)


class SIMDDiv(torch.autograd.Function):
    """ Supports Tensor*Tensor or Tensor*Const
        Fwd: y = x1 / x2
        Bwd: dy/dx1 = 1/x2, dy/dx2 = -x1/(x2^2)
    """
    @staticmethod
    def forward(ctx, in1, in2, mx_specs=None):
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        assert(isinstance(in1, torch.Tensor))

        qin1 = vec_quantize(in1, mx_specs=mx_specs)

        if isinstance(in2, torch.Tensor):
            ctx.in2_const = False
            ctx.in1_shape = list(in1.shape)
            ctx.in2_shape = list(in2.shape)

            qin2 = vec_quantize(in2, mx_specs=mx_specs)
            out = vec_div(qin1, qin2, mx_specs=mx_specs)

            if mx_specs['quantize_backprop']:
                ctx.save_for_backward(out, qin2)
            else:
                ctx.save_for_backward(out, in2)
        else:
            ctx.in2_const = True
            ctx.in2 = in2

            out = vec_div(qin1, in2, mx_specs=mx_specs)

            if mx_specs['quantize_backprop']:
                ctx.save_for_backward(qin1)
            else:
                ctx.save_for_backward(in1)

        return out

    @staticmethod
    def backward(ctx, g):
        g = vec_quantize(g, mx_specs=ctx.mx_specs)

        if not ctx.in2_const:
            out, in2 = ctx.saved_tensors
            g1 = vec_div(g, in2, mx_specs=ctx.mx_specs)
            g2 = vec_div(-out, in2, mx_specs=ctx.mx_specs)
            g2 = vec_mul(g, g2, mx_specs=ctx.mx_specs)

            g1 = _broadcast_gradient(g1, ctx.in1_shape, ctx.mx_specs)
            g2 = _broadcast_gradient(g2, ctx.in2_shape, ctx.mx_specs)
            return (g1, g2, None)
        else:
            in1, = ctx.saved_tensors
            g1 = vec_div(g, ctx.in2, mx_specs=ctx.mx_specs)
            return (g1, None, None)


class SIMDSplit(torch.autograd.Function):
    """ Fwd: x1, x2 = x
        Bwd: dy/dx = dy/dx1 + dy/dx2
    """
    @staticmethod
    def forward(ctx, in1, mx_specs=None):
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        ctx.set_materialize_grads(False)
        return in1.clone(), in1.clone()

    @staticmethod
    def backward(ctx, g1, g2):
        # If only one branch has grad, don't compute anything
        if g1 is None:
            return (g2, None)
        if g2 is None:
            return (g1, None)

        g1 = vec_quantize(g1, mx_specs=ctx.mx_specs)
        g2 = vec_quantize(g2, mx_specs=ctx.mx_specs)
        grad_in = vec_add(g1, g2, mx_specs=ctx.mx_specs)
        return (grad_in, None)


class SIMDSquare(torch.autograd.Function):
    """ Fwd: y = x**2
        Bwd: dy/dx = 2*x
    """
    @staticmethod
    def forward(ctx, in1, mx_specs=None):
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        qin1 = vec_quantize(in1, mx_specs=mx_specs)

        if mx_specs['quantize_backprop']:
            ctx.save_for_backward(qin1)
        else:
            ctx.save_for_backward(in1)

        return vec_quantize(qin1**2, mx_specs=mx_specs)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        g = vec_quantize(g, mx_specs=ctx.mx_specs)
        x2 = vec_mul(x, 2, mx_specs=ctx.mx_specs)
        grad_in = vec_mul(g, x2, mx_specs=ctx.mx_specs)
        return (grad_in, None)


class SIMDSqrt(torch.autograd.Function):
    """ Fwd: y = sqrt(x)
        Bwd: dy/dx = 0.5/sqrt(x)
    """
    @staticmethod
    def forward(ctx, in1, mx_specs=None):
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)

        in1 = vec_quantize(in1, mx_specs=mx_specs)
        out = vec_sqrt(in1, mx_specs=mx_specs)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, g):
        sqrt_x, = ctx.saved_tensors
        g = vec_quantize(g, mx_specs=ctx.mx_specs)
        g = vec_mul(g, 0.5, mx_specs=ctx.mx_specs)
        grad_in = vec_div(
                g, sqrt_x, mx_specs=ctx.mx_specs)
        return (grad_in, None)


class SIMDExp(torch.autograd.Function):
    """ Fwd: y = e^x
        Bwd: dy/dx = e^x
    """
    @staticmethod
    def forward(ctx, in1, mx_specs=None):
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)

        in1 = vec_quantize(in1, mx_specs=mx_specs)
        out = vec_exp(in1, mx_specs=mx_specs)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, g):
        exp_x, = ctx.saved_tensors
        g = vec_quantize(g, mx_specs=ctx.mx_specs)
        g = vec_mul(g, exp_x, mx_specs=ctx.mx_specs)
        return (g, None)


class SIMDLog(torch.autograd.Function):
    """ Fwd: y = log_e(x)
        Bwd: dy/dx = 1/x
    """
    @staticmethod
    def forward(ctx, in1, mx_specs=None):
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)

        qin1 = vec_quantize(in1, mx_specs=mx_specs)
        out = torch_log(qin1)
        out = vec_quantize(out, mx_specs=mx_specs)

        if mx_specs['quantize_backprop']:
            ctx.save_for_backward(qin1)
        else:
            ctx.save_for_backward(in1)

        return out

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        g = vec_quantize(g, mx_specs=ctx.mx_specs)
        g = vec_div(g, x, mx_specs=ctx.mx_specs)
        return (g, None)


class SIMDReduceSum(torch.autograd.Function):
    """ Fwd: y = sum(x, dim)
        Bwd: dy/dx = 1, expanded in summed dims
    """
    @staticmethod
    def forward(ctx, in1, dim, keepdim=False, mx_specs=None):
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)

        dim = [dim] if type(dim) == int else dim
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = list(in1.shape)

        in1 = vec_quantize(in1, mx_specs=mx_specs)
        out = vec_reduce_sum(in1, dim, keepdim=keepdim,
                             mx_specs=mx_specs)
        return out

    @staticmethod
    def backward(ctx, g):
        # Make sure dim is a list of positives
        ndim = len(ctx.input_shape)
        dim = ctx.dim
        dim = [(i+ndim if i < 0 else i) for i in dim]

        # unsqueeze g to the same ndims as input
        g = vec_quantize(g, mx_specs=ctx.mx_specs)
        if not ctx.keepdim:
            for i in sorted(dim):
                g = g.unsqueeze(i)

        # Expand g in each summed dim
        expand_sizes = [-1 for _ in range(ndim)]
        for i in dim:
            expand_sizes[i] = ctx.input_shape[i]

        # g.expand returns a view of g,
        # User cannot modify grad_in inplace
        grad_in = g.expand(expand_sizes)
        return (grad_in, None, None, None)


#----------------------------------------------------
# User-facing functions
#----------------------------------------------------
def simd_add(in1, in2, mx_specs=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return in1 + in2

    mx_specs = apply_mx_specs(mx_specs)
    return SIMDAdd.apply(in1, in2, mx_specs)


def simd_sub(in1, in2, mx_specs=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return in1 - in2

    mx_specs = apply_mx_specs(mx_specs)
    return SIMDSub.apply(in1, in2, mx_specs)


def simd_mul(in1, in2, mx_specs=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return in1 * in2

    mx_specs = apply_mx_specs(mx_specs)
    return SIMDMul.apply(in1, in2, mx_specs)


def simd_div(in1, in2, mx_specs=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return in1 / in2

    mx_specs = apply_mx_specs(mx_specs)
    return SIMDDiv.apply(in1, in2, mx_specs)


def simd_split(in1, mx_specs=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return in1, in1

    mx_specs = apply_mx_specs(mx_specs)
    return SIMDSplit.apply(in1, mx_specs)


def simd_square(in1, mx_specs=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return torch_square(in1)

    mx_specs = apply_mx_specs(mx_specs)
    return SIMDSquare.apply(in1, mx_specs)


def simd_sqrt(in1, mx_specs=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return torch_sqrt(in1)

    mx_specs = apply_mx_specs(mx_specs)
    return SIMDSqrt.apply(in1, mx_specs)


def simd_exp(in1, mx_specs=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return torch_exp(in1)

    mx_specs = apply_mx_specs(mx_specs)
    return SIMDExp.apply(in1, mx_specs)


def simd_log(in1, mx_specs=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return torch_log(in1)

    mx_specs = apply_mx_specs(mx_specs)
    return SIMDLog.apply(in1, mx_specs)


def simd_reduce_sum(in1, dim=None, keepdim=False, mx_specs=None):
    mx_assert_test(mx_specs)

    if dim is None:
        dim = list(range(in1.ndim))

    if mx_specs is None:
        return torch_sum(in1, dim, keepdim=keepdim)

    if dim is None:
        dim = list(range(in1.ndim))

    mx_specs = apply_mx_specs(mx_specs)
    return SIMDReduceSum.apply(in1, dim, keepdim, mx_specs)


def simd_reduce_mean(in1, dim=None, keepdim=False, mx_specs=None):
    mx_assert_test(mx_specs)
    
    if dim is None:
        dim = list(range(in1.ndim))

    if mx_specs is None:
        return torch_mean(in1, dim, keepdim=keepdim)

    mx_specs = apply_mx_specs(mx_specs)

    # np.prod returns 1.0 for empty list
    dim = dim if type(dim) is list else [dim]
    denom = np.prod([in1.shape[i] for i in dim])

    s = SIMDReduceSum.apply(in1, dim, keepdim, mx_specs)
    return SIMDMul.apply(s, 1/denom, mx_specs)


def simd_norm(in1, keepdim=False, mx_specs=None):
    """ Computes Frobenius norm sqrt(sum(in1**2)), same as
        torch.linalg.norm(in1) with no other args """
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return torch.linalg.norm(in1, keepdim=keepdim)

    mx_specs = apply_mx_specs(mx_specs)

    in1 = SIMDSquare.apply(in1, mx_specs)
    s = SIMDReduceSum.apply(
            in1, list(range(in1.ndim)), keepdim, mx_specs)
    return SIMDSqrt.apply(s, mx_specs)

