"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch

from .vector_ops import *
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test

torch_sigmoid = torch.sigmoid
torch_tanh = torch.tanh
torch_relu = torch.relu
torch_relu_ = torch.relu_
f_silu = torch.nn.functional.silu
f_gelu = torch.nn.functional.gelu
f_relu = torch.nn.functional.relu
f_relu6 = torch.nn.functional.relu6
f_leaky_relu = torch.nn.functional.leaky_relu


#-------------------------------------------------------------------------
# User-facing functions
#-------------------------------------------------------------------------
def sigmoid(input, mx_specs=None, name=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return torch_sigmoid(input)

    mx_specs = apply_mx_specs(mx_specs)
    return SigmoidFunction.apply(input, mx_specs, name)


def tanh(input, mx_specs=None, name=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return torch_tanh(input)

    mx_specs = apply_mx_specs(mx_specs)
    return TanhFunction.apply(input, mx_specs, name)


def relu(input, inplace=False, mx_specs=None, name=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_relu(input, inplace=inplace)

    mx_specs = apply_mx_specs(mx_specs)
    return ReLUFunction.apply(input, inplace, mx_specs, name)


def relu6(input, inplace=False, mx_specs=None, name=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_relu6(input, inplace=inplace)

    mx_specs = apply_mx_specs(mx_specs)
    return ReLU6Function.apply(input, inplace, mx_specs, name)


def leaky_relu(input, negative_slope=0.01, inplace=False,
               mx_specs=None, name=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_leaky_relu(input, negative_slope=negative_slope,
                            inplace=inplace)

    mx_specs = apply_mx_specs(mx_specs)
    return LeakyReLUFunction.apply(
        input, negative_slope, inplace, mx_specs, name)


def silu(input, inplace=False, mx_specs=None, name=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_silu(input, inplace=inplace)

    mx_specs = apply_mx_specs(mx_specs)
    return SiLUFunction.apply(input, inplace, mx_specs, name)


def gelu(input, mx_specs=None, first_order_gelu=False,
         approximate=None, name=None):
    """ The approximate arg is to match the torch function signature,
        but the arg is mostly ignored.

        In torch, approximate can be None or 'tanh'.
        In our lib, we support tanh and first_order.
        We hardcode the tanh approx for the baseline.
    """
    mx_assert_test(mx_specs)
    if mx_specs is None and first_order_gelu == False:
        try:
            out = f_gelu(input, approximate='tanh')
        except TypeError:
            out = f_gelu(input)
        return out

    mx_specs = apply_mx_specs(mx_specs)
    return GELUFunction.apply(
        input, mx_specs, first_order_gelu, name)


#-------------------------------------------------------------------------
# User-facing classes
#-------------------------------------------------------------------------
class Sigmoid(torch.nn.Sigmoid):
    def __init__(self, mx_specs=None, name=None):
        super().__init__()

        mx_assert_test(mx_specs)
        self.mx_none = (mx_specs is None)

        self.mx_specs = apply_mx_specs(mx_specs)
        self.name = name

    def forward(self, input):
        if self.mx_none:
            return super().forward(input)
        return SigmoidFunction.apply(input, self.mx_specs, self.name)


class Tanh(torch.nn.Tanh):
    def __init__(self, mx_specs=None, name=None):
        super().__init__()

        mx_assert_test(mx_specs)
        self.mx_none = (mx_specs is None)

        self.mx_specs = apply_mx_specs(mx_specs)
        self.name = name

    def forward(self, input):
        if self.mx_none:
            return super().forward(input)
        return TanhFunction.apply(input, self.mx_specs, self.name)


class ReLU(torch.nn.ReLU):
    def __init__(self, inplace=False, mx_specs=None, name=None):
        super().__init__(inplace=inplace)

        mx_assert_test(mx_specs)
        self.mx_none = (mx_specs is None)

        self.mx_specs = apply_mx_specs(mx_specs)
        self.name = name

    def forward(self, input):
        if self.mx_none:
            return super().forward(input)
        return ReLUFunction.apply(
            input, self.inplace, self.mx_specs, self.name)


class ReLU6(torch.nn.ReLU6):
    def __init__(self, inplace=False, mx_specs=None, name=None):
        super().__init__(inplace=inplace)

        mx_assert_test(mx_specs)
        self.mx_none = (mx_specs is None)

        self.mx_specs = apply_mx_specs(mx_specs)
        self.name = name

    def forward(self, input):
        if self.mx_none:
            return super().forward(input)
        return ReLU6Function.apply(
            input, self.inplace, self.mx_specs, self.name)


class LeakyReLU(torch.nn.LeakyReLU):
    def __init__(self, negative_slope=0.01, inplace=False,
                 mx_specs=None, name=None):
        super().__init__(
                negative_slope=negative_slope, inplace=inplace)

        mx_assert_test(mx_specs)
        self.mx_none = (mx_specs is None)

        self.mx_specs = apply_mx_specs(mx_specs)
        self.name = name

    def forward(self, input):
        if self.mx_none:
            return super().forward(input)
        return LeakyReLUFunction.apply(
            input, self.negative_slope, self.inplace,
            self.mx_specs, self.name)

class SiLU(torch.nn.SiLU):
    def __init__(self, inplace=False, mx_specs=None, name=None):
        super().__init__(inplace=inplace)

        mx_assert_test(mx_specs)
        self.mx_none = (mx_specs is None)

        self.mx_specs = apply_mx_specs(mx_specs) 
        self.name = name

    def forward(self, input):
        if self.mx_none:
            return super().forward(input)
        return SiLUFunction.apply(
            input, self.inplace, self.mx_specs, self.name)


class GELU(torch.nn.GELU):
    """ Note that the torch baseline is hardcoded to use the tanh
        axpproximation to GELU. The approximate kwarg is ignored. """
    def __init__(self, mx_specs=None, first_order_gelu=False,
                 approximate=None, name=None):
        try:
            super().__init__(approximate='tanh')
        except TypeError:
            super().__init__()

        mx_assert_test(mx_specs)
        self.mx_none = (mx_specs is None and not first_order_gelu)

        self.mx_specs = apply_mx_specs(mx_specs)
        self.first_order_gelu = first_order_gelu
        self.name = name

    def forward(self, input):
        if self.mx_none:
            return super().forward(input)
        return GELUFunction.apply(
            input, self.mx_specs, self.first_order_gelu, self.name)


#-------------------------------------------------------------------------
# Internal functions
#-------------------------------------------------------------------------
class SigmoidFunction(torch.autograd.Function):
    """
    Forward pass: 1 / (1 + exp(-x))
    Backward pass: sigmoid(x) * (1 - sigmoid(x))
    """
    @staticmethod
    def forward(ctx, input, mx_specs=None, name=None):
        ctx.name = name

        input           = vec_quantize(input, mx_specs=mx_specs)
        exp_nx          = vec_exp(-input, mx_specs=mx_specs)
        exp_nx_plus_1   = vec_add(exp_nx, 1., mx_specs=mx_specs)
        output          = vec_recip(exp_nx_plus_1,
                                    mx_specs=mx_specs)

        ctx.save_for_backward(output)
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors

        grad_output     = vec_quantize(grad_output,
                                       mx_specs=ctx.mx_specs)
        temp            = vec_sub(1, output, mx_specs=ctx.mx_specs)
        grad_sigmoid    = vec_mul(output, temp, mx_specs=ctx.mx_specs)
        grad_input      = vec_mul(grad_sigmoid, grad_output,
                                  mx_specs=ctx.mx_specs)

        return (grad_input, None, None)


class TanhFunction(torch.autograd.Function):
    """
    Forward pass: torch.tanh
    Backward pass: 1 - tanh(x) * tanh(x)
    """
    @staticmethod
    def forward(ctx, input, mx_specs=None, name=None):
        ctx.name = name

        input       = vec_quantize(input, mx_specs=mx_specs)
        output      = vec_tanh(input, mx_specs=mx_specs)

        ctx.save_for_backward(output)
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors

        grad_output = vec_quantize(grad_output,
                                   mx_specs=ctx.mx_specs)
        output2     = vec_mul(output, output, mx_specs=ctx.mx_specs)
        grad_tanh   = vec_sub(1, output2, mx_specs=ctx.mx_specs)
        grad_input  = vec_mul(grad_tanh, grad_output,
                              mx_specs=ctx.mx_specs)

        return (grad_input, None, None)


class ReLUFunction(torch.autograd.Function):
    """
    Forward pass: torch.relu
    Backward pass: (output > 0) dy else 0
    """
    @staticmethod
    def forward(ctx, input, inplace=False, mx_specs=None, name=None):
        ctx.name = name

        # No need to quantize input first since ReLU just masks
        if inplace:
            ctx.mark_dirty(input)
            input = torch_relu_(input)
            output = vec_quantize(input, mx_specs=mx_specs)
            input.copy_(output)
            output = input
        else:
            output = torch_relu(input)
            output = vec_quantize(output, mx_specs=mx_specs)

        # Stash the ReLU mask
        mask = output > 0
        ctx.save_for_backward(mask)

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors

        # No need to quantize grad_output ReLU just masks
        zs = torch.zeros([1], dtype=grad_output.dtype,
                         device=grad_output.device)
        grad_input = torch.where(mask, grad_output, zs)

        grad_input = vec_quantize(grad_input, mx_specs=ctx.mx_specs)

        return (grad_input, None, None, None)


class ReLU6Function(torch.autograd.Function):
    """
    Forward pass: torch.relu6
    Backward pass: (output > 0 and output < 6) dy else 0
    """
    @staticmethod
    def forward(ctx, input, inplace=False, mx_specs=None, name=None):
        ctx.name = name

        # No need to quantize input first since ReLU just masks
        if inplace:
            ctx.mark_dirty(input)
            input = f_relu6(input, inplace=True)
            output = vec_quantize(input, mx_specs=mx_specs)
            input.copy_(output)
            output = input
        else:
            output = f_relu6(input)
            output = vec_quantize(output, mx_specs=mx_specs)

        # Stash the ReLU6 mask
        mask = torch.logical_and(output > 0, output < 6)
        ctx.save_for_backward(mask)

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors

        # No need to quantize grad_output ReLU just masks
        zs = torch.zeros([1], dtype=grad_output.dtype,
                         device=grad_output.device)
        grad_input = torch.where(mask, grad_output, zs)

        grad_input = vec_quantize(grad_input, mx_specs=ctx.mx_specs)

        return (grad_input, None, None, None)



class LeakyReLUFunction(torch.autograd.Function):
    """ Relu with a non-zero negative slope """
    @staticmethod
    def forward(ctx, input, negative_slope=0.01, inplace=False,
                mx_specs=None, name=None):
        ctx.negative_slope = negative_slope
        ctx.name = name

        q_in       = vec_quantize(input, mx_specs=mx_specs)
        output     = f_leaky_relu(q_in, negative_slope=negative_slope)
        output     = vec_quantize(output, mx_specs=mx_specs)

        if inplace:
            ctx.mark_dirty(input)
            input.copy_(output)
            output = input

        mask = output > 0
        ctx.save_for_backward(mask)
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors

        grad_output = vec_quantize(grad_output,
                                   mx_specs=ctx.mx_specs)
        grad_neg    = vec_mul(grad_output, ctx.negative_slope,
                              mx_specs=ctx.mx_specs)
        grad_input  = torch.where(mask, grad_output, grad_neg)

        return (grad_input, None, None, None, None)


class SiLUFunction(torch.autograd.Function):
    """
    Forward pass: x * sigmoid(x)
    Backward pass: sigmoid(x) + y * (1 - sigmoid(x))
    """
    @staticmethod
    def forward(ctx, input, inplace=False, mx_specs=None, name=None):
        ctx.name = name

        q_in            = vec_quantize(input, mx_specs=mx_specs)
        exp_nx          = vec_exp(-q_in, mx_specs=mx_specs)
        exp_nx_plus_1   = vec_add(exp_nx, 1., mx_specs=mx_specs)
        sig_x           = vec_recip(exp_nx_plus_1,
                                    mx_specs=mx_specs)
        output          = vec_mul(q_in, sig_x, mx_specs=mx_specs)

        if inplace:
            ctx.mark_dirty(input)
            input.copy_(output)
            output = input

        ctx.save_for_backward(output, sig_x)
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        y, sig_x, = ctx.saved_tensors

        grad_output = vec_quantize(grad_output,
                                   mx_specs=ctx.mx_specs)
        temp       = vec_sub(1., sig_x, mx_specs=ctx.mx_specs)
        temp       = vec_mul(y, temp, mx_specs=ctx.mx_specs)
        grad_silu  = vec_add(sig_x, temp, mx_specs=ctx.mx_specs)
        grad_input = vec_mul(grad_silu, grad_output,
                             mx_specs=ctx.mx_specs)

        return (grad_input, None, None, None)


class GELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mx_specs=None, first_order_gelu=False, name=None):
        '''
        GELU function is defined by:
            x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        First order approximation of GELU (aka Swish):
            x * sigmoid(1.702 * x)
        Detailed approximation of GELU:
            x * sigmoid(1.5958 * (x + 0.044715 * x^3))
        See the details in Gaussian Error Linear Units (GELUs)
            https://arxiv.org/pdf/1606.08415.pdf

        Bfloat16 Coefficients used in this implementation
            1.702    ~= 1.703125     # 2**0  * (1 + 2**-1 + 2**-3 + 2**-4 + 2**-6)
            0.044715 ~= 0.044677734  # 2**-5 * (1 + 2**-2 + 2**-3 + 2**-5 + 2**-6 + 2**-7)
            1.5958   ~= 1.59375      # 2**0  * (1 + 2**-1 + 2**-4 + 2**-5)

        '''
        ctx.first_order_gelu = first_order_gelu
        ctx.name = name

        q_in = vec_quantize(input, mx_specs=mx_specs)

        if first_order_gelu:
            # compute 1.702 * x
            sigmoid_input = vec_mul(1.703125, q_in, mx_specs=mx_specs)
        else:
            # compute 1.5958 * (x + 0.044715 * x^3)
            sigmoid_input = vec_mul(q_in, q_in,
                                    mx_specs=mx_specs)
            sigmoid_input = vec_mul(sigmoid_input, q_in,
                                    mx_specs=mx_specs)
            sigmoid_input = vec_mul(0.044677734, sigmoid_input,
                                    mx_specs=mx_specs)
            sigmoid_input = vec_add(sigmoid_input, q_in,
                                    mx_specs=mx_specs)
            sigmoid_input = vec_mul(1.59375, sigmoid_input,
                                    mx_specs=mx_specs)

        # compute Phi(x) using sigmoid
        phi = vec_exp(-sigmoid_input, mx_specs=mx_specs)
        phi = vec_add(phi, 1., mx_specs=mx_specs)
        phi = vec_recip(phi, mx_specs=mx_specs)

        if mx_specs['quantize_backprop']:
            ctx.save_for_backward(q_in, phi)
        else:
            ctx.save_for_backward(input, phi)
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)

        # return x * Phi(x)
        return vec_mul(q_in, phi, mx_specs=mx_specs)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        We compute the gradient based on the approximation
            (x * sigmoid(y))' = sigmoid(y) +
                                x * sigmoid(y) * (1 - sigmoid(y)) * y'
        for the first order approximation of GELU:
            y  = 1.702 * x
            y' = 1.702
        for detailed approximation of GELU:
            y  = 1.5958 * (x + 0.044715 * x^3)
            y' = 1.5958 + 0.21406859 * x^2

        FYI, the gradient of GELU by definition:
            (x * Phi(x))' = Phi(x) + x * Phi'(x)
                          = Phi(x) + x * (1 / sqrt(2 * pi)) * exp(-x^2 / 2)

        Bfloat16 Coefficients used in this implementation
            1.702      ~= 1.703125     # 2**0  * (1 + 2**-1 + 2**-3 + 2**-4 + 2**-6)
            0.044715   ~= 0.044677734  # 2**-5 * (1 + 2**-2 + 2**-3 + 2**-5 + 2**-6 + 2**-7)
            1.5958     ~= 1.59375      # 2**0  * (1 + 2**-1 + 2**-4 + 2**-5)
            0.21406859 ~= 0.21386719   # 2**-3 * (1 + 2**-1 + 2**-3 + 2**-4 + 2**-6 + 2**-7)
        '''
        input, phi = ctx.saved_tensors

        grad_output = vec_quantize(grad_output,
                                   mx_specs=ctx.mx_specs)

        # compute Phi'(x)
        dphi = vec_sub(1, phi, mx_specs=ctx.mx_specs)
        dphi = vec_mul(phi, dphi, mx_specs=ctx.mx_specs)

        if ctx.first_order_gelu:
            dphi = vec_mul(1.703125, dphi, mx_specs=ctx.mx_specs)
        else:
            dy   = vec_mul(input, input, mx_specs=ctx.mx_specs)
            dy   = vec_mul(0.21386719, dy, mx_specs=ctx.mx_specs)
            dy   = vec_add(1.59375, dy, mx_specs=ctx.mx_specs)
            dphi = vec_mul(dy, dphi, mx_specs=ctx.mx_specs)

        # compute x * Phi'(x)
        x_dphi = vec_mul(input, dphi, mx_specs=ctx.mx_specs)

        # compute Phi(x) + x * Phi'(x)
        grad_gelu = vec_add(phi, x_dphi, mx_specs=ctx.mx_specs)
        grad_input = vec_mul(grad_gelu, grad_output,
                             mx_specs=ctx.mx_specs)

        return (grad_input, None, None, None)
