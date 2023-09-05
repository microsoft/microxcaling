"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch
import torch.nn.functional as F

from .vector_ops import *

def _get_group_shape(x, axis, groups):
    """ Compute the shape to reshape to when doing GroupNorm.
        (N, C, ...) -> (N, groups, C//groups, ...)
    """
    H = x.shape[axis]
    assert(H % groups == 0)

    orig_shape = list(x.shape)
    grouped_shape = list(x.shape)
    grouped_shape[axis] = groups
    grouped_shape.insert(axis+1, H//groups)
    return orig_shape, grouped_shape


def _norm_forward(x, axes, weight, bias, eps, mx_specs,
                  groups=None, weight_axis=None,
                  use_running_stats=False,
                  running_mean=None, running_var=None):
    """ Forward pass for BatchNorm, LayerNorm, GroupNorm.
        It computes:
            z = (x - mean(x)) / sqrt(var(x) + eps)
            output = z * weight + bias

        Args:
            groups: divide the reduction dim (singular) into
                groups, each group is normalized separately
            weight_axis: axis that is scaled by weight+bias
            use_running_stats: instead of computing mean
                and var, use running_mean and running_var

        Weight and bias should already be quantized to bfloat
    """
    if type(axes) is not list:
        assert(type(axes) is int)
        axes = [axes]

    # weights and biases will be reshaped to w_shape
    # if their orig shapes cannot be broadcast correctly
    if weight_axis is not None:
        w_shape = [1 for _ in range(x.ndim)]
        w_shape[weight_axis] = x.shape[weight_axis]
    else:
        w_shape = None

    if groups:
        # Reshape reduce dim from H to (groups, H//groups)
        # Assume dim to group is axes[0]
        orig_shape, grouped_shape = _get_group_shape(
                x, axes[0], groups)

        x = x.view(grouped_shape)
        axes = [a+1 for a in axes]

    reduced_shape = list(x.shape)
    reduced_shape[0] = 1
    for i in axes:
        reduced_shape[i] = 1

    if not use_running_stats:
        # mean (reduced_shape)
        x_mean = vec_reduce_mean(x, axes, keepdim=True,
                                 mx_specs=mx_specs)
        # x_shift
        x_shift = vec_sub(x, x_mean,
                          mx_specs=mx_specs)
        # x_var (reduced_shape)
        x_shift_pow2 = vec_mul(x_shift, x_shift,
                               mx_specs=mx_specs)
        x_var = vec_reduce_mean(x_shift_pow2, axes,
                                keepdim=True,
                                mx_specs=mx_specs)
    else:
        assert(running_mean != None)
        assert(running_var != None)
        x_mean = vec_quantize(running_mean, mx_specs=mx_specs)
        x_mean = x_mean.view(reduced_shape)
        x_shift = vec_sub(x, x_mean, mx_specs=mx_specs)
        x_var = vec_quantize(running_var, mx_specs=mx_specs)
        x_var = x_var.view(reduced_shape)

    # x_vare (reduced_shape)
    x_vare = vec_add(x_var, eps, mx_specs=mx_specs)

    # x_std_inv (reduced_shape)
    x_std = vec_sqrt(x_vare, mx_specs=mx_specs)
    x_std_inv = vec_recip(x_std, mx_specs=mx_specs)

    # x_norm
    x_norm = vec_mul(x_shift, x_std_inv, mx_specs=mx_specs)

    if groups:
        x_norm = x_norm.view(orig_shape)

    if w_shape is not None:
        # Reshape weight and bias for elemwise mul
        weight = weight.view(w_shape)
        bias = bias.view(w_shape)

    # output
    x_scale = vec_mul(weight, x_norm, mx_specs=mx_specs)
    output = vec_add(x_scale, bias, mx_specs=mx_specs)

    #Deepspeed FWD pass returns x_vare instead of x_var
    return output, x_shift, x_norm, x_std_inv, x_mean, x_vare


def _norm_backward(grad_output, axes, weight,
                   x_shift, x_std_inv, mx_specs,
                   groups=None, weight_axis=None):
    """ Backward pass for BatchNorm, LayerNorm, GroupNorm.
        Computes the gradient wrt the input
    """
    if type(axes) is not list:
        assert(type(axes) is int)
        axes = [axes]

    # weights and biases will be reshaped to w_shape
    # if their orig shapes cannot be broadcast correctly
    if weight_axis is not None:
        w_shape = [1 for _ in range(grad_output.ndim)]
        w_shape[weight_axis] = grad_output.shape[weight_axis]
    else:
        w_shape = None

    if groups:
        # Reshape reduce dim from H to (groups, H//groups)
        # Assume dim to group is axes[0]
        orig_shape, grouped_shape = _get_group_shape(
                grad_output, axes[0], groups)

        axes = [a+1 for a in axes]

    # Norm backwards consists of 3 terms:
    #  dx = dx_shift + dx_mean + dx_shift2
    #   dx_shift = (grad * w / x_std)
    #   dx_mean = -mean(grad * w / x_std) = -mean(dx_shift)
    #   dx_shift2 = -x_shift * mean(grad * w * x_shift) / x_std**3

    if w_shape is not None:
        # Reshape weight for elemwise mul
        weight = weight.view(w_shape)

    # dx_norm (N, H, ...) = grad * w
    dx_norm = vec_mul(grad_output, weight,
                      mx_specs=mx_specs)

    if groups:
        dx_norm = dx_norm.view(grouped_shape)

    # dx_shift (N, H, ...) = grad * w / x_std
    dx_shift = vec_mul(dx_norm, x_std_inv, mx_specs=mx_specs)
    # dx_mean (1, H, 1s)
    dx_mean = vec_reduce_mean(-dx_shift, axes, keepdim=True,
                              mx_specs=mx_specs)

    # dx_std (1, H, 1s) = mean(grad * w * x_shift) / x_std**3
    dx_std = vec_mul(dx_norm, x_shift, mx_specs=mx_specs)
    dx_std = vec_reduce_mean(dx_std, axes, keepdim=True,
                             mx_specs=mx_specs)
    x_vare_inv = vec_mul(x_std_inv, x_std_inv, mx_specs=mx_specs)
    dx_std = vec_mul(dx_std, x_vare_inv, mx_specs=mx_specs)
    dx_std = vec_mul(dx_std, x_std_inv, mx_specs=mx_specs)
    # dx_shift2 (N, H, ...) = dx_std * x_shift
    dx_shift2 = vec_mul(-dx_std, x_shift, mx_specs=mx_specs)

    # dx (N, H, ...) = dx_shift + dx_shift2 + dx_mean
    dx = vec_add(dx_shift, dx_shift2, mx_specs=mx_specs)
    dx = vec_add(dx, dx_mean, mx_specs=mx_specs)

    if groups:
        dx = dx.view(orig_shape)

    grad_input = dx
    return grad_input

def _norm_backward_LN(grad_output, axes, weight,
                   x_norm, x_var, mx_specs,
                   groups=None, weight_axis=None):
    """ Backward pass for BatchNorm, LayerNorm, GroupNorm.
        Computes the gradient wrt the input
    """
    if type(axes) is not list:
        assert(type(axes) is int)
        axes = [axes]

    # weights and biases will be reshaped to w_shape
    # if their orig shapes cannot be broadcast correctly
    if weight_axis is not None:
        w_shape = [1 for _ in range(grad_output.ndim)]
        w_shape[weight_axis] = grad_output.shape[weight_axis]
    else:
        w_shape = None

    if groups:
        # Reshape reduce dim from H to (groups, H//groups)
        # Assume dim to group is axes[0]
        orig_shape, grouped_shape = _get_group_shape(
                grad_output, axes[0], groups)

        axes = [a+1 for a in axes]

    # Norm backwards consists of 3 terms:
    #  dx = dx_shift + dx_mean + dx_shift2
    #   dx_shift = (grad * w / x_std)
    #   dx_mean = -mean(grad * w / x_std) = -mean(dx_shift)
    #   dx_shift2 = -x_shift * mean(grad * w * x_shift) / x_std**3

    if w_shape is not None:
        # Reshape weight for elemwise mul
        weight = weight.view(w_shape)

    # dx_norm (N, H, ...) = grad * w
    dx_norm = vec_mul(grad_output, weight,
                      mx_specs=mx_specs)

    if groups:
        dx_norm = dx_norm.view(grouped_shape)

    x_std = vec_sqrt(x_var, mx_specs=mx_specs)
    x_std_inv = vec_div(1.0, x_std, mx_specs=mx_specs)

    # dx_shift (N, H, ...) = grad * w / x_std
    dx_shift = vec_mul(dx_norm, x_std_inv, mx_specs=mx_specs)

    # dx_std (1, H, 1s) = mean(grad * w * x_shift) / x_std**3
    # dx_std_tmp (1, H, 1s) = mean(grad * w * x_shift) / x_var, used for dx_shift2 calculation as Deepspeed does
    dx_std_tmp = vec_mul(dx_norm, x_norm, mx_specs=mx_specs)
    dx_std_tmp = vec_mul(dx_std_tmp, x_std, mx_specs=mx_specs)
    dx_std_tmp = vec_reduce_mean(dx_std_tmp, axes, keepdim=True,
                             mx_specs=mx_specs)
    x_vare_inv = vec_div(1.0, x_var, mx_specs=mx_specs)
    dx_std_tmp = vec_mul(dx_std_tmp, x_vare_inv, mx_specs=mx_specs)
    # dx_shift2 (N, H, ...) = dx_std * x_shift = dx_std_tmp * x_norm
    dx_shift2 = vec_mul(-dx_std_tmp, x_norm, mx_specs=mx_specs)

    # dx (N, H, ...) = dx_shift + dx_shift2 + dx_mean
    dx = vec_add(dx_shift, dx_shift2, mx_specs=mx_specs)
    # dx_mean (1, H, 1s)
    #dx_mean is calculated the same way that Deepspeed does
    dx_mean = vec_reduce_mean(dx, axes, keepdim=True,
                                mx_specs=mx_specs)
    dx = vec_add(dx, -dx_mean, mx_specs=mx_specs)

    if groups:
        dx = dx.view(orig_shape)

    grad_input = dx
    return grad_input
