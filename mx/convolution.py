"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch
import packaging.version as version

from torch.nn import grad
from torch.nn.modules.utils import _single, _pair, _triple

from .mx_ops import quantize_mx_op
from .elemwise_ops import quantize_elemwise_op
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test

f_conv1d = torch.nn.functional.conv1d
f_conv2d = torch.nn.functional.conv2d
f_conv3d = torch.nn.functional.conv3d


def conv_weight(
    input, weight_shape, grad_output, stride=1, padding=0, dilation=1, groups=1
):
    """Computes the gradient of conv2d wrt the weight.
    nn.grad.conv2d_weight is bugged in Pytorch < v1.13.0
    This function implements a fix.
    See https://github.com/pytorch/pytorch/issues/51430
    and https://github.com/geohot/tinygrad/commit/8864b373338886a9173d3f823154815535104f28
    """
    num_spatial_dims = input.ndim - 2
    if num_spatial_dims == 1:
        _p = _single
        _conv = f_conv1d
        _conv_weight = grad.conv1d_weight
    elif num_spatial_dims == 2:
        _p = _pair
        _conv = f_conv2d
        _conv_weight = grad.conv2d_weight
    elif num_spatial_dims == 3:
        _p = _triple
        _conv = f_conv3d
        _conv_weight = grad.conv3d_weight
    else:
        raise ValueError(
            "conv_weight does not work with " "input with ndim=%d" % input.ndims
        )

    # For pytorch v1.13.0+, use the built-in convNd_weight.
    # Otherwise use our function
    if version.parse(torch.__version__) >= version.parse("1.13.dev0"):
        return _conv_weight(
            input,
            weight_shape,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    stride = _p(stride)
    padding = _p(padding)
    dilation = _p(dilation)

    bs = input.shape[0]
    cin = weight_shape[1]
    cout = weight_shape[0]
    assert grad_output.shape[0] == bs
    assert cout % groups == 0

    # Get the spatial dims for each tensor
    sdin = list(input.shape[2:])
    sdw = list(weight_shape[2:])
    sdout = list(grad_output.shape[2:])
    sd1s = [1] * len(sdout)

    grad_output = grad_output.reshape(bs, groups, cout // groups, *sdout).repeat(
        1, 1, cin, *sd1s
    )
    grad_output = grad_output.view(bs * cout * cin, 1, *sdout)

    input = input.reshape(1, bs * groups * cin, *sdin)

    grad_weight = _conv(
        input,
        grad_output,
        stride=dilation,
        padding=padding,
        dilation=stride,
        groups=bs * groups * cin,
    )

    # Sum over the batch dim, preserve current spatial dims
    sdgw = list(grad_weight.shape[2:])
    grad_weight = grad_weight.reshape(bs, -1, *sdgw).sum(dim=0)

    # If stride > 1, we only need to keep a subset
    # of the grad_weight spatial dims
    for i in range(num_spatial_dims):
        if stride[i] > 1:
            grad_weight = grad_weight.narrow(i + 1, 0, sdw[i])

    # Transpose and reshape to final shape
    grad_weight = grad_weight.view(groups, cin, cout // groups, *sdw).transpose(2, 1)
    grad_weight = grad_weight.contiguous().view(groups * cout // groups, cin, *sdw)

    return grad_weight


class ConvFunction(torch.autograd.Function):
    """Note that stride, padding, etc will be stored as
    tuples in torch.nn.Conv2d/Conv3d"""

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        mx_specs=None,
        name=None,
    ):
        # input: input tensor (minibatch x in_channels x ...)
        # weight: weight tensor (out_channels x in_channels/groups x ...)
        # bias: optional bias tensor of shape (out_channels)

        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.name = name

        num_spatial_dims = input.ndim - 2
        assert num_spatial_dims in (1, 2, 3)
        if num_spatial_dims == 1:
            fwd_func = f_conv1d
            ctx.conv_input = grad.conv1d_input
        elif num_spatial_dims == 2:
            fwd_func = f_conv2d
            ctx.conv_input = grad.conv2d_input
        elif num_spatial_dims == 3:
            fwd_func = f_conv3d
            ctx.conv_input = grad.conv3d_input

        # round with mx_specs['round_output']
        bf_in = quantize_elemwise_op(
            input, mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        # element-wise quantize for weight and bias
        bf_weight = quantize_elemwise_op(
            weight, mx_specs=mx_specs, round=mx_specs["round_weight"]
        )

        if bias is not None:
            bf_bias = quantize_elemwise_op(
                bias, mx_specs=mx_specs, round=mx_specs["round_weight"]
            )
        else:
            bf_bias = None

        # save context after quantize
        if mx_specs["quantize_backprop"]:
            ctx.save_for_backward(bf_in, bf_weight)
        else:
            ctx.save_for_backward(input, weight)

        assert input.shape[1] % groups == 0

        #####################################################
        # MX conv for output
        #####################################################
        #   input is (batch, in_channels, ...)
        #   weight is (out_channels, in_channels/groups, ..)
        # quantize along in_channels
        qid_input = quantize_mx_op(
            bf_in,
            mx_specs,
            elem_format=mx_specs['a_elem_format'],
            axes=[1],
        )
        qid_weight = quantize_mx_op(
            bf_weight,
            mx_specs,
            elem_format=mx_specs['w_elem_format'],
            axes=[1],
        )

        # compute output
        output = fwd_func(
            qid_input, qid_weight, bf_bias, stride, padding, dilation, groups
        )

        # element-wise quantize for output
        output = quantize_elemwise_op(
            output, mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.shape[1] % ctx.groups == 0

        # load context
        input, weight = ctx.saved_tensors

        grad_output = quantize_elemwise_op(
            grad_output,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # MX conv for grad_weight
        #####################################################
        #   input is  (batch, in_channels, ...)
        #   output is (batch, out_channels, ...)
        # quantize along the batch dim
        qex_input = quantize_mx_op(
            input,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format'],
            axes=[0],
        )
        qex_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format'],
            axes=[0],
        )

        # compute grad_weight
        # don't use nn.grad.conv2d_weight because it is bugged
        grad_weight = conv_weight(
            qex_input,
            weight.shape,
            qex_grad_output,
            stride=ctx.stride,
            padding=ctx.padding,
            dilation=ctx.dilation,
            groups=ctx.groups,
        )

        # element-wise quantize for grad_weight
        grad_weight = quantize_elemwise_op(
            grad_weight,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_weight"],
        )

        #####################################################
        # MX conv_transpose for grad_input
        #####################################################
        # grad_input = conv_transpose2d(output, weight)
        #   weight is (out_channels, in_channels/groups, ...)
        #   output is (batch, out_channels, ...)
        # reduction dim is out_channels
        qod_weight = quantize_mx_op(
            weight,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['w_elem_format'],
            axes=[0],
        )
        qod_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format'],
            axes=[1],
        )

        # compute grad_input
        grad_input = ctx.conv_input(
            input.shape,
            qod_weight,
            qod_grad_output,
            stride=ctx.stride,
            padding=ctx.padding,
            dilation=ctx.dilation,
            groups=ctx.groups,
        )

        # element-wise quantize for grad_input
        grad_input = quantize_elemwise_op(
            grad_input,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # Compute grad_bias
        #####################################################
        if not ctx.has_bias:
            grad_bias = None
        else:
            sum_axes = [0] + list(range(2, grad_output.ndim))
            grad_bias = grad_output.sum(sum_axes)
            grad_bias = quantize_elemwise_op(
                grad_bias,
                mx_specs=ctx.mx_specs,
                round=ctx.mx_specs["round_grad_weight"],
            )

        return (grad_input, grad_weight, grad_bias, None, None, None, None, None, None)


def conv1d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    mx_specs=None,
    name=None,
):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_conv1d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    mx_specs = apply_mx_specs(mx_specs)

    return ConvFunction.apply(
        input, weight, bias, stride, padding, dilation, groups, mx_specs, name
    )


def conv2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    mx_specs=None,
    name=None,
):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_conv2d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    mx_specs = apply_mx_specs(mx_specs)

    return ConvFunction.apply(
        input, weight, bias, stride, padding, dilation, groups, mx_specs, name
    )


def conv3d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    mx_specs=None,
    name=None,
):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_conv3d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    mx_specs = apply_mx_specs(mx_specs)

    return ConvFunction.apply(
        input, weight, bias, stride, padding, dilation, groups, mx_specs, name
    )


class Conv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        mx_specs=None,
        name=None,
    ):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None

        self.name = name
        self.mx_specs = apply_mx_specs(mx_specs)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def apply_mx_specs(self, mx_specs):
        self.mx_specs = mx_specs
        self.mx_none = mx_specs is None
        self.mx_specs = apply_mx_specs(mx_specs)

    def append_name(self, postfix):
        self.name += postfix

    def forward(self, inputs):
        if self.mx_none:
            return super()._conv_forward(inputs, self.weight, self.bias)

        return ConvFunction.apply(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.mx_specs,
            self.name,
        )


class Conv2d(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        mx_specs=None,
        name=None,
    ):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None

        self.name = name
        self.mx_specs = apply_mx_specs(mx_specs)

        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def apply_mx_specs(self, mx_specs):
        self.mx_specs = mx_specs
        self.mx_none = mx_specs is None
        self.mx_specs = apply_mx_specs(mx_specs)

    def append_name(self, postfix):
        self.name += postfix

    def forward(self, inputs):
        if self.mx_none:
            return super()._conv_forward(inputs, self.weight, self.bias)

        return ConvFunction.apply(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.mx_specs,
            self.name,
        )


class Conv3d(torch.nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        mx_specs=None,
        name=None,
    ):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None

        self.name = name
        self.mx_specs = apply_mx_specs(mx_specs)

        super(Conv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def apply_mx_specs(self, mx_specs):
        self.mx_specs = mx_specs
        self.mx_none = mx_specs is None
        self.mx_specs = apply_mx_specs(mx_specs)

    def append_name(self, postfix):
        self.name += postfix

    def forward(self, inputs):
        if self.mx_none:
            return super()._conv_forward(inputs, self.weight, self.bias)

        return ConvFunction.apply(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.mx_specs,
            self.name,
        )
