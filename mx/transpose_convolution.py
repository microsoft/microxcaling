"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch
import torch.nn.functional as F

from .mx_ops import quantize_mx_op
from .elemwise_ops import quantize_elemwise_op
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test
from .convolution import conv_weight

f_conv_transpose2d = F.conv_transpose2d
torch_conv2d = torch.conv2d

class ConvTranspose2dFunction(torch.autograd.Function):
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
        output_padding=0,
        groups=1,
        dilation=1,
        mx_specs=None,
        name=None,
    ):
        # input: input tensor (minibatch x in_channels x iH x iW)
        # weight: weight tensor (out_channels x in_channels/groups x kH x kW)
        # bias: optional bias tensor of shape (out_channels)

        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.name = name

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

        if mx_specs["quantize_backprop"]:
            ctx.save_for_backward(bf_in, bf_weight)
        else:
            ctx.save_for_backward(input, weight)

        assert input.shape[1] % groups == 0

        #####################################################
        # MX conv_transpose2d for output
        #####################################################
        #   input is (batch, in_channels, ...)
        #   weight is (in_channels, out_channels/groups, ...)
        # quantize along in_channels
        qid_input = quantize_mx_op(
            bf_in,
            mx_specs,
            elem_format=mx_specs["a_elem_format"],
            axes=[1],
        )
        qid_weight = quantize_mx_op(
            bf_weight,
            mx_specs,
            elem_format=mx_specs["w_elem_format"],
            axes=[0],
        )

        # compute output
        output = f_conv_transpose2d(
            qid_input,
            qid_weight,
            bf_bias,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
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
            elem_format=ctx.mx_specs["a_elem_format"],
            axes=[0],
        )
        qex_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs["a_elem_format"],
            axes=[0],
        )

        grad_weight = conv_weight(
            qex_grad_output,
            weight.shape,
            qex_input,
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
        # MX conv for grad_input
        #####################################################
        #   weight is (in_channels, out_channels/groups, ...)
        #   output is (batch, out_channels, ...)
        # reduction dim is out_channels
        qod_weight = quantize_mx_op(
            weight,
            ctx.mx_specs,
            elem_format=ctx.mx_specs["w_elem_format"],
            axes=[1],
        )
        qod_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs["a_elem_format"],
            axes=[1],
        )

        # compute grad_input
        grad_input = torch_conv2d(
            qod_grad_output,
            qod_weight,
            bias=None,
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
            grad_bias = grad_output.sum(dim=(3, 2, 0))
            grad_bias = quantize_elemwise_op(
                grad_bias,
                mx_specs=ctx.mx_specs,
                round=ctx.mx_specs["round_grad_weight"],
            )

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class ConvTranspose2d(torch.nn.ConvTranspose2d):
    """Padding mode is assumed to be zeros"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
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
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def apply_mx_specs(self, mx_specs):
        self.mx_specs = mx_specs
        self.mx_none = mx_specs is None
        self.mx_specs = apply_mx_specs(mx_specs)

    def append_name(self, postfix):
        self.name += postfix

    def forward(self, inputs, output_size=None):
        if self.mx_none:
            return super().forward(input, output_size=output_size)

        try:
            # Pytorch 1.12.0+ has an extra argument
            num_spatial_dims = 2
            output_padding = self._output_padding(
                input,
                output_size,
                self.stride,
                self.padding,
                self.kernel_size,
                num_spatial_dims,
                self.dilation,
            )
        except:
            output_padding = self._output_padding(
                input,
                output_size,
                self.stride,
                self.padding,
                self.kernel_size,
                self.dilation,
            )

        return ConvTranspose2dFunction.apply(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
            self.mx_specs,
            self.name,
        )
