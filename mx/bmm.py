"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch

from .mx_ops import quantize_mx_op
from .elemwise_ops import quantize_elemwise_op
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test
from .matmul_precision import set_matmul_precision


torch_bmm = torch.bmm


class BMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in1, in2, mx_specs, name):
        """This function is similar to torch.bmm, but accepts any
        number of outer dims instead of just 1.
        in1: (..., out_rows, features)
        in2: (..., features, out_cols)
        out: (..., out_rows, out_cols)
        """
        bf_in1 = quantize_elemwise_op(
            in1, mx_specs=mx_specs, round=mx_specs["round_output"]
        )
        bf_in2 = quantize_elemwise_op(
            in2, mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        if mx_specs["quantize_backprop"]:
            ctx.save_for_backward(bf_in1, bf_in2)
        else:
            ctx.save_for_backward(in1, in2)

        # quantize everything along the reduction dim
        qin1 = quantize_mx_op(
            bf_in1,
            mx_specs,
            elem_format=mx_specs["a_elem_format"],
            axes=[-1],
            round=mx_specs["round_mx_output"],
        )
        qin2 = quantize_mx_op(
            bf_in2,
            mx_specs,
            elem_format=mx_specs["a_elem_format"],
            axes=[-2],
            round=mx_specs["round_mx_output"],
        )

        # compute output
        with set_matmul_precision(qin1, qin2,
                    mx_specs['a_elem_format'],
                    mx_specs['a_elem_format']):
            out = torch_bmm(qin1, qin2)

        # element-wise quantize for output
        out = quantize_elemwise_op(
            out, mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        in1, in2 = ctx.saved_tensors

        grad_out = quantize_elemwise_op(
            grad_out,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # perform madtile operation for grad_in1, grad_in2
        #####################################################
        qin1 = quantize_mx_op(
            in1,
            ctx.mx_specs,
            elem_format=ctx.mx_specs["a_elem_format_bp"],
            axes=[-2],
            round=ctx.mx_specs["round_mx_input_grad_input"],
        )
        qin2 = quantize_mx_op(
            in2,
            ctx.mx_specs,
            elem_format=ctx.mx_specs["a_elem_format_bp"],
            axes=[-1],
            round=ctx.mx_specs["round_mx_input_grad_input"],
        )

        qgrad_out1 = quantize_mx_op(
            grad_out,
            ctx.mx_specs,
            elem_format=ctx.mx_specs["a_elem_format_bp_os"],
            axes=[-1],
            round=ctx.mx_specs["round_mx_grad_output_grad_input"],
        )
        qgrad_out2 = quantize_mx_op(
            grad_out,
            ctx.mx_specs,
            elem_format=ctx.mx_specs["a_elem_format_bp_os"],
            axes=[-2],
            round=ctx.mx_specs["round_mx_grad_output_grad_input"],
        )

        # compute grad_in1 and grad_in2
        with set_matmul_precision(qgrad_out1, qin2,
                ctx.mx_specs['a_elem_format_bp_os'],
                ctx.mx_specs['a_elem_format_bp']):
            grad_in1 = torch_bmm(qgrad_out1, qin2.transpose(-1, -2))
        
        with set_matmul_precision(qin1, qgrad_out2,
                ctx.mx_specs['a_elem_format_bp'],
                ctx.mx_specs['a_elem_format_bp_os']):
            grad_in2 = torch_bmm(qin1.transpose(-1, -2), qgrad_out2)

        # element-wise quantize for grad_in1 and grad_in2
        grad_in1 = quantize_elemwise_op(
            grad_in1,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )
        grad_in2 = quantize_elemwise_op(
            grad_in2,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        return (grad_in1, grad_in2, None, None)


def bmm(in1, in2, mx_specs=None, name=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return torch_bmm(in1, in2)

    mx_specs = apply_mx_specs(mx_specs)

    return BMMFunction.apply(in1, in2, mx_specs, name)
