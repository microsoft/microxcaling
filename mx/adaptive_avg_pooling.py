"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .vector_ops import *
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test

f_adaptive_avg_pool2d = F.adaptive_avg_pool2d

#-------------------------------------------------------------------------
# User-facing functions
#-------------------------------------------------------------------------
def adaptive_avg_pool2d(input, output_size, mx_specs=None, name=None):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_adaptive_avg_pool2d(input, output_size)
    
    mx_specs = apply_mx_specs(mx_specs)
    return AdaptiveAvgPool2dFunction.apply(input, output_size, mx_specs, name)


#-------------------------------------------------------------------------
# User-facing classes
#-------------------------------------------------------------------------
class AdaptiveAvgPool2d(nn.Module):
    def __init__(self,  output_size, mx_specs=None, name=None):
        super().__init__()

        mx_assert_test(mx_specs)
        self.mx_none = (mx_specs is None)

        self.output_size = output_size
        self.mx_specs = apply_mx_specs(mx_specs)
        self.name = name

    def forward(self, inputs):
        if self.mx_none:
            return f_adaptive_avg_pool2d(inputs, self.output_size)
        return AdaptiveAvgPool2dFunction.apply(inputs, self.output_size, self.mx_specs, self.name)


#-------------------------------------------------------------------------
# Internal functions
#-------------------------------------------------------------------------

def start_index(a, b, c):
    return math.floor((float(a) * float(c)) / b)

def end_index(a, b, c):
    return math.ceil((float(a + 1) * float(c)) / b)

class AdaptiveAvgPool2dFunction(torch.autograd.Function):
    """
    Forward pass:  compute the average for the slice in input
    Backward pass: distribute gradient in output equally to the correct slice in input
    """
    @staticmethod
    def forward(ctx, input, output_size, mx_specs=None, name=None):
        ctx.name = name

        sizeB, sizeD, isizeH, isizeW = input.size()

        if isinstance(output_size, tuple) and len(output_size) == 2:  # image H x W
            osizeH = output_size[0] if output_size[0] else isizeH  # need to support the case of None in output size
            osizeW = output_size[1] if output_size[1] else isizeW  # need to support the case of None in output size
        elif isinstance(output_size, int):  #square image H x H
            osizeH, osizeW = output_size, output_size
        elif output_size == None:
            osizeH, osizeW = isizeH, isizeW
        else:
            raise ValueError('expected 1D or 2D output_size (got {}D output_size)'
                             .format(len(output_size)))

        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        

        device = input.device
        output = torch.zeros(sizeB, sizeD, osizeH, osizeW, device=device)

        # loop over output
        for oh in range(osizeH):
            istartH = start_index(oh, osizeH, isizeH)
            iendH = end_index(oh, osizeH, isizeH)
            kH = iendH - istartH

            for ow in range(osizeW):
                istartW = start_index(ow, osizeW, isizeW)
                iendW = end_index(ow, osizeW, isizeW)
                kW = iendW - istartW

                input_slice = input[:, :, istartH:iendH, istartW:iendW]
                output[:, :, oh, ow] = vec_reduce_mean(input_slice, [2,3], keepdim=False, mx_specs=mx_specs)

        ctx.osizeH = osizeH
        ctx.osizeW = osizeW
        ctx.sizeB = sizeB
        ctx.sizeD = sizeD
        ctx.isizeH = isizeH
        ctx.isizeW = isizeW
        ctx.device = device
        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        return output


    @staticmethod
    def backward(ctx, grad_output):
        osizeH, osizeW, sizeB, sizeD, isizeH, isizeW, device = ctx.osizeH, ctx.osizeW, ctx.sizeB, ctx.sizeD, ctx.isizeH, ctx.isizeW, ctx.device

        grad_input = torch.zeros(sizeB, sizeD, isizeH, isizeW, device=device)

        # loop over output, calculate average
        for oh in range(osizeH):
            istartH = start_index(oh, osizeH, isizeH)
            iendH = end_index(oh, osizeH, isizeH)
            kH = iendH - istartH

            for ow in range(osizeW):
                istartW = start_index(ow, osizeW, isizeW)
                iendW = end_index(ow, osizeW, isizeW)
                kW = iendW - istartW

                grad_delta = grad_output[:,:, oh, ow] / kH / kW

                target_shape = [sizeB, sizeD, kH, kW]
            
                expanded_grad_delta = grad_delta.view(*grad_delta.shape, *(1,)*(len(target_shape)-grad_delta.ndim)).expand(target_shape)
               
                grad_input[:, :, istartH:iendH, istartW:iendW] = vec_add(grad_input[:, :, istartH:iendH, istartW:iendW], expanded_grad_delta, mx_specs=ctx.mx_specs)


        return (grad_input, None, None, None)    






