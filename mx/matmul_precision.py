"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
import torch
from .formats import ElemFormat
from contextlib import contextmanager

def get_elem_format(inp):
    if inp == None:
        elem_format = 'Unknown'
    elif type(inp) is str:
        elem_format = ElemFormat.from_str(inp)
    return elem_format

"""
A contextmanager that allows us to perform torch matmuls in BF16 if the emulated datatype
can be perfectly represented in the BF16 format.

You can use it by wrapping a matmul in it as follows:
with set_matmul_precision(a, b, a_elem_format, b_elem_format):
    c = a @ b

We use torch's set_float32_matmul_precision instead of casting to BF16 before performing
the matmul because Torch does not currently allow matmuls to return a different type than their
inputs.  Since we normally emulate returning FP32, even if the inputs can be perfectly represented
by BF16, the output cannot.  For example, in MXINT2, we could have inputs to a matmul like the following:
a = torch.zeros(4096, 4096)
b = torch.full((4096, 4096), 1)
a[0,0] = 1
a[0,-1] = 2^-22
c = a @ b
In this case, c[0,0] should be 1+2^-22, which is representable in FP32 but not BF16. torch.set_float32_matmul_precision("medium") 
actually causes a matmul with FP32 inputs to be downcast to BF16, but the result to be returned as FP32 as we desire.

You can also force this fast execution mode even if the element format is not fully representable in BF16 by setting the force_bf16 option to True.
"""
@contextmanager
def set_matmul_precision(a, b, a_elem_format, b_elem_format, force_bf16=False):
    a_elem_format, b_elem_format = get_elem_format(a_elem_format), get_elem_format(b_elem_format)
    narrow_precision = [ElemFormat.int4, ElemFormat.int2, ElemFormat.fp6_e3m2, ElemFormat.fp6_e2m3,
                        ElemFormat.fp4, ElemFormat.fp4_e2m1]
    fp32_matmul_precision = torch.get_float32_matmul_precision()
    allow_bf16_reduced_precision_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    if a.device.type == "cuda" and b.device.type == "cuda" \
        and torch.cuda.is_bf16_supported() and \
        ((a_elem_format in narrow_precision and b_elem_format in narrow_precision) or
         force_bf16):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.set_float32_matmul_precision("medium")
    yield
    torch.set_float32_matmul_precision(fp32_matmul_precision)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = allow_bf16_reduced_precision_reduction
