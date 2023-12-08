import torch

from .specs import apply_mx_specs

from .adaptive_avg_pooling import AdaptiveAvgPool2d, adaptive_avg_pool2d
from .batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d, batch_norm
from .bmm import bmm
from .convolution import Conv1d, Conv2d, Conv3d, conv1d, conv2d, conv3d
from .transpose_convolution import ConvTranspose2d
from .activations import GELU, ReLU, SiLU, ReLU6, Sigmoid, Tanh, LeakyReLU, gelu, leaky_relu, relu, relu6, sigmoid, silu, \
    tanh
from .groupnorm import GroupNorm, group_norm
from .layernorm import LayerNorm, layer_norm
from .linear import Linear, linear
from .matmul import MatMulFunction, matmul
from .rnn import LSTM
from .softmax import Softmax, softmax
from .simd_ops import simd_add, simd_sub, simd_mul, simd_div, simd_exp, simd_log, simd_reduce_sum, simd_reduce_mean, \
    simd_norm, simd_square

DEBUG = False

torch_addmm = torch.addmm


def tracer_decorator(func, mx_specs):
    def wrapper(*args, **kwargs):
        if 'dtype' in kwargs:
            dtype = kwargs.pop('dtype')
        else:
            dtype = None
        if DEBUG:
            print(func.__name__, mx_specs)
        res = func(*args, mx_specs=mx_specs, **kwargs)
        if dtype is not None:
            res = res.to(dtype)
        return res
    return wrapper


def inject_pyt_ops(mx_specs):
    """
    Injects PyTorch operators into the PyTorch namespace, replacing them. The ops need to have their own reference
    to the original PyTorch operator, so that they can call it.
    """
    def mx_class_factory(cls):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, mx_specs=mx_specs, **kwargs)
        return type(f'{cls.__name__}_inj', (cls,), {'__init__': __init__})

    for k, v in FUNCTION_MAPPING.items():
        if k in torch.__dict__:
            torch.__dict__[k] = tracer_decorator(v, mx_specs)
        if k in torch.nn.functional.__dict__:
            torch.nn.functional.__dict__[k] = tracer_decorator(v, mx_specs)
    for k,v in MODULE_MAPPING.items():
        torch.nn.__dict__[k] = mx_class_factory(v)


def addmm_mx(bias, in1, in2, mx_specs=None, name=None):
    if mx_specs is None:
        out = torch_addmm(bias, in1, in2)
        return out
    mx_specs = apply_mx_specs(mx_specs)
    return MatMulFunction.apply(in1, in2, bias, mx_specs, name)


MODULE_MAPPING = {
    "AdaptiveAvgPool2d": AdaptiveAvgPool2d,
    "BatchNorm1d": BatchNorm1d,
    "BatchNorm2d": BatchNorm2d,
    "BatchNorm3d": BatchNorm3d,
    "Conv1d": Conv1d,
    "Conv2d": Conv2d,
    "Conv3d": Conv3d,
    "ConvTranspose2d": ConvTranspose2d,
    "GELU": GELU,
    "GroupNorm": GroupNorm,
    "LSTM": LSTM,
    "LayerNorm": LayerNorm,
    "LeakyReLU": LeakyReLU,
    "Linear": Linear,
    "ReLU": ReLU,
    "ReLU6": ReLU6,
    "SiLU": SiLU,
    "Sigmoid": Sigmoid,
    "Softmax": Softmax,
    "Tanh": Tanh,
}

FUNCTION_MAPPING = {
    "adaptive_avg_pool2d": adaptive_avg_pool2d,
    "batch_norm": batch_norm,
    "conv1d": conv1d,
    "conv2d": conv2d,
    "conv3d": conv3d,
    "gelu": gelu,
    "group_norm": group_norm,
    "layer_norm": layer_norm,
    "leaky_relu": leaky_relu,
    "linear": linear,
    "relu": relu,
    "relu6": relu6,
    "sigmoid": sigmoid,
    "silu": silu,
    "softmax": softmax, # Causes NaNs?
    "tanh": tanh,
    "add": simd_add,
    "sub": simd_sub,
    "mul": simd_mul,
    "div": simd_div,
    "exp": simd_exp,
    "log": simd_log,
    "sum": simd_reduce_sum,
    "mean": simd_reduce_mean,
    "norm": simd_norm,
    "square": simd_square,
    "matmul": matmul,
    "mm": matmul,
    "addmm": addmm_mx,
    "bmm": bmm,
}
