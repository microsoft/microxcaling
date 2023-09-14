"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Python interface for custom CUDA implementations of functions.
"""

import os
from torch.utils.cpp_extension import load

sources = [
    "funcs.cpp",
    "mx.cu",
    "elemwise.cu",
    "reduce.cu",
]
file_dir = os.path.dirname(__file__)
sources = [os.path.join(file_dir, "cpp", x) for x in sources]
funcs = load(name="custom_extensions", sources=sources)

