"""
For pre-building the custom cpp extensions for cuda - currently not used
"""

from setuptools import setup, Extension
from torch.utils import cpp_extension

sources = [
    "funcs.cpp",
    "mx.cu",
    "elemwise.cu",
    "reduce.cu",
]

setup(name='custom_extensions',
      ext_modules=[cpp_extension.CppExtension('custom_extensions', sources)],
      cmdclass={'build_ext': cpp_extension.BuildExtension})