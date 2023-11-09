from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mx_ext",
    ext_modules=[
        CUDAExtension(
            # args here are passed to setuptools.Extension
            name="mx_ext",
            sources=[
                "funcs.cpp",
                "mx.cu",
                "elemwise.cu",
                "reduce.cu"
            ],
            extra_compile_args={
                'cxx': ["-O3"],
                'nvcc': ["-O3"],
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
