# C++ and CUDA Extensions

This directory includes custom C++ and CUDA implementations of functions. These
are automatically compiled and pulled into the Python APIs using
PyTorch's JIT compilation of extensions.

The following are some references for creating custom extensions for PyTorch:

* [Custom C++ and CUDA Extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)
* [Tensor class](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html)
* [Tensor creation API](https://pytorch.org/cppdocs/notes/tensor_creation.html)

The following are some general CUDA references:

* Introduction to CUDA: https://devblogs.nvidia.com/even-easier-introduction-cuda/
* Threads, blocks, and grid: 
  * https://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid
  * https://stackoverflow.com/questions/16619274/cuda-griddim-and-blockdim
* How to Query Device Properties and Handle Errors in CUDA C/C++: https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc/

# Dependencies

This library has been built and tested on a machine with the following:
* Ubuntu 18.04
* Nvidia Tesla V100
* Nvidia driver 530.30.02
* CUDA 12.1
* libcudnn7\_7.6.4.38
* Python 3.9.12
* PyTorch 1.12.1+cu113
