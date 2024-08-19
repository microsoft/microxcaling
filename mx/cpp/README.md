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
| Information       | Nvidia V100                | Nvidia A100                | Nvidia H100                |
|-------------------|:--------------------------:|:--------------------------:|:--------------------------:|
| Container Image   | nvcr.io/nvidia/pytorch:24.06-py3 | nvcr.io/nvidia/pytorch:24.06-py3 | nvcr.io/nvidia/pytorch:24.06-py3 |
| OS                | Ubuntu 20.04               | Ubuntu 22.04               | Ubuntu 22.04               |
| Nvidia Driver     | 535.171.04                 | 535.183.01                 | 550.54.15                  |
| CUDA              | 12.5                       | 12.5                       | 12.5                       |
| cuDNN             | 9.1.0.70                   | 9.1.0.70                   | 9.1.0.70                   |
| Python            | 3.10.12                    | 3.10.12                    | 3.10.12                    |
| PyTorch           | 2.4.0                      | 2.4.0                      | 2.4.0                      |