#include <torch/types.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include "elemwise.cuh"

//-----------------------------------------------------------------------
// quantize_elementwise
//-----------------------------------------------------------------------
torch::Tensor quantize_elemwise_cuda(
    const torch::Tensor input,
    const int bits,
    const int exp_bits,
    const float max_norm,
    const RoundingMode rounding_mode = rd_away,
    const bool saturate_normals = false,
    const bool allow_denorm = true
) {
    at::Device device = input.device();
    const at::cuda::CUDAGuard device_guard{device};
    auto output = torch::empty_like(input);
    output = output.to(device);

    // Calculate total size of the input
    const int ndim = input.dim();
    const auto input_sizes = input.sizes();
    long total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= input_sizes[i];
    }

    // Each block contains ELEMWISE_ROW_SIZE threads
    // Each thread handles ELEMWISE_ROWS_PER_THREAD elements
    const int elemwise_tile_size = ELEMWISE_ROW_SIZE * ELEMWISE_ROWS_PER_THREAD;
    long blocks = total_size / elemwise_tile_size;
    if (total_size % elemwise_tile_size != 0)
        blocks += 1;
    const int threads = ELEMWISE_ROW_SIZE;

    // Call CUDA kernel
    if (input.dtype() == torch::ScalarType::Half) {
        #if (__CUDACC_VER_MAJOR__ >= 10 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ >= 2))
        quantize_elemwise_cuda_kernel<<<blocks, threads>>>(
            (__half*)input.data_ptr<at::Half>(),
            bits,
            exp_bits,
            max_norm,
            total_size,
            rounding_mode,
            saturate_normals,
            allow_denorm,
            (__half*)output.data_ptr<at::Half>()
        );
        #else
        AT_ASSERTM(0, " fp16 not supported on this CUDA device");
        #endif
    } else {
        quantize_elemwise_cuda_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            bits,
            exp_bits,
            max_norm,
            total_size,
            rounding_mode,
            saturate_normals,
            allow_denorm,
            output.data_ptr<float>()
        );
    }

    gpuErrchk(cudaPeekAtLastError());
    return output;
}

