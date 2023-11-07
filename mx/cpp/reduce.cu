/*
 * Microsoft Confidential
 */

#include <torch/types.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include "common.cuh"
#include "reduce.cuh"

//-----------------------------------------------------------------------
// Reduction over innermost dim, with following assumptions:
//  * WARP_SIZE <= inner_dim_size <= MAX_REDUCE_THREADS
//  * inner_dim_size is a multiple of WARP_SIZE
//-----------------------------------------------------------------------
torch::Tensor reduce_sum_inner_dim(
  const torch::Tensor input
) {
    at::Device device = input.device();
    const at::cuda::CUDAGuard device_guard{device};
    const long inner_dim_size = get_inner_dim_size(input);
    const long total_size = get_total_size(input);

    // Allocate output
    c10::ArrayRef<int64_t> output_sizes(input.sizes().data(), input.dim()-1);
    //TODO: break this up into multiple lines
    auto output = torch::zeros(output_sizes, torch::dtype(
          input.dtype()).device(input.device()).layout(
          input.layout()).requires_grad(input.requires_grad()));

    AT_ASSERTM(inner_dim_size >= WARP_SIZE);
    AT_ASSERTM(inner_dim_size % WARP_SIZE == 0);

    // Get number of kernel threads
    const long blocks = get_blocks(total_size, MAX_REDUCE_THREADS);
    const int threads = get_threads(total_size, MAX_REDUCE_THREADS);

    // Call CUDA kernel 1
    if (input.dtype() == torch::ScalarType::Half) {
        AT_ASSERTM(0, "fp16 reduce is not supported");
    } else {
        reduce_sum_kernel<float><<<blocks, threads>>>(
            input.data_ptr<float>(),
            total_size,
            inner_dim_size,
            output.data_ptr<float>()
        );
    }

    gpuErrchk(cudaPeekAtLastError());
    return output;
}

torch::Tensor reduce_max_inner_dim(
    const torch::Tensor input
) {
    at::Device device = input.device();
    const at::cuda::CUDAGuard device_guard{device};
    const long inner_dim_size = get_inner_dim_size(input);
    const long total_size = get_total_size(input);

    // Allocate output
    c10::ArrayRef<int64_t> output_sizes(input.sizes().data(), input.dim()-1);
    //TODO: break this up into multiple lines
    auto output = torch::zeros(output_sizes, torch::dtype(
          input.dtype()).device(input.device()).layout(
          input.layout()).requires_grad(input.requires_grad()));

    AT_ASSERTM(inner_dim_size >= WARP_SIZE);
    AT_ASSERTM(inner_dim_size % WARP_SIZE == 0);

    // Get number of kernel threads
    const long blocks = get_blocks(total_size, MAX_REDUCE_THREADS);
    const int threads = get_threads(total_size, MAX_REDUCE_THREADS);

    // Call CUDA kernel 1
    if (input.dtype() == torch::ScalarType::Half) {
        AT_ASSERTM(0, "fp16 reduce is not supported");
    } else {
        reduce_max_kernel<float><<<blocks, threads>>>(
            input.data_ptr<float>(),
            total_size,
            inner_dim_size,
            output.data_ptr<float>()
        );
    }

    gpuErrchk(cudaPeekAtLastError());
    return output;
  }
