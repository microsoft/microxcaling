#include <torch/types.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include "common.cuh"
#include "mx.cuh"

//-----------------------------------------------------------------------
// quantize_mx_cuda
//-----------------------------------------------------------------------
torch::Tensor quantize_mx_cuda(
    const torch::Tensor input,
    const int scale_bits,
    const int elem_ebits,
    const int elem_mbits,
    const float elem_max_norm,
    const torch::Tensor max_values,
    const int axis,
    const bool flush_fp32_subnorms = false,
    const RoundingMode rounding_mode = rd_away
) {
    at::Device device = input.device();
    const at::cuda::CUDAGuard device_guard{device};
    auto output = torch::empty_like(input);
    output = output.to(device);

    const int ndim = input.dim();
    auto input_sizes = input.sizes();

    // Size of shared axis
    const int axis_size = input_sizes[axis];
    // Size of axes before shared axis
    long pre_axis_size = 1;
    for (int i = 0; i < axis; i++) {
        pre_axis_size *= input_sizes[i];
    }
    // Size of axes after shared axis
    long post_axis_size = 1;
    for (int i = axis + 1; i < ndim; i++) {
        post_axis_size *= input_sizes[i];
    }

    long total_size = pre_axis_size * axis_size * post_axis_size;
    // 1 thread per element, up to max number of threads
    const long blocks = get_blocks(total_size);
    const int threads = get_threads(total_size);

    // Call CUDA kernel
    if (input.dtype() == torch::ScalarType::Half) {
        AT_ASSERTM(0, " fp16 not supported for MX");
    } else {
        quantize_mx_cuda_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            scale_bits,
            elem_ebits,
            elem_mbits,
            elem_max_norm,
            max_values.data_ptr<float>(),
            total_size,
            axis_size,
            post_axis_size,
            flush_fp32_subnorms,
            rounding_mode,
            output.data_ptr<float>()
        );
    }

    gpuErrchk(cudaPeekAtLastError());
    return output;
}

//-----------------------------------------------------------------------
// quantize_mx_by_tile
//-----------------------------------------------------------------------
torch::Tensor quantize_mx_by_tile_cuda(
    const torch::Tensor input,
    const int scale_bits,
    const int elem_ebits,
    const int elem_mbits,
    const float elem_max_norm,
    const int tile_size,
    const int axis,
    const bool flush_fp32_subnorms = false,
    const RoundingMode rounding_mode = rd_away
) {
    at::Device device = input.device();
    const at::cuda::CUDAGuard device_guard{device};
    auto output = torch::empty_like(input);
    output = output.to(device);

    const int ndim = input.dim();
    auto input_sizes = input.sizes();

    // Size of shared axis
    const int axis_size = input_sizes[axis];
    int tsize = (tile_size > 0) ? tile_size : axis_size;
    // Size of axes before shared axis
    long pre_axis_size = 1;
    for (int i = 0; i < axis; i++) {
        pre_axis_size *= input_sizes[i];
    }
    // Size of axes after shared axis
    long post_axis_size = 1;
    for (int i = axis + 1; i < ndim; i++) {
        post_axis_size *= input_sizes[i];
    }
    // Number of tiles along the chosen axis
    int num_tiles = axis_size / tsize;
    if (axis_size % tsize) {
        num_tiles += 1;
    }

    // Call quantize innermost if the shared exponent axis is the
    // innermost axis and tile size is small
    if (axis == ndim-1 && axis_size % tsize == 0 &&
        tsize <= WARP_SIZE && is_power_of_two(tsize))
    {
        const long total_size = pre_axis_size * axis_size * post_axis_size;
        const long blocks = get_blocks(total_size);
        const int threads = get_threads(total_size);

        if (input.dtype() == torch::ScalarType::Half) {
            AT_ASSERTM(0, " fp16 not supported for MX");
        } else {
            quantize_mx_innermost_cuda_kernel<<<blocks, threads>>>(
                input.data_ptr<float>(),
                scale_bits,
                elem_ebits,
                elem_mbits,
                elem_max_norm,
                total_size,
                tsize,
                flush_fp32_subnorms,
                rounding_mode,
                output.data_ptr<float>()
            );
        }
    }
    // Otherwise call quantize_mx_by_tile
    else {
        // 1 thread per tile, up to max number of threads
        const long total_tiles = pre_axis_size * num_tiles * post_axis_size;
        const long blocks = get_blocks(total_tiles);
        const int threads = get_threads(total_tiles);

        // Call CUDA kernel
        if (input.dtype() == torch::ScalarType::Half) {
            AT_ASSERTM(0, " fp16 not supported for MX");
        } else {
            quantize_mx_by_tile_cuda_kernel<<<blocks, threads>>>(
                input.data_ptr<float>(),
                scale_bits,
                elem_ebits,
                elem_mbits,
                elem_max_norm,
                total_tiles,
                tsize,
                num_tiles,
                axis_size,
                post_axis_size,
                flush_fp32_subnorms,
                rounding_mode,
                output.data_ptr<float>()
            );
        }
    }

    gpuErrchk(cudaPeekAtLastError());
    return output;
}
