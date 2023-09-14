/*
 * Microsoft Confidential
 */

#ifndef PYT_MX_MX_CUH
#define PYT_MX_MX_CUH

#include "common.cuh"
#include "shared_exp.cuh"
#include "quantize.cuh"

//-----------------------------------------------------------------------
// quantize_mx_cuda_kernel
//-----------------------------------------------------------------------
template<typename T>
__global__ void quantize_mx_cuda_kernel(
    const T* __restrict__ input,
    const int scale_bits,
    const int elem_ebits,
    const int elem_mbits,
    const float elem_max_norm,
    const float* __restrict__ max_values,
    const long total_size,
    const int axis_size,
    const int post_axis_size,
    const bool flush_fp32_subnorms,
    const RoundingMode rounding_mode,
    T* __restrict__ output
) {
    const long offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= total_size) return;

    // Compute index of the max value for this element
    const long post_axis_i = offset % post_axis_size;
    const long pre_axis_i = offset / (post_axis_size * axis_size);

    // Get shared exponent
    const long m_i = pre_axis_i * post_axis_size + post_axis_i;
    int shared_exp = (int) get_biased_exponent(max_values[m_i]);
    bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);

    // Compute the shared scale
    const float scale = mx_get_shared_scale(
          shared_exp, scale_bits, elem_max_norm);

    T scaled_in = (flush_tile) ? 0 : input[offset] / scale;

    T scaled_out = quantize_elemwise(
            scaled_in, elem_mbits, elem_ebits, elem_max_norm,
            rounding_mode, true, true);

    output[offset] = scaled_out * scale;
}

//-----------------------------------------------------------------------
// quantize_innermost, fast MX quantization for axis=[-1]
// input requirements:
//  - the axis is dim-1 (the innermost dim),
//  - tile_size divides axis_size evenly
//  - tile_size is a power of 2
//  - tile_size <= WARP_SIZE
//-----------------------------------------------------------------------
template<typename T>
__global__ void quantize_mx_innermost_cuda_kernel (
    const T* __restrict__ in,
    const int scale_bits,
    const int elem_ebits,
    const int elem_mbits,
    const float elem_max_norm,
    const long total_size,
    const int tile_size,
    const bool flush_fp32_subnorms,
    const RoundingMode rounding_mode,
    T* __restrict__ out
) {
    const long offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= total_size) return;
    const T elem = in[offset];

    // allreduce to get the max value in each tile
    int shared_exp = get_biased_exponent(elem);
    for (int mask = tile_size/2; mask > 0; mask /= 2) {
        int _tmp = __shfl_xor_sync(0xFFFFFFFF, shared_exp, mask);
        shared_exp = (_tmp > shared_exp) ? _tmp : shared_exp;
    }

    bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);

    // Compute the shared scale
    const float scale = mx_get_shared_scale(
          shared_exp, scale_bits, elem_max_norm);

    T scaled_in = (flush_tile) ? 0 : elem / scale;

    T scaled_out = quantize_elemwise(
            scaled_in, elem_mbits, elem_ebits, elem_max_norm,
            rounding_mode, true, true);

    out[offset] = scaled_out * scale;
}

//-----------------------------------------------------------------------
// quantize_mx_by_tile kernel
// Each thread loops across the tile to get the max exponent, then
// loops across it again to perform quantization.
//-----------------------------------------------------------------------
template<typename T>
__global__ void quantize_mx_by_tile_cuda_kernel (
    const T* __restrict__ in,
    const int scale_bits,
    const int elem_ebits,
    const int elem_mbits,
    const float elem_max_norm,
    const int total_tiles,
    const int tile_size,
    const int num_tiles,
    const int axis_size,
    const int post_axis_size,
    const bool flush_fp32_subnorms,
    const RoundingMode rounding_mode,
    T* __restrict__ out
) {
    const long offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= total_tiles) return;

    // Calculate indices on different dimensions
    const long post_axis_i = offset % post_axis_size;
    const long num_tiles_i = (offset / post_axis_size) % num_tiles;
    const long pre_axis_i = offset / (num_tiles * post_axis_size);

    // Handle non-full bounding box/tile
    int adjusted_tile_size;
    if ((num_tiles_i + 1) * tile_size > axis_size) {
        adjusted_tile_size = axis_size % tile_size;
    } else {
        adjusted_tile_size = tile_size;
    }

    // Find biased shared_exp
    int shared_exp = 0; // biased exp must be >= 0
    for (int i = 0; i < adjusted_tile_size; i++) {
        long in_i = pre_axis_i * axis_size * post_axis_size +
            (num_tiles_i * tile_size + i) * post_axis_size +
            post_axis_i;

        int exp = get_biased_exponent(in[in_i]);
        shared_exp = (exp > shared_exp) ? exp : shared_exp;
    }

    bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);

    // Compute the shared scale
    const float scale = mx_get_shared_scale(
          shared_exp, scale_bits, elem_max_norm);

    // Loop over bounding box to quantize
    for (int i = 0; i < adjusted_tile_size; i++) {
        long in_i = pre_axis_i * axis_size * post_axis_size +
            (num_tiles_i * tile_size + i) * post_axis_size +
            post_axis_i;

        T scaled_in = (flush_tile) ? 0 : in[in_i] / scale;

        T scaled_out = quantize_elemwise(
                scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                rounding_mode, true, true);

        out[in_i] = scaled_out * scale;
    }
}

#endif
