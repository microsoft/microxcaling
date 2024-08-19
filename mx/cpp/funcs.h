#ifndef PYT_MX_FUNCS_H
#define PYT_MX_FUNCS_H

#include <torch/extension.h>
#include "common.cuh"
#include "shared_exp.cuh"
#include "quantize.cuh"

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//-----------------------------------------------------------------------
// CUDA forward declarations
//-----------------------------------------------------------------------
torch::Tensor quantize_mx_cuda(
    const torch::Tensor A, // Input tensor
    const int scale_bits,
    const int elem_ebits,
    const int elem_mbits,
    const float elem_max_norm,
    const torch::Tensor max_values,
    const int axis,
    const bool flush_fp32_subnorms = false,
    const RoundingMode rounding_mode = rd_away
);

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
);

torch::Tensor quantize_elemwise_cuda(
    const torch::Tensor A, // Input tensor
    const int bits,     // Number of bits (sign + magnitude) to quantize to
    const int exp_bits, // Number of exponent bits to quantize to
    const float max_norm,
    const RoundingMode rounding_mode = rd_away,
    const bool saturate_normals = false,
    const bool allow_denorm = true
);

torch::Tensor reduce_sum_inner_dim(
    const torch::Tensor A
);

torch::Tensor reduce_max_inner_dim(
    const torch::Tensor A
);

//-----------------------------------------------------------------------
// Templated quantization funcs
//-----------------------------------------------------------------------
template<typename T>
void quantize_mx_cpp(
    const T* A_data,
    const int scale_bits,
    const int elem_ebits,
    const int elem_mbits,
    const float elem_max_norm,
    const T* max_data,
    const int axis_size,
    const int pre_axis_size,
    const int post_axis_size,
    const bool flush_fp32_subnorms,
    const RoundingMode rounding_mode,
    T* B_data
) {

    // Loop over dimension before shared axis
    for (int i = 0; i < pre_axis_size; i++) {
    // Loop over dimension after shared axis
    for (int j = 0; j < post_axis_size; j++) {
        // Get shared exponent
        const long m_i = i * post_axis_size + j;
        int shared_exp = (int) get_biased_exponent(max_data[m_i]);
        bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);

        // Compute the shared scale
        const float scale = mx_get_shared_scale(
              shared_exp, scale_bits, elem_max_norm);

        // Loop over axis
        for (int k = 0; k < axis_size; k++) {
            int A_i = i * post_axis_size * axis_size +
                      k * post_axis_size + j;

            B_data[A_i] = quantize_mx_elem(
                    A_data[A_i], scale, flush_tile, elem_ebits, elem_mbits,
                    elem_max_norm, rounding_mode);
        }
    } }
}

#endif
