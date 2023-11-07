/*
 * Microsoft Confidential
 *
 * funcs.cpp
 *
 * This file provides various C++ implementations and torch interface
 */

#include <torch/extension.h>
#include <vector>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <cassert>
#include "funcs.h"
#include "common.cuh"
#include "shared_exp.cuh"
#include "quantize.cuh"


//-----------------------------------------------------------------------
// CPU custom code using C++
//-----------------------------------------------------------------------
torch::Tensor quantize_mx_func_cpp(
    torch::Tensor A, // Input tensor
    int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    torch::Tensor max_values,  // Max values in each block
    int axis, // Axis along which exponents are shared
    const bool flush_fp32_subnorms,
    const int rmode = 0
) {
    const int ndim = A.dim();
    AT_ASSERTM(1 <= ndim && ndim <= 5, " number of dims outside of range [1,5]");
    AT_ASSERTM(axis >= 0 && axis < ndim, " shared axis < 0 or shared axis >= ndim");

    // Explicitly cast int to enum
    RoundingMode rounding_mode = static_cast<RoundingMode>(rmode);

    // Output
    auto B = torch::empty_like(A);

    // Get data pointers
    auto max_value_data = max_values.data_ptr<float>();
    auto A_data = A.data_ptr<float>();
    auto B_data = B.data_ptr<float>();

    // Size of shared axis
    auto A_sizes = A.sizes();
    const int axis_size = A_sizes[axis];
    // Size of axes before shared axis
    int pre_axis_size = 1;
    for (int i = 0 ; i < axis; i++) {
        pre_axis_size *= A_sizes[i];
    }
    // Size of axes after shared axis
    int post_axis_size = 1;
    for (int i = axis + 1; i < ndim; i++) {
        post_axis_size *= A_sizes[i];
    }

    // Loop over dimension before shared axis
    for (int i = 0; i < pre_axis_size; i++) {
        // Loop over dimension after shared axis
        for (int j = 0; j < post_axis_size; j++) {
            // Get shared exponent
            const long m_i = i * post_axis_size + j;
            int shared_exp = (int) get_biased_exponent(max_value_data[m_i]);
            bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);

            // Compute the shared scale
            const float scale = mx_get_shared_scale(
                  shared_exp, scale_bits, elem_max_norm);

            // Loop over axis
            for (int k = 0; k < axis_size; k++) {
                int A_i = i * post_axis_size * axis_size +
                          k * post_axis_size + j;

                float scaled_in = (flush_tile) ? 0 : A_data[A_i] / scale;

                float scaled_out = quantize_elemwise(
                        scaled_in, elem_mbits, elem_ebits, elem_max_norm,
                        rounding_mode, true, true);

                B_data[A_i] = scaled_out * scale;
            }
        }
    }

    return B;
}

torch::Tensor quantize_elemwise_func_cpp(
    torch::Tensor input, // Input tensor
    int bits,    // Number of bits (sign + magnitude) to quantize to
    int exp_bits, // Number of exponent bits to quantize to
    float max_norm,
    const int rmode = 0,
    const bool saturate_normals = false,
    const bool allow_denorm = true
) {
    // Explicitly cast int to enum
    RoundingMode rounding_mode = static_cast<RoundingMode>(rmode);

    // Calculate total size of the input
    const int ndim = input.dim();
    const auto input_sizes = input.sizes();
    long total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= input_sizes[i];
    }

    // Output
    auto output = torch::empty_like(input);

    // Get data pointers
    auto i_data = input.data_ptr<float>();
    auto o_data = output.data_ptr<float>();

    // Loop over dimension before shared axis
    for (long i = 0; i < total_size; i++) {
        o_data[i] = quantize_elemwise(
            i_data[i], bits, exp_bits, max_norm,
            rounding_mode, saturate_normals, allow_denorm);
    }

    return output;
}

//-----------------------------------------------------------------------
// GPU custom code using CUDA
//-----------------------------------------------------------------------
torch::Tensor quantize_mx_func_cuda(
    torch::Tensor A, // Input tensor
    int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    torch::Tensor max_values, // Max values along axis of interest
    int axis,  // Axis along which exponents are shared
    const bool flush_fp32_subnorms = false,
    const int rmode = 0
) {
    CHECK_INPUT(A);
    AT_ASSERTM(axis < A.dim(), " quantize_mx_func_cuda axis exceeds input dimensions");
    CHECK_INPUT(max_values);

    // Explicitly cast int to enum
    RoundingMode rounding_mode = static_cast<RoundingMode>(rmode);

    return quantize_mx_cuda(
        A, scale_bits, elem_ebits, elem_mbits, elem_max_norm,
        max_values, axis, flush_fp32_subnorms, rounding_mode);
}

torch::Tensor quantize_mx_by_tile_func_cuda(
    torch::Tensor A, // Input tensor
    int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    int tile_size,
    int axis,
    const bool flush_fp32_subnorms = false,
    const int rmode = 0
) {
    CHECK_INPUT(A);
    AT_ASSERTM(axis < A.dim(), " quantize_mx_by_tile axis exceeds input dimensions");

    // Explicitly cast int to enum
    RoundingMode rounding_mode = static_cast<RoundingMode>(rmode);

    return quantize_mx_by_tile_cuda(
        A, scale_bits, elem_ebits, elem_mbits, elem_max_norm,
        tile_size, axis, flush_fp32_subnorms, rounding_mode);
}

torch::Tensor quantize_elemwise_func_cuda(
    torch::Tensor A, // Input tensor
    int bits,     // Number of bits (sign + magnitude) to quantize to
    int exp_bits, // Exponent bits to quantize to
    float max_norm,
    const int rmode = 0,
    const bool saturate_normals = false,
    const bool allow_denorm = true
) {
    CHECK_INPUT(A);
    AT_ASSERTM(bits <= 24, " quantize_elemwise with bits > 24 leads to negative shifts");

    // Explicitly cast int to enum
    RoundingMode rounding_mode = static_cast<RoundingMode>(rmode);

    return quantize_elemwise_cuda(
        A, bits, exp_bits, max_norm,
        rounding_mode, saturate_normals, allow_denorm);
}

torch::Tensor reduce_sum_inner_dim_cuda(
    torch::Tensor A
) {
    CHECK_INPUT(A);
    return reduce_sum_inner_dim(A);
}

torch::Tensor reduce_max_inner_dim_cuda(
    torch::Tensor A
) {
    CHECK_INPUT(A);
    return reduce_max_inner_dim(A);
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_mx_func_cpp", &quantize_mx_func_cpp, "MX quantization function in C++");
    m.def("quantize_elemwise_func_cpp", &quantize_elemwise_func_cpp, "Element-wise quantization function in C++");
    m.def("quantize_mx_func_cuda", &quantize_mx_func_cuda, "MX quantization function with CUDA");
    m.def("quantize_mx_by_tile_func_cuda", &quantize_mx_by_tile_func_cuda, "MX quantization function with CUDA");
    m.def("quantize_elemwise_func_cuda", &quantize_elemwise_func_cuda, "Element-wise quantization function with CUDA");
    m.def("reduce_sum_inner_dim", &reduce_sum_inner_dim_cuda, "Sum reduction over innermost dim");
    m.def("reduce_max_inner_dim", &reduce_max_inner_dim_cuda, "Max reduction over innermost dim");
}
