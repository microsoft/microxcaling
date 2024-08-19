/*
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

    if (A.dtype() == torch::ScalarType::Float) {
        quantize_mx_cpp(
            A.data_ptr<float>(),
            scale_bits,
            elem_ebits, elem_mbits, elem_max_norm,
            max_values.data_ptr<float>(),
            axis_size, pre_axis_size, post_axis_size,
            flush_fp32_subnorms,
            rounding_mode,
            B.data_ptr<float>()
        );
    }
    else if (A.dtype() == torch::ScalarType::Half) {
        quantize_mx_cpp(
            A.data_ptr<at::Half>(),
            scale_bits,
            elem_ebits, elem_mbits, elem_max_norm,
            max_values.data_ptr<at::Half>(),
            axis_size, pre_axis_size, post_axis_size,
            flush_fp32_subnorms,
            rounding_mode,
            B.data_ptr<at::Half>()
        );
    }
    else if (A.dtype() == torch::ScalarType::BFloat16) {
        quantize_mx_cpp(
            A.data_ptr<at::BFloat16>(),
            scale_bits,
            elem_ebits, elem_mbits, elem_max_norm,
            max_values.data_ptr<at::BFloat16>(),
            axis_size, pre_axis_size, post_axis_size,
            flush_fp32_subnorms,
            rounding_mode,
            B.data_ptr<at::BFloat16>()
        );
    }
    else {
        AT_ASSERTM(0, " Tensor dtype not supported");
    }
    return B;
}

template <typename T>
void quantize_tensor(const torch::Tensor& input, torch::Tensor& output,
                     int bits, int exp_bits, float max_norm,
                     int rmode, bool saturate_normals,
                     bool allow_denorm) {
    auto i_data = input.data_ptr<T>();
    auto o_data = output.data_ptr<T>();
    long get_total_size = input.numel();  
    // Explicitly cast int to enum
    RoundingMode rounding_mode = static_cast<RoundingMode>(rmode);
    for (long i = 0; i < get_total_size; i++) {
        o_data[i] = quantize_elemwise(
            i_data[i], bits, exp_bits, max_norm,
            rounding_mode, saturate_normals, allow_denorm);
    }
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
    // Calculate total size of the input
    const int ndim = input.dim();
    const auto input_sizes = input.sizes();

    // Output
    auto output = torch::empty_like(input);

    // Example usage:
    if (input.dtype() == torch::ScalarType::Float) {
        quantize_tensor<float>(input, output, bits, exp_bits, max_norm,
                            rmode, saturate_normals, allow_denorm);
    } else if (input.dtype() == torch::ScalarType::Half) {
        quantize_tensor<at::Half>(input, output, bits, exp_bits, max_norm,
                                rmode, saturate_normals, allow_denorm);
    } else if (input.dtype() == torch::ScalarType::BFloat16) {
        quantize_tensor<at::BFloat16>(input, output, bits, exp_bits, max_norm,
                                    rmode, saturate_normals, allow_denorm);
    }
    else {
        AT_ASSERTM(0, " Tensor dtype not supported");
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
