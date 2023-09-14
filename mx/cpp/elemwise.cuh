/*
 * Microsoft Confidential
 */

#ifndef PYT_MX_ELEMWISE_CUH
#define PYT_MX_ELEMWISE_CUH

#include "common.cuh"
#include "quantize.cuh"

//-----------------------------------------------------------------------
// quantize_elementwise kernel
//-----------------------------------------------------------------------
const int ELEMWISE_ROW_SIZE = 1024;
const int ELEMWISE_ROWS_PER_THREAD = 16;

template<typename T>
__global__ void quantize_elemwise_cuda_kernel(
    const T* __restrict__ input,
    const int bits,
    const int exp_bits,
    const float max_norm,
    const long total_size,
    const RoundingMode rounding_mode,
    const bool saturate_normals,
    const bool allow_denorm,
    T* __restrict__ output
) {
    long offset = blockIdx.x * blockDim.x * ELEMWISE_ROWS_PER_THREAD + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < ELEMWISE_ROWS_PER_THREAD; ++i) {
        if (offset >= total_size) return;
        output[offset] = quantize_elemwise(
            input[offset], bits, exp_bits, max_norm,
            rounding_mode, saturate_normals, allow_denorm);
        offset += ELEMWISE_ROW_SIZE;
    }
}

#endif
