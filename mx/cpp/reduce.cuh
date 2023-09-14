/*
 * Microsoft Confidential
 */

#ifndef PYT_MX_REDUCE_CUH
#define PYT_MX_REDUCE_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// We choose the max possible reduce threads such that we can reduce
// across MAX_REDUCE_THREADS using a two-level reduction.
// We avoid reducing across different thread blocks.
const int MAX_REDUCE_THREADS = 1024;    // WARP_SIZE * WARP_SIZE
const int REDUCE_SHARED_MEM_SIZE = MAX_REDUCE_THREADS / WARP_SIZE;

// Basic assumptions:
// reduce_width (last dim size) is a multiple of WARP_SIZE

//-----------------------------------------------------------------------
// Wrap reduction
// Each template class defines:
//  * a start_val for the reduction
//  * a function to reduce the 32 threads within a warp
//  * a function to atomically update a partial reduction at some
//    address in global memory
//-----------------------------------------------------------------------
template<typename T>
class SumReduceHelper {
public:
    static const T start_val;

    // Reduces two values
    static T reduce(const T a, const T b) {
        return a+b;
    }

    // Reduce across the threads of a single warp, result in thread 0
    __forceinline__ __device__
    static T reduce_warp(T val, const int size=WARP_SIZE) {
        for (int offset = size/2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset, size);
        }
        return val;
    }

    // Atomically reduces val and the value at addr
    __forceinline__ __device__
    static void atomic_accum(T* addr, T val) {
        atomicAdd(addr, val);
    }
};

template<typename T>
class MaxReduceHelper {
public:
    static const T start_val;

    // Reduces two values
    static T reduce(const T a, const T b) {
        return (a > b) ? a : b;
    }

    // Reduce across the threads of a single warp, result in thread 0
    __forceinline__ __device__
    static T reduce_warp(T val, const int size=WARP_SIZE) {
        for (int offset = size/2; offset > 0; offset /= 2) {
            T tmp = __shfl_down_sync(0xffffffff, val, offset, size);
            val = (tmp > val) ? tmp : val;
        }
        return val;
    }

    // Atomically reduces val and the value at addr
    __forceinline__ __device__
    static void atomic_accum(T* addr, T val) {
        // atomicMax doesn't work for floats
        if (val >= 0) {
            atomicMax((int *)addr, __float_as_int(val));
        } else {
            atomicMin((unsigned int *)addr, __float_as_uint(val));
        }
    }
};

template<typename T> const T SumReduceHelper<T>::start_val = 0;
template<typename T> const T MaxReduceHelper<T>::start_val = -65504.f;

//-----------------------------------------------------------------------
// Block reduction
// This function reduces values across warps within a block.
// Threads with idx >= total_size must have already exited.
//-----------------------------------------------------------------------
template<template<typename T> class ReduceHelper, typename T>
__forceinline__ __device__
T reduce_block(
    T val,
    const long total_size,
    const long inner_dim_size
) {
    // Shared memory to communicate between warps within a block.
    // Each block gets its own copy of shared memory.
    // See https://jhui.github.io/2017/03/06/CUDA/ for CUDA memory model.
    static __shared__ T shared[REDUCE_SHARED_MEM_SIZE];

    // The threads in a warp all have the same WID and different LANE
    const int wid = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    // First level reduction
    // Each warp performs reduction across WARP_SIZE threads, then the first
    // thread in each warp writes this partial sum into shared memory.
    val = ReduceHelper<T>::reduce_warp(val);
    if (lane==0) shared[wid] = val;
    __syncthreads();                  // Wait for all partial reductions

    // Second level reduction
    const long rows_per_block = blockDim.x / inner_dim_size;
    const long warps_per_block = blockDim.x / WARP_SIZE;
    const long warps_per_row = inner_dim_size / WARP_SIZE;
    val = ReduceHelper<T>::start_val;
    // Case 1: Each warp (32 warps max) reduces a single row
    if ((wid < rows_per_block && lane < warps_per_row) ||
    // Case 2: Warp 0 reduces the 32 results from 32 warps
       (wid == 0 && rows_per_block == 0 && lane < warps_per_block))
    {
        val = shared[wid * warps_per_row + lane];
    }
    val = ReduceHelper<T>::reduce_warp(val);
    return val;
}

template<typename T>
__forceinline__ __device__
T reduce_sum_block(T val, const long total_size, const long inner_dim_size)
{
    return reduce_block<SumReduceHelper, T>(val, total_size, inner_dim_size);
}

template<typename T>
__forceinline__ __device__
T reduce_max_block(T val, const long total_size, const long inner_dim_size)
{
    return reduce_block<MaxReduceHelper, T>(val, total_size, inner_dim_size);
}


//-----------------------------------------------------------------------
// Array reduction, assumptions:
//  * WARP_SIZE <= inner_dim_size <= MAX_REDUCE_THREADS
//  * inner_dim_size is a multiple of WARP_SIZE
//-----------------------------------------------------------------------
template <template<typename T> class ReduceHelper, typename T>
__forceinline__ __device__
void reduce_kernel(
    const T* __restrict__ in,
    const long total_size,
    const long inner_dim_size,
    T* __restrict__ out
) {
    const long offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= total_size) return;
    // The threads in a warp all have the same WID and different LANE
    const int wid = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    T val = in[offset];
    val = reduce_block<ReduceHelper, T>(val, total_size, inner_dim_size);

    // The first thread of each warp produces the val for one inner dim row
    const long inner_dim_rows = total_size / inner_dim_size;
    const long rows_per_block = blockDim.x / inner_dim_size;
    // Each block produces rows_per_block vals
    const long out_addr = blockIdx.x * rows_per_block + wid;

    // Case 1: inner_dim_size <= max threads per block
    if (out_addr < inner_dim_rows &&   // valid output address
        wid < rows_per_block &&     // rows_per_block vals per block
        lane == 0)                  // first thread of each warp
    {
        out[out_addr] = val;
    }
    // Case 2: inner_dim_size is a multiple of threads per block
    else if (rows_per_block == 0 && threadIdx.x == 0)
    {
        const int adj_out_addr = offset / inner_dim_size;
        ReduceHelper<T>::atomic_accum(out+adj_out_addr, val);
    }
}

template <typename T>
__global__ void reduce_sum_kernel(
    const T* __restrict__ in,
    const long total_size,
    const long inner_dim_size,
    T* __restrict__ out
) {
    reduce_kernel<SumReduceHelper, T>(in, total_size, inner_dim_size, out);
}

template <typename T>
__global__ void reduce_max_kernel(
    const T* __restrict__ in,
    const long total_size,
    const long inner_dim_size,
    T* __restrict__ out
) {
    reduce_kernel<MaxReduceHelper, T>(in, total_size, inner_dim_size, out);
}

#endif
