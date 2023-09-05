/*
 * Microsoft Confidential
 */

#ifndef PYT_MX_COMMON_CUH
#define PYT_MX_COMMON_CUH

#include <stdio.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Max threads per block for CUDA compute capability 2.x - 7.5 is 1024
// Max threads for some CUDA random number generators is 256
#define MAX_THREADS 1024
#define WARP_SIZE 32

#define FLOAT32_EXP_BIAS 127
#define FLOAT32_EXP_MAX 255
#define FLOAT32_TRAILING_MBITS 23
#define FLOAT32_IMPLIED1 (1 << FLOAT32_TRAILING_MBITS)
#define FLOAT32_FULL_MBITS (FLOAT32_TRAILING_MBITS + 1)
#define FLOAT32_INF 0x7fe00000
#define FLOAT32_EXP_OFFSET 23
#define FLOAT32_SIGN_OFFSET 31
#define FLOAT32_EXP_MASK 0x7f800000
#define FLOAT32_MANTISSA_MASK 0x007fffff

#define FLOAT16_MIN_NORMAL_EXP -14
#define FLOAT16_MAX_EXP 15
#define FLOAT16_EXP_BIAS 15

//----------------------------------------------------------------------
// Definitions from Deepspeed
//----------------------------------------------------------------------
#define DS_CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#define DS_MAX_THREADS 1024
#define DS_THREADS 256

#define DS_MAX_THREAD_STRIDE 32
#define DS_TILE_DIM 32

// Maximum sequence-length support based on the number of threads (2048) allowed in each block and
// this MAX is 8K For higher sequence length we need to use higher Max, like for 64K : 32
#define DS_MAX_THREAD_ITERATIONS 8  // Maximum 8K
#define DS_MAX_WARP_NUM 32

#define DS_MAX_REGISTERS 256

#define DS_MAX_REG 256

#define DS_WARP_SIZE_BITS 5


//-----------------------------------------------------------------------
// Misc. helper functions
//-----------------------------------------------------------------------
inline bool is_power_of_two(const int x) {
    // Will return true for x == 0
    return !(x & (x - 1));
}

inline long get_inner_dim_size(const torch::Tensor A) {
    const int ndim = A.dim();
    auto input_sizes = A.sizes();
    return input_sizes[ndim-1];
}

inline long get_total_size(const torch::Tensor A) {
    const int ndim = A.dim();
    auto input_sizes = A.sizes();
    long total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= input_sizes[i];
    }
    return total_size;
}

//-----------------------------------------------------------------------
// Divides a parallelism factor 'size' into either:
//  - 1 block of size threads
//  - B blocks of MAX_THREADS s.t. B*MAX_THREADS > size
//-----------------------------------------------------------------------
inline long get_blocks(
    long size,
    int max_threads=MAX_THREADS
) {
    long blocks;
    if (size <= max_threads) {
        blocks = 1;
    } else {
        blocks = size / max_threads;
        if (size % max_threads) {
            blocks += 1;
        }
    }
    return blocks;
}

inline int get_threads(
    long size,
    int max_threads=MAX_THREADS
) {
    int final_threads = int(std::min(size, long(max_threads)));
    if (final_threads < 0) //If overflow happens.
    {
        return 2147483647; //return INT MAX
    }
    return final_threads;
}

//---------------------------------------------------------
// Helper types/structs
//---------------------------------------------------------
typedef union {
    unsigned int i;
    float f;
} u_float_int;

typedef enum _RoundingMode {
  rd_away = 0,    // round nearest, ties to away
  rd_floor = 1,   // floor
  rd_even = 2     // round nearest, ties to even
} RoundingMode;

//---------------------------------------------------------
// Helper functions for quantization
//---------------------------------------------------------
__host__ __device__ __forceinline__
int get_sign(
    const u_float_int input
) {
    int sign = input.i >> FLOAT32_SIGN_OFFSET;
    return sign;
}

__host__ __device__ __forceinline__
int get_biased_exponent(
    const u_float_int input
) {
    // Mask only exponent bits
    int exp = input.i & FLOAT32_EXP_MASK;
    // Shift down to lowest bits
    exp = exp >> FLOAT32_EXP_OFFSET;
    return exp;
}

__host__ __device__ __forceinline__
int get_biased_exponent(
    const float input
) {
    u_float_int u;
    u.f = input;
    return get_biased_exponent(u);
}

// get_unbiased_exponent supports denorms
__host__ __device__ __forceinline__
int get_unbiased_exponent(
    const float input
) {
    u_float_int u;
    u.f = input;
    int exp = get_biased_exponent(u);
    if (exp == 0) {
        // Denorm
        return 1 - FLOAT32_EXP_BIAS;
    } else {
        return exp - FLOAT32_EXP_BIAS;
    }
}

__host__ __device__ __forceinline__
int get_biased_exponent(
    const __half input
) {
    u_float_int u;
    u.f = __half2float(input);
    return get_biased_exponent(u);
}

__host__ __device__ __forceinline__
int get_trailing_mantissa(
    const u_float_int input
) {
    return input.i & FLOAT32_MANTISSA_MASK;
}

// Construct float from sign, biased exponent, and mantissa
__host__ __device__ __forceinline__
float construct_float(
    int sign,
    int biased_exp,
    int trailing_mantissa
) {
    u_float_int x;
    x.i = trailing_mantissa | (biased_exp << FLOAT32_EXP_OFFSET) | (sign << FLOAT32_SIGN_OFFSET);
    return x.f;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif
