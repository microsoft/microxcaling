/*
 * Microsoft Confidential
 */

#ifndef PYT_MX_SHARED_EXP_CUH
#define PYT_MX_SHARED_EXP_CUH

#include "common.cuh"

//-----------------------------------------------------------------------
// Bound the shared_exp based on ebits
//-----------------------------------------------------------------------
__host__ __device__ __forceinline__
int clamp_shared_exp(
    int shared_exp,
    const int ebits
) {
    // Set overflowing shared exps to NaN and
    // bound underflowing shared exps to -emax
    // Note that (for 8 bits) the min scale is -127, not -126
    int emax = ebits != 0 ? (1 << (ebits-1)) - 1 : FLOAT32_EXP_MAX;
    int shared_ub = shared_exp - FLOAT32_EXP_BIAS;
    shared_exp = shared_ub > emax ? FLOAT32_EXP_MAX : shared_exp;
    shared_exp = shared_ub < -emax ? FLOAT32_EXP_BIAS-emax : shared_exp;
    return shared_exp;
}

//-----------------------------------------------------------------------
// Compute shared scale for MX
//-----------------------------------------------------------------------
__host__ __device__ __forceinline__
float mx_get_shared_scale(
    int shared_exp,
    const int scale_bits,
    const float elem_max_norm
) {
    // Offset shared exponent by elem_emax, preserve NaNs
    const int elem_emax = get_unbiased_exponent(elem_max_norm);
    shared_exp = (shared_exp != FLOAT32_EXP_MAX) ? \
                 shared_exp - elem_emax : shared_exp;

    // Clamp to scale_bits range
    shared_exp = clamp_shared_exp(shared_exp, scale_bits);

    // Scale mantissa is 1 on the MSB mantissa bit if
    // scale is subnormal or NaN, otherwise 0
    const int scale_mant = \
            (shared_exp == 0 || shared_exp == FLOAT32_EXP_MAX) ? \
            (FLOAT32_IMPLIED1 >> 1) : 0;

    // Construct scale
    return construct_float(0, shared_exp, scale_mant);
}

#endif
