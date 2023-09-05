/*
 * Microsoft Confidential
 */

#ifndef PYT_MX_QUANTIZE_CUH
#define PYT_MX_QUANTIZE_CUH

//---------------------------------------------------------
// Shift right and round a float32 mantissa
// Example of "allow_overflow". Say we are rounding 11111 to 4 bits
// If allow_overflow is False, it will floor the result to 1111
// If allow_overflow is True,  it will round up to 10000, overflowing 4 bits
//---------------------------------------------------------
__host__ __device__ __forceinline__
void shift_right_round_mantissa(
    int &mantissa,      // 23-bit float32 trailing mantissa
    const bool is_subnorm,  // is the input a subnorm?
    const int mbits,    // number to bits to round to
    const int exp_diff, // extra right shifts
    const RoundingMode rounding_mode,
    const bool allow_overflow = false
) {
    // Implied 1
    mantissa = is_subnorm ? mantissa : mantissa + FLOAT32_IMPLIED1;
    int fp32_sig_bits = is_subnorm ? 23 : 24;

    // RNE logic
    bool tie = false;
    bool even = false;
    if (rounding_mode == rd_even) {
        // tbits is the no. of bits that will be removed
        int tbits = exp_diff + (fp32_sig_bits - mbits);
        // 1 at all truncation locations except the first truncation location
        int mask = (1 << (tbits - 1)) - 1;
        // We have a tie only if all the truncation bits except the first
        // one are zero. If the first truncation bit is 1, we have an
        // actual tie. For rounding, we don't care if the first bits
        // is 1 or 0. If it is 0, no rounding happens.
        tie = !(mantissa & mask);
        mask = (1 << tbits); // 1 at the first non-truncated location
        even = !(mantissa & mask); // True if the last bit before truncation location is 0
    }

    // Adjust for shared exponent
    mantissa = mantissa >> exp_diff;
    // Shift down to target bit width + 1
    mantissa = mantissa >> (fp32_sig_bits - mbits - 1);
    // Rounding using floor(x+1), with overflow check
    if ((rounding_mode == rd_away || rounding_mode == rd_even) &&
        (allow_overflow || mantissa != ((1 << (mbits + 1)) - 1))) {
        if (!(tie && even))
            mantissa = mantissa + 1;
    }
    // Shift last bit away
    mantissa = mantissa >> 1;
}

//---------------------------------------------------------
// Shift back up to restore a float32 mantissa
// Use in pair with shift_right_round_mantissa to check for
// overflows caused by rounding
//---------------------------------------------------------
__host__ __device__ __forceinline__
bool shift_left_mantissa(
    int &mantissa,    // From shift_right_round_mantissa
    const bool is_subnorm,  // is the input a subnorm?
    const int mbits,
    const int exp_diff
) {
    int fp32_sig_bits = is_subnorm ? 23 : 24;
    mantissa = mantissa << (fp32_sig_bits - mbits + exp_diff);
    // Handle overflow.
    // When a subnorm overflows (into a normal) we don't rshift
    const bool overflow = (mantissa >= (1 << fp32_sig_bits));
    mantissa = (overflow && !is_subnorm) ? mantissa >> 1 : mantissa;
    // Remove implied 1
    mantissa = mantissa & (FLOAT32_IMPLIED1 - 1);
    return overflow;
}

//---------------------------------------------------------
// Quantize a float32 to arbitrary exponent and mantissa
// (trailing mantissa + implied 1 + sign) bits.
// The 'bits' arg should include sign and implicit 1. For
// bfloat16 bits=9.
//---------------------------------------------------------
__host__ __device__ __forceinline__
float quantize_elemwise(
    float input,
    int bits,     // bits = mantissa bits + sign bit
    int exp_bits, // exp_bits == 0 indicates integer dtype
    float max_norm,
    const RoundingMode rounding_mode = rd_away,
    bool saturate_normals = false,
    bool allow_denorm = true
) {
    u_float_int input_;
    input_.f = input;
    int biased_exp = get_biased_exponent(input_);
    int sign = get_sign(input_);
    int tmant = get_trailing_mantissa(input_);

    // Mantissa bits to quantize to (remove sign)
    const int mbits = bits - 1;
    const bool is_int = exp_bits == 0;

    // Integers can be treated has having exp bias of 1
    const int new_bias = is_int ? 1 : (1 << (exp_bits-1)) - 1;
    const int new_biased_exp = biased_exp - FLOAT32_EXP_BIAS + new_bias;

    // Skip denorms
    if ((!is_int) && (!allow_denorm) && (new_biased_exp < 1)) {
        return 0.0;
    }

    // Use exp_diff to truncate additional bits for subnorms
    // mbits includes implicit 1, so when new_biased_exp==0
    // we want exp_diff = 1 to truncate away 1 bit
    int exp_diff = (new_biased_exp <= 0) ? 1-new_biased_exp : 0;
    exp_diff = (exp_diff > FLOAT32_FULL_MBITS) ? FLOAT32_FULL_MBITS : exp_diff;

    // Shift down and round mantissa, allow overflow except for integers
    // This converts tmant into a full mantissa
    shift_right_round_mantissa(
          tmant, biased_exp==0, mbits, exp_diff, rounding_mode, !is_int);

    if (tmant == 0) {
        return 0.0;
    }

    // Shift back up to restore mantissa
    // This converts back to a trailing mantissa
    const bool overflow = shift_left_mantissa(
          tmant, biased_exp==0, mbits, exp_diff);
    biased_exp = overflow ? biased_exp+1 : biased_exp;

    // Reconstruct float number
    float output = construct_float(sign, biased_exp, tmant);

    // Return Inf if rounded value is out of bounds,
    // unless target format is integer or saturate_normals==True
    if (abs(output) > max_norm) {
        if (is_int || saturate_normals)
            output = sign ? -max_norm : max_norm;
        else
            output = construct_float(sign, 0xFF, 0);
    }
    return output;
}

//-----------------------------------------------------------------------
// Half-precision
//-----------------------------------------------------------------------
__host__ __device__ __forceinline__
__half quantize_elemwise(
    __half input,
    int bits,
    int exp_bits,
    float max_norm,
    const RoundingMode rounding_mode = rd_away,
    bool saturate_normals = false,
    bool allow_denorm = true
) {
    float in_fp32 = __half2float(input);
    float out_fp32 = quantize_elemwise(in_fp32, bits, exp_bits, max_norm,
                                       rounding_mode, saturate_normals,
                                       allow_denorm);
    return __float2half(out_fp32);
}

#endif
