use ark_std::test_rng;
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::field::JoltField;

pub mod errors;
pub mod gaussian_elimination;
pub mod instruction_utils;
pub mod math;
pub mod msm;
pub mod profiling;
pub mod thread;
pub mod transcript;

/// utility: precompute f(x) = eq(g,x)
pub fn precompute_eq<F: JoltField>(g: &[F]) -> Vec<F> {
    let dim = g.len();
    let mut dp = vec![F::zero(); 1 << dim];
    dp[0] = F::one() - g[0];
    dp[1] = g[0];
    for i in 1..dim {
        for b in 0..(1 << i) {
            let prev = dp[b];
            dp[b + (1 << i)] = prev * g[i];
            dp[b] = prev - dp[b + (1 << i)];
        }
    }
    dp
}

/// swap the bits of `x` from position `a..a+n` to `b..b+n` and from `b..b+n` to `a..a+n` in little endian order
pub fn swap_bits(x: usize, a: usize, b: usize, n: usize) -> usize {
    let a_bits = (x >> a) & ((1usize << n) - 1);
    let b_bits = (x >> b) & ((1usize << n) - 1);
    let local_xor_mask = a_bits ^ b_bits;
    let global_xor_mask = (local_xor_mask << a) | (local_xor_mask << b);
    x ^ global_xor_mask
}

/// Converts an integer value to a bitvector (all values {0,1}) of field elements.
/// Note: ordering has the MSB in the highest index. All of the following represent the integer 1:
/// - [1]
/// - [0, 0, 1]
/// - [0, 0, 0, 0, 0, 0, 0, 1]
/// ```ignore
/// use jolt_core::utils::index_to_field_bitvector;
/// # use ark_bn254::Fr;
/// # use ark_std::{One, Zero};
/// let zero = Fr::zero();
/// let one = Fr::one();
///
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 1), vec![one]);
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 3), vec![zero, zero, one]);
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 7), vec![zero, zero, zero, zero, zero, zero, one]);
/// ```
pub fn index_to_field_bitvector<F: JoltField>(value: usize, bits: usize) -> Vec<F> {
    assert!(value < 1 << bits);

    let mut bitvector: Vec<F> = Vec::with_capacity(bits);

    for i in (0..bits).rev() {
        if (value >> i) & 1 == 1 {
            bitvector.push(F::one());
        } else {
            bitvector.push(F::zero());
        }
    }
    bitvector
}

#[tracing::instrument(skip_all)]
pub fn compute_dotproduct<F: JoltField>(a: &[F], b: &[F]) -> F {
    ark_std::cfg_iter!(a)
        .zip_eq(ark_std::cfg_iter!(b))
        .map(|(a_i, b_i)| *a_i * *b_i)
        .sum()
}

/// Compute dotproduct optimized for values being 0 / 1
#[tracing::instrument(skip_all)]
pub fn compute_dotproduct_low_optimized<F: JoltField>(a: &[F], b: &[F]) -> F {
    ark_std::cfg_iter!(a)
        .zip_eq(ark_std::cfg_iter!(b))
        .map(|(a_i, b_i)| mul_0_1_optimized(a_i, b_i))
        .sum()
}

#[inline(always)]
pub fn mul_0_1_optimized<F: JoltField>(a: &F, b: &F) -> F {
    if a.is_zero() || b.is_zero() {
        F::zero()
    } else if a.is_one() {
        *b
    } else if b.is_one() {
        *a
    } else {
        *a * *b
    }
}

#[inline(always)]
pub fn mul_0_optimized<F: JoltField>(likely_zero: &F, x: &F) -> F {
    if likely_zero.is_zero() {
        F::zero()
    } else {
        *likely_zero * *x
    }
}

/// Checks if `num` is a power of 2.
pub fn is_power_of_two(num: usize) -> bool {
    num != 0 && (num & (num - 1)) == 0
}

/// Splits `item` into two chunks of `num_bits` size where each is less than 2^num_bits.
/// Ex: split_bits(0b101_000, 3) -> (101, 000)
pub fn split_bits(item: usize, num_bits: usize) -> (usize, usize) {
    let max_value = (1 << num_bits) - 1; // Calculate the maximum value that can be represented with num_bits

    let low_chunk = item & max_value; // Extract the lower bits
    let high_chunk = (item >> num_bits) & max_value; // Shift the item to the right and extract the next set of bits

    (high_chunk, low_chunk)
}

pub fn gen_random_point<F: JoltField>(memory_bits: usize) -> Vec<F> {
    let mut rng = test_rng();
    let mut r_i: Vec<F> = Vec::with_capacity(memory_bits);
    for _ in 0..memory_bits {
        r_i.push(F::random(&mut rng));
    }
    r_i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split() {
        assert_eq!(split_bits(0b00_01, 2), (0, 1));
        assert_eq!(split_bits(0b10_01, 2), (2, 1));
    }
}
