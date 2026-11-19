//! The crate's only `unsafe` code, behind the `simd` feature on x86-64 and aarch64: kernels that the
//! portable code calls through the safe functions here, chosen at run time on processors that
//! have the instructions. Each returns exactly what the portable code that it replaces returns
//! (the tests compare them, and the known-answer and fn-dsa parity tests pass with the feature),
//! so no output depends on the processor or on the feature. Setting the environment variable
//! `NTRU_TRAPDOOR_SCALAR`, to any value, keeps the portable code everywhere.
//!
//! - Keccak-f\[1600\] (x86-64): the portable code compiled for BMI1 and BMI2 (`andn`, `rorx`).
//! - The two-prime NTT: the layers with half-blocks of at least 8 values (x86-64, AVX2, 8 lanes)
//!   or 4 values (aarch64, NEON, 4 lanes), and the pointwise products.
#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
mod ntt_avx2;
#[cfg(target_arch = "aarch64")]
mod ntt_neon;
#[cfg(test)]
mod tests;

use std::sync::OnceLock;

use crate::ring::ntt::NttPrime;

/// Whether `NTRU_TRAPDOOR_SCALAR` asks for the portable code.
fn scalar_forced() -> bool {
    std::env::var_os("NTRU_TRAPDOOR_SCALAR").is_some()
}

/// Whether the AVX2 kernels run. Decided once per process.
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !scalar_forced() && std::is_x86_feature_detected!("avx2"))
}

/// Whether the BMI1 and BMI2 Keccak runs. Decided once per process.
#[cfg(target_arch = "x86_64")]
pub(crate) fn bmi() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !scalar_forced()
            && std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
    })
}

/// Whether the NEON kernels run (NEON is part of the aarch64 baseline). Decided once per
/// process.
#[cfg(target_arch = "aarch64")]
pub(crate) fn neon() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !scalar_forced())
}

/// The permutation compiled for BMI1 and BMI2, when the processor has them: `true` if it ran
/// (then `state` is permuted), `false` otherwise (`state` untouched).
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn keccak_f1600(state: &mut [u64; 25]) -> bool {
    if bmi() {
        // SAFETY: the processor has BMI1 and BMI2 (`bmi`).
        unsafe { crate::prng::keccak_f1600_bmi(state) };
        return true;
    }
    false
}

/// The layers of the forward NTT whose half-blocks have at least 8 values, when AVX2 runs:
/// returns the block size and the number of blocks of the next layer, which is where the
/// portable code continues (`(n, 1)` when nothing ran).
#[inline]
pub(crate) fn ntt_wide_layers(pr: &NttPrime, logn: u32, a: &mut [u32]) -> (usize, usize) {
    let n = 1usize << logn;
    #[cfg(target_arch = "x86_64")]
    if logn >= 4 && a.len() >= n && avx2() {
        // SAFETY: the processor has AVX2 (`avx2`), and the kernel reads and writes `a[..n]`
        // and the tables within their lengths.
        return unsafe { ntt_avx2::ntt_wide_layers(pr, logn, &mut a[..n]) };
    }
    #[cfg(target_arch = "aarch64")]
    if logn >= 3 && a.len() >= n && neon() {
        // SAFETY: NEON is part of the aarch64 baseline, and the kernel stays within `a[..n]`
        // and the tables.
        return unsafe { ntt_neon::ntt_wide_layers(pr, logn, &mut a[..n]) };
    }
    (n, 1)
}

/// The layers of the inverse NTT from half-block size `t` (with `hm` blocks) to the end, when
/// a kernel takes that size (`t >= 8` with AVX2, `t >= 4` with NEON): `true` if they ran.
#[inline]
pub(crate) fn intt_wide_layers(
    pr: &NttPrime,
    logn: u32,
    a: &mut [u32],
    t: usize,
    hm: usize,
) -> bool {
    let n = 1usize << logn;
    #[cfg(target_arch = "x86_64")]
    if t >= 8 && t < n && a.len() >= n && avx2() {
        // SAFETY: as in `ntt_wide_layers`.
        unsafe { ntt_avx2::intt_wide_layers(pr, logn, &mut a[..n], t, hm) };
        return true;
    }
    #[cfg(target_arch = "aarch64")]
    if t >= 4 && t < n && a.len() >= n && neon() {
        // SAFETY: as in `ntt_wide_layers`.
        unsafe { ntt_neon::intt_wide_layers(pr, logn, &mut a[..n], t, hm) };
        return true;
    }
    false
}

/// `a[i] <- mmul(a[i], b[i])` when AVX2 runs and the length is a multiple of 8: `true` if done.
#[inline]
pub(crate) fn mmul_slices(pr: &NttPrime, a: &mut [u32], b: &[u32]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if a.len() == b.len() && a.len().is_multiple_of(8) && avx2() {
        // SAFETY: AVX2 is present, and the slices have the same length, a multiple of 8.
        unsafe { ntt_avx2::mmul_slices(pr, a, b) };
        return true;
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() == b.len() && a.len().is_multiple_of(4) && neon() {
        // SAFETY: NEON is part of the aarch64 baseline; equal lengths, multiples of 4.
        unsafe { ntt_neon::mmul_slices(pr, a, b) };
        return true;
    }
    false
}

/// `a[i] <- mmul(mmul(a[i], b[i]), c)` (or `mmul(a[i], c)` without `b`) when AVX2 runs and
/// the length is a multiple of 8: `true` if done.
#[inline]
pub(crate) fn mmul_slices_const(pr: &NttPrime, a: &mut [u32], b: Option<&[u32]>, c: u32) -> bool {
    #[cfg(target_arch = "x86_64")]
    if b.is_none_or(|b| b.len() == a.len()) && a.len().is_multiple_of(8) && avx2() {
        // SAFETY: as in `mmul_slices`.
        unsafe { ntt_avx2::mmul_slices_const(pr, a, b, c) };
        return true;
    }
    #[cfg(target_arch = "aarch64")]
    if b.is_none_or(|b| b.len() == a.len()) && a.len().is_multiple_of(4) && neon() {
        // SAFETY: as in `mmul_slices`.
        unsafe { ntt_neon::mmul_slices_const(pr, a, b, c) };
        return true;
    }
    false
}

/// `acc[i] <- acc[i] + mmul(x[i], y[i]) mod p` when AVX2 runs and the lengths are equal
/// multiples of 8: `true` if done.
#[inline]
pub(crate) fn mmul_add_slices(pr: &NttPrime, acc: &mut [u32], x: &[u32], y: &[u32]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if acc.len() == x.len() && acc.len() == y.len() && acc.len().is_multiple_of(8) && avx2() {
        // SAFETY: as in `mmul_slices`.
        unsafe { ntt_avx2::mmul_add_slices(pr, acc, x, y) };
        return true;
    }
    #[cfg(target_arch = "aarch64")]
    if acc.len() == x.len() && acc.len() == y.len() && acc.len().is_multiple_of(4) && neon() {
        // SAFETY: as in `mmul_slices`.
        unsafe { ntt_neon::mmul_add_slices(pr, acc, x, y) };
        return true;
    }
    false
}
