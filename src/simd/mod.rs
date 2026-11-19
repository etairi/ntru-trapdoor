//! The crate's only `unsafe` code, behind the `simd` feature on x86-64 and aarch64: kernels that the
//! portable code calls through the safe functions here, chosen at run time on processors that
//! have the instructions. Each returns exactly what the portable code that it replaces returns
//! (the tests compare them, and the known-answer and fn-dsa parity tests pass with the feature),
//! so no output depends on the processor or on the feature. Setting the environment variable
//! `NTRU_TRAPDOOR_SCALAR`, to any value, keeps the portable code everywhere.
//!
//! - Keccak-f\[1600\] on one state: on x86-64 AVX-512 (AVX-512F and AVX-512VL), else the
//!   portable code compiled for AVX2, BMI1, BMI2, LZCNT and POPCNT (`andn`, `rorx`). On Apple's
//!   aarch64 processors the SHA3 extension.
//! - Keccak-f\[1600\] on several independent states at once, for batches of SHAKE256 streams:
//!   four with AVX-512VL or AVX2 and eight with AVX-512 (x86-64), two with the SHA3 extension
//!   (aarch64).
//! - The two-prime NTT: the layers with half-blocks of at least 8 values (x86-64: AVX-512F with
//!   16 lanes from 16 values when the processor has it, AVX2 with 8 lanes otherwise) or 4 values
//!   (aarch64: NEON, 4 lanes), the narrow layers below them with the lanes regrouped in
//!   registers (AVX2 or NEON), and the pointwise products (AVX-512F, AVX2 or NEON).
//! - The code around the transforms: the CRT and the reduction modulo $`q`$ (odd $`q`$), the
//!   reduction of signed values modulo $`q`$, the centred lift modulo $`p_1`$, the norm of the
//!   inversion tower in the NTT domain, the check of the NTRU equation, and Verify's norm and
//!   final comparison.
#![allow(unsafe_code)]

#[macro_use]
mod keccak_lanes;
#[cfg(target_arch = "x86_64")]
pub(crate) mod fft_avx2;
#[cfg(target_arch = "aarch64")]
mod fft_neon;
#[cfg(target_arch = "aarch64")]
mod keccak_neon;
#[cfg(target_arch = "x86_64")]
mod keccak_x86;
#[cfg(target_arch = "x86_64")]
mod ntt_avx2;
#[cfg(target_arch = "x86_64")]
mod ntt_avx512;
#[cfg(target_arch = "aarch64")]
mod ntt_neon;
#[cfg(test)]
mod tests;

use std::sync::OnceLock;

use crate::ring::ntt::{NttPrime, P1, P2};

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

/// Whether the Keccak compiled for AVX2, BMI1, BMI2, LZCNT and POPCNT runs. Decided once per
/// process.
#[cfg(target_arch = "x86_64")]
pub(crate) fn bmi() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !scalar_forced()
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("lzcnt")
            && std::is_x86_feature_detected!("popcnt")
    })
}

/// Whether the AVX-512 kernels run (AVX-512F and AVX-512VL: the Keccak kernels need both, the
/// NTT kernels AVX-512F alone). Decided once per process.
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx512() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !scalar_forced()
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vl")
    })
}

/// Whether SampPre runs through its copy compiled for x86-64-v3 (AVX2, BMI1, BMI2, LZCNT and
/// POPCNT). Decided once per process.
#[cfg(target_arch = "x86_64")]
pub(crate) fn v3() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !scalar_forced()
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
            && std::is_x86_feature_detected!("lzcnt")
            && std::is_x86_feature_detected!("popcnt")
    })
}

/// One SampPre attempt through its x86-64-v3 copy (`sign::samp_pre_attempt_v3`), when the
/// processor has the instructions: `None` otherwise, and then the caller runs the portable
/// code. Both compute the same IEEE operations in the same order, so the result is the same.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn samp_pre_attempt_v3<P: crate::prng::Prng>(
    ek: &crate::ExpandedKey,
    t: &crate::RqPoly,
    prng: &mut P,
) -> Option<Option<crate::Preimage>> {
    if v3() {
        // SAFETY: the processor has every feature that the copy is compiled for (`v3`).
        return Some(unsafe { crate::sign::samp_pre_attempt_v3(ek, t, prng) });
    }
    None
}

/// Whether the NEON kernels run (NEON is part of the aarch64 baseline). Decided once per
/// process.
#[cfg(target_arch = "aarch64")]
pub(crate) fn neon() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !scalar_forced())
}

/// Whether the kernels of the SHA3 extension run. Decided once per process.
#[cfg(target_arch = "aarch64")]
pub(crate) fn sha3() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !scalar_forced() && std::arch::is_aarch64_feature_detected!("sha3"))
}

/// Keccak-f\[1600\] on one state, when the processor has the instructions: AVX-512, else AVX2,
/// BMI1, BMI2, LZCNT and POPCNT (x86-64), or the SHA3 extension on Apple's processors
/// (aarch64). `true` if a kernel ran (then `state` is permuted), `false` otherwise (`state`
/// untouched).
///
/// The SHA3 kernel computes one state in registers that hold two, so it needs processors that
/// run the SHA3 instructions as fast as the other vector instructions. An Apple M4 does, and the
/// kernel is faster than the portable code there. A Neoverse V2 runs one SHA3 instruction per
/// cycle, and the kernel took 517 ns per permutation there, against 293 ns for the portable code.
/// Other aarch64 processors keep the portable code.
#[inline]
pub(crate) fn keccak_f1600(state: &mut [u64; 25]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512() {
            // SAFETY: the processor has AVX-512F and AVX-512VL (`avx512`).
            unsafe { keccak_x86::keccak_f1600_avx512(state) };
            return true;
        }
        if bmi() {
            // SAFETY: the processor has AVX2, BMI1, BMI2, LZCNT and POPCNT (`bmi`).
            unsafe { crate::prng::keccak_f1600_bmi(state) };
            return true;
        }
    }
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    if sha3() {
        // SAFETY: the processor has the SHA3 extension (`sha3`).
        unsafe { keccak_neon::keccak_f1600(state) };
        return true;
    }
    #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
    let _ = state;
    false
}

/// Keccak-f\[1600\] on two independent states at once, with the SHA3 extension (aarch64):
/// `true` if it ran (then each state is permuted, as by the portable code), `false` otherwise
/// (the states untouched). Together with [`keccak_f1600_x4`] and [`keccak_f1600_x8`], a caller
/// can try the widest kernel first.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "for batches of SHAKE256 streams")
)]
#[inline]
pub(crate) fn keccak_f1600_x2(states: &mut [[u64; 25]; 2]) -> bool {
    #[cfg(target_arch = "aarch64")]
    if sha3() {
        // SAFETY: the processor has the SHA3 extension (`sha3`).
        unsafe { keccak_neon::keccak_f1600_x2(states) };
        return true;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = states;
    false
}

/// Keccak-f\[1600\] on four independent states at once, with AVX-512VL, else AVX2 (x86-64):
/// `true` if a kernel ran (then each state is permuted), `false` otherwise (the states
/// untouched).
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "for batches of SHAKE256 streams")
)]
#[inline]
pub(crate) fn keccak_f1600_x4(states: &mut [[u64; 25]; 4]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512() {
            // SAFETY: the processor has AVX-512F and AVX-512VL (`avx512`).
            unsafe { keccak_x86::keccak_f1600_x4_avx512(states) };
            return true;
        }
        if avx2() {
            // SAFETY: the processor has AVX2 (`avx2`).
            unsafe { keccak_x86::keccak_f1600_x4_avx2(states) };
            return true;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = states;
    false
}

/// Keccak-f\[1600\] on eight independent states at once, with AVX-512 (x86-64): `true` if it
/// ran (then each state is permuted), `false` otherwise (the states untouched).
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "for batches of SHAKE256 streams")
)]
#[inline]
pub(crate) fn keccak_f1600_x8(states: &mut [[u64; 25]; 8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if avx512() {
        // SAFETY: the processor has AVX-512F and AVX-512VL (`avx512`).
        unsafe { keccak_x86::keccak_f1600_x8_avx512(states) };
        return true;
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = states;
    false
}

/// `fft::poly_split_fft` through the NEON kernel (`gm = GM[2 hn..3 hn]`, `logn >= 3`): `true` if
/// it ran.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn fft_split(logn: u32, f0: &mut [f64], f1: &mut [f64], f: &[f64], gm: &[f64]) -> bool {
    if logn >= 3 && neon() {
        // SAFETY: NEON is part of the aarch64 baseline, and the kernel checks its lengths.
        unsafe { fft_neon::poly_split_fft(logn, f0, f1, f, gm) };
        return true;
    }
    false
}

/// `fft::poly_merge_fft` through the NEON kernel (`gm = GM[2 hn..3 hn]`, `logn >= 3`): `true` if
/// it ran.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn fft_merge(logn: u32, f: &mut [f64], f0: &[f64], f1: &[f64], gm: &[f64]) -> bool {
    if logn >= 3 && neon() {
        // SAFETY: as in `fft_split`.
        unsafe { fft_neon::poly_merge_fft(logn, f, f0, f1, gm) };
        return true;
    }
    false
}

/// `fft::merge_sub_mul_add_split` through the NEON kernel (`gm = GM[2 hn..3 hn]`, `logn >= 3`):
/// `true` if it ran.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn fft_fused(
    logn: u32,
    w: &mut [f64],
    t0: &[f64],
    t1: &mut [f64],
    l10: &[f64],
    gm: &[f64],
) -> bool {
    if logn >= 3 && neon() {
        // SAFETY: as in `fft_split`.
        unsafe { fft_neon::merge_sub_mul_add_split(logn, w, t0, t1, l10, gm) };
        return true;
    }
    false
}

/// One layer of `fft::fft` (blocks of `t >= 4` values) through the NEON kernel: `true` if it
/// ran.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn fft_layer(re: &mut [f64], im: &mut [f64], t: usize, m: usize, gm: &[f64]) -> bool {
    if t >= 4 && neon() {
        // SAFETY: as in `fft_split`.
        unsafe { fft_neon::fft_layer(re, im, t, m, gm) };
        return true;
    }
    false
}

/// One layer of `fft::ifft` (half-blocks of `t >= 2` values) through the NEON kernel: `true` if
/// it ran.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn ifft_layer(re: &mut [f64], im: &mut [f64], t: usize, hm: usize, gm: &[f64]) -> bool {
    if t >= 2 && neon() {
        // SAFETY: as in `fft_split`.
        unsafe { fft_neon::ifft_layer(re, im, t, hm, gm) };
        return true;
    }
    false
}

/// The layers of the forward NTT whose half-blocks have at least 8 values (AVX-512 from 16
/// values, then AVX2) or 4 values (NEON), when a kernel runs: returns the block size and the
/// number of blocks of the next layer, which is where the narrow kernels or the portable code
/// continue (`(n, 1)` when nothing ran).
#[inline]
pub(crate) fn ntt_wide_layers(pr: &NttPrime, logn: u32, a: &mut [u32]) -> (usize, usize) {
    let n = 1usize << logn;
    #[cfg(target_arch = "x86_64")]
    if logn >= 5 && a.len() >= n && avx512() {
        // SAFETY: the processor has AVX-512F (`avx512`), which implies AVX2, and the kernels
        // read and write `a[..n]` and the tables within their lengths.
        return unsafe {
            let (t, m) = ntt_avx512::ntt_wide_layers(pr, logn, &mut a[..n]);
            ntt_avx2::ntt_wide_layers_from(pr, logn, &mut a[..n], t, m)
        };
    }
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
/// a kernel takes that size (`t >= 8` with AVX-512 or AVX2, `t >= 4` with NEON): `true` if
/// they ran.
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
    if t >= 8 && t < n && n >= 32 && a.len() >= n && avx512() {
        // SAFETY: as in `ntt_wide_layers`, and AVX-512F implies AVX2, which runs the layers
        // below half-blocks of 16.
        unsafe {
            ntt_avx2::intt_wide_layers_until(pr, logn, &mut a[..n], t, hm, 16);
            let (t16, hm16) = if t >= 16 { (t, hm) } else { (16, hm * t / 16) };
            ntt_avx512::intt_wide_layers(pr, logn, &mut a[..n], t16, hm16);
        }
        return true;
    }
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
    if a.len() == b.len() && a.len().is_multiple_of(16) && avx512() {
        // SAFETY: AVX-512F is present, and the slices have the same length, a multiple of 16.
        unsafe { ntt_avx512::mmul_slices(pr, a, b) };
        return true;
    }
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
    if b.is_none_or(|b| b.len() == a.len()) && a.len().is_multiple_of(16) && avx512() {
        // SAFETY: as in `mmul_slices`.
        unsafe { ntt_avx512::mmul_slices_const(pr, a, b, c) };
        return true;
    }
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
    if acc.len() == x.len() && acc.len() == y.len() && acc.len().is_multiple_of(16) && avx512() {
        // SAFETY: as in `mmul_slices`.
        unsafe { ntt_avx512::mmul_add_slices(pr, acc, x, y) };
        return true;
    }
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

/// The constants of the kernels modulo $`q`$, for $`3\le q<2^{25}`$.
pub(crate) struct QConsts {
    q: u32,
    /// $`-q^{-1}\bmod2^{32}`$ for an odd $`q`$ (0 for an even one, which the CRT kernels leave to
    /// the portable code).
    q0i: u32,
    /// $`2^{32}\bmod q`$.
    r: u32,
    /// $`p_12^{32}\bmod q`$.
    p1r: u32,
    /// $`P\bmod q`$, $`P=p_1p_2`$.
    pq: u32,
    /// $`\lfloor2^{32}/q\rfloor`$.
    m: u32,
}

impl QConsts {
    fn new(q: u32) -> QConsts {
        let qw = q as u64;
        // q^-1 mod 2^32 for an odd q: q is its own inverse modulo 8, and each Newton step
        // doubles the number of correct low bits (3, 6, 12, 24, 48).
        let mut inv = q;
        for _ in 0..4 {
            inv = inv.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(inv)));
        }
        QConsts {
            q,
            q0i: if q & 1 == 1 { inv.wrapping_neg() } else { 0 },
            r: ((1u64 << 32) % qw) as u32,
            p1r: (((P1.p as u64 % qw) << 32) % qw) as u32,
            pq: ((P1.p as u64 * P2.p as u64) % qw) as u32,
            m: ((1u64 << 32) / qw) as u32,
        }
    }
}

/// The last layers of the forward NTT, from blocks of `t` values (where [`ntt_wide_layers`]
/// stopped: 8 with AVX2, 4 with NEON) to the end, for $`n\ge16`$: `true` if they ran.
#[inline]
pub(crate) fn ntt_narrow_layers(pr: &NttPrime, logn: u32, a: &mut [u32], t: usize) -> bool {
    let n = 1usize << logn;
    #[cfg(target_arch = "x86_64")]
    if t == 8 && n >= 16 && a.len() >= n && avx2() {
        // SAFETY: the processor has AVX2 (`avx2`), and the kernel reads and writes `a[..n]`
        // and the tables within their lengths.
        unsafe { ntt_avx2::ntt_narrow_layers(pr, logn, &mut a[..n]) };
        return true;
    }
    #[cfg(target_arch = "aarch64")]
    if t == 4 && n >= 16 && a.len() >= n && neon() {
        // SAFETY: NEON is part of the aarch64 baseline, and the kernel stays within `a[..n]`
        // and the tables.
        unsafe { ntt_neon::ntt_narrow_layers(pr, logn, &mut a[..n]) };
        return true;
    }
    false
}

/// The first layers of the inverse NTT (half-blocks of 1, 2 and 4 values with AVX2, of 1 and
/// 2 with NEON), for $`n\ge16`$: the half-block size and the block count of the next layer,
/// where [`intt_wide_layers`] continues, or `None` if no kernel ran.
#[inline]
pub(crate) fn intt_narrow_layers(
    pr: &NttPrime,
    logn: u32,
    a: &mut [u32],
) -> Option<(usize, usize)> {
    let n = 1usize << logn;
    #[cfg(target_arch = "x86_64")]
    if n >= 16 && a.len() >= n && avx2() {
        // SAFETY: as in `ntt_narrow_layers`.
        unsafe { ntt_avx2::intt_narrow_layers(pr, logn, &mut a[..n]) };
        return Some((8, n >> 4));
    }
    #[cfg(target_arch = "aarch64")]
    if n >= 16 && a.len() >= n && neon() {
        // SAFETY: as in `ntt_narrow_layers`.
        unsafe { ntt_neon::intt_narrow_layers(pr, logn, &mut a[..n]) };
        return Some((4, n >> 3));
    }
    None
}

/// Whether the 8-lane (AVX2) or 4-lane (NEON) kernels run.
#[inline]
fn kernels() -> bool {
    #[cfg(target_arch = "x86_64")]
    return avx2();
    #[cfg(target_arch = "aarch64")]
    return neon();
}

/// The lane count of the kernels of [`kernels`].
#[cfg(target_arch = "x86_64")]
const LANES: usize = 8;
#[cfg(target_arch = "aarch64")]
const LANES: usize = 4;

/// Whether the kernels of the code around the transforms take slices of length `len`: they
/// run, and `len` is a multiple of their lane count.
#[inline]
fn lanes_fit(len: usize) -> bool {
    len.is_multiple_of(LANES) && kernels()
}

/// `out[i] <- ModQ::crt(u1[i], u2[i])`, or `mp_add(out[i], ModQ::crt(u1[i], u2[i]), q)` with
/// `acc`, for an odd $`q`$ and equal lengths that the kernels take: `true` if done.
#[inline]
pub(crate) fn crt_slices(q: u32, u1: &[u32], u2: &[u32], out: &mut [u32], acc: bool) -> bool {
    if q & 1 == 0 || u1.len() != out.len() || u2.len() != out.len() || !lanes_fit(out.len()) {
        return false;
    }
    let c = QConsts::new(q);
    // SAFETY: `lanes_fit` checked that the kernels run on this processor (AVX2, or NEON which
    // is part of the aarch64 baseline), and the slices have equal lengths, multiples of the
    // lane count.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if acc {
            ntt_avx2::crt_slices::<true>(&c, u1, u2, out)
        } else {
            ntt_avx2::crt_slices::<false>(&c, u1, u2, out)
        }
        #[cfg(target_arch = "aarch64")]
        if acc {
            ntt_neon::crt_slices::<true>(&c, u1, u2, out)
        } else {
            ntt_neon::crt_slices::<false>(&c, u1, u2, out)
        }
    }
    true
}

/// `out[i] <- ModQ::reduce_i32(x[i])` for equal lengths that the kernels take: `true` if done.
#[inline]
pub(crate) fn reduce_i32_slices(q: u32, x: &[i32], out: &mut [u32]) -> bool {
    if x.len() != out.len() || !lanes_fit(out.len()) {
        return false;
    }
    let c = QConsts::new(q);
    // SAFETY: as in `crt_slices`.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        ntt_avx2::reduce_i32_slices(&c, x, out);
        #[cfg(target_arch = "aarch64")]
        ntt_neon::reduce_i32_slices(&c, x, out);
    }
    true
}

/// The OR over `i` of `mp_add(ModQ::reduce_i32(x[i]), y[i], q) ^ z[i]` (canonical `y`), for
/// equal lengths that the kernels take, or `None`.
#[inline]
pub(crate) fn add_i32_diff(q: u32, x: &[i32], y: &[u32], z: &[u32]) -> Option<u32> {
    if y.len() != x.len() || z.len() != x.len() || !lanes_fit(x.len()) {
        return None;
    }
    let c = QConsts::new(q);
    // SAFETY: as in `crt_slices`.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        return Some(ntt_avx2::add_i32_diff(&c, x, y, z));
        #[cfg(target_arch = "aarch64")]
        return Some(ntt_neon::add_i32_diff(&c, x, y, z));
    }
}

/// `out[i] <- ModQ::reduce_i32(c)` for the centred lift `c` of the residue `u[i]` modulo $`p_1`$,
/// for equal lengths that the kernels take: `true` if done.
#[inline]
pub(crate) fn centred_p1_mod_q(q: u32, u: &[u32], out: &mut [u32]) -> bool {
    if u.len() != out.len() || !lanes_fit(out.len()) {
        return false;
    }
    let c = QConsts::new(q);
    // SAFETY: as in `crt_slices`.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        ntt_avx2::centred_p1_mod_q(&c, u, out);
        #[cfg(target_arch = "aarch64")]
        ntt_neon::centred_p1_mod_q(&c, u, out);
    }
    true
}

/// $`\sum_ix_i^2`$ over `a` and `b`, exactly, when a kernel runs (any lengths), or `None`.
#[inline]
pub(crate) fn norm_sq(a: &[i32], b: &[i32]) -> Option<u128> {
    if !kernels() {
        return None;
    }
    // SAFETY: `kernels` checked that the kernels run on this processor, and they take any
    // lengths.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        return Some(ntt_avx2::norm_sq(a, b));
        #[cfg(target_arch = "aarch64")]
        return Some(ntt_neon::norm_sq(a, b));
    }
}

/// The norm of the inversion tower in the NTT domain (`ring::inv::norm_ntt`) of size
/// $`h=2^{\mathrm{logh}}`$, when the kernels take $`h`$ and the lengths suffice: `true` if done.
#[inline]
pub(crate) fn norm_ntt(pr: &NttPrime, logh: u32, e: &[u32], o: &[u32], out: &mut [u32]) -> bool {
    let h = 1usize << logh;
    if h < 8 || e.len() < h || o.len() < h || out.len() < h || !lanes_fit(h) {
        return false;
    }
    // SAFETY: as in `crt_slices`, and the kernels stay within `[..h]` of each slice and the
    // tables.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        ntt_avx2::norm_ntt(pr, logh, e, o, out);
        #[cfg(target_arch = "aarch64")]
        ntt_neon::norm_ntt(pr, logh, e, o, out);
    }
    true
}

/// The OR over `i` of `mp_sub(mul(f[i], G[i]), mul(g[i], F[i]), p) ^ q` (`NttPrime::mul`), for
/// equal lengths that the kernels take, or `None`.
#[inline]
pub(crate) fn ntru_diff(
    pr: &NttPrime,
    q: u32,
    f: &[u32],
    g: &[u32],
    big_f: &[u32],
    big_g: &[u32],
) -> Option<u32> {
    let n = f.len();
    if g.len() != n || big_f.len() != n || big_g.len() != n || !lanes_fit(n) {
        return None;
    }
    // SAFETY: as in `crt_slices`.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        return Some(ntt_avx2::ntru_diff(pr, q, f, g, big_f, big_g));
        #[cfg(target_arch = "aarch64")]
        return Some(ntt_neon::ntru_diff(pr, q, f, g, big_f, big_g));
    }
}
