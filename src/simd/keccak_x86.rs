//! Keccak-f\[1600\] kernels for x86-64. Each returns, for each of its states, what
//! `prng::keccak_f1600_portable` returns.
//!
//! - [`keccak_f1600_avx512`]: one state, with row $`y`$ (lanes $`5y`$ to $`5y+4`$) in the low
//!   five 64-bit elements of a 512-bit register, the layout of XKCP's AVX-512 code. θ's column
//!   parities are the XOR of the five registers, and its two shifts are permutations of that one
//!   register. ρ rotates each element by its own offset. π is split in two: a permutation inside
//!   each register turns row $`k`$ into column $`k`$ of π's output, so that χ combines whole
//!   registers, and a 5×5 transposition then brings the rows back.
//! - `keccak_f1600_x4_avx2`, `keccak_f1600_x4_avx512` (AVX-512VL on 256-bit registers) and
//!   `keccak_f1600_x8_avx512` (512-bit registers): one state per 64-bit element
//!   (`keccak_lanes`), the states transposed in and out with 128-bit loads, stores and
//!   interleaves.
use core::arch::x86_64::*;

use crate::prng::RC;

/// ι's constant of the round before each round, 0 before the first: the single-state kernel
/// XORs it into lane $`(0,0)`$ at the start of the next round, beside the parities, rather than
/// after χ, where it would lengthen the chain of dependent instructions.
const RC_PREV: [u64; 24] = {
    let mut t = [0u64; 24];
    let mut i = 1;
    while i < 24 {
        t[i] = RC[i - 1];
        i += 1;
    }
    t
};

/// $`a\oplus b\oplus c`$.
#[target_feature(enable = "avx512f")]
#[inline]
fn xor3(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_ternarylogic_epi64::<0x96>(a, b, c)
}

/// $`a\oplus(\lnot b\wedge c)`$.
#[target_feature(enable = "avx512f")]
#[inline]
fn chi(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_ternarylogic_epi64::<0xD2>(a, b, c)
}

/// The register with elements `e`, element 0 first.
#[target_feature(enable = "avx512f")]
#[inline]
fn v(e: [i64; 8]) -> __m512i {
    _mm512_setr_epi64(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7])
}

/// Keccak-f\[1600\] on one state in five 512-bit registers. Elements 5 to 7 of each register
/// are don't-cares: the masked loads clear them, and no permutation moves them into elements 0
/// to 4.
#[target_feature(enable = "avx512f,avx512vl")]
pub(super) fn keccak_f1600_avx512(state: &mut [u64; 25]) {
    // θ: element x of the permuted parities is C[x - 1], C[x + 1].
    let prev = v([4, 0, 1, 2, 3, 5, 6, 7]);
    let next = v([1, 2, 3, 4, 0, 5, 6, 7]);
    // ρ: the offsets of row y.
    let rho = [
        v([0, 1, 62, 28, 27, 0, 0, 0]),
        v([36, 44, 6, 55, 20, 0, 0, 0]),
        v([3, 10, 43, 25, 39, 0, 0, 0]),
        v([41, 45, 15, 21, 8, 0, 0, 0]),
        v([18, 2, 61, 56, 14, 0, 0, 0]),
    ];
    // π inside register k (row k): element j takes lane ((k + 3j) mod 5, k), which π moves to
    // (k, j). The register then holds column k of π's output.
    let pi = [
        v([0, 3, 1, 4, 2, 5, 6, 7]),
        v([1, 4, 2, 0, 3, 5, 6, 7]),
        v([2, 0, 3, 1, 4, 5, 6, 7]),
        v([3, 1, 4, 2, 0, 5, 6, 7]),
        v([4, 2, 0, 3, 1, 5, 6, 7]),
    ];
    // The transposition from columns to rows (indices 8 to 15 select from the second register).
    let t_lo = v([0, 1, 2, 3, 4, 5, 8, 10]);
    let t_hi = v([0, 1, 2, 3, 4, 5, 9, 11]);
    let t_01 = v([0, 1, 8, 9, 6, 5, 6, 7]);
    let t_23 = v([2, 3, 10, 11, 7, 5, 6, 7]);
    let t_4 = v([4, 5, 12, 13, 4, 5, 6, 7]);

    let p = state.as_mut_ptr() as *mut i64;
    // SAFETY: each masked load reads only the five lanes 5y to 5y + 4 of `state` (mask 0x1F),
    // all within its 25 lanes.
    let mut r = unsafe {
        [
            _mm512_maskz_loadu_epi64(0x1F, p),
            _mm512_maskz_loadu_epi64(0x1F, p.add(5)),
            _mm512_maskz_loadu_epi64(0x1F, p.add(10)),
            _mm512_maskz_loadu_epi64(0x1F, p.add(15)),
            _mm512_maskz_loadu_epi64(0x1F, p.add(20)),
        ]
    };
    for &rc in RC_PREV.iter() {
        // ι of the previous round, in element 0.
        let r0 = _mm512_mask_xor_epi64(r[0], 1, r[0], _mm512_set1_epi64(rc as i64));
        // θ.
        let c = xor3(xor3(r[1], r[2], r[3]), r0, r[4]);
        let dp = _mm512_permutexvar_epi64(prev, c);
        let dn = _mm512_rol_epi64::<1>(_mm512_permutexvar_epi64(next, c));
        // θ's sums, ρ, and π inside each register: b[k] is column k of π's output.
        let b0 = _mm512_permutexvar_epi64(pi[0], _mm512_rolv_epi64(xor3(r0, dp, dn), rho[0]));
        let b1 = _mm512_permutexvar_epi64(pi[1], _mm512_rolv_epi64(xor3(r[1], dp, dn), rho[1]));
        let b2 = _mm512_permutexvar_epi64(pi[2], _mm512_rolv_epi64(xor3(r[2], dp, dn), rho[2]));
        let b3 = _mm512_permutexvar_epi64(pi[3], _mm512_rolv_epi64(xor3(r[3], dp, dn), rho[3]));
        let b4 = _mm512_permutexvar_epi64(pi[4], _mm512_rolv_epi64(xor3(r[4], dp, dn), rho[4]));
        // χ: column x of the round's output.
        let c0 = chi(b0, b1, b2);
        let c1 = chi(b1, b2, b3);
        let c2 = chi(b2, b3, b4);
        let c3 = chi(b3, b4, b0);
        let c4 = chi(b4, b0, b1);
        // Columns to rows. u0 holds elements 0, 2, 4 of c0 and c1 (interleaved), u2 elements 1
        // and 3. Elements 6 and 7 of w0 and w2 take elements 0, 2 and 1, 3 of c4.
        let u0 = _mm512_unpacklo_epi64(c0, c1);
        let u1 = _mm512_unpacklo_epi64(c2, c3);
        let u2 = _mm512_unpackhi_epi64(c0, c1);
        let u3 = _mm512_unpackhi_epi64(c2, c3);
        let w0 = _mm512_permutex2var_epi64(u0, t_lo, c4);
        let w2 = _mm512_permutex2var_epi64(u2, t_hi, c4);
        r = [
            _mm512_permutex2var_epi64(w0, t_01, u1),
            _mm512_permutex2var_epi64(w2, t_01, u3),
            _mm512_permutex2var_epi64(w0, t_23, u1),
            _mm512_permutex2var_epi64(w2, t_23, u3),
            _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(w0, t_4, u1), c4),
        ];
    }
    // ι of the last round.
    r[0] = _mm512_mask_xor_epi64(r[0], 1, r[0], _mm512_set1_epi64(RC[23] as i64));
    for (y, ry) in r.into_iter().enumerate() {
        // SAFETY: the masked store writes only the five lanes 5y to 5y + 4 of `state` (mask
        // 0x1F), all within its 25 lanes.
        unsafe { _mm512_mask_storeu_epi64(p.add(5 * y), 0x1F, ry) };
    }
}

/// Lanes `j` and `j + 1` of a state.
#[inline]
fn load2(s: &[u64; 25], j: usize) -> __m128i {
    let c = &s[j..j + 2];
    // SAFETY: `c` is two lanes, 16 bytes. The load is unaligned, and SSE2 is part of the x86-64
    // baseline.
    unsafe { _mm_loadu_si128(c.as_ptr().cast()) }
}

/// Writes lanes `j` and `j + 1` of a state.
#[inline]
fn store2(s: &mut [u64; 25], j: usize, x: __m128i) {
    let c = &mut s[j..j + 2];
    // SAFETY: as in `load2`.
    unsafe { _mm_storeu_si128(c.as_mut_ptr().cast(), x) }
}

/// Four states transposed: element k of `a[i]` is lane i of state k.
#[target_feature(enable = "avx2")]
#[inline]
fn load_x4(s: &[[u64; 25]; 4]) -> [__m256i; 25] {
    let mut a = [_mm256_setzero_si256(); 25];
    for i in 0..12 {
        let j = 2 * i;
        // Lanes j and j + 1 of states 0 and 2, then of states 1 and 3.
        let e = _mm256_castsi128_si256(load2(&s[0], j));
        let e = _mm256_inserti128_si256::<1>(e, load2(&s[2], j));
        let o = _mm256_castsi128_si256(load2(&s[1], j));
        let o = _mm256_inserti128_si256::<1>(o, load2(&s[3], j));
        a[j] = _mm256_unpacklo_epi64(e, o);
        a[j + 1] = _mm256_unpackhi_epi64(e, o);
    }
    a[24] = _mm256_set_epi64x(
        s[3][24] as i64,
        s[2][24] as i64,
        s[1][24] as i64,
        s[0][24] as i64,
    );
    a
}

/// The inverse of [`load_x4`].
#[target_feature(enable = "avx2")]
#[inline]
fn store_x4(s: &mut [[u64; 25]; 4], a: &[__m256i; 25]) {
    for i in 0..12 {
        let j = 2 * i;
        // Lanes j and j + 1 of states 0 and 2, then of states 1 and 3.
        let e = _mm256_unpacklo_epi64(a[j], a[j + 1]);
        let o = _mm256_unpackhi_epi64(a[j], a[j + 1]);
        store2(&mut s[0], j, _mm256_castsi256_si128(e));
        store2(&mut s[2], j, _mm256_extracti128_si256::<1>(e));
        store2(&mut s[1], j, _mm256_castsi256_si128(o));
        store2(&mut s[3], j, _mm256_extracti128_si256::<1>(o));
    }
    let mut t = [0u64; 4];
    // SAFETY: `t` is 32 bytes, and the store is unaligned.
    unsafe { _mm256_storeu_si256(t.as_mut_ptr().cast(), a[24]) };
    for (sk, tk) in s.iter_mut().zip(t) {
        sk[24] = tk;
    }
}

/// Four states with AVX2, which has no 64-bit rotation: two shifts and an OR, or a byte
/// shuffle for the offsets 8 and 56.
mod avx2 {
    use core::arch::x86_64::*;

    type V = __m256i;

    #[target_feature(enable = "avx2")]
    #[inline]
    fn xor5(a: V, b: V, c: V, d: V, e: V) -> V {
        let ab = _mm256_xor_si256(a, b);
        let cd = _mm256_xor_si256(c, d);
        _mm256_xor_si256(_mm256_xor_si256(ab, cd), e)
    }

    /// Rotation left by `R`, with `L = 64 - R`.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn rol<const R: i32, const L: i32>(x: V) -> V {
        if R == 8 {
            let m = _mm256_setr_epi8(
                7, 0, 1, 2, 3, 4, 5, 6, 15, 8, 9, 10, 11, 12, 13, 14, 7, 0, 1, 2, 3, 4, 5, 6, 15,
                8, 9, 10, 11, 12, 13, 14,
            );
            _mm256_shuffle_epi8(x, m)
        } else if R == 56 {
            let m = _mm256_setr_epi8(
                1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 8, 1, 2, 3, 4, 5, 6, 7, 0, 9,
                10, 11, 12, 13, 14, 15, 8,
            );
            _mm256_shuffle_epi8(x, m)
        } else {
            _mm256_or_si256(_mm256_slli_epi64::<R>(x), _mm256_srli_epi64::<L>(x))
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn theta_d(c: V, e: V) -> V {
        _mm256_xor_si256(c, rol::<1, 63>(e))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn theta_xor(a: V, d: V) -> V {
        _mm256_xor_si256(a, d)
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn rho<const R: i32, const L: i32>(a: V, d: V) -> V {
        rol::<R, L>(_mm256_xor_si256(a, d))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn chi(a: V, b: V, c: V) -> V {
        _mm256_xor_si256(a, _mm256_andnot_si256(b, c))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn iota(a: V, rc: u64) -> V {
        _mm256_xor_si256(a, _mm256_set1_epi64x(rc as i64))
    }

    /// Keccak-f\[1600\] on four states.
    #[target_feature(enable = "avx2")]
    pub(in crate::simd) fn keccak_f1600_x4(states: &mut [[u64; 25]; 4]) {
        let mut a = super::load_x4(states);
        keccak_f1600_lanes!(a);
        super::store_x4(states, &a);
    }
}

pub(super) use avx2::keccak_f1600_x4 as keccak_f1600_x4_avx2;

/// Four states with AVX-512VL: 32 registers of 256 bits, `vprolq` for the rotations and
/// `vpternlogq` for θ's five-way XOR and for χ.
mod avx512vl {
    use core::arch::x86_64::*;

    type V = __m256i;

    #[target_feature(enable = "avx512f,avx512vl")]
    #[inline]
    fn xor5(a: V, b: V, c: V, d: V, e: V) -> V {
        let abc = _mm256_ternarylogic_epi64::<0x96>(a, b, c);
        _mm256_ternarylogic_epi64::<0x96>(abc, d, e)
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    #[inline]
    fn theta_d(c: V, e: V) -> V {
        _mm256_xor_si256(c, _mm256_rol_epi64::<1>(e))
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    #[inline]
    fn theta_xor(a: V, d: V) -> V {
        _mm256_xor_si256(a, d)
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    #[inline]
    fn rho<const R: i32, const L: i32>(a: V, d: V) -> V {
        _mm256_rol_epi64::<R>(_mm256_xor_si256(a, d))
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    #[inline]
    fn chi(a: V, b: V, c: V) -> V {
        _mm256_ternarylogic_epi64::<0xD2>(a, b, c)
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    #[inline]
    fn iota(a: V, rc: u64) -> V {
        _mm256_xor_si256(a, _mm256_set1_epi64x(rc as i64))
    }

    /// Keccak-f\[1600\] on four states.
    #[target_feature(enable = "avx512f,avx512vl")]
    pub(in crate::simd) fn keccak_f1600_x4(states: &mut [[u64; 25]; 4]) {
        let mut a = super::load_x4(states);
        keccak_f1600_lanes!(a);
        super::store_x4(states, &a);
    }
}

pub(super) use avx512vl::keccak_f1600_x4 as keccak_f1600_x4_avx512;

/// Four 128-bit values as one 512-bit register, `x0` in the low bits.
#[target_feature(enable = "avx512f")]
#[inline]
fn join4(x0: __m128i, x1: __m128i, x2: __m128i, x3: __m128i) -> __m512i {
    let v = _mm512_castsi128_si512(x0);
    let v = _mm512_inserti32x4::<1>(v, x1);
    let v = _mm512_inserti32x4::<2>(v, x2);
    _mm512_inserti32x4::<3>(v, x3)
}

/// Eight states transposed: element k of `a[i]` is lane i of state k.
#[target_feature(enable = "avx512f")]
#[inline]
fn load_x8(s: &[[u64; 25]; 8]) -> [__m512i; 25] {
    let mut a = [_mm512_setzero_si512(); 25];
    for i in 0..12 {
        let j = 2 * i;
        // Lanes j and j + 1 of the even states, then of the odd states.
        let e = join4(
            load2(&s[0], j),
            load2(&s[2], j),
            load2(&s[4], j),
            load2(&s[6], j),
        );
        let o = join4(
            load2(&s[1], j),
            load2(&s[3], j),
            load2(&s[5], j),
            load2(&s[7], j),
        );
        a[j] = _mm512_unpacklo_epi64(e, o);
        a[j + 1] = _mm512_unpackhi_epi64(e, o);
    }
    a[24] = _mm512_setr_epi64(
        s[0][24] as i64,
        s[1][24] as i64,
        s[2][24] as i64,
        s[3][24] as i64,
        s[4][24] as i64,
        s[5][24] as i64,
        s[6][24] as i64,
        s[7][24] as i64,
    );
    a
}

/// The inverse of [`load_x8`].
#[target_feature(enable = "avx512f")]
#[inline]
fn store_x8(s: &mut [[u64; 25]; 8], a: &[__m512i; 25]) {
    for i in 0..12 {
        let j = 2 * i;
        // Lanes j and j + 1 of the even states, then of the odd states.
        let e = _mm512_unpacklo_epi64(a[j], a[j + 1]);
        let o = _mm512_unpackhi_epi64(a[j], a[j + 1]);
        store2(&mut s[0], j, _mm512_castsi512_si128(e));
        store2(&mut s[2], j, _mm512_extracti32x4_epi32::<1>(e));
        store2(&mut s[4], j, _mm512_extracti32x4_epi32::<2>(e));
        store2(&mut s[6], j, _mm512_extracti32x4_epi32::<3>(e));
        store2(&mut s[1], j, _mm512_castsi512_si128(o));
        store2(&mut s[3], j, _mm512_extracti32x4_epi32::<1>(o));
        store2(&mut s[5], j, _mm512_extracti32x4_epi32::<2>(o));
        store2(&mut s[7], j, _mm512_extracti32x4_epi32::<3>(o));
    }
    let mut t = [0u64; 8];
    // SAFETY: `t` is 64 bytes, and the store is unaligned.
    unsafe { _mm512_storeu_si512(t.as_mut_ptr().cast(), a[24]) };
    for (sk, tk) in s.iter_mut().zip(t) {
        sk[24] = tk;
    }
}

/// Eight states with AVX-512F: the primitives of `avx512vl` on 512-bit registers.
mod avx512 {
    use core::arch::x86_64::*;

    type V = __m512i;

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn xor5(a: V, b: V, c: V, d: V, e: V) -> V {
        let abc = _mm512_ternarylogic_epi64::<0x96>(a, b, c);
        _mm512_ternarylogic_epi64::<0x96>(abc, d, e)
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn theta_d(c: V, e: V) -> V {
        _mm512_xor_si512(c, _mm512_rol_epi64::<1>(e))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn theta_xor(a: V, d: V) -> V {
        _mm512_xor_si512(a, d)
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn rho<const R: i32, const L: i32>(a: V, d: V) -> V {
        _mm512_rol_epi64::<R>(_mm512_xor_si512(a, d))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn chi(a: V, b: V, c: V) -> V {
        _mm512_ternarylogic_epi64::<0xD2>(a, b, c)
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    fn iota(a: V, rc: u64) -> V {
        _mm512_xor_si512(a, _mm512_set1_epi64(rc as i64))
    }

    /// Keccak-f\[1600\] on eight states.
    #[target_feature(enable = "avx512f")]
    pub(in crate::simd) fn keccak_f1600_x8(states: &mut [[u64; 25]; 8]) {
        let mut a = super::load_x8(states);
        keccak_f1600_lanes!(a);
        super::store_x8(states, &a);
    }
}

pub(super) use avx512::keccak_f1600_x8 as keccak_f1600_x8_avx512;
