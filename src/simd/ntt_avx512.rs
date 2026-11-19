//! AVX-512 kernels of the two-prime NTT (16 lanes of 32 bits, AVX-512F only): the layers with
//! half-blocks of at least 16 values and the pointwise products. The arithmetic of the AVX2
//! kernels (`ntt_avx2`) on twice the lanes, lane by lane equal to the portable code: Montgomery
//! multiplication with `vpmuludq` on the even and the odd lanes, $`t<1.5p`$, and the final
//! reductions as unsigned minima. The layers below half-blocks of 16 stay with AVX2.
//!
//! On a Xeon 8581C these kernels take 2 to 6 % less time per product or Verify than the AVX2
//! ones, far from the factor of two of the lane count. With a root or a constant operand, the
//! odd lanes come from `vpshufd` and one `vpermt2d` merges the high halves (port 5), which
//! measured 2 to 3 % faster than the shifts and the blend of the two-operand product.
use core::arch::x86_64::*;

use crate::ring::ntt::NttPrime;

/// Montgomery multiplication of 16 lanes (canonical inputs, canonical output).
#[target_feature(enable = "avx512f")]
#[inline]
fn mmul16(a: __m512i, b: __m512i, p: __m512i, p0i: __m512i) -> __m512i {
    let ze = _mm512_mul_epu32(a, b);
    let zo = _mm512_mul_epu32(_mm512_srli_epi64::<32>(a), _mm512_srli_epi64::<32>(b));
    let se = _mm512_add_epi64(ze, _mm512_mul_epu32(_mm512_mul_epu32(ze, p0i), p));
    let so = _mm512_add_epi64(zo, _mm512_mul_epu32(_mm512_mul_epu32(zo, p0i), p));
    let t = _mm512_mask_blend_epi32(0xAAAA, _mm512_srli_epi64::<32>(se), so);
    _mm512_min_epu32(t, _mm512_sub_epi32(t, p))
}

/// [`mmul16`] for a `b` with the same value in every lane (a root or a constant): `b` needs no
/// move for the odd lanes, `vpshufd` brings the odd lanes of `a` to the even positions (where
/// `vpmuludq` reads them), and `vpermt2d` takes the high halves of the even and the odd
/// products.
#[target_feature(enable = "avx512f")]
#[inline]
fn mmul16c(a: __m512i, b: __m512i, p: __m512i, p0i: __m512i) -> __m512i {
    let ze = _mm512_mul_epu32(a, b);
    let zo = _mm512_mul_epu32(_mm512_shuffle_epi32::<0xF5>(a), b);
    let se = _mm512_add_epi64(ze, _mm512_mul_epu32(_mm512_mul_epu32(ze, p0i), p));
    let so = _mm512_add_epi64(zo, _mm512_mul_epu32(_mm512_mul_epu32(zo, p0i), p));
    let idx = _mm512_setr_epi32(1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31);
    let t = _mm512_permutex2var_epi32(se, idx, so);
    _mm512_min_epu32(t, _mm512_sub_epi32(t, p))
}

/// $`a+b \bmod p`$ on 16 lanes.
#[target_feature(enable = "avx512f")]
#[inline]
fn add16(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
    let s = _mm512_add_epi32(a, b);
    _mm512_min_epu32(s, _mm512_sub_epi32(s, p))
}

/// $`a-b \bmod p`$ on 16 lanes.
#[target_feature(enable = "avx512f")]
#[inline]
fn sub16(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
    let d = _mm512_sub_epi32(a, b);
    _mm512_min_epu32(d, _mm512_add_epi32(d, p))
}

/// 16 lanes from `s`.
#[target_feature(enable = "avx512f")]
#[inline]
fn load(s: &[u32; 16]) -> __m512i {
    // SAFETY: `s` is 64 bytes, and the load is unaligned.
    unsafe { _mm512_loadu_si512(s.as_ptr() as *const __m512i) }
}

/// 16 lanes into `s`.
#[target_feature(enable = "avx512f")]
#[inline]
fn store(s: &mut [u32; 16], v: __m512i) {
    // SAFETY: `s` is 64 bytes, and the store is unaligned.
    unsafe { _mm512_storeu_si512(s.as_mut_ptr() as *mut __m512i, v) }
}

/// The constants of a prime as lanes.
#[target_feature(enable = "avx512f")]
#[inline]
fn consts(pr: &NttPrime) -> (__m512i, __m512i) {
    (
        _mm512_set1_epi32(pr.p as i32),
        _mm512_set1_epi32(pr.p0i as i32),
    )
}

/// `NttPrime::ntt`'s layers with half-blocks of at least 16 values, from the first: returns
/// the block size and the block count of the next layer (block size 16).
#[target_feature(enable = "avx512f")]
pub(super) fn ntt_wide_layers(pr: &NttPrime, logn: u32, a: &mut [u32]) -> (usize, usize) {
    let n = 1usize << logn;
    let (p, p0i) = consts(pr);
    let mut t = n;
    let mut m = 1usize;
    while t >= 32 {
        let ht = t >> 1;
        for (blk, &s) in a[..n].chunks_exact_mut(t).zip(pr.gm[m..2 * m].iter()) {
            let sv = _mm512_set1_epi32(s as i32);
            let (x, y) = blk.split_at_mut(ht);
            for (xc, yc) in x
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(y.as_chunks_mut::<16>().0)
            {
                let (xv, yv) = (load(xc), load(yc));
                let v = mmul16c(yv, sv, p, p0i);
                store(xc, add16(xv, v, p));
                store(yc, sub16(xv, v, p));
            }
        }
        t = ht;
        m <<= 1;
    }
    (t, m)
}

/// `NttPrime::intt_n`'s layers from half-block size `t >= 16` (with `hm` blocks) to the end.
#[target_feature(enable = "avx512f")]
pub(super) fn intt_wide_layers(pr: &NttPrime, logn: u32, a: &mut [u32], t: usize, hm: usize) {
    let n = 1usize << logn;
    let (p, p0i) = consts(pr);
    let (mut t, mut hm) = (t, hm);
    while t < n {
        let dt = t << 1;
        for (blk, &s) in a[..n].chunks_exact_mut(dt).zip(pr.igm[hm..2 * hm].iter()) {
            let sv = _mm512_set1_epi32(s as i32);
            let (x, y) = blk.split_at_mut(t);
            for (xc, yc) in x
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(y.as_chunks_mut::<16>().0)
            {
                let (u, v) = (load(xc), load(yc));
                store(xc, add16(u, v, p));
                store(yc, mmul16c(sub16(u, v, p), sv, p, p0i));
            }
        }
        t = dt;
        hm >>= 1;
    }
}

/// `a[i] <- mmul(a[i], b[i])`, lengths equal multiples of 16.
#[target_feature(enable = "avx512f")]
pub(super) fn mmul_slices(pr: &NttPrime, a: &mut [u32], b: &[u32]) {
    let (p, p0i) = consts(pr);
    for (ac, bc) in a
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(b.as_chunks::<16>().0)
    {
        store(ac, mmul16(load(ac), load(bc), p, p0i));
    }
}

/// `a[i] <- mmul(mmul(a[i], b[i]), c)`, or `mmul(a[i], c)` without `b`, for a length that is a
/// multiple of 16.
#[target_feature(enable = "avx512f")]
pub(super) fn mmul_slices_const(pr: &NttPrime, a: &mut [u32], b: Option<&[u32]>, c: u32) {
    let (p, p0i) = consts(pr);
    let cv = _mm512_set1_epi32(c as i32);
    match b {
        Some(b) => {
            for (ac, bc) in a
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(b.as_chunks::<16>().0)
            {
                store(ac, mmul16c(mmul16(load(ac), load(bc), p, p0i), cv, p, p0i));
            }
        }
        None => {
            for ac in a.as_chunks_mut::<16>().0.iter_mut() {
                store(ac, mmul16c(load(ac), cv, p, p0i));
            }
        }
    }
}

/// `acc[i] <- mp_add(acc[i], mmul(x[i], y[i]))`, lengths equal multiples of 16.
#[target_feature(enable = "avx512f")]
pub(super) fn mmul_add_slices(pr: &NttPrime, acc: &mut [u32], x: &[u32], y: &[u32]) {
    let (p, p0i) = consts(pr);
    for ((sc, xc), yc) in acc
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<16>().0)
        .zip(y.as_chunks::<16>().0)
    {
        store(sc, add16(load(sc), mmul16(load(xc), load(yc), p, p0i), p));
    }
}
