//! AVX2 kernels of the two-prime NTT (8 lanes of 32 bits). Every function computes, lane by
//! lane, the residue that the portable code computes: Montgomery multiplication
//! $`ab\,2^{-32}\bmod p`$ and the modular addition and subtraction, all canonical in $`[0,p)`$.
//!
//! Montgomery multiplication on 8 lanes: `vpmuludq` multiplies the even and the odd 32-bit
//! lanes separately into 64 bits, $`w=(z\bmod2^{32})\,p'\bmod2^{32}`$ and $`t=(z+wp)/2^{32}`$
//! follow as in `mp_mmul`, and $`t<1.5p`$ (since $`z<p^2<2^{31}p`$ and $`wp<2^{32}p`$). The final
//! reduction is $`\min(t,t-p)`$ on unsigned lanes: when $`t<p`$, $`t-p`$ wraps above $`t`$.
//! Addition: $`s=a+b<2p<2^{32}`$, then $`\min(s,s-p)`$. Subtraction: $`d=a-b`$, then
//! $`\min(d,d+p)`$ (when $`a<b`$, $`d`$ wraps and $`d+p=a-b+p`$ is the smaller one).
use core::arch::x86_64::*;

use crate::ring::ntt::NttPrime;

/// Montgomery multiplication of 8 lanes (canonical inputs, canonical output).
#[target_feature(enable = "avx2")]
#[inline]
fn mmul8(a: __m256i, b: __m256i, p: __m256i, p0i: __m256i) -> __m256i {
    let ze = _mm256_mul_epu32(a, b);
    let zo = _mm256_mul_epu32(_mm256_srli_epi64::<32>(a), _mm256_srli_epi64::<32>(b));
    let se = _mm256_add_epi64(ze, _mm256_mul_epu32(_mm256_mul_epu32(ze, p0i), p));
    let so = _mm256_add_epi64(zo, _mm256_mul_epu32(_mm256_mul_epu32(zo, p0i), p));
    let t = _mm256_blend_epi32::<0b1010_1010>(_mm256_srli_epi64::<32>(se), so);
    _mm256_min_epu32(t, _mm256_sub_epi32(t, p))
}

/// $`a+b \bmod p`$ on 8 lanes.
#[target_feature(enable = "avx2")]
#[inline]
fn add8(a: __m256i, b: __m256i, p: __m256i) -> __m256i {
    let s = _mm256_add_epi32(a, b);
    _mm256_min_epu32(s, _mm256_sub_epi32(s, p))
}

/// $`a-b \bmod p`$ on 8 lanes.
#[target_feature(enable = "avx2")]
#[inline]
fn sub8(a: __m256i, b: __m256i, p: __m256i) -> __m256i {
    let d = _mm256_sub_epi32(a, b);
    _mm256_min_epu32(d, _mm256_add_epi32(d, p))
}

/// 8 lanes from `s`.
#[target_feature(enable = "avx2")]
#[inline]
fn load(s: &[u32; 8]) -> __m256i {
    // SAFETY: `s` is 32 bytes; the load is unaligned.
    unsafe { _mm256_loadu_si256(s.as_ptr() as *const __m256i) }
}

/// 8 lanes into `s`.
#[target_feature(enable = "avx2")]
#[inline]
fn store(s: &mut [u32; 8], v: __m256i) {
    // SAFETY: `s` is 32 bytes; the store is unaligned.
    unsafe { _mm256_storeu_si256(s.as_mut_ptr() as *mut __m256i, v) }
}

/// The constants of a prime as lanes.
#[target_feature(enable = "avx2")]
#[inline]
fn consts(pr: &NttPrime) -> (__m256i, __m256i) {
    (
        _mm256_set1_epi32(pr.p as i32),
        _mm256_set1_epi32(pr.p0i as i32),
    )
}

/// `NttPrime::ntt`'s layers with half-blocks of at least 8 values (block `i` of layer `m`
/// with the root `gm[m + i]`): returns the block size and the block count of the next layer.
#[target_feature(enable = "avx2")]
pub(super) fn ntt_wide_layers(pr: &NttPrime, logn: u32, a: &mut [u32]) -> (usize, usize) {
    let n = 1usize << logn;
    let (p, p0i) = consts(pr);
    let mut t = n;
    let mut m = 1usize;
    while t >= 16 {
        let ht = t >> 1;
        for (blk, &s) in a[..n].chunks_exact_mut(t).zip(pr.gm[m..2 * m].iter()) {
            let sv = _mm256_set1_epi32(s as i32);
            let (x, y) = blk.split_at_mut(ht);
            for (xc, yc) in x
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(y.as_chunks_mut::<8>().0)
            {
                let (xv, yv) = (load(xc), load(yc));
                let v = mmul8(yv, sv, p, p0i);
                store(xc, add8(xv, v, p));
                store(yc, sub8(xv, v, p));
            }
        }
        t = ht;
        m <<= 1;
    }
    (t, m)
}

/// `NttPrime::intt_n`'s layers from half-block size `t >= 8` (with `hm` blocks) to the end
/// (block `i` with the root `igm[hm + i]`).
#[target_feature(enable = "avx2")]
pub(super) fn intt_wide_layers(pr: &NttPrime, logn: u32, a: &mut [u32], t: usize, hm: usize) {
    let n = 1usize << logn;
    let (p, p0i) = consts(pr);
    let (mut t, mut hm) = (t, hm);
    while t < n {
        let dt = t << 1;
        for (blk, &s) in a[..n].chunks_exact_mut(dt).zip(pr.igm[hm..2 * hm].iter()) {
            let sv = _mm256_set1_epi32(s as i32);
            let (x, y) = blk.split_at_mut(t);
            for (xc, yc) in x
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(y.as_chunks_mut::<8>().0)
            {
                let (u, v) = (load(xc), load(yc));
                store(xc, add8(u, v, p));
                store(yc, mmul8(sub8(u, v, p), sv, p, p0i));
            }
        }
        t = dt;
        hm >>= 1;
    }
}

/// `a[i] <- mmul(a[i], b[i])`, lengths equal multiples of 8.
#[target_feature(enable = "avx2")]
pub(super) fn mmul_slices(pr: &NttPrime, a: &mut [u32], b: &[u32]) {
    let (p, p0i) = consts(pr);
    for (ac, bc) in a
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(b.as_chunks::<8>().0)
    {
        store(ac, mmul8(load(ac), load(bc), p, p0i));
    }
}

/// `a[i] <- mmul(mmul(a[i], b[i]), c)`, or `mmul(a[i], c)` without `b`; length a multiple of 8.
#[target_feature(enable = "avx2")]
pub(super) fn mmul_slices_const(pr: &NttPrime, a: &mut [u32], b: Option<&[u32]>, c: u32) {
    let (p, p0i) = consts(pr);
    let cv = _mm256_set1_epi32(c as i32);
    match b {
        Some(b) => {
            for (ac, bc) in a
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(b.as_chunks::<8>().0)
            {
                store(ac, mmul8(mmul8(load(ac), load(bc), p, p0i), cv, p, p0i));
            }
        }
        None => {
            for ac in a.as_chunks_mut::<8>().0.iter_mut() {
                store(ac, mmul8(load(ac), cv, p, p0i));
            }
        }
    }
}

/// `acc[i] <- mp_add(acc[i], mmul(x[i], y[i]))`, lengths equal multiples of 8.
#[target_feature(enable = "avx2")]
pub(super) fn mmul_add_slices(pr: &NttPrime, acc: &mut [u32], x: &[u32], y: &[u32]) {
    let (p, p0i) = consts(pr);
    for ((sc, xc), yc) in acc
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
        .zip(y.as_chunks::<8>().0)
    {
        store(sc, add8(load(sc), mmul8(load(xc), load(yc), p, p0i), p));
    }
}
