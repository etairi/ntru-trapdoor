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
//!
//! The narrow layers of the transforms (half-blocks of 4, 2 and 1 values) run on two blocks
//! of 8 values at a time. Between two layers the lanes are regrouped (`vperm2i128`,
//! `vpunpck*`, shifts and blends), so that every layer is one butterfly on two vectors, and the
//! last step puts them back in order. (Merging the layer with half-blocks of 8 into the same
//! pass measured 2 to 4 % slower on a Xeon 8581C.) The CRT, the reductions modulo $`q`$ and
//! Verify's norm follow the portable functions value by value (each kernel documents its
//! bounds).
use core::arch::x86_64::*;

use super::QConsts;
use crate::ring::ntt::{NttPrime, P1, P1_INV_MOD_P2_MONT, P2};

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
    ntt_wide_layers_from(pr, logn, a, 1usize << logn, 1)
}

/// [`ntt_wide_layers`] from blocks of `t` values (`m` blocks), where a wider kernel stopped.
#[target_feature(enable = "avx2")]
pub(super) fn ntt_wide_layers_from(
    pr: &NttPrime,
    logn: u32,
    a: &mut [u32],
    t: usize,
    m: usize,
) -> (usize, usize) {
    let n = 1usize << logn;
    let (p, p0i) = consts(pr);
    let (mut t, mut m) = (t, m);
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
    intt_wide_layers_until(pr, logn, a, t, hm, 1usize << logn);
}

/// [`intt_wide_layers`] up to the half-block size `stop` (excluded), where a wider kernel
/// continues.
#[target_feature(enable = "avx2")]
pub(super) fn intt_wide_layers_until(
    pr: &NttPrime,
    logn: u32,
    a: &mut [u32],
    t: usize,
    hm: usize,
    stop: usize,
) {
    let n = 1usize << logn;
    let (p, p0i) = consts(pr);
    let (mut t, mut hm) = (t, hm);
    while t < stop.min(n) {
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

/// 4 lanes from `s`.
#[target_feature(enable = "avx2")]
#[inline]
fn load4(s: &[u32; 4]) -> __m128i {
    // SAFETY: `s` is 16 bytes, and the load is unaligned.
    unsafe { _mm_loadu_si128(s.as_ptr() as *const __m128i) }
}

/// 2 lanes from `s` (the other lanes zero).
#[target_feature(enable = "avx2")]
#[inline]
fn load2(s: &[u32; 2]) -> __m128i {
    // SAFETY: `s` is 8 bytes, and the load is unaligned.
    unsafe { _mm_loadl_epi64(s.as_ptr() as *const __m128i) }
}

/// 8 signed lanes from `s`.
#[target_feature(enable = "avx2")]
#[inline]
fn load_i32(s: &[i32; 8]) -> __m256i {
    // SAFETY: `s` is 32 bytes, and the load is unaligned.
    unsafe { _mm256_loadu_si256(s.as_ptr() as *const __m256i) }
}

/// Lanes 0 to 3 of `v` holding `r[0], r[0], r[1], r[1]` and so on: the roots of a layer with
/// half-blocks of 2 values, for 4 blocks of 4.
#[target_feature(enable = "avx2")]
#[inline]
fn roots_dup2(r: &[u32; 4]) -> __m256i {
    let idx = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(load4(r)), idx)
}

/// `r[0]` in lanes 0 to 3 and `r[1]` in lanes 4 to 7: the roots of a layer with half-blocks of
/// 4 values, for 2 blocks of 8.
#[target_feature(enable = "avx2")]
#[inline]
fn roots_dup4(r: &[u32; 2]) -> __m256i {
    let idx = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
    _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(load2(r)), idx)
}

/// The forward butterfly `(x + y s, x - y s)` on 8 lanes.
#[target_feature(enable = "avx2")]
#[inline]
fn fwd8(x: __m256i, y: __m256i, s: __m256i, p: __m256i, p0i: __m256i) -> (__m256i, __m256i) {
    let v = mmul8(y, s, p, p0i);
    (add8(x, v, p), sub8(x, v, p))
}

/// The inverse butterfly `(x + y, (x - y) s)` on 8 lanes.
#[target_feature(enable = "avx2")]
#[inline]
fn inv8(x: __m256i, y: __m256i, s: __m256i, p: __m256i, p0i: __m256i) -> (__m256i, __m256i) {
    (add8(x, y, p), mmul8(sub8(x, y, p), s, p, p0i))
}

/// Swaps the odd lanes of `x` with the even lanes of `y`, pair by pair: from
/// `[x0 x1 x2 x3 ..]`, `[y0 y1 y2 y3 ..]` to `[x0 y0 x2 y2 ..]`, `[x1 y1 x3 y3 ..]`. Its own
/// inverse.
#[target_feature(enable = "avx2")]
#[inline]
fn swap32(x: __m256i, y: __m256i) -> (__m256i, __m256i) {
    (
        _mm256_blend_epi32::<0b1010_1010>(x, _mm256_slli_epi64::<32>(y)),
        _mm256_blend_epi32::<0b1010_1010>(_mm256_srli_epi64::<32>(x), y),
    )
}

/// `NttPrime::ntt`'s last three layers, from blocks of 8 values (`n / 8` blocks, block `i`
/// with the root `gm[n/8 + i]`, then `gm[n/4 + j]` for its 4-blocks and `gm[n/2 + k]` for its
/// pairs), two blocks of 8 per iteration. Lane `l` of the vectors `x`, `y` below holds the
/// values `(a_l, b_l)` of one butterfly.
#[target_feature(enable = "avx2")]
pub(super) fn ntt_narrow_layers(pr: &NttPrime, logn: u32, a: &mut [u32]) {
    let n = 1usize << logn;
    let m = n >> 3;
    let (p, p0i) = consts(pr);
    let ra = pr.gm[m..2 * m].as_chunks::<2>().0;
    let rb = pr.gm[2 * m..4 * m].as_chunks::<4>().0;
    let rc = pr.gm[4 * m..8 * m].as_chunks::<8>().0;
    let blocks = a[..n].as_chunks_mut::<8>().0.as_chunks_mut::<2>().0;
    for (((pair, sa), sb), sc) in blocks.iter_mut().zip(ra).zip(rb).zip(rc) {
        let [u, v] = pair;
        let (r0, r1) = (load(u), load(v));
        // Half-blocks of 4: x = [u0..u3 | v0..v3], y = [u4..u7 | v4..v7].
        let x = _mm256_permute2x128_si256::<0x20>(r0, r1);
        let y = _mm256_permute2x128_si256::<0x31>(r0, r1);
        let (x, y) = fwd8(x, y, roots_dup4(sa), p, p0i);
        // Half-blocks of 2: x = [u0 u1 u4 u5 | ..], y = [u2 u3 u6 u7 | ..].
        let (x, y) = (_mm256_unpacklo_epi64(x, y), _mm256_unpackhi_epi64(x, y));
        let (x, y) = fwd8(x, y, roots_dup2(sb), p, p0i);
        // Pairs: x = [u0 u2 u4 u6 | ..], y = [u1 u3 u5 u7 | ..].
        let (x, y) = swap32(x, y);
        let (x, y) = fwd8(x, y, load(sc), p, p0i);
        // Back in order.
        let (lo, hi) = (_mm256_unpacklo_epi32(x, y), _mm256_unpackhi_epi32(x, y));
        store(u, _mm256_permute2x128_si256::<0x20>(lo, hi));
        store(v, _mm256_permute2x128_si256::<0x31>(lo, hi));
    }
}

/// `NttPrime::intt_n`'s first three layers (pairs with `igm[n/2 + k]`, half-blocks of 2 with
/// `igm[n/4 + j]`, half-blocks of 4 with `igm[n/8 + i]`), two blocks of 8 per iteration: the
/// steps of [`ntt_narrow_layers`] in reverse order.
#[target_feature(enable = "avx2")]
pub(super) fn intt_narrow_layers(pr: &NttPrime, logn: u32, a: &mut [u32]) {
    let n = 1usize << logn;
    let m = n >> 3;
    let (p, p0i) = consts(pr);
    let rc = pr.igm[4 * m..8 * m].as_chunks::<8>().0;
    let rb = pr.igm[2 * m..4 * m].as_chunks::<4>().0;
    let ra = pr.igm[m..2 * m].as_chunks::<2>().0;
    let blocks = a[..n].as_chunks_mut::<8>().0.as_chunks_mut::<2>().0;
    for (((pair, sc), sb), sa) in blocks.iter_mut().zip(rc).zip(rb).zip(ra) {
        let [u, v] = pair;
        let (r0, r1) = (load(u), load(v));
        // Pairs: x = [u0 u2 u4 u6 | v0 ..], y = [u1 u3 u5 u7 | v1 ..].
        let lo = _mm256_castsi256_ps(_mm256_permute2x128_si256::<0x20>(r0, r1));
        let hi = _mm256_castsi256_ps(_mm256_permute2x128_si256::<0x31>(r0, r1));
        let x = _mm256_castps_si256(_mm256_shuffle_ps::<0x88>(lo, hi));
        let y = _mm256_castps_si256(_mm256_shuffle_ps::<0xDD>(lo, hi));
        let (x, y) = inv8(x, y, load(sc), p, p0i);
        // Half-blocks of 2: x = [u0 u1 u4 u5 | ..], y = [u2 u3 u6 u7 | ..].
        let (x, y) = swap32(x, y);
        let (x, y) = inv8(x, y, roots_dup2(sb), p, p0i);
        // Half-blocks of 4: x = [u0..u3 | v0..v3], y = [u4..u7 | v4..v7].
        let (x, y) = (_mm256_unpacklo_epi64(x, y), _mm256_unpackhi_epi64(x, y));
        let (x, y) = inv8(x, y, roots_dup4(sa), p, p0i);
        store(u, _mm256_permute2x128_si256::<0x20>(x, y));
        store(v, _mm256_permute2x128_si256::<0x31>(x, y));
    }
}

/// `ModQ::crt` on 8 lanes, for an odd $`q`$. With $`t=(u_2-u_1)/p_1\bmod p_2`$ (as the portable
/// code), $`u_1+p_1t\bmod q`$ is the Montgomery reduction modulo $`q`$ of
/// $`z=u_1(2^{32}\bmod q)+t(p_12^{32}\bmod q)<2^{31}q+2^{31}q=2^{32}q`$, so it lies in $`[0,2q)`$
/// before its final subtraction. The integer $`u_1+p_1t\in[0,P)`$ exceeds $`(P-1)/2=p_1h_2+h_1`$
/// ($`h_i=(p_i-1)/2`$) iff $`t>h_2`$, or $`t=h_2`$ and $`u_1>h_1`$: then $`P\bmod q`$ is subtracted.
#[target_feature(enable = "avx2")]
#[inline]
fn crt8(u1: __m256i, u2: __m256i, k: &Crt8) -> __m256i {
    let d = sub8(u2, _mm256_min_epu32(u1, _mm256_sub_epi32(u1, k.p2)), k.p2);
    let t = mmul8(d, k.inv, k.p2, k.p2i);
    let ze = _mm256_add_epi64(_mm256_mul_epu32(u1, k.r), _mm256_mul_epu32(t, k.p1r));
    let zo = _mm256_add_epi64(
        _mm256_mul_epu32(_mm256_srli_epi64::<32>(u1), k.r),
        _mm256_mul_epu32(_mm256_srli_epi64::<32>(t), k.p1r),
    );
    let se = _mm256_add_epi64(ze, _mm256_mul_epu32(_mm256_mul_epu32(ze, k.q0i), k.q));
    let so = _mm256_add_epi64(zo, _mm256_mul_epu32(_mm256_mul_epu32(zo, k.q0i), k.q));
    let r = _mm256_blend_epi32::<0b1010_1010>(_mm256_srli_epi64::<32>(se), so);
    let r = _mm256_min_epu32(r, _mm256_sub_epi32(r, k.q));
    let neg = _mm256_or_si256(
        _mm256_cmpgt_epi32(t, k.h2),
        _mm256_and_si256(_mm256_cmpeq_epi32(t, k.h2), _mm256_cmpgt_epi32(u1, k.h1)),
    );
    sub8(r, _mm256_and_si256(k.pq, neg), k.q)
}

/// The constants of [`crt8`] as lanes.
struct Crt8 {
    p2: __m256i,
    p2i: __m256i,
    inv: __m256i,
    h1: __m256i,
    h2: __m256i,
    q: __m256i,
    q0i: __m256i,
    r: __m256i,
    p1r: __m256i,
    pq: __m256i,
}

#[target_feature(enable = "avx2")]
#[inline]
fn crt_consts(c: &QConsts) -> Crt8 {
    let s = |x: u32| _mm256_set1_epi32(x as i32);
    Crt8 {
        p2: s(P2.p),
        p2i: s(P2.p0i),
        inv: s(P1_INV_MOD_P2_MONT),
        h1: s((P1.p - 1) / 2),
        h2: s((P2.p - 1) / 2),
        q: s(c.q),
        q0i: s(c.q0i),
        r: s(c.r),
        p1r: s(c.p1r),
        pq: s(c.pq),
    }
}

/// `out[i] <- crt(u1[i], u2[i])`, or `mp_add(out[i], crt(u1[i], u2[i]), q)` with `ACC`, for an
/// odd $`q`$. Equal lengths, multiples of 8.
#[target_feature(enable = "avx2")]
pub(super) fn crt_slices<const ACC: bool>(c: &QConsts, u1: &[u32], u2: &[u32], out: &mut [u32]) {
    let k = crt_consts(c);
    for ((o, x), y) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(u1.as_chunks::<8>().0)
        .zip(u2.as_chunks::<8>().0)
    {
        let r = crt8(load(x), load(y), &k);
        store(o, if ACC { add8(load(o), r, k.q) } else { r });
    }
}

/// `ModQ::reduce_i32` on 8 lanes: $`|x|\le2^{31}`$ (`vpabsd` leaves $`-2^{31}`$ as $`2^{31}`$ unsigned),
/// the Barrett quotient $`\hat k=\lfloor|x|\,m/2^{32}\rfloor`$ with $`m=\lfloor2^{32}/q\rfloor`$ is
/// $`\lfloor|x|/q\rfloor`$ or one less (the error of $`m`$ costs less than $`|x|/2^{32}\le1/2`$),
/// so $`|x|-\hat kq\in[0,2q)`$, then $`q-r`$ for $`x<0`$.
#[target_feature(enable = "avx2")]
#[inline]
fn reduce_i32x8(x: __m256i, q: __m256i, m: __m256i) -> __m256i {
    let ax = _mm256_abs_epi32(x);
    let he = _mm256_srli_epi64::<32>(_mm256_mul_epu32(ax, m));
    let ho = _mm256_mul_epu32(_mm256_srli_epi64::<32>(ax), m);
    let k = _mm256_blend_epi32::<0b1010_1010>(he, ho);
    let r = _mm256_sub_epi32(ax, _mm256_mullo_epi32(k, q));
    let r = _mm256_min_epu32(r, _mm256_sub_epi32(r, q));
    let nr = sub8(_mm256_setzero_si256(), r, q);
    _mm256_blendv_epi8(r, nr, _mm256_srai_epi32::<31>(x))
}

/// `out[i] <- x[i] mod q` (`ModQ::reduce_i32`). Equal lengths, multiples of 8.
#[target_feature(enable = "avx2")]
pub(super) fn reduce_i32_slices(c: &QConsts, x: &[i32], out: &mut [u32]) {
    let (q, m) = (_mm256_set1_epi32(c.q as i32), _mm256_set1_epi32(c.m as i32));
    for (o, v) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<8>().0)
    {
        store(o, reduce_i32x8(load_i32(v), q, m));
    }
}

/// The OR over `i` of `(x[i] mod q + y[i] mod q) XOR z[i]`, for signed `x` and canonical `y`:
/// zero iff `x + y = z` modulo `q` everywhere. Equal lengths, multiples of 8.
#[target_feature(enable = "avx2")]
pub(super) fn add_i32_diff(c: &QConsts, x: &[i32], y: &[u32], z: &[u32]) -> u32 {
    let (q, m) = (_mm256_set1_epi32(c.q as i32), _mm256_set1_epi32(c.m as i32));
    let mut acc = _mm256_setzero_si256();
    for ((xc, yc), zc) in x
        .as_chunks::<8>()
        .0
        .iter()
        .zip(y.as_chunks::<8>().0)
        .zip(z.as_chunks::<8>().0)
    {
        let s = add8(reduce_i32x8(load_i32(xc), q, m), load(yc), q);
        acc = _mm256_or_si256(acc, _mm256_xor_si256(s, load(zc)));
    }
    let mut lanes = [0u32; 8];
    store(&mut lanes, acc);
    lanes.iter().fold(0, |d, &l| d | l)
}

/// `out[i] <- c mod q` for the centred lift `c` of `u[i] mod p1` (`u` if $`u\le(p_1-1)/2`$, else
/// $`u-p_1`$, so $`|c|<2^{30}`$). Equal lengths, multiples of 8.
#[target_feature(enable = "avx2")]
pub(super) fn centred_p1_mod_q(c: &QConsts, u: &[u32], out: &mut [u32]) {
    let (q, m) = (_mm256_set1_epi32(c.q as i32), _mm256_set1_epi32(c.m as i32));
    let (p1, h1) = (
        _mm256_set1_epi32(P1.p as i32),
        _mm256_set1_epi32((P1.p >> 1) as i32),
    );
    for (o, uc) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(u.as_chunks::<8>().0)
    {
        let v = load(uc);
        let lifted = _mm256_sub_epi32(v, _mm256_and_si256(p1, _mm256_cmpgt_epi32(v, h1)));
        store(o, reduce_i32x8(lifted, q, m));
    }
}

/// $`\sum_ix_i^2`$ over both slices, exactly: the squares ($`\le2^{62}`$) split into their low and
/// high 32 bits, which 64-bit lanes add without overflow for any slice length below $`2^{34}`$.
/// The tails (lengths not a multiple of 8) are added in `u128`.
#[target_feature(enable = "avx2")]
pub(super) fn norm_sq(a: &[i32], b: &[i32]) -> u128 {
    let lo32 = _mm256_set1_epi64x(0xFFFF_FFFF);
    let (mut lo, mut hi) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    let mut tail = 0u128;
    for s in [a, b] {
        let (chunks, rest) = s.as_chunks::<8>();
        for c in chunks {
            let x = _mm256_abs_epi32(load_i32(c));
            let xo = _mm256_srli_epi64::<32>(x);
            for sq in [_mm256_mul_epu32(x, x), _mm256_mul_epu32(xo, xo)] {
                lo = _mm256_add_epi64(lo, _mm256_and_si256(sq, lo32));
                hi = _mm256_add_epi64(hi, _mm256_srli_epi64::<32>(sq));
            }
        }
        tail += rest
            .iter()
            .map(|&x| (x.unsigned_abs() as u128).pow(2))
            .sum::<u128>();
    }
    let (mut l, mut h) = ([0u64; 4], [0u64; 4]);
    // SAFETY: each array is 32 bytes, and the stores are unaligned.
    unsafe {
        _mm256_storeu_si256(l.as_mut_ptr() as *mut __m256i, lo);
        _mm256_storeu_si256(h.as_mut_ptr() as *mut __m256i, hi);
    }
    let sum: u128 = l
        .iter()
        .zip(&h)
        .map(|(&l, &h)| ((h as u128) << 32) + l as u128)
        .sum();
    sum + tail
}

/// The norm of `ring::inv` in the NTT domain (`norm_ntt`), on 8 lanes: point `2i` is evaluated
/// at $`\zeta_i`$ and point `2i + 1` at $`-\zeta_i`$, $`\zeta_i`$ = `gm[h/2 + i]`, so
/// `out[k] = e^2 c - zeta o^2 c` for even `k` and `e^2 c + zeta o^2 c` for odd `k`, with
/// `c = ninv_r2[logh]`. Length `h` a multiple of 8.
#[target_feature(enable = "avx2")]
pub(super) fn norm_ntt(pr: &NttPrime, logh: u32, e: &[u32], o: &[u32], out: &mut [u32]) {
    let h = 1usize << logh;
    let (p, p0i) = consts(pr);
    let c = _mm256_set1_epi32(pr.ninv_r2[logh as usize] as i32);
    let z = pr.gm[h / 2..h].as_chunks::<4>().0;
    for (((oc, ec), occ), zc) in out[..h]
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(e[..h].as_chunks::<8>().0)
        .zip(o[..h].as_chunks::<8>().0)
        .zip(z)
    {
        let (ev, ov) = (load(ec), load(occ));
        let e2 = mmul8(mmul8(ev, ev, p, p0i), c, p, p0i);
        let o2 = mmul8(mmul8(ov, ov, p, p0i), c, p, p0i);
        let zo2 = mmul8(o2, roots_dup2(zc), p, p0i);
        store(
            oc,
            _mm256_blend_epi32::<0b1010_1010>(sub8(e2, zo2, p), add8(e2, zo2, p)),
        );
    }
}

/// The OR over `i` of `(mul(f[i], G[i]) - mul(g[i], F[i])) XOR q` (`NttPrime::mul`, the plain
/// product): zero iff $`\hat f\hat G-\hat g\hat F=q`$ in every slot. Equal lengths, multiples of
/// 8.
#[target_feature(enable = "avx2")]
pub(super) fn ntru_diff(
    pr: &NttPrime,
    q: u32,
    f: &[u32],
    g: &[u32],
    bf: &[u32],
    bg: &[u32],
) -> u32 {
    let (p, p0i) = consts(pr);
    let r2 = _mm256_set1_epi32(pr.r2 as i32);
    let qv = _mm256_set1_epi32(q as i32);
    let mut acc = _mm256_setzero_si256();
    for (((fc, gc), bfc), bgc) in f
        .as_chunks::<8>()
        .0
        .iter()
        .zip(g.as_chunks::<8>().0)
        .zip(bf.as_chunks::<8>().0)
        .zip(bg.as_chunks::<8>().0)
    {
        let fg = mmul8(mmul8(load(fc), load(bgc), p, p0i), r2, p, p0i);
        let gf = mmul8(mmul8(load(gc), load(bfc), p, p0i), r2, p, p0i);
        acc = _mm256_or_si256(acc, _mm256_xor_si256(sub8(fg, gf, p), qv));
    }
    let mut lanes = [0u32; 8];
    store(&mut lanes, acc);
    lanes.iter().fold(0, |d, &l| d | l)
}
