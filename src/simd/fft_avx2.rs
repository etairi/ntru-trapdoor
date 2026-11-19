//! AVX2 kernels of the FFT passes of SampPre, four values per instruction. Each performs, for
//! every output value, the IEEE operations of its portable counterpart in fft/mod.rs (ports of
//! fn-dsa's portable branches) in the same order, without FMA, so the results are the same
//! bits: the products and sums of a complex multiplication are those of `flc_mul`, the negated
//! twiddle of the split is a sign flip as `-x` is, and the halvings are products by 0.5.
//!
//! Lane layout: four consecutive indices `i` per register. The pairs `(f[2i], f[2i + 1])` of the
//! split and merge are separated into even and odd lanes by `deinterleave` and put back by
//! `interleave`, and the interleaved twiddles `(GM[2(hn + i)], GM[2(hn + i) + 1])` the same way.
use core::arch::x86_64::*;

/// Four values.
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
fn ld4(s: &[f64; 4]) -> __m256d {
    // SAFETY: `s` is 32 bytes, and the load is unaligned.
    unsafe { _mm256_loadu_pd(s.as_ptr()) }
}

/// Stores four values.
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
fn st4(s: &mut [f64; 4], v: __m256d) {
    // SAFETY: `s` is 32 bytes, and the store is unaligned.
    unsafe { _mm256_storeu_pd(s.as_mut_ptr(), v) }
}

/// Eight consecutive values `[x0 .. x7]` as their even lanes `[x0 x2 x4 x6]` and odd lanes
/// `[x1 x3 x5 x7]`.
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
fn ld_pairs(s: &[f64; 8]) -> (__m256d, __m256d) {
    // SAFETY: `s` is 64 bytes, two unaligned loads of 32.
    let (a, b) = unsafe {
        (
            _mm256_loadu_pd(s.as_ptr()),
            _mm256_loadu_pd(s.as_ptr().add(4)),
        )
    };
    let lo = _mm256_unpacklo_pd(a, b);
    let hi = _mm256_unpackhi_pd(a, b);
    (
        _mm256_permute4x64_pd::<0xD8>(lo),
        _mm256_permute4x64_pd::<0xD8>(hi),
    )
}

/// Stores even lanes `e` and odd lanes `o` as `[e0 o0 e1 o1 e2 o2 e3 o3]`.
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
fn st_pairs(s: &mut [f64; 8], e: __m256d, o: __m256d) {
    let lo = _mm256_unpacklo_pd(e, o);
    let hi = _mm256_unpackhi_pd(e, o);
    let (a, b) = (
        _mm256_permute2f128_pd::<0x20>(lo, hi),
        _mm256_permute2f128_pd::<0x31>(lo, hi),
    );
    // SAFETY: `s` is 64 bytes, two unaligned stores of 32.
    unsafe {
        _mm256_storeu_pd(s.as_mut_ptr(), a);
        _mm256_storeu_pd(s.as_mut_ptr().add(4), b);
    }
}

/// `flc_mul` on four lanes: `(x_re * y_re - x_im * y_im, x_re * y_im + x_im * y_re)`.
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
fn cmul(x_re: __m256d, x_im: __m256d, y_re: __m256d, y_im: __m256d) -> (__m256d, __m256d) {
    (
        _mm256_sub_pd(_mm256_mul_pd(x_re, y_re), _mm256_mul_pd(x_im, y_im)),
        _mm256_add_pd(_mm256_mul_pd(x_re, y_im), _mm256_mul_pd(x_im, y_re)),
    )
}

/// `-x` on four lanes: the sign bit flipped, as the scalar negation does.
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[inline]
fn neg(x: __m256d) -> __m256d {
    _mm256_xor_pd(x, _mm256_set1_pd(-0.0))
}

/// `fft::poly_split_fft` with `gm = GM[2 hn..3 hn]`, for `n/4` a multiple of 4.
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
pub(crate) fn poly_split_fft(logn: u32, f0: &mut [f64], f1: &mut [f64], f: &[f64], gm: &[f64]) {
    let hn = 1usize << (logn - 1);
    let qn = hn >> 1;
    assert!(qn.is_multiple_of(4) && gm.len() >= hn);
    let half = _mm256_set1_pd(0.5);
    let (f_re, f_im) = f[..2 * hn].split_at(hn);
    let (f0_re, f0_im) = f0[..hn].split_at_mut(qn);
    let (f1_re, f1_im) = f1[..hn].split_at_mut(qn);
    for (((((re, im), g), (o0r, o0i)), o1r), o1i) in f_re
        .as_chunks::<8>()
        .0
        .iter()
        .zip(f_im.as_chunks::<8>().0)
        .zip(gm[..hn].as_chunks::<8>().0)
        .zip(
            f0_re
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(f0_im.as_chunks_mut::<4>().0),
        )
        .zip(f1_re.as_chunks_mut::<4>().0)
        .zip(f1_im.as_chunks_mut::<4>().0)
    {
        let (g_re, g_im) = ld_pairs(g);
        let (a_re, b_re) = ld_pairs(re);
        let (a_im, b_im) = ld_pairs(im);
        st4(o0r, _mm256_mul_pd(_mm256_add_pd(a_re, b_re), half));
        st4(o0i, _mm256_mul_pd(_mm256_add_pd(a_im, b_im), half));
        let (t_re, t_im) = (_mm256_sub_pd(a_re, b_re), _mm256_sub_pd(a_im, b_im));
        let (u_re, u_im) = cmul(t_re, t_im, g_re, neg(g_im));
        st4(o1r, _mm256_mul_pd(u_re, half));
        st4(o1i, _mm256_mul_pd(u_im, half));
    }
}

/// `fft::poly_merge_fft` with `gm = GM[2 hn..3 hn]`, for `n/4` a multiple of 4.
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
pub(crate) fn poly_merge_fft(logn: u32, f: &mut [f64], f0: &[f64], f1: &[f64], gm: &[f64]) {
    let hn = 1usize << (logn - 1);
    let qn = hn >> 1;
    assert!(qn.is_multiple_of(4) && gm.len() >= hn);
    let (f_re, f_im) = f[..2 * hn].split_at_mut(hn);
    let (f0_re, f0_im) = f0[..hn].split_at(qn);
    let (f1_re, f1_im) = f1[..hn].split_at(qn);
    for (((((o_re, o_im), g), (a_re, a_im)), c_re), c_im) in f_re
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(f_im.as_chunks_mut::<8>().0)
        .zip(gm[..hn].as_chunks::<8>().0)
        .zip(
            f0_re
                .as_chunks::<4>()
                .0
                .iter()
                .zip(f0_im.as_chunks::<4>().0),
        )
        .zip(f1_re.as_chunks::<4>().0)
        .zip(f1_im.as_chunks::<4>().0)
    {
        let (g_re, g_im) = ld_pairs(g);
        let (a_re, a_im) = (ld4(a_re), ld4(a_im));
        let (b_re, b_im) = cmul(ld4(c_re), ld4(c_im), g_re, g_im);
        st_pairs(o_re, _mm256_add_pd(a_re, b_re), _mm256_sub_pd(a_re, b_re));
        st_pairs(o_im, _mm256_add_pd(a_im, b_im), _mm256_sub_pd(a_im, b_im));
    }
}

/// `fft::merge_sub_mul_add_split` with `gm = GM[2 hn..3 hn]`, for `n/4` a multiple of 4: the
/// merge of `w` into `t1` (whose old value is read first), `tb0 = t0 + (t1 - z1) * l10`, and the
/// split of `tb0` into `w`, with the operations of the portable pass on every value.
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
pub(crate) fn merge_sub_mul_add_split(
    logn: u32,
    w: &mut [f64],
    t0: &[f64],
    t1: &mut [f64],
    l10: &[f64],
    gm: &[f64],
) {
    let n = 1usize << logn;
    let hn = n >> 1;
    let qn = hn >> 1;
    assert!(qn.is_multiple_of(4) && gm.len() >= hn);
    let half = _mm256_set1_pd(0.5);
    let (w0, w1) = w[..n].split_at_mut(hn);
    let (w0_re, w0_im) = w0.split_at_mut(qn);
    let (w1_re, w1_im) = w1.split_at_mut(qn);
    let (t0_re, t0_im) = t0[..n].split_at(hn);
    let (t1_re, t1_im) = t1[..n].split_at_mut(hn);
    let (l_re, l_im) = l10[..n].split_at(hn);
    let inputs = t0_re
        .as_chunks::<8>()
        .0
        .iter()
        .zip(t0_im.as_chunks::<8>().0)
        .zip(l_re.as_chunks::<8>().0.iter().zip(l_im.as_chunks::<8>().0))
        .zip(gm[..hn].as_chunks::<8>().0);
    let targets = t1_re
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(t1_im.as_chunks_mut::<8>().0);
    let halves = w0_re
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(w0_im.as_chunks_mut::<4>().0)
        .zip(
            w1_re
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(w1_im.as_chunks_mut::<4>().0),
        );
    for ((((t0r, t0i), (lr, li)), g), ((t1r, t1i), ((w0r, w0i), (w1r, w1i)))) in
        inputs.zip(targets.zip(halves))
    {
        let (g_re, g_im) = ld_pairs(g);
        // poly_merge_fft: z1 at 2i (a + b) and 2i + 1 (a - b).
        let (a_re, a_im) = (ld4(w0r), ld4(w0i));
        let (b_re, b_im) = cmul(ld4(w1r), ld4(w1i), g_re, g_im);
        let z_re = [_mm256_add_pd(a_re, b_re), _mm256_sub_pd(a_re, b_re)];
        let z_im = [_mm256_add_pd(a_im, b_im), _mm256_sub_pd(a_im, b_im)];
        // poly_sub, poly_mul_fft and poly_add at 2i and 2i + 1.
        let (x0, x1) = ld_pairs(t1r);
        let t1_re = [x0, x1];
        let (x0, x1) = ld_pairs(t1i);
        let t1_im = [x0, x1];
        let (x0, x1) = ld_pairs(t0r);
        let t0_re = [x0, x1];
        let (x0, x1) = ld_pairs(t0i);
        let t0_im = [x0, x1];
        let (x0, x1) = ld_pairs(lr);
        let l_re = [x0, x1];
        let (x0, x1) = ld_pairs(li);
        let l_im = [x0, x1];
        let mut tb_re = [_mm256_setzero_pd(); 2];
        let mut tb_im = [_mm256_setzero_pd(); 2];
        for k in 0..2 {
            let x_re = _mm256_sub_pd(t1_re[k], z_re[k]);
            let x_im = _mm256_sub_pd(t1_im[k], z_im[k]);
            let (y_re, y_im) = cmul(x_re, x_im, l_re[k], l_im[k]);
            tb_re[k] = _mm256_add_pd(t0_re[k], y_re);
            tb_im[k] = _mm256_add_pd(t0_im[k], y_im);
        }
        // z1 into t1.
        st_pairs(t1r, z_re[0], z_re[1]);
        st_pairs(t1i, z_im[0], z_im[1]);
        // poly_split_fft of tb0.
        let [a_re, b_re] = tb_re;
        let [a_im, b_im] = tb_im;
        st4(w0r, _mm256_mul_pd(_mm256_add_pd(a_re, b_re), half));
        st4(w0i, _mm256_mul_pd(_mm256_add_pd(a_im, b_im), half));
        let (t_re, t_im) = (_mm256_sub_pd(a_re, b_re), _mm256_sub_pd(a_im, b_im));
        let (u_re, u_im) = cmul(t_re, t_im, g_re, neg(g_im));
        st4(w1r, _mm256_mul_pd(u_re, half));
        st4(w1i, _mm256_mul_pd(u_im, half));
    }
}

/// One butterfly layer of `fft::fft` on blocks of `t` values (half-blocks `ht = t/2`, a multiple
/// of 4), with the twiddle `(gm[2(m + i)], gm[2(m + i) + 1])` for block `i` (`gm` = `GM`).
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
pub(crate) fn fft_layer(re: &mut [f64], im: &mut [f64], t: usize, m: usize, gm: &[f64]) {
    let ht = t >> 1;
    assert!(ht.is_multiple_of(4) && re.len() == im.len());
    for (i, (bre, bim)) in re
        .chunks_exact_mut(t)
        .zip(im.chunks_exact_mut(t))
        .enumerate()
    {
        let s_re = _mm256_set1_pd(gm[(m + i) << 1]);
        let s_im = _mm256_set1_pd(gm[((m + i) << 1) + 1]);
        let (x_re, y_re) = bre.split_at_mut(ht);
        let (x_im, y_im) = bim.split_at_mut(ht);
        for (((xr, xi), yr), yi) in x_re
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(x_im.as_chunks_mut::<4>().0)
            .zip(y_re.as_chunks_mut::<4>().0)
            .zip(y_im.as_chunks_mut::<4>().0)
        {
            let (a_re, a_im) = (ld4(xr), ld4(xi));
            let (z_re, z_im) = cmul(ld4(yr), ld4(yi), s_re, s_im);
            st4(xr, _mm256_add_pd(a_re, z_re));
            st4(xi, _mm256_add_pd(a_im, z_im));
            st4(yr, _mm256_sub_pd(a_re, z_re));
            st4(yi, _mm256_sub_pd(a_im, z_im));
        }
    }
}

/// One butterfly layer of `fft::ifft` on blocks of `dt = 2t` values (`t` a multiple of 4), with
/// the conjugated twiddle `(gm[2(hm + i)], -gm[2(hm + i) + 1])` for block `i` (`gm` = `GM`).
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
pub(crate) fn ifft_layer(re: &mut [f64], im: &mut [f64], t: usize, hm: usize, gm: &[f64]) {
    assert!(t.is_multiple_of(4) && re.len() == im.len());
    for (i, (bre, bim)) in re
        .chunks_exact_mut(2 * t)
        .zip(im.chunks_exact_mut(2 * t))
        .enumerate()
    {
        let s_re = _mm256_set1_pd(gm[(hm + i) << 1]);
        let s_im = _mm256_set1_pd(-gm[((hm + i) << 1) + 1]);
        let (x_re, y_re) = bre.split_at_mut(t);
        let (x_im, y_im) = bim.split_at_mut(t);
        for (((xr, xi), yr), yi) in x_re
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(x_im.as_chunks_mut::<4>().0)
            .zip(y_re.as_chunks_mut::<4>().0)
            .zip(y_im.as_chunks_mut::<4>().0)
        {
            let (a_re, a_im) = (ld4(xr), ld4(xi));
            let (b_re, b_im) = (ld4(yr), ld4(yi));
            st4(xr, _mm256_add_pd(a_re, b_re));
            st4(xi, _mm256_add_pd(a_im, b_im));
            let (d_re, d_im) = (_mm256_sub_pd(a_re, b_re), _mm256_sub_pd(a_im, b_im));
            let (z_re, z_im) = cmul(d_re, d_im, s_re, s_im);
            st4(yr, z_re);
            st4(yi, z_im);
        }
    }
}

/// `x <- x * e` on all values, for a length that is a multiple of 4.
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
pub(crate) fn scale(f: &mut [f64], e: f64) {
    assert!(f.len().is_multiple_of(4));
    let ev = _mm256_set1_pd(e);
    for x in f.as_chunks_mut::<4>().0 {
        st4(x, _mm256_mul_pd(ld4(x), ev));
    }
}
