//! NEON kernels of the FFT passes of SampPre, two values per instruction. Each performs, for
//! every output value, the IEEE operations of its portable counterpart in fft/mod.rs in the
//! same order, without FMA (`vmulq_f64` then `vaddq_f64` or `vsubq_f64`, never `vfmaq_f64`), so
//! the results are the same bits: the products and sums of a complex multiplication are those
//! of `flc_mul`, the negated twiddle of the split is `vnegq_f64` (a sign flip, as `-x` is), and
//! the halvings are products by 0.5.
//!
//! Lane layout: two consecutive indices `i` per register. `vld2q_f64` and `vst2q_f64` separate
//! the pairs `(f[2i], f[2i + 1])` of the split and merge into even and odd lanes and put them
//! back, and load the interleaved twiddles `(GM[2(hn + i)], GM[2(hn + i) + 1])` the same way.
use core::arch::aarch64::*;

/// Two values.
#[target_feature(enable = "neon")]
#[inline]
fn ld2v(s: &[f64; 2]) -> float64x2_t {
    // SAFETY: `s` is 16 bytes.
    unsafe { vld1q_f64(s.as_ptr()) }
}

/// Stores two values.
#[target_feature(enable = "neon")]
#[inline]
fn st2v(s: &mut [f64; 2], v: float64x2_t) {
    // SAFETY: `s` is 16 bytes.
    unsafe { vst1q_f64(s.as_mut_ptr(), v) }
}

/// Four consecutive values `[x0 x1 x2 x3]` as their even lanes `[x0 x2]` and odd lanes
/// `[x1 x3]`.
#[target_feature(enable = "neon")]
#[inline]
fn ld_pairs(s: &[f64; 4]) -> (float64x2_t, float64x2_t) {
    // SAFETY: `s` is 32 bytes.
    let v = unsafe { vld2q_f64(s.as_ptr()) };
    (v.0, v.1)
}

/// Stores even lanes `e` and odd lanes `o` as `[e0 o0 e1 o1]`.
#[target_feature(enable = "neon")]
#[inline]
fn st_pairs(s: &mut [f64; 4], e: float64x2_t, o: float64x2_t) {
    // SAFETY: `s` is 32 bytes.
    unsafe { vst2q_f64(s.as_mut_ptr(), float64x2x2_t(e, o)) }
}

/// `flc_mul` on two lanes: `(x_re * y_re - x_im * y_im, x_re * y_im + x_im * y_re)`.
#[target_feature(enable = "neon")]
#[inline]
fn cmul(
    x_re: float64x2_t,
    x_im: float64x2_t,
    y_re: float64x2_t,
    y_im: float64x2_t,
) -> (float64x2_t, float64x2_t) {
    (
        vsubq_f64(vmulq_f64(x_re, y_re), vmulq_f64(x_im, y_im)),
        vaddq_f64(vmulq_f64(x_re, y_im), vmulq_f64(x_im, y_re)),
    )
}

/// `fft::poly_split_fft` with `gm = GM[2 hn..3 hn]`, for `n/4` a multiple of 2.
#[target_feature(enable = "neon")]
pub(super) fn poly_split_fft(logn: u32, f0: &mut [f64], f1: &mut [f64], f: &[f64], gm: &[f64]) {
    let hn = 1usize << (logn - 1);
    let qn = hn >> 1;
    assert!(qn.is_multiple_of(2) && gm.len() >= hn);
    let half = vdupq_n_f64(0.5);
    let (f_re, f_im) = f[..2 * hn].split_at(hn);
    let (f0_re, f0_im) = f0[..hn].split_at_mut(qn);
    let (f1_re, f1_im) = f1[..hn].split_at_mut(qn);
    for (((((re, im), g), (o0r, o0i)), o1r), o1i) in f_re
        .as_chunks::<4>()
        .0
        .iter()
        .zip(f_im.as_chunks::<4>().0)
        .zip(gm[..hn].as_chunks::<4>().0)
        .zip(
            f0_re
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(f0_im.as_chunks_mut::<2>().0),
        )
        .zip(f1_re.as_chunks_mut::<2>().0)
        .zip(f1_im.as_chunks_mut::<2>().0)
    {
        let (g_re, g_im) = ld_pairs(g);
        let (a_re, b_re) = ld_pairs(re);
        let (a_im, b_im) = ld_pairs(im);
        st2v(o0r, vmulq_f64(vaddq_f64(a_re, b_re), half));
        st2v(o0i, vmulq_f64(vaddq_f64(a_im, b_im), half));
        let (t_re, t_im) = (vsubq_f64(a_re, b_re), vsubq_f64(a_im, b_im));
        let (u_re, u_im) = cmul(t_re, t_im, g_re, vnegq_f64(g_im));
        st2v(o1r, vmulq_f64(u_re, half));
        st2v(o1i, vmulq_f64(u_im, half));
    }
}

/// `fft::poly_merge_fft` with `gm = GM[2 hn..3 hn]`, for `n/4` a multiple of 2.
#[target_feature(enable = "neon")]
pub(super) fn poly_merge_fft(logn: u32, f: &mut [f64], f0: &[f64], f1: &[f64], gm: &[f64]) {
    let hn = 1usize << (logn - 1);
    let qn = hn >> 1;
    assert!(qn.is_multiple_of(2) && gm.len() >= hn);
    let (f_re, f_im) = f[..2 * hn].split_at_mut(hn);
    let (f0_re, f0_im) = f0[..hn].split_at(qn);
    let (f1_re, f1_im) = f1[..hn].split_at(qn);
    for (((((o_re, o_im), g), (a_re, a_im)), c_re), c_im) in f_re
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(f_im.as_chunks_mut::<4>().0)
        .zip(gm[..hn].as_chunks::<4>().0)
        .zip(
            f0_re
                .as_chunks::<2>()
                .0
                .iter()
                .zip(f0_im.as_chunks::<2>().0),
        )
        .zip(f1_re.as_chunks::<2>().0)
        .zip(f1_im.as_chunks::<2>().0)
    {
        let (g_re, g_im) = ld_pairs(g);
        let (a_re, a_im) = (ld2v(a_re), ld2v(a_im));
        let (b_re, b_im) = cmul(ld2v(c_re), ld2v(c_im), g_re, g_im);
        st_pairs(o_re, vaddq_f64(a_re, b_re), vsubq_f64(a_re, b_re));
        st_pairs(o_im, vaddq_f64(a_im, b_im), vsubq_f64(a_im, b_im));
    }
}

/// `fft::merge_sub_mul_add_split` with `gm = GM[2 hn..3 hn]`, for `n/4` a multiple of 2: the
/// merge of `w` into `t1` (whose old value is read first), `tb0 = t0 + (t1 - z1) * l10`, and the
/// split of `tb0` into `w`, with the operations of the portable pass on every value.
#[target_feature(enable = "neon")]
pub(super) fn merge_sub_mul_add_split(
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
    assert!(qn.is_multiple_of(2) && gm.len() >= hn);
    let half = vdupq_n_f64(0.5);
    let (w0, w1) = w[..n].split_at_mut(hn);
    let (w0_re, w0_im) = w0.split_at_mut(qn);
    let (w1_re, w1_im) = w1.split_at_mut(qn);
    let (t0_re, t0_im) = t0[..n].split_at(hn);
    let (t1_re, t1_im) = t1[..n].split_at_mut(hn);
    let (l_re, l_im) = l10[..n].split_at(hn);
    let inputs = t0_re
        .as_chunks::<4>()
        .0
        .iter()
        .zip(t0_im.as_chunks::<4>().0)
        .zip(l_re.as_chunks::<4>().0.iter().zip(l_im.as_chunks::<4>().0))
        .zip(gm[..hn].as_chunks::<4>().0);
    let targets = t1_re
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(t1_im.as_chunks_mut::<4>().0);
    let halves = w0_re
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(w0_im.as_chunks_mut::<2>().0)
        .zip(
            w1_re
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(w1_im.as_chunks_mut::<2>().0),
        );
    for ((((t0r, t0i), (lr, li)), g), ((t1r, t1i), ((w0r, w0i), (w1r, w1i)))) in
        inputs.zip(targets.zip(halves))
    {
        let (g_re, g_im) = ld_pairs(g);
        // poly_merge_fft: z1 at 2i (a + b) and 2i + 1 (a - b).
        let (a_re, a_im) = (ld2v(w0r), ld2v(w0i));
        let (b_re, b_im) = cmul(ld2v(w1r), ld2v(w1i), g_re, g_im);
        let z_re = [vaddq_f64(a_re, b_re), vsubq_f64(a_re, b_re)];
        let z_im = [vaddq_f64(a_im, b_im), vsubq_f64(a_im, b_im)];
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
        let mut tb_re = [vdupq_n_f64(0.0); 2];
        let mut tb_im = [vdupq_n_f64(0.0); 2];
        for k in 0..2 {
            let x_re = vsubq_f64(t1_re[k], z_re[k]);
            let x_im = vsubq_f64(t1_im[k], z_im[k]);
            let (y_re, y_im) = cmul(x_re, x_im, l_re[k], l_im[k]);
            tb_re[k] = vaddq_f64(t0_re[k], y_re);
            tb_im[k] = vaddq_f64(t0_im[k], y_im);
        }
        // z1 into t1.
        st_pairs(t1r, z_re[0], z_re[1]);
        st_pairs(t1i, z_im[0], z_im[1]);
        // poly_split_fft of tb0.
        let [a_re, b_re] = tb_re;
        let [a_im, b_im] = tb_im;
        st2v(w0r, vmulq_f64(vaddq_f64(a_re, b_re), half));
        st2v(w0i, vmulq_f64(vaddq_f64(a_im, b_im), half));
        let (t_re, t_im) = (vsubq_f64(a_re, b_re), vsubq_f64(a_im, b_im));
        let (u_re, u_im) = cmul(t_re, t_im, g_re, vnegq_f64(g_im));
        st2v(w1r, vmulq_f64(u_re, half));
        st2v(w1i, vmulq_f64(u_im, half));
    }
}

/// One butterfly layer of `fft::fft` on blocks of `t` values (half-blocks `ht = t/2`, a multiple
/// of 2), with the twiddle `(gm[2(m + i)], gm[2(m + i) + 1])` for block `i` (`gm` = `GM`).
#[target_feature(enable = "neon")]
pub(super) fn fft_layer(re: &mut [f64], im: &mut [f64], t: usize, m: usize, gm: &[f64]) {
    let ht = t >> 1;
    assert!(ht.is_multiple_of(2) && re.len() == im.len());
    for (i, (bre, bim)) in re
        .chunks_exact_mut(t)
        .zip(im.chunks_exact_mut(t))
        .enumerate()
    {
        let s_re = vdupq_n_f64(gm[(m + i) << 1]);
        let s_im = vdupq_n_f64(gm[((m + i) << 1) + 1]);
        let (x_re, y_re) = bre.split_at_mut(ht);
        let (x_im, y_im) = bim.split_at_mut(ht);
        for (((xr, xi), yr), yi) in x_re
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(x_im.as_chunks_mut::<2>().0)
            .zip(y_re.as_chunks_mut::<2>().0)
            .zip(y_im.as_chunks_mut::<2>().0)
        {
            let (a_re, a_im) = (ld2v(xr), ld2v(xi));
            let (z_re, z_im) = cmul(ld2v(yr), ld2v(yi), s_re, s_im);
            st2v(xr, vaddq_f64(a_re, z_re));
            st2v(xi, vaddq_f64(a_im, z_im));
            st2v(yr, vsubq_f64(a_re, z_re));
            st2v(yi, vsubq_f64(a_im, z_im));
        }
    }
}

/// One butterfly layer of `fft::ifft` on blocks of `dt = 2t` values (`t` a multiple of 2), with
/// the conjugated twiddle `(gm[2(hm + i)], -gm[2(hm + i) + 1])` for block `i` (`gm` = `GM`).
#[target_feature(enable = "neon")]
pub(super) fn ifft_layer(re: &mut [f64], im: &mut [f64], t: usize, hm: usize, gm: &[f64]) {
    assert!(t.is_multiple_of(2) && re.len() == im.len());
    for (i, (bre, bim)) in re
        .chunks_exact_mut(2 * t)
        .zip(im.chunks_exact_mut(2 * t))
        .enumerate()
    {
        let s_re = vdupq_n_f64(gm[(hm + i) << 1]);
        let s_im = vdupq_n_f64(-gm[((hm + i) << 1) + 1]);
        let (x_re, y_re) = bre.split_at_mut(t);
        let (x_im, y_im) = bim.split_at_mut(t);
        for (((xr, xi), yr), yi) in x_re
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(x_im.as_chunks_mut::<2>().0)
            .zip(y_re.as_chunks_mut::<2>().0)
            .zip(y_im.as_chunks_mut::<2>().0)
        {
            let (a_re, a_im) = (ld2v(xr), ld2v(xi));
            let (b_re, b_im) = (ld2v(yr), ld2v(yi));
            st2v(xr, vaddq_f64(a_re, b_re));
            st2v(xi, vaddq_f64(a_im, b_im));
            let (d_re, d_im) = (vsubq_f64(a_re, b_re), vsubq_f64(a_im, b_im));
            let (z_re, z_im) = cmul(d_re, d_im, s_re, s_im);
            st2v(yr, z_re);
            st2v(yi, z_im);
        }
    }
}
