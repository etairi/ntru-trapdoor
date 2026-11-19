//! Native-`f64` FFT over $`\mathbb R[x]/(x^n+1)`$ and the FFT-domain polynomial operations.
//!
//! Port of the portable (non-SIMD) branches of fn-dsa-sign/src/poly.rs (Pornin,
//! The Unlicense), commit 0629bb1, with `FLR` replaced by native `f64`:
//! `flc_mul` :34-38, `FFT` :52-231 (portable branch :196-230), `iFFT` :234-468 (:428-467),
//! `poly_set_small` :471-540, `poly_add` :543-596, `poly_sub` :599-652, `poly_neg` :655-703,
//! `poly_mul_fft` :718-805, `poly_muladj_fft` :809-896, `poly_mulownadj_fft` :901-969,
//! `poly_mulconst` :972-1023, `poly_invnorm2_fft` :1043-1052 and `poly_div_selfadj_fft`
//! :1088-1095 (both commented out upstream), `poly_LDL_fft` :1107-1223 (:1204-1222),
//! `poly_split_fft` :1250-1364, `poly_split_selfadj_fft` :1369-1461, `poly_merge_fft`
//! :1466-1558, and the twiddle table `GM` :1574-2602 (module `gm`, checked against fn-dsa's
//! `check_GM` digest, poly.rs:2694-2706). `FLR` helpers map to `f64`: `half` is `* 0.5`,
//! `square` is `x * x`, `slice_div2e(e)` multiplies by $`2^{-e}`$ exactly
//! (flr_native.rs:341-371).
//!
//! Representation (fn-dsa's): a polynomial of degree $`n=2^{\mathrm{logn}}`$ in FFT form is $`n`$
//! `f64` values, the real parts of the $`n/2`$ evaluations first, then their imaginary parts; a
//! self-adjoint polynomial uses only the first $`n/2`$. `logn >= 1`.
//!
//! Every function performs, for every output value, exactly fn-dsa's IEEE-754 operations in
//! fn-dsa's order (no FMA: Rust never contracts `a * b + c`; no reassociation). The loops are
//! written over slices instead of indices, which changes only the order in which independent
//! outputs are computed, never an output value. fn-dsa's NEON branches compute the same values
//! (they differ from the portable branch only by commuted IEEE additions and multiplications,
//! which are exact symmetries, and in the sign of some zero results), so the parity tests
//! against fn-dsa hold on aarch64 too.
//!
//! The fused passes of SampPre and ffSampling (`poly_target_fft`, `poly_lattice_point_fft`,
//! `merge_sub_mul_add_split`) run several of these functions in one loop, with each function's
//! operations on each value, so they are bit-identical to the sequence of calls (tests
//! `split_merge_and_fused_node_bit_exact` and `fused_target_and_lattice_point_bit_exact`).
//!
//! Also used by key generation (the Gram–Schmidt norms and the Babai reduction of NTRUSolve):
//! every `pub fn` below.

mod gm;

use gm::GM;

/// The twiddle `(GM[4], GM[5])` of the split and merge at `logn = 2`, for ffSampling's unrolled
/// `logn = 2` node.
pub(crate) const GM_LOGN2: (f64, f64) = (GM[4], GM[5]);

/// Defines a function and, with the `simd` feature on x86-64, its copy under the name after the
/// slash: the same body compiled for AVX2, BMI1, BMI2, LZCNT and POPCNT, without FMA. SampPre
/// runs through these copies on processors that have the instructions (sign.rs). Rust neither
/// contracts nor reassociates floating-point operations, so both compilations perform the same
/// IEEE operations in the same order and return the same bits.
macro_rules! with_v3_copy {
    (
        $(#[$m:meta])*
        $vis:vis fn $name:ident / $v3:ident ($($arg:ident: $ty:ty),* $(,)?) $body:block
    ) => {
        $(#[$m])*
        $vis fn $name($($arg: $ty),*) $body

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
        pub(crate) fn $v3($($arg: $ty),*) $body
    };
}

/// Complex multiplication (fn-dsa poly.rs:34-38).
#[inline(always)]
pub(crate) fn flc_mul(x_re: f64, x_im: f64, y_re: f64, y_im: f64) -> (f64, f64) {
    (x_re * y_re - x_im * y_im, x_re * y_im + x_im * y_re)
}

/// Normal representation to FFT representation (fn-dsa `FFT`, poly.rs:52-231, portable branch
/// :196-230).
pub fn fft(logn: u32, f: &mut [f64]) {
    assert!(logn >= 1);
    let n = 1usize << logn;
    let hn = n >> 1;
    let (re, im) = f[..n].split_at_mut(hn);
    // The first iteration of the textbook FFT is a no-op in this representation (poly.rs:53-57).
    let mut t = hn;
    for lm in 1..logn {
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        if crate::simd::fft_layer(re, im, t, 1usize << lm, &GM) {
            t >>= 1;
            continue;
        }
        fft_layer(re, im, t, 1usize << lm);
        t >>= 1;
    }
}

/// One layer of [`fft`]: the `m/2` blocks of `t` values, block `i` with the twiddle
/// `(GM[2(m + i)], GM[2(m + i) + 1])`.
#[inline(always)]
fn fft_layer(re: &mut [f64], im: &mut [f64], t: usize, m: usize) {
    let hm = m >> 1;
    let ht = t >> 1;
    for i in 0..hm {
        let j0 = i * t;
        let s_re = GM[(m + i) << 1];
        let s_im = GM[((m + i) << 1) + 1];
        let (x_re, y_re) = re[j0..j0 + t].split_at_mut(ht);
        let (x_im, y_im) = im[j0..j0 + t].split_at_mut(ht);
        for (((xr, xi), yr), yi) in x_re
            .iter_mut()
            .zip(x_im.iter_mut())
            .zip(y_re.iter_mut())
            .zip(y_im.iter_mut())
        {
            let (z_re, z_im) = flc_mul(*yr, *yi, s_re, s_im);
            let (a_re, a_im) = (*xr, *xi);
            *xr = a_re + z_re;
            *xi = a_im + z_im;
            *yr = a_re - z_re;
            *yi = a_im - z_im;
        }
    }
}

/// [`fft`] in the x86-64-v3 copy of SampPre: the AVX2 kernel on the layers whose half-blocks
/// hold at least 4 values, [`fft_layer`] on the others.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
pub(crate) fn fft_v3(logn: u32, f: &mut [f64]) {
    assert!(logn >= 1);
    let n = 1usize << logn;
    let hn = n >> 1;
    let (re, im) = f[..n].split_at_mut(hn);
    let mut t = hn;
    for lm in 1..logn {
        if t >= 8 {
            crate::simd::fft_avx2::fft_layer(re, im, t, 1usize << lm, &GM);
        } else {
            fft_layer(re, im, t, 1usize << lm);
        }
        t >>= 1;
    }
}

/// FFT representation to normal representation (fn-dsa `iFFT`, poly.rs:234-468, portable
/// branch :428-467).
pub fn ifft(logn: u32, f: &mut [f64]) {
    assert!(logn >= 1);
    let n = 1usize << logn;
    let hn = n >> 1;
    {
        let (re, im) = f[..n].split_at_mut(hn);
        let mut t = 1usize;
        for lm in 1..logn {
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            if crate::simd::ifft_layer(re, im, t, 1usize << (logn - lm), &GM) {
                t <<= 1;
                continue;
            }
            ifft_layer(re, im, t, 1usize << (logn - lm));
            t <<= 1;
        }
    }
    // logn - 1 delayed halvings: FLR::slice_div2e (flr_native.rs:361-366) multiplies by the
    // exact power of two 2^-(logn-1).
    let e = 1.0 / ((1u64 << (logn - 1)) as f64);
    for x in f[..n].iter_mut() {
        *x *= e;
    }
}

/// One layer of [`ifft`]: the `hm/2` blocks of `2t` values, block `i` with the conjugated
/// twiddle `(GM[2(hm + i)], -GM[2(hm + i) + 1])`.
#[inline(always)]
fn ifft_layer(re: &mut [f64], im: &mut [f64], t: usize, hm: usize) {
    let dt = t << 1;
    for i in 0..(hm >> 1) {
        let j0 = i * dt;
        // 1/w is the conjugate of w (poly.rs:235-238).
        let s_re = GM[(hm + i) << 1];
        let s_im = -GM[((hm + i) << 1) + 1];
        let (x_re, y_re) = re[j0..j0 + dt].split_at_mut(t);
        let (x_im, y_im) = im[j0..j0 + dt].split_at_mut(t);
        for (((xr, xi), yr), yi) in x_re
            .iter_mut()
            .zip(x_im.iter_mut())
            .zip(y_re.iter_mut())
            .zip(y_im.iter_mut())
        {
            let (a_re, a_im) = (*xr, *xi);
            let (b_re, b_im) = (*yr, *yi);
            *xr = a_re + b_re;
            *xi = a_im + b_im;
            let d_re = a_re - b_re;
            let d_im = a_im - b_im;
            let (z_re, z_im) = flc_mul(d_re, d_im, s_re, s_im);
            *yr = z_re;
            *yi = z_im;
        }
    }
}

/// [`ifft`] in the x86-64-v3 copy of SampPre: the AVX2 kernel on the layers whose half-blocks
/// hold at least 4 values and on the final scaling, [`ifft_layer`] on the others.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
pub(crate) fn ifft_v3(logn: u32, f: &mut [f64]) {
    assert!(logn >= 1);
    let n = 1usize << logn;
    let hn = n >> 1;
    {
        let (re, im) = f[..n].split_at_mut(hn);
        let mut t = 1usize;
        for lm in 1..logn {
            if t >= 4 {
                crate::simd::fft_avx2::ifft_layer(re, im, t, 1usize << (logn - lm), &GM);
            } else {
                ifft_layer(re, im, t, 1usize << (logn - lm));
            }
            t <<= 1;
        }
    }
    let e = 1.0 / ((1u64 << (logn - 1)) as f64);
    if n.is_multiple_of(4) {
        crate::simd::fft_avx2::scale(&mut f[..n], e);
    } else {
        for x in f[..n].iter_mut() {
            *x *= e;
        }
    }
}

/// `d <- f` as `f64` (normal representation), for small integer coefficients (fn-dsa
/// `poly_set_small`, poly.rs:471-540, with `i16` instead of `i8`).
pub fn poly_set_small(logn: u32, d: &mut [f64], f: &[i16]) {
    let n = 1usize << logn;
    for (x, &v) in d[..n].iter_mut().zip(f[..n].iter()) {
        *x = v as f64;
    }
}

/// `a <- a + b` (either representation; fn-dsa poly.rs:543-596).
pub fn poly_add(logn: u32, a: &mut [f64], b: &[f64]) {
    let n = 1usize << logn;
    for (x, &y) in a[..n].iter_mut().zip(b[..n].iter()) {
        *x += y;
    }
}

/// `a <- a - b` (either representation; fn-dsa poly.rs:599-652). ffSampling now computes this
/// step inside [`merge_sub_mul_add_split`]; the tests use it as the unfused reference.
#[allow(dead_code)] // callers: the tests
pub fn poly_sub(logn: u32, a: &mut [f64], b: &[f64]) {
    let n = 1usize << logn;
    for (x, &y) in a[..n].iter_mut().zip(b[..n].iter()) {
        *x -= y;
    }
}

/// `a <- -a` (either representation; fn-dsa poly.rs:655-703).
pub fn poly_neg(logn: u32, a: &mut [f64]) {
    let n = 1usize << logn;
    for x in a[..n].iter_mut() {
        *x = -*x;
    }
}

/// `a <- a * b` (FFT representation; fn-dsa poly.rs:718-805, portable branch :793-804).
#[allow(dead_code)] // callers: the tests (references of the fused passes)
pub fn poly_mul_fft(logn: u32, a: &mut [f64], b: &[f64]) {
    let hn = 1usize << (logn - 1);
    let (a_re, a_im) = a[..2 * hn].split_at_mut(hn);
    let (b_re, b_im) = b[..2 * hn].split_at(hn);
    for (((ar, ai), &br), &bi) in a_re
        .iter_mut()
        .zip(a_im.iter_mut())
        .zip(b_re.iter())
        .zip(b_im.iter())
    {
        let (re, im) = flc_mul(*ar, *ai, br, bi);
        *ar = re;
        *ai = im;
    }
}

/// `a <- a * adj(b)` (FFT representation; fn-dsa poly.rs:809-896, portable branch :884-895).
pub fn poly_muladj_fft(logn: u32, a: &mut [f64], b: &[f64]) {
    let hn = 1usize << (logn - 1);
    let (a_re, a_im) = a[..2 * hn].split_at_mut(hn);
    let (b_re, b_im) = b[..2 * hn].split_at(hn);
    for (((ar, ai), &br), &bi) in a_re
        .iter_mut()
        .zip(a_im.iter_mut())
        .zip(b_re.iter())
        .zip(b_im.iter())
    {
        let (re, im) = flc_mul(*ar, *ai, br, -bi);
        *ar = re;
        *ai = im;
    }
}

/// `a <- a * adj(a)` (FFT representation); the result is self-adjoint and the imaginary half
/// is set to zero (fn-dsa poly.rs:901-969, portable branch :958-968).
pub fn poly_mulownadj_fft(logn: u32, a: &mut [f64]) {
    let hn = 1usize << (logn - 1);
    let (a_re, a_im) = a[..2 * hn].split_at_mut(hn);
    for (ar, ai) in a_re.iter_mut().zip(a_im.iter_mut()) {
        *ar = *ar * *ar + *ai * *ai;
        *ai = 0.0;
    }
}

/// `a <- a * x` for a real constant `x` (either representation; fn-dsa poly.rs:972-1023).
#[allow(dead_code)] // callers: the tests (references of the fused passes)
pub fn poly_mulconst(logn: u32, a: &mut [f64], x: f64) {
    let n = 1usize << logn;
    for v in a[..n].iter_mut() {
        *v *= x;
    }
}

/// `a <- a / b` with `b` self-adjoint (FFT representation; only `b[..n/2]` is read). Used by
/// keygen (the Babai reduction of NTRUSolve). fn-dsa
/// poly.rs:1084-1096 (commented out upstream): one division `1/b`, then two products.
#[allow(dead_code)] // caller: keygen
pub fn poly_div_selfadj_fft(logn: u32, a: &mut [f64], b: &[f64]) {
    let hn = 1usize << (logn - 1);
    let (a_re, a_im) = a[..2 * hn].split_at_mut(hn);
    for ((ar, ai), &bv) in a_re.iter_mut().zip(a_im.iter_mut()).zip(b[..hn].iter()) {
        let x = 1.0 / bv;
        *ar *= x;
        *ai *= x;
    }
}

/// `d <- 1 / (f adj(f) + g adj(g))` (FFT representation; self-adjoint, so only `d[..n/2]` is
/// written). Used by keygen (the second Gram–Schmidt norm). fn-dsa poly.rs:1038-1053 (commented
/// out upstream).
#[allow(dead_code)] // caller: keygen
pub fn poly_invnorm2_fft(logn: u32, d: &mut [f64], f: &[f64], g: &[f64]) {
    let hn = 1usize << (logn - 1);
    for i in 0..hn {
        let nf = f[i] * f[i] + f[i + hn] * f[i + hn];
        let ng = g[i] * g[i] + g[i + hn] * g[i + hn];
        d[i] = 1.0 / (nf + ng);
    }
}

/// LDL decomposition of the self-adjoint $`2\times2`$ matrix
/// `[[g00, g01], [adj(g01), g11]]` in place: `g01 <- l10`, `g11 <- d11`; `g00` (= `d00`) is
/// unchanged (fn-dsa `poly_LDL_fft`, poly.rs:1107-1223, portable branch :1204-1222).
pub fn poly_ldl_fft(logn: u32, g00: &[f64], g01: &mut [f64], g11: &mut [f64]) {
    let hn = 1usize << (logn - 1);
    let (g01_re, g01_im) = g01[..2 * hn].split_at_mut(hn);
    for (((&g00_re, l_re), l_im), d11) in g00[..hn]
        .iter()
        .zip(g01_re.iter_mut())
        .zip(g01_im.iter_mut())
        .zip(g11[..hn].iter_mut())
    {
        // g00 and g11 are self-adjoint.
        let (x_re, x_im) = (*l_re, *l_im);
        let inv_g00_re = 1.0 / g00_re;
        let (mu_re, mu_im) = (x_re * inv_g00_re, x_im * inv_g00_re);
        let zo_re = mu_re * x_re + mu_im * x_im;
        *d11 -= zo_re;
        *l_re = mu_re;
        *l_im = -mu_im;
    }
}

/// Split `f = f0(x^2) + x f1(x^2)` (FFT representation, half-size outputs; fn-dsa
/// `poly_split_fft`, poly.rs:1250-1364, portable branch :1342-1363).
pub fn poly_split_fft(logn: u32, f0: &mut [f64], f1: &mut [f64], f: &[f64]) {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if logn >= 3 && crate::simd::fft_split(logn, f0, f1, f, &GM[1 << logn..3 << (logn - 1)]) {
        return;
    }
    // If logn = 1 then the loop is entirely skipped.
    if logn == 1 {
        f0[0] = f[0];
        f1[0] = f[1];
        return;
    }
    let hn = 1usize << (logn - 1);
    let qn = hn >> 1;
    // The pairs (f[2i], f[2i + 1]) of each half, and the twiddles GM[2(i + hn)], read through
    // chunks of two: the same operations on the same values as fn-dsa's indexed loop.
    let (f_re, f_im) = f[..2 * hn].split_at(hn);
    let (f0_re, f0_im) = f0[..hn].split_at_mut(qn);
    let (f1_re, f1_im) = f1[..hn].split_at_mut(qn);
    let (re2, _) = f_re.as_chunks::<2>();
    let (im2, _) = f_im.as_chunks::<2>();
    let (gm2, _) = GM[2 * hn..3 * hn].as_chunks::<2>();
    for i in 0..qn {
        let [a_re, b_re] = re2[i];
        let [a_im, b_im] = im2[i];

        let (t_re, t_im) = (a_re + b_re, a_im + b_im);
        f0_re[i] = t_re * 0.5;
        f0_im[i] = t_im * 0.5;

        let (t_re, t_im) = (a_re - b_re, a_im - b_im);
        let (u_re, u_im) = flc_mul(t_re, t_im, gm2[i][0], -gm2[i][1]);
        f1_re[i] = u_re * 0.5;
        f1_im[i] = u_im * 0.5;
    }
}

/// [`poly_split_fft`] for a self-adjoint `f` (only `f[..n/2]` is read; `f0` is self-adjoint and
/// its imaginary half is set to zero). fn-dsa `poly_split_selfadj_fft`, poly.rs:1369-1461,
/// portable branch :1441-1460.
pub fn poly_split_selfadj_fft(logn: u32, f0: &mut [f64], f1: &mut [f64], f: &[f64]) {
    // If logn = 1 then the loop is entirely skipped.
    if logn == 1 {
        f0[0] = f[0];
        f1[0] = 0.0;
        return;
    }
    let hn = 1usize << (logn - 1);
    let qn = hn >> 1;
    for i in 0..qn {
        let a_re = f[i << 1];
        let b_re = f[(i << 1) + 1];

        let t_re = a_re + b_re;
        f0[i] = t_re * 0.5;
        f0[i + qn] = 0.0;

        let t_re = (a_re - b_re) * 0.5;
        f1[i] = t_re * GM[(i + hn) << 1];
        f1[i + qn] = t_re * -GM[((i + hn) << 1) + 1];
    }
}

/// Merge `f <- f0(x^2) + x f1(x^2)` (FFT representation; fn-dsa `poly_merge_fft`,
/// poly.rs:1466-1558, portable branch :1541-1557).
pub fn poly_merge_fft(logn: u32, f: &mut [f64], f0: &[f64], f1: &[f64]) {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if logn >= 3 && crate::simd::fft_merge(logn, f, f0, f1, &GM[1 << logn..3 << (logn - 1)]) {
        return;
    }
    // If logn = 1 then the loop is entirely skipped.
    if logn == 1 {
        f[0] = f0[0];
        f[1] = f1[0];
        return;
    }
    let hn = 1usize << (logn - 1);
    let qn = hn >> 1;
    // As in poly_split_fft: pairs of each half and the twiddles through chunks of two.
    let (f_re, f_im) = f[..2 * hn].split_at_mut(hn);
    let (f0_re, f0_im) = f0[..hn].split_at(qn);
    let (f1_re, f1_im) = f1[..hn].split_at(qn);
    let (re2, _) = f_re.as_chunks_mut::<2>();
    let (im2, _) = f_im.as_chunks_mut::<2>();
    let (gm2, _) = GM[2 * hn..3 * hn].as_chunks::<2>();
    for i in 0..qn {
        let (a_re, a_im) = (f0_re[i], f0_im[i]);
        let (b_re, b_im) = flc_mul(f1_re[i], f1_im[i], gm2[i][0], gm2[i][1]);
        re2[i] = [a_re + b_re, a_re - b_re];
        im2[i] = [a_im + b_im, a_im - b_im];
    }
}

with_v3_copy! {
/// SampPre's target in one pass (fn-dsa lib.rs:621-640): from `t0` = FFT(t), `t1 <- t0 * b01 *
/// (-inv_q)` and `t0 <- t0 * b11 * inv_q`. The same operations per value as `poly_mul_fft`
/// followed by `poly_mulconst`, for each of the two products.
pub(crate) fn poly_target_fft / poly_target_fft_v3 (
    logn: u32,
    t0: &mut [f64],
    t1: &mut [f64],
    b01: &[f64],
    b11: &[f64],
    inv_q: f64,
) {
    let hn = 1usize << (logn - 1);
    let ninv_q = -inv_q;
    let (t0_re, t0_im) = t0[..2 * hn].split_at_mut(hn);
    let (t1_re, t1_im) = t1[..2 * hn].split_at_mut(hn);
    let (b01_re, b01_im) = b01[..2 * hn].split_at(hn);
    let (b11_re, b11_im) = b11[..2 * hn].split_at(hn);
    for j in 0..hn {
        let (x_re, x_im) = (t0_re[j], t0_im[j]);
        let (y_re, y_im) = flc_mul(x_re, x_im, b01_re[j], b01_im[j]);
        t1_re[j] = y_re * ninv_q;
        t1_im[j] = y_im * ninv_q;
        let (y_re, y_im) = flc_mul(x_re, x_im, b11_re[j], b11_im[j]);
        t0_re[j] = y_re * inv_q;
        t0_im[j] = y_im * inv_q;
    }
}
}

with_v3_copy! {
/// SampPre's lattice point in one pass (fn-dsa lib.rs:673-689): `(z0, z1)` in `(t0, t1)` becomes
/// `v0 = z0 b00 + z1 b10` in `t0` and `v1 = z1 b11 + z0 b01` in `t1`, with the operations of
/// `poly_mul_fft` and `poly_add` and the operand order of fn-dsa's sequence.
pub(crate) fn poly_lattice_point_fft / poly_lattice_point_fft_v3 (
    logn: u32,
    t0: &mut [f64],
    t1: &mut [f64],
    basis: (&[f64], &[f64], &[f64], &[f64]),
) {
    let hn = 1usize << (logn - 1);
    let (b00, b01, b10, b11) = basis;
    let (t0_re, t0_im) = t0[..2 * hn].split_at_mut(hn);
    let (t1_re, t1_im) = t1[..2 * hn].split_at_mut(hn);
    let (b00_re, b00_im) = b00[..2 * hn].split_at(hn);
    let (b01_re, b01_im) = b01[..2 * hn].split_at(hn);
    let (b10_re, b10_im) = b10[..2 * hn].split_at(hn);
    let (b11_re, b11_im) = b11[..2 * hn].split_at(hn);
    for j in 0..hn {
        let (z0_re, z0_im) = (t0_re[j], t0_im[j]);
        let (z1_re, z1_im) = (t1_re[j], t1_im[j]);
        let (a_re, a_im) = flc_mul(z0_re, z0_im, b00_re[j], b00_im[j]);
        let (b_re, b_im) = flc_mul(z1_re, z1_im, b10_re[j], b10_im[j]);
        t0_re[j] = a_re + b_re;
        t0_im[j] = a_im + b_im;
        let (c_re, c_im) = flc_mul(z0_re, z0_im, b01_re[j], b01_im[j]);
        let (d_re, d_im) = flc_mul(z1_re, z1_im, b11_re[j], b11_im[j]);
        t1_re[j] = d_re + c_re;
        t1_im[j] = d_im + c_im;
    }
}
}

/// One node of ffSampling between its two recursive calls, fused into a single pass for
/// `logn >= 2`: merge `z1` from the right subtree's output `(w0, w1)` into `t1` (whose old
/// value is read first), form `tb0 = t0 + (t1 - z1) * l10`, and split `tb0` into `(w0, w1)` for
/// the left subtree. `t0` is only read (the caller overwrites it with the final merge).
///
/// Every value is computed by the operations, in the order, of the unfused sequence
/// `poly_merge_fft(z1, w0, w1)`, `poly_sub(t1 - z1)`, `poly_mul_fft(_, l10)`, `poly_add(t0, _)`,
/// `poly_split_fft(w0, w1, tb0)`, so the results are bit-identical. `w` holds `w0` then `w1`
/// (`n/2` values each).
pub(crate) fn merge_sub_mul_add_split(
    logn: u32,
    w: &mut [f64],
    t0: &[f64],
    t1: &mut [f64],
    l10: &[f64],
) {
    debug_assert!(logn >= 2);
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if logn >= 3 && crate::simd::fft_fused(logn, w, t0, t1, l10, &GM[1 << logn..3 << (logn - 1)]) {
        return;
    }
    let n = 1usize << logn;
    let hn = n >> 1;
    let qn = hn >> 1;
    let (w0, w1) = w[..n].split_at_mut(hn);
    let (w0_re, w0_im) = w0.split_at_mut(qn);
    let (w1_re, w1_im) = w1.split_at_mut(qn);
    let (t0_re, t0_im) = t0[..n].split_at(hn);
    let (t1_re, t1_im) = t1[..n].split_at_mut(hn);
    let (l_re, l_im) = l10[..n].split_at(hn);
    let (t0_re2, _) = t0_re.as_chunks::<2>();
    let (t0_im2, _) = t0_im.as_chunks::<2>();
    let (t1_re2, _) = t1_re.as_chunks_mut::<2>();
    let (t1_im2, _) = t1_im.as_chunks_mut::<2>();
    let (l_re2, _) = l_re.as_chunks::<2>();
    let (l_im2, _) = l_im.as_chunks::<2>();
    let (gm2, _) = GM[2 * hn..3 * hn].as_chunks::<2>();
    for i in 0..qn {
        let [g_re, g_im] = gm2[i];
        // poly_merge_fft: z1 at 2i (a + b) and 2i + 1 (a - b).
        let (a_re, a_im) = (w0_re[i], w0_im[i]);
        let (b_re, b_im) = flc_mul(w1_re[i], w1_im[i], g_re, g_im);
        let z_re = [a_re + b_re, a_re - b_re];
        let z_im = [a_im + b_im, a_im - b_im];
        // poly_sub, poly_mul_fft and poly_add at 2i and 2i + 1; z1 into t1.
        let mut tb_re = [0.0f64; 2];
        let mut tb_im = [0.0f64; 2];
        for k in 0..2 {
            let x_re = t1_re2[i][k] - z_re[k];
            let x_im = t1_im2[i][k] - z_im[k];
            let (y_re, y_im) = flc_mul(x_re, x_im, l_re2[i][k], l_im2[i][k]);
            tb_re[k] = t0_re2[i][k] + y_re;
            tb_im[k] = t0_im2[i][k] + y_im;
        }
        t1_re2[i] = z_re;
        t1_im2[i] = z_im;
        // poly_split_fft of tb0 at i.
        let [a_re, b_re] = tb_re;
        let [a_im, b_im] = tb_im;
        let (t_re, t_im) = (a_re + b_re, a_im + b_im);
        w0_re[i] = t_re * 0.5;
        w0_im[i] = t_im * 0.5;
        let (t_re, t_im) = (a_re - b_re, a_im - b_im);
        let (u_re, u_im) = flc_mul(t_re, t_im, g_re, -g_im);
        w1_re[i] = u_re * 0.5;
        w1_im[i] = u_im * 0.5;
    }
}

/// [`poly_split_fft`] in the x86-64-v3 copy of SampPre: the AVX2 kernel from `logn = 4` on.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
pub(crate) fn poly_split_fft_v3(logn: u32, f0: &mut [f64], f1: &mut [f64], f: &[f64]) {
    if logn >= 4 {
        let hn = 1usize << (logn - 1);
        crate::simd::fft_avx2::poly_split_fft(logn, f0, f1, f, &GM[2 * hn..3 * hn]);
    } else {
        poly_split_fft(logn, f0, f1, f);
    }
}

/// [`poly_merge_fft`] in the x86-64-v3 copy of SampPre: the AVX2 kernel from `logn = 4` on.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
pub(crate) fn poly_merge_fft_v3(logn: u32, f: &mut [f64], f0: &[f64], f1: &[f64]) {
    if logn >= 4 {
        let hn = 1usize << (logn - 1);
        crate::simd::fft_avx2::poly_merge_fft(logn, f, f0, f1, &GM[2 * hn..3 * hn]);
    } else {
        poly_merge_fft(logn, f, f0, f1);
    }
}

/// [`merge_sub_mul_add_split`] in the x86-64-v3 copy of SampPre: the AVX2 kernel from
/// `logn = 4` on.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
pub(crate) fn merge_sub_mul_add_split_v3(
    logn: u32,
    w: &mut [f64],
    t0: &[f64],
    t1: &mut [f64],
    l10: &[f64],
) {
    if logn >= 4 {
        let hn = 1usize << (logn - 1);
        crate::simd::fft_avx2::merge_sub_mul_add_split(logn, w, t0, t1, l10, &GM[2 * hn..3 * hn]);
    } else {
        merge_sub_mul_add_split(logn, w, t0, t1, l10);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prng::{Prng, Shake256Prng, shake256};

    /// fn-dsa `check_GM` (poly.rs:2694-2706): SHAKE256 of the table's little-endian encoding.
    #[test]
    fn gm_digest() {
        let mut enc = Vec::with_capacity(2048 * 8);
        for x in GM.iter() {
            enc.extend_from_slice(&x.to_le_bytes());
        }
        let mut d = [0u8; 32];
        shake256(&[&enc], &mut d);
        assert_eq!(
            hex::encode(d),
            "f45a496cf56ccc6e3e3395a20209206d81d71a7905a661447bd5bc0e24e0af1e"
        );
    }

    fn rand_poly(rng: &mut Shake256Prng, f: &mut [f64]) {
        for x in f.iter_mut() {
            *x = (((rng.next_u16() & 0x3FF) as i64) - 512) as f64;
        }
    }

    /// Port of fn-dsa's `poly` test (poly.rs:2604-2690), extended to logn = 1: FFT round trip,
    /// FFT multiplication against the exact negacyclic product, split and merge.
    fn poly_inner(logn: u32) {
        let n = 1usize << logn;
        let hn = n >> 1;
        let mut rng = Shake256Prng::new(&[logn as u8]);
        let mut f = vec![0.0; n];
        let mut g = vec![0.0; n];
        let mut h = vec![0.0; n];
        let mut f0 = vec![0.0; hn];
        let mut f1 = vec![0.0; hn];
        let mut g0 = vec![0.0; hn];
        let mut g1 = vec![0.0; hn];
        for ctr in 0..(1u32 << (15 - logn)) {
            rand_poly(&mut rng, &mut f);
            g.copy_from_slice(&f);
            fft(logn, &mut g);
            ifft(logn, &mut g);
            for i in 0..n {
                assert_eq!(f[i].round_ties_even(), g[i].round_ties_even());
                assert!((f[i] - g[i]).abs() < 1e-9);
            }

            if ctr < 5 {
                rand_poly(&mut rng, &mut g);
                for x in h.iter_mut() {
                    *x = 0.0;
                }
                for i in 0..n {
                    for j in 0..n {
                        let s = f[i] * g[j];
                        if i + j < n {
                            h[i + j] += s;
                        } else {
                            h[i + j - n] -= s;
                        }
                    }
                }
                fft(logn, &mut f);
                fft(logn, &mut g);
                poly_mul_fft(logn, &mut f, &g);
                ifft(logn, &mut f);
                for i in 0..n {
                    assert_eq!(f[i].round_ties_even(), h[i].round_ties_even());
                }
            }

            if logn >= 2 {
                rand_poly(&mut rng, &mut f);
                h.copy_from_slice(&f);
                fft(logn, &mut f);
                poly_split_fft(logn, &mut f0, &mut f1, &f);
                g0.copy_from_slice(&f0);
                g1.copy_from_slice(&f1);
                ifft(logn - 1, &mut g0);
                ifft(logn - 1, &mut g1);
                for i in 0..hn {
                    assert_eq!(g0[i].round_ties_even(), h[2 * i].round_ties_even());
                    assert_eq!(g1[i].round_ties_even(), h[2 * i + 1].round_ties_even());
                }
                poly_merge_fft(logn, &mut g, &f0, &f1);
                ifft(logn, &mut g);
                for i in 0..n {
                    assert_eq!(g[i].round_ties_even(), h[i].round_ties_even());
                }
            }
        }
    }

    #[test]
    fn poly_fn_dsa_test() {
        for logn in 1..11 {
            poly_inner(logn);
        }
    }

    /// At logn = 1 the FFT is the identity on (re, im) = (f0, f1), and split/merge are trivial.
    #[test]
    fn logn1_conventions() {
        let mut f = [3.0, -5.0];
        fft(1, &mut f);
        assert_eq!(f, [3.0, -5.0]);
        ifft(1, &mut f);
        assert_eq!(f, [3.0, -5.0]);
        let (mut a, mut b) = ([0.0], [0.0]);
        poly_split_fft(1, &mut a, &mut b, &f);
        assert_eq!((a[0], b[0]), (3.0, -5.0));
        let mut g = [0.0; 2];
        poly_merge_fft(1, &mut g, &a, &b);
        assert_eq!(g, f);
    }

    /// The FFT evaluates f at the roots exp(i*pi*(2j+1)/n) (in fn-dsa's bit-reversed order):
    /// check |f(w)|^2 sums (Parseval) and the FFT of x.
    #[test]
    fn fft_of_monomial_and_parseval() {
        for logn in 1..=10u32 {
            let n = 1usize << logn;
            let hn = n / 2;
            // FFT of x: the stored values are one root of each conjugate pair of the n roots
            // exp(i pi (2j+1)/n) of x^n + 1, so their |arguments| are pi (2j+1)/n, j < n/2.
            let mut f = vec![0.0; n];
            f[1] = 1.0;
            fft(logn, &mut f);
            let mut angles: Vec<f64> = (0..hn).map(|j| f[j + hn].atan2(f[j]).abs()).collect();
            angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (j, a) in angles.iter().enumerate() {
                let want = core::f64::consts::PI * (2 * j + 1) as f64 / n as f64;
                assert!((a - want).abs() < 1e-12, "logn {logn} j {j}: {a} vs {want}");
            }
            // Parseval: sum_i f_i^2 = (2/n) sum_j |F_j|^2.
            let mut rng = Shake256Prng::new(&[0xAA, logn as u8]);
            let mut g = vec![0.0; n];
            rand_poly(&mut rng, &mut g);
            let e1: f64 = g.iter().map(|x| x * x).sum();
            fft(logn, &mut g);
            let e2: f64 = (0..hn)
                .map(|j| g[j] * g[j] + g[j + hn] * g[j + hn])
                .sum::<f64>()
                * 2.0
                / n as f64;
            assert!((e1 - e2).abs() <= 1e-9 * e1, "logn {logn}: {e1} vs {e2}");
        }
    }

    /// muladj, mulownadj, invnorm2, div_selfadj, ldl, split_selfadj against their definitions.
    #[test]
    fn fft_domain_operations() {
        for logn in 1..=10u32 {
            let n = 1usize << logn;
            let hn = n / 2;
            let mut rng = Shake256Prng::new(&[0xBB, logn as u8]);
            let mut a = vec![0.0; n];
            let mut b = vec![0.0; n];
            let mut c = vec![0.0; n];
            rand_poly(&mut rng, &mut a);
            rand_poly(&mut rng, &mut b);
            rand_poly(&mut rng, &mut c);
            fft(logn, &mut a);
            fft(logn, &mut b);
            fft(logn, &mut c);
            let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * (1.0 + x.abs().max(y.abs()));

            // a * adj(b) = a * conj(b) pointwise.
            let mut t = a.clone();
            poly_muladj_fft(logn, &mut t, &b);
            for j in 0..hn {
                let re = a[j] * b[j] + a[j + hn] * b[j + hn];
                let im = a[j + hn] * b[j] - a[j] * b[j + hn];
                assert!(close(t[j], re) && close(t[j + hn], im));
            }
            // a * adj(a) = |a|^2, imaginary half zero.
            let mut t = a.clone();
            poly_mulownadj_fft(logn, &mut t);
            for j in 0..hn {
                assert!(close(t[j], a[j] * a[j] + a[j + hn] * a[j + hn]));
                assert_eq!(t[j + hn], 0.0);
            }
            // invnorm2.
            let mut d = vec![0.0; hn];
            poly_invnorm2_fft(logn, &mut d, &a, &b);
            for j in 0..hn {
                let nn = a[j] * a[j] + a[j + hn] * a[j + hn] + b[j] * b[j] + b[j + hn] * b[j + hn];
                assert!(close(d[j] * nn, 1.0));
            }
            // div_selfadj: (a / s) * s = a for s = |b|^2 + 1.
            let mut s = b.clone();
            poly_mulownadj_fft(logn, &mut s);
            for v in s[..hn].iter_mut() {
                *v += 1.0;
            }
            let mut t = a.clone();
            poly_div_selfadj_fft(logn, &mut t, &s);
            for j in 0..hn {
                assert!(close(t[j] * s[j], a[j]) && close(t[j + hn] * s[j], a[j + hn]));
            }
            // LDL of the Gram matrix of [[a, b], [c, a]]: G = L D adj(L).
            let mut g00 = a.clone();
            poly_mulownadj_fft(logn, &mut g00);
            let mut t = b.clone();
            poly_mulownadj_fft(logn, &mut t);
            poly_add(logn, &mut g00, &t);
            let mut g01 = a.clone();
            poly_muladj_fft(logn, &mut g01, &c);
            let mut t = b.clone();
            poly_muladj_fft(logn, &mut t, &a);
            poly_add(logn, &mut g01, &t);
            let mut g11 = c.clone();
            poly_mulownadj_fft(logn, &mut g11);
            let mut t = a.clone();
            poly_mulownadj_fft(logn, &mut t);
            poly_add(logn, &mut g11, &t);
            let (o01, o11) = (g01.clone(), g11.clone());
            poly_ldl_fft(logn, &g00, &mut g01, &mut g11);
            for j in 0..hn {
                // g01 = g00 * adj(l10); g11 = d11 + |l10|^2 g00.
                let (l_re, l_im) = (g01[j], g01[j + hn]);
                assert!(close(o01[j], g00[j] * l_re) && close(o01[j + hn], -g00[j] * l_im));
                assert!(close(o11[j], g11[j] + (l_re * l_re + l_im * l_im) * g00[j]));
                assert!(g11[j] > 0.0);
            }
            // split_selfadj agrees with split on a self-adjoint input.
            if logn >= 1 {
                let mut full = g00.clone();
                for v in full[hn..].iter_mut() {
                    *v = 0.0;
                }
                let (mut x0, mut x1) = (vec![0.0; hn.max(1)], vec![0.0; hn.max(1)]);
                let (mut y0, mut y1) = (vec![0.0; hn.max(1)], vec![0.0; hn.max(1)]);
                poly_split_fft(logn, &mut x0, &mut x1, &full);
                poly_split_selfadj_fft(logn, &mut y0, &mut y1, &full);
                for j in 0..hn {
                    assert!(close(x0[j], y0[j]) && close(x1[j], y1[j]));
                }
            }
        }
    }

    /// fn-dsa's indexed `poly_split_fft` loop (poly.rs:1342-1363), verbatim: the reference of
    /// the chunked rewrite.
    fn split_fndsa(logn: u32, f0: &mut [f64], f1: &mut [f64], f: &[f64]) {
        if logn == 1 {
            f0[0] = f[0];
            f1[0] = f[1];
            return;
        }
        let hn = 1usize << (logn - 1);
        let qn = hn >> 1;
        for i in 0..qn {
            let (a_re, a_im) = (f[i << 1], f[(i << 1) + hn]);
            let (b_re, b_im) = (f[(i << 1) + 1], f[(i << 1) + 1 + hn]);
            let (t_re, t_im) = (a_re + b_re, a_im + b_im);
            f0[i] = t_re * 0.5;
            f0[i + qn] = t_im * 0.5;
            let (t_re, t_im) = (a_re - b_re, a_im - b_im);
            let (u_re, u_im) = flc_mul(t_re, t_im, GM[(i + hn) << 1], -GM[((i + hn) << 1) + 1]);
            f1[i] = u_re * 0.5;
            f1[i + qn] = u_im * 0.5;
        }
    }

    /// fn-dsa's indexed `poly_merge_fft` loop (poly.rs:1541-1557), verbatim.
    fn merge_fndsa(logn: u32, f: &mut [f64], f0: &[f64], f1: &[f64]) {
        if logn == 1 {
            f[0] = f0[0];
            f[1] = f1[0];
            return;
        }
        let hn = 1usize << (logn - 1);
        let qn = hn >> 1;
        for i in 0..qn {
            let (a_re, a_im) = (f0[i], f0[i + qn]);
            let (b_re, b_im) = flc_mul(
                f1[i],
                f1[i + qn],
                GM[(i + hn) << 1],
                GM[((i + hn) << 1) + 1],
            );
            f[i << 1] = a_re + b_re;
            f[(i << 1) + hn] = a_im + b_im;
            f[(i << 1) + 1] = a_re - b_re;
            f[(i << 1) + 1 + hn] = a_im - b_im;
        }
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// fn-dsa's indexed FFT loop (poly.rs:196-230, portable branch), verbatim: the reference of
    /// the layer functions.
    fn fft_fndsa(logn: u32, f: &mut [f64]) {
        let n = 1usize << logn;
        let hn = n >> 1;
        let mut t = hn;
        for lm in 1..logn {
            let m = 1usize << lm;
            let hm = m >> 1;
            let ht = t >> 1;
            let mut j0 = 0;
            for i in 0..hm {
                let s_re = GM[(m + i) << 1];
                let s_im = GM[((m + i) << 1) + 1];
                for j in 0..ht {
                    let j1 = j0 + j;
                    let j2 = j1 + ht;
                    let (x_re, x_im) = (f[j1], f[j1 + hn]);
                    let (y_re, y_im) = (f[j2], f[j2 + hn]);
                    let (z_re, z_im) = flc_mul(y_re, y_im, s_re, s_im);
                    f[j1] = x_re + z_re;
                    f[j1 + hn] = x_im + z_im;
                    f[j2] = x_re - z_re;
                    f[j2 + hn] = x_im - z_im;
                }
                j0 += t;
            }
            t = ht;
        }
    }

    /// fn-dsa's indexed inverse FFT loop (poly.rs:428-467, portable branch), verbatim.
    fn ifft_fndsa(logn: u32, f: &mut [f64]) {
        let n = 1usize << logn;
        let hn = n >> 1;
        let mut t = 1usize;
        for lm in 1..logn {
            let hm = 1usize << (logn - lm);
            let dt = t << 1;
            let mut j0 = 0;
            for i in 0..(hm >> 1) {
                let s_re = GM[(hm + i) << 1];
                let s_im = -GM[((hm + i) << 1) + 1];
                for j in 0..t {
                    let j1 = j0 + j;
                    let j2 = j1 + t;
                    let (x_re, x_im) = (f[j1], f[j1 + hn]);
                    let (y_re, y_im) = (f[j2], f[j2 + hn]);
                    f[j1] = x_re + y_re;
                    f[j1 + hn] = x_im + y_im;
                    let (z_re, z_im) = flc_mul(x_re - y_re, x_im - y_im, s_re, s_im);
                    f[j2] = z_re;
                    f[j2 + hn] = z_im;
                }
                j0 += dt;
            }
            t = dt;
        }
        let e = 1.0 / ((1u64 << (logn - 1)) as f64);
        for x in f[..n].iter_mut() {
            *x *= e;
        }
    }

    /// The FFT and the inverse FFT (by blocks and lanes, and through the NEON kernels with the
    /// `simd` feature on aarch64) equal fn-dsa's indexed loops bit for bit, at every size, on
    /// random values and on signed zeros.
    #[test]
    fn fft_layers_bit_exact() {
        let mut rng = Shake256Prng::new(b"fft-layers");
        for logn in 1..=10u32 {
            let n = 1usize << logn;
            for round in 0..8 {
                let mut f = vec![0.0; n];
                rand_poly(&mut rng, &mut f);
                if round == 0 {
                    for (i, x) in f.iter_mut().enumerate() {
                        if i % 5 == 0 {
                            *x = if i % 2 == 0 { 0.0 } else { -0.0 };
                        }
                    }
                }
                let (mut a, mut b) = (f.clone(), f.clone());
                fft(logn, &mut a);
                fft_fndsa(logn, &mut b);
                assert_eq!(bits(&a), bits(&b), "fft {logn}");
                let (mut c, mut d) = (a.clone(), a.clone());
                ifft(logn, &mut c);
                ifft_fndsa(logn, &mut d);
                assert_eq!(bits(&c), bits(&d), "ifft {logn}");
            }
        }
    }

    /// The chunked split and merge, and the fused node pass of ffSampling, equal their unfused
    /// fn-dsa sequences bit for bit at every size (FFT-domain inputs, so non-integer values).
    #[test]
    fn split_merge_and_fused_node_bit_exact() {
        for logn in 1..=10u32 {
            let n = 1usize << logn;
            let hn = n / 2;
            let mut rng = Shake256Prng::new(&[0xCC, logn as u8]);
            for _ in 0..4 {
                let mut f = vec![0.0; n];
                rand_poly(&mut rng, &mut f);
                fft(logn, &mut f);
                let (mut a0, mut a1) = (vec![0.0; hn], vec![0.0; hn]);
                let (mut b0, mut b1) = (vec![0.0; hn], vec![0.0; hn]);
                poly_split_fft(logn, &mut a0, &mut a1, &f);
                split_fndsa(logn, &mut b0, &mut b1, &f);
                assert_eq!(
                    (bits(&a0), bits(&a1)),
                    (bits(&b0), bits(&b1)),
                    "split {logn}"
                );
                let (mut m, mut mr) = (vec![0.0; n], vec![0.0; n]);
                poly_merge_fft(logn, &mut m, &a0, &a1);
                merge_fndsa(logn, &mut mr, &a0, &a1);
                assert_eq!(bits(&m), bits(&mr), "merge {logn}");
                if logn < 2 {
                    continue;
                }
                // The node pass: w = (w0, w1) from the right subtree, targets t0, t1, l10.
                let mut t0 = vec![0.0; n];
                let mut t1 = vec![0.0; n];
                let mut l10 = vec![0.0; n];
                let mut w = vec![0.0; n];
                for v in [&mut t0, &mut t1, &mut l10, &mut w] {
                    rand_poly(&mut rng, v);
                    fft(logn, v);
                }
                for x in l10.iter_mut() {
                    *x /= 1024.0 + 0.5;
                }
                // Unfused: z1 = merge(w0, w1); x = t1 - z1; t1 = z1; x *= l10; t0 += x;
                // (w0, w1) = split(t0).
                let mut z1 = vec![0.0; n];
                merge_fndsa(logn, &mut z1, &w[..hn], &w[hn..]);
                let mut x = t1.clone();
                poly_sub(logn, &mut x, &z1);
                let t1_ref = z1.clone();
                poly_mul_fft(logn, &mut x, &l10);
                let mut tb0 = t0.clone();
                poly_add(logn, &mut tb0, &x);
                let (mut r0, mut r1) = (vec![0.0; hn], vec![0.0; hn]);
                split_fndsa(logn, &mut r0, &mut r1, &tb0);
                // Fused.
                let t0_in = t0.clone();
                merge_sub_mul_add_split(logn, &mut w, &t0, &mut t1, &l10);
                assert_eq!(bits(&t0), bits(&t0_in), "t0 is read only");
                assert_eq!(bits(&t1), bits(&t1_ref), "z1 {logn}");
                assert_eq!(bits(&w[..hn]), bits(&r0), "w0 {logn}");
                assert_eq!(bits(&w[hn..]), bits(&r1), "w1 {logn}");
            }
        }
    }

    /// The fused target and lattice-point passes of SampPre equal fn-dsa's unfused sequences
    /// bit for bit.
    #[test]
    fn fused_target_and_lattice_point_bit_exact() {
        for logn in 1..=10u32 {
            let n = 1usize << logn;
            let mut rng = Shake256Prng::new(&[0xDD, logn as u8]);
            let mut v: Vec<Vec<f64>> = (0..6).map(|_| vec![0.0; n]).collect();
            for x in v.iter_mut() {
                rand_poly(&mut rng, x);
                fft(logn, x);
            }
            let (b00, b01, b10, b11, z0, z1) = (&v[0], &v[1], &v[2], &v[3], &v[4], &v[5]);
            let inv_q = 1.0 / 1048573.0;
            // Target, unfused (the code of samp_pre_with before the fusion).
            let mut r0 = z0.clone();
            let mut r1 = r0.clone();
            poly_mul_fft(logn, &mut r1, b01);
            poly_mulconst(logn, &mut r1, -inv_q);
            poly_mul_fft(logn, &mut r0, b11);
            poly_mulconst(logn, &mut r0, inv_q);
            let (mut f0, mut f1) = (z0.clone(), vec![0.0; n]);
            poly_target_fft(logn, &mut f0, &mut f1, b01, b11, inv_q);
            assert_eq!(
                (bits(&f0), bits(&f1)),
                (bits(&r0), bits(&r1)),
                "target {logn}"
            );
            // Lattice point, unfused.
            let (mut tx, mut ty) = (z0.clone(), z1.clone());
            poly_mul_fft(logn, &mut tx, b00);
            poly_mul_fft(logn, &mut ty, b10);
            poly_add(logn, &mut tx, &ty);
            let mut ty = z0.clone();
            poly_mul_fft(logn, &mut ty, b01);
            let mut w1 = z1.clone();
            poly_mul_fft(logn, &mut w1, b11);
            poly_add(logn, &mut w1, &ty);
            let (mut g0, mut g1) = (z0.clone(), z1.clone());
            poly_lattice_point_fft(logn, &mut g0, &mut g1, (b00, b01, b10, b11));
            assert_eq!(
                (bits(&g0), bits(&g1)),
                (bits(&tx), bits(&w1)),
                "point {logn}"
            );
        }
    }
}
