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
//! Also used by key generation (the Gram–Schmidt norms and the Babai reduction of NTRUSolve):
//! every `pub fn` below.

mod gm;

use gm::GM;

/// Complex multiplication (fn-dsa poly.rs:34-38).
#[inline(always)]
fn flc_mul(x_re: f64, x_im: f64, y_re: f64, y_im: f64) -> (f64, f64) {
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
        let m = 1usize << lm;
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
        t = ht;
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
            let hm = 1usize << (logn - lm);
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
            t = dt;
        }
    }
    // logn - 1 delayed halvings: FLR::slice_div2e (flr_native.rs:361-366) multiplies by the
    // exact power of two 2^-(logn-1).
    let e = 1.0 / ((1u64 << (logn - 1)) as f64);
    for x in f[..n].iter_mut() {
        *x *= e;
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

/// `a <- a - b` (either representation; fn-dsa poly.rs:599-652).
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
    // If logn = 1 then the loop is entirely skipped.
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
    // If logn = 1 then the loop is entirely skipped.
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
}
