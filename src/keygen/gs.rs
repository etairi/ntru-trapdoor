//! Falcon's two Gram–Schmidt norms of a candidate $`(f,g)`$ (design.md §5.9, steps 2-3).
//!
//! The basis $`B=\begin{pmatrix}g&-f\\G&-F\end{pmatrix}`$ has Gram–Schmidt norm
//! $`\max(\|(g,-f)\|,\ \|(q\bar f/(f\bar f+g\bar g),\,q\bar g/(f\bar f+g\bar g))\|)`$, and TrapGen
//! accepts when both are at most $`1.17\sqrt q`$ (Falcon spec. Alg. 5 line 9; falcon-rust
//! src/math.rs:1536-1568; fn-dsa-kgen/src/lib.rs:292-302 and src/ntru.rs:18-43 in fixed point).
//!
//! - $`N_1=\|(g,-f)\|^2`$ exactly in integers; the test $`10000N_1\le13689q`$ is exact.
//! - $`N_2`$: at each evaluation point the vector has squared modulus
//!   $`q^2(|\hat f|^2+|\hat g|^2)/(|\hat f|^2+|\hat g|^2)^2=q^2/(|\hat f|^2+|\hat g|^2)`$, and
//!   Parseval over the $`n/2`$ conjugate pairs gives $`\|a\|^2=\frac2n\sum_{j<n/2}|\hat a_j|^2`$,
//!   so $`N_2=\frac{2q^2}{n}\sum_{j<n/2}\big(|\hat f_j|^2+|\hat g_j|^2\big)^{-1}`$ [derived].
//!   Computed in the crate's `f64` FFT: `poly_set_small`, `fft`, `poly_invnorm2_fft`, a sum.
//!   The relative error is about $`2^{-43}`$ [derived: FFT error $`\approx2^{-40}`$ absolute on
//!   $`|\hat f_j|\approx10^3`$, and a 512-term sum]; only keys within that distance of the bound
//!   can be misclassified, and the leaf check of TrapGen protects the sampler regardless.

use crate::poly::wipe_vec;
use crate::{Params, fft};

/// $`N_1=\|f\|^2+\|g\|^2`$, exactly.
pub(crate) fn sq_norm_fg(f: &[i16], g: &[i16]) -> u64 {
    f.iter()
        .chain(g)
        .map(|&x| {
            let x = x.unsigned_abs() as u64;
            x * x
        })
        .sum()
}

/// $`N_2=\frac{2q^2}{n}\sum_{j<n/2}(|\hat f_j|^2+|\hat g_j|^2)^{-1}`$ in `f64` (infinite if
/// $`\hat f_j=\hat g_j=0`$ somewhere). `params.logn >= 1`; the lengths are `params.n()`.
pub(crate) fn ortho_sq_norm(params: &Params, f: &[i16], g: &[i16]) -> f64 {
    let n = params.n();
    let logn = params.logn;
    assert!(f.len() == n && g.len() == n, "ortho_sq_norm: lengths");
    let mut fx = vec![0.0f64; n];
    let mut gx = vec![0.0f64; n];
    let mut d = vec![0.0f64; n];
    fft::poly_set_small(logn, &mut fx, f);
    fft::poly_set_small(logn, &mut gx, g);
    fft::fft(logn, &mut fx);
    fft::fft(logn, &mut gx);
    fft::poly_invnorm2_fft(logn, &mut d, &fx, &gx);
    let s = pairwise_sum(&d[..n / 2]);
    let q = params.q as f64;
    let n2 = (2.0 * q * q / n as f64) * s;
    // Slice wipes (`crate::poly::wipe_vec`): `Vec::zeroize` costs about ten times more.
    wipe_vec(&mut fx);
    wipe_vec(&mut gx);
    wipe_vec(&mut d);
    n2
}

/// Pairwise summation (error $`O(\varepsilon\log m)`$ instead of $`O(\varepsilon m)`$).
fn pairwise_sum(x: &[f64]) -> f64 {
    if x.len() <= 8 {
        return x.iter().sum();
    }
    let (a, b) = x.split_at(x.len() / 2);
    pairwise_sum(a) + pairwise_sum(b)
}

/// [`crate::hazmat::gram_schmidt_sq_norms`].
pub(crate) fn gram_schmidt_sq_norms(params: &Params, f: &[i16], g: &[i16]) -> (u64, f64) {
    (sq_norm_fg(f, g), ortho_sq_norm(params, f, g))
}

/// The exact first test: $`N_1\le(1.17\sqrt q)^2`$, i.e. `gs_bound_den * N1 <= gs_bound_num`.
pub(crate) fn passes_norm_fg(params: &Params, n1: u64) -> bool {
    (params.gs_bound_den as u128) * (n1 as u128) <= params.gs_bound_num as u128
}

/// The second test: $`N_2\le(1.17\sqrt q)^2`$ (false for NaN).
pub(crate) fn passes_norm_ortho(params: &Params, n2: f64) -> bool {
    n2 <= params.gs_bound()
}
