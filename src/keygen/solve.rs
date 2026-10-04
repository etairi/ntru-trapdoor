//! NTRUSolve, stage 1 (big integers): given $`f,g\in\mathbb Z[x]/(x^n+1)`$, find $`F,G`$ with
//! $`fG-gF=q`$, Babai-reduced against $`(f,g)`$ (design.md §5.9).
//!
//! Structure: Falcon specification Alg. 6, as in falcon-rust's `ntru_solve` (src/math.rs:
//! 1242-1303; Szepieniec, MIT) and the exact Sage oracle
//! `tools/ntru_sizes.py`:
//! 1. the field-norm tower $`f_{k+1}=N(f_k)`$ down to degree 1 (the resultants), every level
//!    kept ([`bigpoly::field_norm`], Kronecker squarings);
//! 2. the deepest level: $`uf_L+vg_L=1`$ by Lehmer's extended gcd ([`xgcd`]), so
//!    $`(F_L,G_L)=(-vq,uq)`$; `None` if the resultants are not coprime;
//! 3. the lift $`F_k=F_{k+1}(x^2)g_k(-x)`$, $`G_k=G_{k+1}(x^2)f_k(-x)`$ ([`flat::lift`]: two
//!    half-size Kronecker products per polynomial, the digits written straight into limbs);
//! 4. Babai's round-off at every level $`k<L`$: [`babai_reduce`], a port of falcon-rust's
//!    `babai_reduce_bigint` (src/math.rs:89-216) with its window schedule `babai_shifts`
//!    (src/math.rs:68-74). Differences from upstream, none of which affects the validity of the
//!    result (another FFT may round a near-half-integer Babai quotient the other way and so
//!    return another, equally valid $`(F,G)`$; the sampler's output distribution does not depend
//!    on which one): the quotient comes from the crate's `f64` FFT (fn-dsa's representation)
//!    instead of falcon-rust's complex FFT; $`(F,G)`$ live in fixed-width two's-complement limbs
//!    ([`Flat`]); $`kf`$ is a schoolbook product with one-limb scalars ($`n\le32`$) or one
//!    Kronecker product ($`n\ge64`$) instead of Karatsuba; and the power of two common to the
//!    coefficients of $`k`$ moves into the shift ([`strip_twos`]). The first `Vec<BigInt>`
//!    version is kept as `babai_reduce_reference` (tests); the two agree bit for bit
//!    (`keygen::tests::babai_flat_matches_reference`).
//!
//! **Parity pre-check** (new): $`\mathrm{Res}(f,x^n+1)\equiv f(1)\pmod 2`$, because
//! $`x^n+1\equiv(x+1)^n\pmod 2`$; if both $`f(1)`$ and $`g(1)`$ are even, the resultants share
//! the factor 2 and there is no solution, so the tower is skipped (about a quarter of random
//! pairs [derived]; fn-dsa instead samples $`f,g`$ of odd parity, fn-dsa-kgen/src/gauss.rs:
//! 30-81). This only saves time: the answer is the same.
//!
//! Variable time. The intermediates are freed without being wiped: num-bigint has no zeroize
//! support (design.md §10), and the crate's own limb buffers ([`Flat`]) are not wiped either,
//! which would cost about 5% and leave the num-bigint copies anyway.

use num_bigint::BigInt;
use num_traits::One;

use super::bigpoly::{self, max_bits};
use super::flat::{self, Flat, KronCache, SignMag};
use super::xgcd::xgcd;
use crate::fft;

/// NTRUSolve; see [`crate::hazmat::ntru_solve`].
pub(crate) fn ntru_solve(q: u32, f: &[i16], g: &[i16]) -> Option<(Vec<i32>, Vec<i32>)> {
    let n = f.len();
    assert!(n.is_power_of_two() && g.len() == n, "ntru_solve: lengths");
    let odd = |p: &[i16]| p.iter().fold(0u16, |acc, &x| acc ^ (x as u16 & 1)) == 1;
    if !odd(f) && !odd(g) {
        return None;
    }
    let logn = n.trailing_zeros() as usize;

    // 1. The field-norm tower: fs[k], gs[k] have n >> k coefficients.
    let mut fs: Vec<Vec<BigInt>> = Vec::with_capacity(logn + 1);
    let mut gs: Vec<Vec<BigInt>> = Vec::with_capacity(logn + 1);
    fs.push(f.iter().map(|&x| BigInt::from(x)).collect());
    gs.push(g.iter().map(|&x| BigInt::from(x)).collect());
    for k in 0..logn {
        let nf = bigpoly::field_norm(&fs[k]);
        let ng = bigpoly::field_norm(&gs[k]);
        fs.push(nf);
        gs.push(ng);
    }

    // 2. The deepest level: u f_L + v g_L = 1, then f_L (u q) - g_L (-v q) = q.
    let (d, u, v) = xgcd(&fs[logn][0], &gs[logn][0]);
    if !d.is_one() {
        return None;
    }
    let qb = BigInt::from(q);
    let (fl, gl) = (-(v * &qb), u * &qb);
    let l = (fl.bits().max(gl.bits()) / 64 + 2) as usize;
    let mut big_f = Flat::from_big(&[fl], l);
    let mut big_g = Flat::from_big(&[gl], l);

    // 3-4. Lift and reduce, up to depth 0, on limbs.
    for k in (0..logn).rev() {
        let (mut nf, mut ng) = flat::lift(&big_f, &big_g, &fs[k], &gs[k], BABAI_HEADROOM);
        babai_reduce(&fs[k], &gs[k], &mut nf, &mut ng);
        big_f = nf;
        big_g = ng;
    }
    Some((big_f.to_i32()?, big_g.to_i32()?))
}

/// Spare bits above the initial size of $`(F,G)`$ for the intermediate values of the Babai
/// reduction (see [`babai_reduce`]).
const BABAI_HEADROOM: u64 = 160;

/// falcon-rust `babai_shifts` (src/math.rs:68-74): for a quotient of $`d`$ bits, either window
/// the top 106 bits of $`(F,G)`$ and shift the 53-bit quotient back by $`d-53`$, or (when
/// $`d\le53`$) window at the scale of $`(f,g)`$ and reduce all $`d`$ bits in one pass.
fn babai_shifts(capital_size: u64, size: u64, d: u64) -> (u64, u64) {
    if d > 53 {
        (capital_size - 106, d - 53)
    } else {
        (size - 53, 0)
    }
}

/// $`\lfloor x/2^{s}\rfloor`$ as `f64` through `i128`: falcon-rust's
/// `i128::try_from(x >> s).unwrap() as f64` (src/math.rs:107, :146; `>>` rounds toward
/// $`-\infty`$), without allocating. The schedule keeps every window below $`2^{107}`$.
fn window_big(x: &BigInt, s: u64) -> f64 {
    let m = x.magnitude();
    let t: u128 = if m.bits() <= s {
        0
    } else {
        debug_assert!(m.bits() - s < 127, "Babai window wider than i128");
        let sh = (s % 64) as u32;
        let mut it = m.iter_u64_digits().skip((s / 64) as usize);
        let (l0, l1, l2) = (
            it.next().unwrap_or(0) as u128,
            it.next().unwrap_or(0) as u128,
            it.next().unwrap_or(0) as u128,
        );
        if sh == 0 {
            l0 | (l1 << 64)
        } else {
            (l0 >> sh) | (l1 << (64 - sh)) | (l2 << (128 - sh))
        }
    };
    let q = if x.sign() == num_bigint::Sign::Minus {
        // floor(-m / 2^s) = -(m >> s) - [m mod 2^s != 0].
        -(t as i128) - m.trailing_zeros().is_some_and(|tz| tz < s) as i128
    } else {
        t as i128
    };
    q as f64
}

/// The rounded Babai quotient $`k=\mathrm{round}((F'\bar f'+G'\bar g')/(f'\bar f'+g'\bar g'))`$
/// for the windows $`F'_i`$, $`G'_i`$ given by `wf` and `wg` (falcon-rust src/math.rs:145-159),
/// with $`\hat f'`$, $`\hat g'`$ and the self-adjoint denominator already in FFT form.
#[allow(clippy::too_many_arguments)]
fn quotient(
    logn: u32,
    wf: impl Fn(usize) -> f64,
    wg: impl Fn(usize) -> f64,
    fa: &[f64],
    ga: &[f64],
    den: &[f64],
    num: &mut [f64],
    tmp: &mut [f64],
) -> Vec<i128> {
    for (i, o) in num.iter_mut().enumerate() {
        *o = wf(i);
    }
    for (i, o) in tmp.iter_mut().enumerate() {
        *o = wg(i);
    }
    fft::fft(logn, num);
    fft::fft(logn, tmp);
    fft::poly_muladj_fft(logn, num, fa);
    fft::poly_muladj_fft(logn, tmp, ga);
    fft::poly_add(logn, num, tmp);
    fft::poly_div_selfadj_fft(logn, num, den);
    fft::ifft(logn, num);
    // `f64::round` (half away from zero) and a saturating cast, as falcon-rust.
    num.iter().map(|&x| x.round() as i128).collect()
}

/// The FFT of the 53-bit windows of $`f`$ and $`g`$ and the self-adjoint denominator
/// $`\hat f'\bar{\hat f'}+\hat g'\bar{\hat g'}`$ (falcon-rust src/math.rs:97-116); returns
/// (`size`, `fa`, `ga`, `den`).
fn babai_setup(f: &[BigInt], g: &[BigInt]) -> (u64, Vec<f64>, Vec<f64>, Vec<f64>) {
    let logn = f.len().trailing_zeros();
    let size = max_bits(f).max(max_bits(g)).max(53);
    let shift = size - 53;
    let mut fa: Vec<f64> = f.iter().map(|x| window_big(x, shift)).collect();
    let mut ga: Vec<f64> = g.iter().map(|x| window_big(x, shift)).collect();
    fft::fft(logn, &mut fa);
    fft::fft(logn, &mut ga);
    let mut den = fa.clone();
    fft::poly_mulownadj_fft(logn, &mut den);
    let mut t = ga.clone();
    fft::poly_mulownadj_fft(logn, &mut t);
    fft::poly_add(logn, &mut den, &t);
    (size, fa, ga, den)
}

/// Polynomials up to this degree use the schoolbook product in the Babai passes (one- or
/// two-limb scalars times $`f`$); larger ones one Kronecker product. Tuned on the M4 at the PCS
/// set (keygen/README.md).
const BABAI_SCHOOLBOOK_MAX_N: usize = 32;

/// Divides $`k`$ by the largest power of two that divides every coefficient and returns its
/// exponent $`t`$, so that $`2^s kf=2^{s+t}(k/2^t)f`$ exactly. A $`k`$ rounded from `f64` values of
/// magnitude $`2^e`$, $`e>53`$, has at most 53 significant bits, so the products shrink
/// [derived].
fn strip_twos(k: &mut [i128]) -> u64 {
    let t = k
        .iter()
        .filter(|&&x| x != 0)
        .map(|x| x.trailing_zeros())
        .min()
        .unwrap_or(0);
    if t > 0 {
        for x in k.iter_mut() {
            *x >>= t;
        }
    }
    t as u64
}

/// `acc` $`\leftarrow kf\bmod(x^n+1)`$.
fn mul_k(
    acc: &mut Flat,
    k: &[i128],
    f: &[BigInt],
    fs: &SignMag,
    f_bits: u64,
    cache: &mut Option<KronCache>,
) {
    if f.len() <= BABAI_SCHOOLBOOK_MAX_N {
        flat::mul_small_into(acc, k, fs);
    } else {
        flat::kron_small_into(acc, k, f, f_bits, cache);
    }
}

/// Babai round-off of $`(F,G)`$ against $`(f,g)`$: falcon-rust `babai_reduce_bigint`
/// (src/math.rs:89-216). Each pass estimates up to 53 bits of
/// $`k=\mathrm{round}((F\bar f+G\bar g)/(f\bar f+g\bar g))`$ from top windows in `f64` and
/// subtracts $`2^{s}k\cdot(f,g)`$; the loop stops when $`(F,G)`$ is below the size of
/// $`(f,g)`$ or stops shrinking. A pass that would enlarge $`(F,G)`$ (an overshoot, when
/// $`|\hat f|^2+|\hat g|^2`$ is tiny at some frequency) falls back to a one-window quotient
/// shifted by $`d`$, as upstream. `f.len()` is at least 2.
///
/// The arithmetic runs on fixed-width limbs ([`Flat`]). Width: with $`c`$ the initial size of
/// $`(F,G)`$, every intermediate value is below $`2^{c+138}`$ [derived: $`|k_i|<2^{127}`$, so
/// $`|(kf)_i|<n2^{127}|f|_\infty\le2^{c+137}`$ at $`n\le1024`$, shifted only when $`d>53`$, where
/// $`2^s|kf|<2^{c+75+\log_2n}`$], so [`BABAI_HEADROOM`] = 160 spare bits suffice; narrower
/// inputs are widened.
pub(crate) fn babai_reduce(f: &[BigInt], g: &[BigInt], cf_in: &mut Flat, cg_in: &mut Flat) {
    let n = f.len();
    let logn = n.trailing_zeros();
    debug_assert!(logn >= 1 && cf_in.n == n && cg_in.n == n);
    let (size, fa, ga, den) = babai_setup(f, g);

    let c0 = cf_in.max_bits().max(cg_in.max_bits());
    let lw = ((c0 + BABAI_HEADROOM) / 64 + 1) as usize;
    let mut cf = if cf_in.l >= lw {
        core::mem::replace(cf_in, Flat::zero(0, 1))
    } else {
        cf_in.widen(lw)
    };
    let mut cg = if cg_in.l >= lw {
        core::mem::replace(cg_in, Flat::zero(0, 1))
    } else {
        cg_in.widen(lw)
    };
    let (fs, gs) = (SignMag::from_big(f), SignMag::from_big(g));
    let (f_bits, g_bits) = (max_bits(f), max_bits(g));
    let mut acc_f = Flat::zero(n, fs.l + 3);
    let mut acc_g = Flat::zero(n, gs.l + 3);
    let (mut cache_f, mut cache_g) = (None, None);
    let mut nf = cf.clone();
    let mut ng = cg.clone();

    let mut num = vec![0.0f64; n];
    let mut tmp = vec![0.0f64; n];
    let mut prev_capital_size = u64::MAX;
    loop {
        let capital_size = cf.max_bits().max(cg.max_bits()).max(53);
        // Stop at the target size, or when the size stopped decreasing (f64 precision).
        if capital_size < size || capital_size >= prev_capital_size {
            break;
        }
        prev_capital_size = capital_size;

        let d = capital_size - size;
        let (capital_shift, back_shift) = babai_shifts(capital_size, size, d);
        let mut k = quotient(
            logn,
            |i| cf.window(i, capital_shift),
            |i| cg.window(i, capital_shift),
            &fa,
            &ga,
            &den,
            &mut num,
            &mut tmp,
        );
        if k.iter().all(|&x| x == 0) {
            break;
        }
        let t = strip_twos(&mut k);
        mul_k(&mut acc_f, &k, f, &fs, f_bits, &mut cache_f);
        mul_k(&mut acc_g, &k, g, &gs, g_bits, &mut cache_g);
        nf.d.copy_from_slice(&cf.d);
        ng.d.copy_from_slice(&cg.d);
        nf.sub_shifted(&acc_f, back_shift + t);
        ng.sub_shifted(&acc_g, back_shift + t);

        // Tentative check (upstream :173-210): an overshoot falls back to the one-window
        // quotient (capital_size - 53), shifted back by d.
        if d > 53 && nf.max_bits().max(ng.max_bits()) >= capital_size {
            // Upstream rounds to i64 (`f64 as i64`, saturating): clamp likewise.
            let s_old = capital_size - 53;
            let mut k_old: Vec<i128> = quotient(
                logn,
                |i| cf.window(i, s_old),
                |i| cg.window(i, s_old),
                &fa,
                &ga,
                &den,
                &mut num,
                &mut tmp,
            )
            .into_iter()
            .map(|x| x.clamp(i64::MIN as i128, i64::MAX as i128))
            .collect();
            if k_old.iter().all(|&x| x == 0) {
                break;
            }
            let t = strip_twos(&mut k_old);
            mul_k(&mut acc_f, &k_old, f, &fs, f_bits, &mut cache_f);
            mul_k(&mut acc_g, &k_old, g, &gs, g_bits, &mut cache_g);
            cf.sub_shifted(&acc_f, d + t);
            cg.sub_shifted(&acc_g, d + t);
            continue;
        }
        core::mem::swap(&mut cf, &mut nf);
        core::mem::swap(&mut cg, &mut ng);
    }
    *cf_in = cf;
    *cg_in = cg;
}

/// The `Vec<BigInt>` implementation of [`babai_reduce`] (the first version, kept as the
/// reference of the differential test): the same schedule, windows and quotients, with
/// big-integer products and subtractions.
#[cfg(test)]
pub(crate) fn babai_reduce_reference(
    f: &[BigInt],
    g: &[BigInt],
    big_f: &mut Vec<BigInt>,
    big_g: &mut Vec<BigInt>,
) {
    let sub_shifted = |p: &[BigInt], q: &[BigInt], s: u64| -> Vec<BigInt> {
        p.iter().zip(q).map(|(a, b)| a - (b << s)).collect()
    };
    let n = f.len();
    let logn = n.trailing_zeros();
    let (size, fa, ga, den) = babai_setup(f, g);
    let mut num = vec![0.0f64; n];
    let mut tmp = vec![0.0f64; n];
    let mut prev_capital_size = u64::MAX;
    loop {
        let capital_size = max_bits(big_f).max(max_bits(big_g)).max(53);
        if capital_size < size || capital_size >= prev_capital_size {
            break;
        }
        prev_capital_size = capital_size;
        let d = capital_size - size;
        let (capital_shift, back_shift) = babai_shifts(capital_size, size, d);
        let (bf, bg) = (big_f.clone(), big_g.clone());
        let k = quotient(
            logn,
            |i| window_big(&bf[i], capital_shift),
            |i| window_big(&bg[i], capital_shift),
            &fa,
            &ga,
            &den,
            &mut num,
            &mut tmp,
        );
        if k.iter().all(|&x| x == 0) {
            break;
        }
        let (kf, kg) = bigpoly::mul_small_pair(&k, f, g);
        let nf = sub_shifted(big_f, &kf, back_shift);
        let ng = sub_shifted(big_g, &kg, back_shift);
        if d > 53 && max_bits(&nf).max(max_bits(&ng)) >= capital_size {
            let s_old = capital_size - 53;
            let k_old: Vec<i128> = quotient(
                logn,
                |i| window_big(&bf[i], s_old),
                |i| window_big(&bg[i], s_old),
                &fa,
                &ga,
                &den,
                &mut num,
                &mut tmp,
            )
            .into_iter()
            .map(|x| x.clamp(i64::MIN as i128, i64::MAX as i128))
            .collect();
            if k_old.iter().all(|&x| x == 0) {
                break;
            }
            let (kf, kg) = bigpoly::mul_small_pair(&k_old, f, g);
            *big_f = sub_shifted(big_f, &kf, d);
            *big_g = sub_shifted(big_g, &kg, d);
            continue;
        }
        *big_f = nf;
        *big_g = ng;
    }
}

/// The resultant $`\mathrm{Res}(f,x^n+1)=N_{K/\mathbb Q}(f)`$ (the deepest level of the field
/// norm tower), for tests.
#[cfg(test)]
pub(crate) fn resultant(f: &[i16]) -> BigInt {
    let mut p: Vec<BigInt> = f.iter().map(|&x| BigInt::from(x)).collect();
    while p.len() > 1 {
        p = bigpoly::field_norm(&p);
    }
    p.pop().unwrap_or_else(|| BigInt::from(0u8))
}
