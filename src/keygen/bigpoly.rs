//! Exact arithmetic on integer polynomials modulo $`x^n+1`$ with big-integer coefficients, for
//! NTRUSolve: negacyclic products by Kronecker substitution, the field norm and the lift.
//!
//! **Kronecker substitution** (design.md §5.9). A polynomial $`a`$ with $`|a_i|<2^{b_a}`$ is
//! evaluated at $`2^w`$, $`A=\sum_i a_i2^{wi}`$, as one signed big integer. If
//! $`w\ge b_a+b_b+\lceil\log_2 n\rceil+2`$, every coefficient of the plain product $`ab`$
//! satisfies $`|c_i|<2^{w-2}`$, so $`AB=\sum_i c_i2^{wi}`$ determines the $`c_i`$ as balanced
//! base-$`2^w`$ digits, and the negacyclic fold $`r_i=c_i-c_{i+n}`$ satisfies $`|r_i|<2^{w-1}`$.
//! One big-integer product (Karatsuba or Toom-3 in num-bigint) replaces $`n^2`$ coefficient
//! products.
//!
//! **Digit extraction** uses an offset: with $`O=\sum_{j<m}2^{w-1}2^{wj}`$, the integer
//! $`AB+O`$ is non-negative and its $`w`$-bit fields are exactly $`c_j+2^{w-1}\in[1,2^w)`$, with
//! no carries between fields; so $`c_j`$ is a field minus $`2^{w-1}`$. Fields of at most 127 bits
//! are read as `u128` (the shallow levels of the solver); wider ones become `BigInt`s.
//!
//! **Small operands.** When $`n\le16`$ and one operand fits in `i128` (the Babai quotient
//! $`k`$), the schoolbook product with scalar big-integer multiplications is cheaper than a
//! Kronecker product whose packed operand is mostly zeros [derived: $`n^2L`$ limb products
//! against roughly $`(nL)^{1.585}`$ for Karatsuba, $`L`$ limbs per coefficient].
//!
//! New code (not a port); the algorithms are textbook. The field norm and the lift are the
//! formulas of falcon-rust (src/polynomial.rs:257-309, Szepieniec, MIT) and of the Falcon
//! specification (Alg. 6), evaluated with Kronecker products instead of Karatsuba on
//! `Polynomial<BigInt>`.

use num_bigint::{BigInt, BigUint, Sign};
#[cfg(test)]
use num_traits::Zero;

/// Polynomials up to this degree multiply `i128 x BigInt` by the schoolbook method (tests).
#[cfg(test)]
const SCHOOLBOOK_MAX_N: usize = 8;

/// The bit length of the largest $`|a_i|`$ (0 for the zero polynomial).
pub(crate) fn max_bits(a: &[BigInt]) -> u64 {
    a.iter().map(|x| x.bits()).max().unwrap_or(0)
}

/// The bit length of $`|x|`$ (tests).
#[cfg(test)]
fn bits_i128(x: i128) -> u64 {
    (128 - x.unsigned_abs().leading_zeros()) as u64
}

/// $`\lceil\log_2 n\rceil`$ for $`n\ge1`$.
pub(crate) fn ceil_log2(n: usize) -> u64 {
    (usize::BITS - (n.max(1) - 1).leading_zeros()) as u64
}

/// The Kronecker width for a product whose coefficients are sums of at most `terms` products
/// of a `ba`-bit and a `bb`-bit integer: $`|c|<2^{w-2}`$.
fn kron_width(ba: u64, bb: u64, terms: usize) -> u64 {
    ba + bb + ceil_log2(terms) + 2
}

/// ORs the 64-bit limbs `limbs` (little-endian) into `buf` at bit offset `off`.
#[inline]
pub(crate) fn place(buf: &mut [u64], off: u64, limbs: impl Iterator<Item = u64>) {
    let li = (off / 64) as usize;
    let sh = (off % 64) as u32;
    for (j, d) in limbs.enumerate() {
        if sh == 0 {
            buf[li + j] |= d;
        } else {
            buf[li + j] |= d << sh;
            let hi = d >> (64 - sh);
            if hi != 0 {
                buf[li + j + 1] |= hi;
            }
        }
    }
}

/// A `BigUint` from little-endian 64-bit limbs (num-bigint 0.4 has no `u64` constructor).
pub(crate) fn biguint_from_limbs(limbs: &[u64]) -> BigUint {
    let mut d = Vec::with_capacity(2 * limbs.len());
    for &x in limbs {
        d.push(x as u32);
        d.push((x >> 32) as u32);
    }
    BigUint::new(d)
}

/// $`\sum_i a_i2^{wi}`$ for big-integer coefficients with $`|a_i|<2^w`$.
pub(crate) fn pack_big(a: &[BigInt], w: u64) -> BigInt {
    pack_big_strided(a, 0, 1, w)
}

/// $`\sum_j a_{s+jt}2^{wj}`$ for the coefficients $`a_s,a_{s+t},\dots`$ (`start` $`=s`$,
/// `step` $`=t`$), each $`|a_i|<2^w`$.
pub(crate) fn pack_big_strided(a: &[BigInt], start: usize, step: usize, w: u64) -> BigInt {
    let len = (a.len() - start).div_ceil(step);
    let nlimbs = ((len as u64 * w) / 64 + 2) as usize;
    let mut pos = vec![0u64; nlimbs];
    let mut neg: Vec<u64> = Vec::new();
    for (i, x) in a.iter().skip(start).step_by(step).enumerate() {
        let off = i as u64 * w;
        match x.sign() {
            Sign::NoSign => {}
            Sign::Plus => place(&mut pos, off, x.magnitude().iter_u64_digits()),
            Sign::Minus => {
                if neg.is_empty() {
                    neg = vec![0u64; nlimbs];
                }
                place(&mut neg, off, x.magnitude().iter_u64_digits());
            }
        }
    }
    let p = BigInt::from(biguint_from_limbs(&pos));
    if neg.is_empty() {
        p
    } else {
        p - BigInt::from(biguint_from_limbs(&neg))
    }
}

/// $`\sum_i a_i2^{wi}`$ for `i128` coefficients with $`|a_i|<2^w`$.
pub(crate) fn pack_i128(a: &[i128], w: u64) -> BigInt {
    let nlimbs = ((a.len() as u64 * w) / 64 + 3) as usize;
    let mut pos = vec![0u64; nlimbs];
    let mut neg = vec![0u64; nlimbs];
    let mut any_neg = false;
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        let m = x.unsigned_abs();
        let limbs = [m as u64, (m >> 64) as u64];
        let len = if limbs[1] == 0 { 1 } else { 2 };
        let buf = if x > 0 {
            &mut pos
        } else {
            any_neg = true;
            &mut neg
        };
        place(buf, i as u64 * w, limbs[..len].iter().copied());
    }
    let p = BigInt::from(biguint_from_limbs(&pos));
    if any_neg {
        p - BigInt::from(biguint_from_limbs(&neg))
    } else {
        p
    }
}

/// The offset $`O=\sum_{j<m}2^{w-1}2^{wj}`$.
pub(crate) fn digit_offset(w: u64, m: usize) -> BigInt {
    let nlimbs = ((m as u64 * w) / 64 + 1) as usize;
    let mut buf = vec![0u64; nlimbs];
    for j in 0..m as u64 {
        let b = j * w + (w - 1);
        buf[(b / 64) as usize] |= 1u64 << (b % 64);
    }
    BigInt::from(biguint_from_limbs(&buf))
}

/// Bits `[off, off + w)` of the little-endian limbs, `w <= 128`.
#[inline]
fn read_u128(limbs: &[u64], off: u64, w: u64) -> u128 {
    let li = (off / 64) as usize;
    let sh = (off % 64) as u32;
    let get = |i: usize| -> u128 { limbs.get(i).copied().unwrap_or(0) as u128 };
    let mut x = get(li) >> sh;
    x |= get(li + 1) << (64 - sh);
    if sh > 0 {
        x |= get(li + 2) << (128 - sh);
    }
    if w < 128 { x & ((1u128 << w) - 1) } else { x }
}

/// Bits `[off, off + w)` of the little-endian limbs, as limbs.
fn read_limbs(limbs: &[u64], off: u64, w: u64) -> Vec<u64> {
    let li = (off / 64) as usize;
    let sh = (off % 64) as u32;
    let out_len = w.div_ceil(64) as usize;
    let get = |i: usize| -> u64 { limbs.get(i).copied().unwrap_or(0) };
    let mut out = Vec::with_capacity(out_len);
    for j in 0..out_len {
        let lo = get(li + j) >> sh;
        let hi = if sh == 0 {
            0
        } else {
            get(li + j + 1) << (64 - sh)
        };
        out.push(lo | hi);
    }
    let rem = (w % 64) as u32;
    if rem != 0 {
        let last = out.len() - 1;
        out[last] &= (1u64 << rem) - 1;
    }
    out
}

/// The balanced base-$`2^w`$ digits of an integer: the `m` digits are either all `i128`
/// (when $`w\le127`$) or all big integers.
enum Digits {
    Small(Vec<i128>),
    Big(Vec<BigInt>),
}

/// The `m` digits $`c_j`$ of $`v=\sum_{j<m}c_j2^{wj}`$, $`|c_j|<2^{w-1}`$ (the caller
/// guarantees the bound; `w >= 2`).
fn digits(v: &BigInt, w: u64, m: usize) -> Digits {
    let t = v + digit_offset(w, m);
    debug_assert!(t.sign() != Sign::Minus, "Kronecker digit bound violated");
    let limbs = t.magnitude().to_u64_digits();
    if w <= 127 {
        let half = 1i128 << (w - 1);
        Digits::Small(
            (0..m as u64)
                .map(|j| read_u128(&limbs, j * w, w) as i128 - half)
                .collect(),
        )
    } else {
        let half = BigInt::from(1u8) << (w - 1);
        Digits::Big(
            (0..m as u64)
                .map(|j| BigInt::from(biguint_from_limbs(&read_limbs(&limbs, j * w, w))) - &half)
                .collect(),
        )
    }
}

/// The negacyclic fold of the `2n - 1` digits of a plain product: $`r_i=c_i-c_{i+n}`$ (tests).
#[cfg(test)]
fn fold_product(p: &BigInt, w: u64, n: usize) -> Vec<BigInt> {
    match digits(p, w, 2 * n - 1) {
        Digits::Small(c) => (0..n)
            .map(|i| {
                let hi = if i + n < 2 * n - 1 { c[i + n] } else { 0 };
                BigInt::from(c[i] - hi)
            })
            .collect(),
        Digits::Big(mut c) => {
            let (lo, hi) = c.split_at_mut(n);
            for (x, y) in lo.iter_mut().zip(hi.iter()) {
                *x -= y;
            }
            c.truncate(n);
            c
        }
    }
}

/// $`ab\bmod(x^n+1)`$, exactly; `a.len() == b.len() == n` (tests and future use).
#[cfg(test)]
pub(crate) fn mul(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    let n = a.len();
    assert_eq!(b.len(), n);
    let w = kron_width(max_bits(a), max_bits(b), n);
    let pa = pack_big(a, w);
    let p = if core::ptr::eq(a, b) {
        &pa * &pa
    } else {
        &pa * &pack_big(b, w)
    };
    fold_product(&p, w, n)
}

/// $`kf\bmod(x^n+1)`$ for an `i128` polynomial $`k`$ (the Babai quotient) and a big-integer
/// $`f`$; `k.len() == f.len() == n`. The reference of `flat::mul_small_into` (tests).
#[cfg(test)]
pub(crate) fn mul_small(k: &[i128], f: &[BigInt]) -> Vec<BigInt> {
    let n = f.len();
    assert_eq!(k.len(), n);
    if n <= SCHOOLBOOK_MAX_N {
        // Schoolbook: r_{i+j mod n} += (-1)^{[i+j >= n]} k_i f_j.
        let mut r = vec![BigInt::zero(); n];
        for (i, &ki) in k.iter().enumerate() {
            if ki == 0 {
                continue;
            }
            for (j, fj) in f.iter().enumerate() {
                if fj.is_zero() {
                    continue;
                }
                let t = fj * ki;
                let idx = i + j;
                if idx < n {
                    r[idx] += t;
                } else {
                    r[idx - n] -= t;
                }
            }
        }
        return r;
    }
    let kb = k.iter().map(|&x| bits_i128(x)).max().unwrap_or(0);
    let w = kron_width(kb, max_bits(f), n);
    let p = pack_i128(k, w) * pack_big(f, w);
    fold_product(&p, w, n)
}

/// Two products $`ka\bmod(x^n+1)`$ and $`kb\bmod(x^n+1)`$ sharing the packed $`k`$ (tests).
#[cfg(test)]
pub(crate) fn mul_small_pair(k: &[i128], a: &[BigInt], b: &[BigInt]) -> (Vec<BigInt>, Vec<BigInt>) {
    let n = a.len();
    if n <= SCHOOLBOOK_MAX_N {
        return (mul_small(k, a), mul_small(k, b));
    }
    let kb = k.iter().map(|&x| bits_i128(x)).max().unwrap_or(0);
    let w = kron_width(kb, max_bits(a).max(max_bits(b)), n);
    let pk = pack_i128(k, w);
    let ra = fold_product(&(&pk * pack_big(a, w)), w, n);
    let rb = fold_product(&(&pk * pack_big(b, w)), w, n);
    (ra, rb)
}

/// The field norm $`N(f)(y)=f_e(y)^2-y\,f_o(y)^2\bmod(y^{n/2}+1)`$, where
/// $`f(x)=f_e(x^2)+x\,f_o(x^2)`$, so that $`N(f)(x^2)=f(x)f(-x)`$ (falcon-rust
/// src/polynomial.rs:257-279; Falcon spec. eq. 3.25). `f.len()` is a power of two, at least 2.
pub(crate) fn field_norm(f: &[BigInt]) -> Vec<BigInt> {
    let n = f.len();
    assert!(n >= 2 && n.is_power_of_two());
    let m = n / 2;
    let fe: Vec<BigInt> = f.iter().step_by(2).cloned().collect();
    let fo: Vec<BigInt> = f.iter().skip(1).step_by(2).cloned().collect();
    let b = max_bits(&fe).max(max_bits(&fo));
    // Coefficients of fe^2 - y fo^2 (before folding) are sums of at most 2m = n products.
    let w = kron_width(b, b, n);
    let pe = pack_big(&fe, w);
    let po = pack_big(&fo, w);
    let p = &pe * &pe - ((&po * &po) << w);
    // fe^2 has degree <= 2m - 2 and y fo^2 degree <= 2m - 1: 2m digits, folded mod y^m + 1.
    match digits(&p, w, 2 * m) {
        Digits::Small(c) => (0..m).map(|i| BigInt::from(c[i] - c[i + m])).collect(),
        Digits::Big(mut c) => {
            let (lo, hi) = c.split_at_mut(m);
            for (x, y) in lo.iter_mut().zip(hi.iter()) {
                *x -= y;
            }
            c.truncate(m);
            c
        }
    }
}

/// The lift of NTRUSolve: $`F(x)=F'(x^2)\,g(-x)`$ and $`G(x)=G'(x^2)\,f(-x)`$ modulo $`x^n+1`$
/// (falcon-rust src/math.rs:1264-1271, src/polynomial.rs:281-309), computed as half-size
/// products: with $`g(x)=g_e(x^2)+x\,g_o(x^2)`$,
/// $`F'(x^2)g(-x)=[F'g_e](x^2)-x\,[F'g_o](x^2)`$. The reference of `flat::lift` (tests).
#[cfg(test)]
pub(crate) fn lift(
    big_fp: &[BigInt],
    big_gp: &[BigInt],
    f: &[BigInt],
    g: &[BigInt],
) -> (Vec<BigInt>, Vec<BigInt>) {
    (lift_one(big_fp, g), lift_one(big_gp, f))
}

/// $`P'(x^2)\,a(-x)\bmod(x^n+1)`$ with $`n=2\,|P'|`$.
#[cfg(test)]
fn lift_one(pp: &[BigInt], a: &[BigInt]) -> Vec<BigInt> {
    let m = pp.len();
    let n = 2 * m;
    assert_eq!(a.len(), n);
    let ae: Vec<BigInt> = a.iter().step_by(2).cloned().collect();
    let ao: Vec<BigInt> = a.iter().skip(1).step_by(2).cloned().collect();
    let (even, odd) = if m <= SCHOOLBOOK_MAX_N / 2 {
        (schoolbook(pp, &ae), schoolbook(pp, &ao))
    } else {
        let w = kron_width(max_bits(pp), max_bits(&ae).max(max_bits(&ao)), m);
        let pk = pack_big(pp, w);
        (
            fold_product(&(&pk * pack_big(&ae, w)), w, m),
            fold_product(&(&pk * pack_big(&ao, w)), w, m),
        )
    };
    let mut out = Vec::with_capacity(n);
    for (e, o) in even.into_iter().zip(odd) {
        out.push(e);
        out.push(-o);
    }
    out
}

/// Schoolbook $`ab\bmod(x^n+1)`$ (tiny degrees).
#[cfg(test)]
fn schoolbook(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    let n = a.len();
    let mut r = vec![BigInt::zero(); n];
    for (i, ai) in a.iter().enumerate() {
        if ai.is_zero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            let t = ai * bj;
            if i + j < n {
                r[i + j] += t;
            } else {
                r[i + j - n] -= t;
            }
        }
    }
    r
}

/// The Kronecker evaluation of $`fG-gF`$ for machine-integer polynomials (one product per term;
/// $`|f_i|,|g_i|<2^{15}`$, $`|F_i|,|G_i|<2^{31}`$): returns the packed value and the width
/// (tests).
#[cfg(test)]
fn ntru_lhs_packed(f: &[i16], g: &[i16], big_f: &[i32], big_g: &[i32]) -> (BigInt, u64) {
    let n = f.len();
    assert!(g.len() == n && big_f.len() == n && big_g.len() == n);
    let to128 = |s: &[i16]| s.iter().map(|&x| x as i128).collect::<Vec<i128>>();
    let to128w = |s: &[i32]| s.iter().map(|&x| x as i128).collect::<Vec<i128>>();
    let (f, g, big_f, big_g) = (to128(f), to128(g), to128w(big_f), to128w(big_g));
    let bits = |s: &[i128]| s.iter().map(|&x| bits_i128(x)).max().unwrap_or(0);
    let bs = bits(&f).max(bits(&g));
    let bb = bits(&big_f).max(bits(&big_g));
    // Each coefficient of fG - gF is a sum of at most 2n products.
    let w = kron_width(bs, bb, 2 * n);
    let p = pack_i128(&f, w) * pack_i128(&big_g, w) - pack_i128(&g, w) * pack_i128(&big_f, w);
    (p, w)
}

/// $`fG-gF\bmod(x^n+1)`$, exactly (tests).
#[cfg(test)]
pub(crate) fn ntru_lhs(f: &[i16], g: &[i16], big_f: &[i32], big_g: &[i32]) -> Vec<BigInt> {
    let (p, w) = ntru_lhs_packed(f, g, big_f, big_g);
    fold_product(&p, w, f.len())
}

/// `true` iff $`fG-gF=q`$ in $`\mathbb Z[x]/(x^n+1)`$. The width is at most
/// $`15+31+\log_2(2n)+2\le59`$ bits at $`n\le1024`$, so the folded digits are compared as `i128`.
/// The tests' independent oracle for the crate's check, `ring::ntt::ntru_equation_holds`
/// (two-prime NTT), which replaced it in key generation and `SecretKey::validate`.
#[cfg(test)]
pub(crate) fn ntru_equation_holds(
    q: u32,
    f: &[i16],
    g: &[i16],
    big_f: &[i32],
    big_g: &[i32],
) -> bool {
    let n = f.len();
    let (p, w) = ntru_lhs_packed(f, g, big_f, big_g);
    match digits(&p, w, 2 * n - 1) {
        Digits::Small(c) => (0..n).all(|i| {
            let r = c[i] - if i + n < 2 * n - 1 { c[i + n] } else { 0 };
            r == if i == 0 { q as i128 } else { 0 }
        }),
        Digits::Big(_) => {
            let r = fold_product(&p, w, n);
            r[0] == BigInt::from(q) && r[1..].iter().all(|x| x.is_zero())
        }
    }
}

/// The value of `x` as `i32`, if it fits (tests).
#[cfg(test)]
pub(crate) fn to_i32(x: &BigInt) -> Option<i32> {
    if x.bits() > 31 {
        return None;
    }
    let m = x.magnitude().iter_u64_digits().next().unwrap_or(0) as i64;
    let v = if x.sign() == Sign::Minus { -m } else { m };
    i32::try_from(v).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic test generator (SplitMix64), independent of the crate's PRNG.
    pub(crate) struct Mix(pub u64);
    impl Mix {
        pub(crate) fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        /// A random signed integer of at most `bits` bits.
        pub(crate) fn big(&mut self, bits: u64) -> BigInt {
            let limbs: Vec<u64> = (0..bits.div_ceil(64)).map(|_| self.next()).collect();
            let mut m = biguint_from_limbs(&limbs);
            let excess = limbs.len() as u64 * 64 - bits;
            m >>= excess;
            let x = BigInt::from(m);
            if self.next() & 1 == 1 { -x } else { x }
        }
    }

    fn naive(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
        schoolbook(a, b)
    }

    #[test]
    fn mul_matches_schoolbook() {
        let mut r = Mix(1);
        for logn in 0..=7u32 {
            let n = 1usize << logn;
            for &(ba, bb) in &[
                (1u64, 1u64),
                (8, 30),
                (60, 60),
                (63, 64),
                (100, 3),
                (300, 700),
                (2000, 50),
            ] {
                let a: Vec<BigInt> = (0..n).map(|_| r.big(ba)).collect();
                let b: Vec<BigInt> = (0..n).map(|_| r.big(bb)).collect();
                assert_eq!(mul(&a, &b), naive(&a, &b), "logn {logn} bits {ba} {bb}");
                assert_eq!(mul(&a, &a), naive(&a, &a), "square logn {logn} bits {ba}");
            }
        }
    }

    #[test]
    fn mul_extreme_values() {
        // All coefficients at -(2^b - 1) or +(2^b - 1): the digit bound is tight.
        for logn in 0..=6u32 {
            let n = 1usize << logn;
            for &b in &[1u64, 7, 62, 63, 64, 65, 126, 127, 128, 129, 200] {
                let big: BigInt = (BigInt::from(1u8) << b) - 1u8;
                let a: Vec<BigInt> = (0..n)
                    .map(|i| {
                        if i % 3 == 0 {
                            -big.clone()
                        } else {
                            big.clone()
                        }
                    })
                    .collect();
                let c: Vec<BigInt> = (0..n).map(|_| -big.clone()).collect();
                assert_eq!(mul(&a, &c), naive(&a, &c), "logn {logn} b {b}");
                assert_eq!(mul(&c, &c), naive(&c, &c), "logn {logn} b {b}");
            }
        }
    }

    #[test]
    fn mul_small_matches_schoolbook() {
        let mut r = Mix(2);
        for logn in 0..=7u32 {
            let n = 1usize << logn;
            for &(kb, fb) in &[(1u32, 5u64), (53, 9), (106, 40), (126, 300), (20, 4000)] {
                let k: Vec<i128> = (0..n)
                    .map(|_| {
                        let v = ((r.next() as u128) << 64 | r.next() as u128) >> (128 - kb);
                        if r.next() & 1 == 1 {
                            -(v as i128)
                        } else {
                            v as i128
                        }
                    })
                    .collect();
                let f: Vec<BigInt> = (0..n).map(|_| r.big(fb)).collect();
                let g: Vec<BigInt> = (0..n).map(|_| r.big(fb / 2 + 1)).collect();
                let kbig: Vec<BigInt> = k.iter().map(|&x| BigInt::from(x)).collect();
                assert_eq!(mul_small(&k, &f), naive(&kbig, &f), "logn {logn}");
                let (x, y) = mul_small_pair(&k, &f, &g);
                assert_eq!(x, naive(&kbig, &f));
                assert_eq!(y, naive(&kbig, &g));
            }
        }
    }

    #[test]
    fn field_norm_is_f_times_f_of_minus_x() {
        let mut r = Mix(3);
        for logn in 1..=7u32 {
            let n = 1usize << logn;
            for &b in &[1u64, 9, 61, 64, 200] {
                let f: Vec<BigInt> = (0..n).map(|_| r.big(b)).collect();
                let fneg: Vec<BigInt> = f
                    .iter()
                    .enumerate()
                    .map(|(i, x)| if i % 2 == 1 { -x.clone() } else { x.clone() })
                    .collect();
                let prod = naive(&f, &fneg);
                // f(x) f(-x) is even: its odd coefficients vanish, the even ones are N(f).
                let nf = field_norm(&f);
                for i in 0..n / 2 {
                    assert_eq!(nf[i], prod[2 * i], "logn {logn} b {b} i {i}");
                    assert!(prod[2 * i + 1].is_zero());
                }
            }
        }
    }

    #[test]
    fn lift_matches_definition() {
        let mut r = Mix(4);
        for logn in 1..=7u32 {
            let n = 1usize << logn;
            let m = n / 2;
            for &(bp, ba) in &[(3u64, 2u64), (40, 20), (300, 100)] {
                let fp: Vec<BigInt> = (0..m).map(|_| r.big(bp)).collect();
                let gp: Vec<BigInt> = (0..m).map(|_| r.big(bp)).collect();
                let f: Vec<BigInt> = (0..n).map(|_| r.big(ba)).collect();
                let g: Vec<BigInt> = (0..n).map(|_| r.big(ba)).collect();
                let up = |p: &[BigInt]| -> Vec<BigInt> {
                    let mut o = vec![BigInt::zero(); n];
                    for (i, x) in p.iter().enumerate() {
                        o[2 * i] = x.clone();
                    }
                    o
                };
                let neg = |p: &[BigInt]| -> Vec<BigInt> {
                    p.iter()
                        .enumerate()
                        .map(|(i, x)| if i % 2 == 1 { -x.clone() } else { x.clone() })
                        .collect()
                };
                let (bf, bg) = lift(&fp, &gp, &f, &g);
                assert_eq!(bf, naive(&up(&fp), &neg(&g)), "logn {logn}");
                assert_eq!(bg, naive(&up(&gp), &neg(&f)), "logn {logn}");
            }
        }
    }

    #[test]
    fn ntru_lhs_matches_schoolbook() {
        let mut r = Mix(5);
        for logn in 0..=10u32 {
            let n = 1usize << logn;
            let f: Vec<i16> = (0..n).map(|_| (r.next() % 701) as i16 - 350).collect();
            let g: Vec<i16> = (0..n).map(|_| (r.next() % 701) as i16 - 350).collect();
            let bf: Vec<i32> = (0..n).map(|_| (r.next() % 8191) as i32 - 4095).collect();
            let bg: Vec<i32> = (0..n).map(|_| (r.next() % 8191) as i32 - 4095).collect();
            let mut expect = vec![0i64; n];
            for i in 0..n {
                for j in 0..n {
                    let t = f[i] as i64 * bg[j] as i64 - g[i] as i64 * bf[j] as i64;
                    if i + j < n {
                        expect[i + j] += t;
                    } else {
                        expect[i + j - n] -= t;
                    }
                }
            }
            let got = ntru_lhs(&f, &g, &bf, &bg);
            let expect: Vec<BigInt> = expect.into_iter().map(BigInt::from).collect();
            assert_eq!(got, expect, "logn {logn}");
        }
    }

    #[test]
    fn to_i32_bounds() {
        assert_eq!(to_i32(&BigInt::from(i32::MAX)), Some(i32::MAX));
        assert_eq!(to_i32(&BigInt::from(-(i32::MAX))), Some(-i32::MAX));
        assert_eq!(to_i32(&BigInt::from(i32::MIN)), None); // 32 bits of magnitude
        assert_eq!(to_i32(&BigInt::from(1u64 << 31)), None);
        assert_eq!(to_i32(&BigInt::zero()), Some(0));
    }
}
