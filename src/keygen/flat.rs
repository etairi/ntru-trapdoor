//! Fixed-width two's-complement multi-limb polynomials: the allocation-free arithmetic of the
//! Babai reduction in NTRUSolve.
//!
//! The reduction subtracts $`2^s\,kf`$ from $`F`$ in many passes, where $`k`$ has `i128`
//! coefficients and $`f`$ big ones. With `Vec<BigInt>` every pass allocates for each coefficient
//! (products, digits, shifts), and a Kronecker product cannot exploit the small $`k`$.
//! A sample profile of the `BigInt` version at the PCS set showed the products at about 3/4 of
//! the reduction [measured, 2026-10-03; keygen/README.md]. Here a polynomial is one `Vec<u64>`:
//! $`n`$ coefficients of `l` little-endian limbs each, in two's complement. A pass then costs:
//! - $`kf\bmod(x^n+1)`$ into an accumulator, by schoolbook with one- or two-limb scalars
//!   ([`mul_small_into`], about $`n^2\ell_f`$ limb products) or, for large $`n`$, by one
//!   Kronecker product whose digits are written straight into limbs ([`kron_small_into`]);
//! - $`F\leftarrow F-2^s\cdot\mathrm{acc}`$ in place ([`Flat::sub_shifted`]);
//! - the windows $`\lfloor F_i/2^s\rfloor`$ read as `i128` ([`Flat::window`]): falcon-rust's
//!   exact semantics (src/math.rs:145-150).
//!
//! Every operation is exact modulo $`2^{64\ell}`$ per coefficient; the caller sizes $`\ell`$ so
//! that no true value overflows (see `solve::babai_reduce`).
//!
//! Not wiped: like num-bigint's buffers (design.md §10), these limbs are freed without
//! zeroization. Wiping them on drop cost about 5% of NTRUSolve [measured, keygen/README.md], and
//! it would not make the solver's memory clean, since the tower and every product also live in
//! num-bigint buffers.

use num_bigint::{BigInt, Sign};
use num_traits::Zero;

use super::bigpoly::{
    biguint_from_limbs, ceil_log2, digit_offset, max_bits, pack_big, pack_big_strided, pack_i128,
    place,
};

/// $`n`$ coefficients of `l` limbs each, two's complement, little-endian.
#[derive(Clone)]
pub(crate) struct Flat {
    pub(crate) n: usize,
    pub(crate) l: usize,
    pub(crate) d: Vec<u64>,
}

/// The sign-extension limb of a two's-complement number.
#[inline]
fn ext(x: &[u64]) -> u64 {
    ((x[x.len() - 1] as i64) >> 63) as u64
}

/// Two's-complement negation in place.
#[inline]
fn neg_in_place(x: &mut [u64]) {
    let mut carry = 1u64;
    for w in x.iter_mut() {
        let (r, o) = (!*w).overflowing_add(carry);
        *w = r;
        carry = o as u64;
    }
}

/// The bit length of $`|x|`$ for a two's-complement $`x`$ (as `BigInt::bits`).
fn bits_abs(x: &[u64]) -> u64 {
    let neg = ext(x) != 0;
    let mut top = 0u64;
    for (k, &w) in x.iter().enumerate().rev() {
        let c = if neg { !w } else { w };
        if c != 0 {
            top = 64 * k as u64 + 64 - c.leading_zeros() as u64;
            break;
        }
    }
    if !neg {
        return top;
    }
    // x < 0: |x| = !x + 1 has the bit length of !x, plus one when !x = 2^top - 1, i.e. when the
    // low `top` bits of x are all zero (x = -2^top).
    let full = (top / 64) as usize;
    let rem = (top % 64) as u32;
    let low_zero =
        x[..full].iter().all(|&w| w == 0) && (rem == 0 || x[full] & ((1u64 << rem) - 1) == 0);
    if low_zero { top + 1 } else { top }
}

impl Flat {
    /// The zero polynomial.
    pub(crate) fn zero(n: usize, l: usize) -> Flat {
        Flat {
            n,
            l,
            d: vec![0u64; n * l],
        }
    }

    /// From big integers; every $`|a_i|<2^{64\ell-1}`$.
    pub(crate) fn from_big(a: &[BigInt], l: usize) -> Flat {
        let mut out = Flat::zero(a.len(), l);
        for (i, x) in a.iter().enumerate() {
            assert!(
                x.bits() < 64 * l as u64,
                "Flat::from_big: coefficient too wide"
            );
            let c = &mut out.d[i * l..(i + 1) * l];
            for (w, m) in c.iter_mut().zip(x.magnitude().iter_u64_digits()) {
                *w = m;
            }
            if x.sign() == Sign::Minus {
                neg_in_place(c);
            }
        }
        out
    }

    /// To big integers (tests).
    #[cfg(test)]
    pub(crate) fn to_big(&self) -> Vec<BigInt> {
        let mut tmp = vec![0u64; self.l];
        (0..self.n)
            .map(|i| {
                let c = self.coeff(i);
                if ext(c) == 0 {
                    BigInt::from(biguint_from_limbs(c))
                } else {
                    tmp.copy_from_slice(c);
                    neg_in_place(&mut tmp);
                    BigInt::from_biguint(Sign::Minus, biguint_from_limbs(&tmp))
                }
            })
            .collect()
    }

    #[inline]
    pub(crate) fn coeff(&self, i: usize) -> &[u64] {
        &self.d[i * self.l..(i + 1) * self.l]
    }

    #[inline]
    fn coeff_mut(&mut self, i: usize) -> &mut [u64] {
        &mut self.d[i * self.l..(i + 1) * self.l]
    }

    /// The largest bit length of $`|F_i|`$.
    pub(crate) fn max_bits(&self) -> u64 {
        (0..self.n)
            .map(|i| bits_abs(self.coeff(i)))
            .max()
            .unwrap_or(0)
    }

    /// $`\lfloor F_i/2^s\rfloor`$ as `f64`, through `i128` (the caller guarantees that the
    /// quotient fits in `i128`): falcon-rust's `i128::try_from(F >> s).unwrap() as f64`.
    pub(crate) fn window(&self, i: usize, s: u64) -> f64 {
        let x = self.coeff(i);
        let e = ext(x);
        let get = |q: usize| if q < x.len() { x[q] } else { e };
        let li = (s / 64) as usize;
        let sh = (s % 64) as u32;
        let (l0, l1, l2) = (get(li), get(li + 1), get(li + 2));
        let (lo, hi) = if sh == 0 {
            (l0, l1)
        } else {
            (
                (l0 >> sh) | (l1 << (64 - sh)),
                (l1 >> sh) | (l2 << (64 - sh)),
            )
        };
        ((((hi as u128) << 64) | lo as u128) as i128) as f64
    }

    /// The same values over `l >= self.l` limbs (sign extension).
    pub(crate) fn widen(&self, l: usize) -> Flat {
        debug_assert!(l >= self.l);
        let mut out = Flat::zero(self.n, l);
        for i in 0..self.n {
            let c = self.coeff(i);
            let e = ext(c);
            let o = out.coeff_mut(i);
            o[..self.l].copy_from_slice(c);
            o[self.l..].fill(e);
        }
        out
    }

    /// The coefficients as `i32`, if they all fit.
    pub(crate) fn to_i32(&self) -> Option<Vec<i32>> {
        (0..self.n)
            .map(|i| {
                let c = self.coeff(i);
                let e = ext(c);
                if c[1..].iter().any(|&w| w != e) {
                    return None;
                }
                let v = c[0] as i64;
                // The sign of the low limb must agree with the extension.
                if (v < 0) != (e != 0) {
                    return None;
                }
                i32::try_from(v).ok()
            })
            .collect()
    }

    /// $`F_i\leftarrow F_i-2^s a_i`$ for every $`i`$ (modulo $`2^{64\ell}`$; exact when the
    /// true results fit).
    pub(crate) fn sub_shifted(&mut self, a: &Flat, s: u64) {
        debug_assert_eq!(self.n, a.n);
        let ls = (s / 64) as isize;
        let b = (s % 64) as u32;
        let la = a.l as isize;
        for i in 0..self.n {
            let ai = a.coeff(i);
            let ea = ext(ai);
            let get = |q: isize| -> u64 {
                if q < 0 {
                    0
                } else if q < la {
                    ai[q as usize]
                } else {
                    ea
                }
            };
            let fi = self.coeff_mut(i);
            let mut borrow = 0u64;
            for (p, w) in fi.iter_mut().enumerate() {
                let q = p as isize - ls;
                let v = if b == 0 {
                    get(q)
                } else {
                    (get(q) << b) | (get(q - 1) >> (64 - b))
                };
                let (r1, o1) = w.overflowing_sub(v);
                let (r2, o2) = r1.overflowing_sub(borrow);
                *w = r2;
                borrow = (o1 | o2) as u64;
            }
        }
    }
}

/// $`acc\mathrel{+}=a\cdot k`$ (unsigned `a`, scalar `k`), the carry propagated through `acc`
/// (modulo $`2^{64|acc|}`$).
#[inline]
fn addmul_1(acc: &mut [u64], a: &[u64], k: u64) {
    let mut carry = 0u64;
    for (w, &x) in acc.iter_mut().zip(a) {
        let v = *w as u128 + (x as u128) * (k as u128) + carry as u128;
        *w = v as u64;
        carry = (v >> 64) as u64;
    }
    for w in acc[a.len()..].iter_mut() {
        if carry == 0 {
            break;
        }
        let (r, o) = w.overflowing_add(carry);
        *w = r;
        carry = o as u64;
    }
}

/// $`acc\mathrel{-}=a\cdot k`$ (unsigned `a`, scalar `k`), the borrow propagated through `acc`
/// (modulo $`2^{64|acc|}`$).
#[inline]
fn submul_1(acc: &mut [u64], a: &[u64], k: u64) {
    let mut borrow = 0u64;
    for (w, &x) in acc.iter_mut().zip(a) {
        let prod = (x as u128) * (k as u128) + borrow as u128;
        let (r, o) = w.overflowing_sub(prod as u64);
        *w = r;
        // (2^64 - 1)^2 + 2^64 - 1 < 2^128: the high word plus 1 does not overflow.
        borrow = (prod >> 64) as u64 + o as u64;
    }
    for w in acc[a.len()..].iter_mut() {
        if borrow == 0 {
            break;
        }
        let (r, o) = w.overflowing_sub(borrow);
        *w = r;
        borrow = o as u64;
    }
}

/// A polynomial in sign-magnitude form (the multiplicand of the schoolbook product).
pub(crate) struct SignMag {
    pub(crate) n: usize,
    pub(crate) l: usize,
    mag: Vec<u64>,
    neg: Vec<bool>,
    nonzero: Vec<bool>,
}

impl SignMag {
    pub(crate) fn from_big(a: &[BigInt]) -> SignMag {
        let bits = a.iter().map(|x| x.bits()).max().unwrap_or(0);
        let l = (bits.div_ceil(64) as usize).max(1);
        let mut mag = vec![0u64; a.len() * l];
        for (i, x) in a.iter().enumerate() {
            for (w, m) in mag[i * l..(i + 1) * l]
                .iter_mut()
                .zip(x.magnitude().iter_u64_digits())
            {
                *w = m;
            }
        }
        SignMag {
            n: a.len(),
            l,
            mag,
            neg: a.iter().map(|x| x.sign() == Sign::Minus).collect(),
            nonzero: a.iter().map(|x| !x.is_zero()).collect(),
        }
    }
}

/// `acc` (with `acc.l >= f.l + 3`) $`\leftarrow kf\bmod(x^n+1)`$ by schoolbook.
pub(crate) fn mul_small_into(acc: &mut Flat, k: &[i128], f: &SignMag) {
    let n = f.n;
    debug_assert!(acc.n == n && k.len() == n && acc.l >= f.l + 3);
    acc.d.fill(0);
    let la = acc.l;
    for (i, &ki) in k.iter().enumerate() {
        if ki == 0 {
            continue;
        }
        let km = ki.unsigned_abs();
        let (k0, k1) = (km as u64, (km >> 64) as u64);
        for j in 0..n {
            if !f.nonzero[j] {
                continue;
            }
            let fj = &f.mag[j * f.l..(j + 1) * f.l];
            let mut idx = i + j;
            // x^n = -1: the wrapped terms change sign.
            let mut negative = (ki < 0) ^ f.neg[j];
            if idx >= n {
                idx -= n;
                negative = !negative;
            }
            let c = &mut acc.d[idx * la..(idx + 1) * la];
            if negative {
                submul_1(c, fj, k0);
                if k1 != 0 {
                    submul_1(&mut c[1..], fj, k1);
                }
            } else {
                addmul_1(c, fj, k0);
                if k1 != 0 {
                    addmul_1(&mut c[1..], fj, k1);
                }
            }
        }
    }
}

/// Packed operands of the Kronecker path, cached for the current width.
pub(crate) struct KronCache {
    w: u64,
    packed: BigInt,
}

/// `acc` $`\leftarrow kf\bmod(x^n+1)`$ by one Kronecker product, the digits written straight
/// into limbs; `fb` is $`f`$ as big integers and `f_bits` its largest bit length.
pub(crate) fn kron_small_into(
    acc: &mut Flat,
    k: &[i128],
    fb: &[BigInt],
    f_bits: u64,
    cache: &mut Option<KronCache>,
) {
    let n = fb.len();
    let kb = k
        .iter()
        .map(|&x| 128 - x.unsigned_abs().leading_zeros() as u64)
        .max()
        .unwrap_or(0);
    let logn = (usize::BITS - (n.max(1) - 1).leading_zeros()) as u64;
    let w = kb + f_bits + logn + 2;
    debug_assert!(w < 64 * acc.l as u64);
    if cache.as_ref().is_none_or(|c| c.w != w) {
        *cache = Some(KronCache {
            w,
            packed: pack_big(fb, w),
        });
    }
    let pf = &cache.as_ref().expect("cache").packed;
    fold_into(&(pack_i128(k, w) * pf), w, acc);
}

/// `out` $`\leftarrow`$ the negacyclic fold $`r_i=c_i-c_{i+m}`$ of the $`2m-1`$ balanced digits
/// $`c_j`$ of $`p=\sum_jc_j2^{wj}`$ ($`|c_j|<2^{w-2}`$, $`m`$ = `out.n`, $`64\,\ell>w`$), written
/// straight into limbs: the digits $`c_j+2^{w-1}`$ are the $`w`$-bit fields of $`p+O`$ (no carries
/// between fields; `bigpoly` module documentation).
pub(crate) fn fold_into(p: &BigInt, w: u64, out: &mut Flat) {
    let m = out.n;
    let t = p + digit_offset(w, 2 * m - 1);
    debug_assert!(t.sign() != Sign::Minus);
    let limbs: Vec<u64> = t.magnitude().iter_u64_digits().collect();
    let la = out.l;
    debug_assert!(64 * la as u64 > w);
    let mut tmp = vec![0u64; la];
    for j in 0..2 * m - 1 {
        read_digit(&limbs, j as u64 * w, w, &mut tmp);
        let i = if j < m { j } else { j - m };
        let c = &mut out.d[i * la..(i + 1) * la];
        if j < m {
            c.copy_from_slice(&tmp);
        } else {
            let mut borrow = 0u64;
            for (x, &y) in c.iter_mut().zip(tmp.iter()) {
                let (r1, o1) = x.overflowing_sub(y);
                let (r2, o2) = r1.overflowing_sub(borrow);
                *x = r2;
                borrow = (o1 | o2) as u64;
            }
        }
    }
}

/// $`\sum_ix_i2^{wi}`$ for two's-complement coefficients with $`|x_i|<2^w`$.
pub(crate) fn pack_flat(x: &Flat, w: u64) -> BigInt {
    let nlimbs = ((x.n as u64 * w) / 64 + 2) as usize;
    let mut pos = vec![0u64; nlimbs];
    let mut neg: Vec<u64> = Vec::new();
    let mut tmp = vec![0u64; x.l];
    for i in 0..x.n {
        let c = x.coeff(i);
        let off = i as u64 * w;
        if ext(c) == 0 {
            let top = c.iter().rposition(|&v| v != 0).map_or(0, |t| t + 1);
            place(&mut pos, off, c[..top].iter().copied());
        } else {
            tmp.copy_from_slice(c);
            neg_in_place(&mut tmp);
            let top = tmp.iter().rposition(|&v| v != 0).map_or(0, |t| t + 1);
            if neg.is_empty() {
                neg = vec![0u64; nlimbs];
            }
            place(&mut neg, off, tmp[..top].iter().copied());
        }
    }
    let p = BigInt::from(biguint_from_limbs(&pos));
    if neg.is_empty() {
        p
    } else {
        p - BigInt::from(biguint_from_limbs(&neg))
    }
}

/// The lift of NTRUSolve on limbs: $`F(x)=F'(x^2)g(-x)`$ and $`G(x)=G'(x^2)f(-x)`$ modulo
/// $`x^n+1`$ (falcon-rust src/math.rs:1264-1271, with `lift_next_cyclotomic` and
/// `galois_adjoint`, src/polynomial.rs:281-309), as `bigpoly::lift` (half-size Kronecker
/// products: with
/// $`g=g_e(x^2)+x\,g_o(x^2)`$, $`F'(x^2)g(-x)=[F'g_e](x^2)-x[F'g_o](x^2)`$), the digits written
/// into limbs with `headroom` spare bits per coefficient (for the Babai reduction).
pub(crate) fn lift(
    pf: &Flat,
    pg: &Flat,
    f: &[BigInt],
    g: &[BigInt],
    headroom: u64,
) -> (Flat, Flat) {
    (lift_one(pf, g, headroom), lift_one(pg, f, headroom))
}

fn lift_one(pp: &Flat, a: &[BigInt], headroom: u64) -> Flat {
    let m = pp.n;
    let n = 2 * m;
    assert_eq!(a.len(), n);
    let a_bits = max_bits(a);
    let w = pp.max_bits() + a_bits + ceil_log2(m) + 2;
    let l = ((w + headroom) / 64 + 1) as usize;
    let pk = pack_flat(pp, w);
    let mut even = Flat::zero(m, l);
    let mut odd = Flat::zero(m, l);
    fold_into(&(&pk * pack_big_strided(a, 0, 2, w)), w, &mut even);
    fold_into(&(&pk * pack_big_strided(a, 1, 2, w)), w, &mut odd);
    let mut out = Flat::zero(n, l);
    for i in 0..m {
        out.coeff_mut(2 * i).copy_from_slice(even.coeff(i));
        let o = out.coeff_mut(2 * i + 1);
        o.copy_from_slice(odd.coeff(i));
        neg_in_place(o);
    }
    out
}

/// The digit $`c=v-2^{w-1}`$ of the field $`v`$ = bits `[off, off + w)` of `limbs`, in two's
/// complement over `out.len()` limbs ($`64|out|>w`$): flip bit $`w-1`$ of $`v`$, then
/// sign-extend from it.
fn read_digit(limbs: &[u64], off: u64, w: u64, out: &mut [u64]) {
    let li = (off / 64) as usize;
    let sh = (off % 64) as u32;
    let get = |i: usize| -> u64 { limbs.get(i).copied().unwrap_or(0) };
    let full = (w / 64) as usize;
    let rem = (w % 64) as u32;
    let nl = full + (rem != 0) as usize;
    for (q, o) in out.iter_mut().enumerate().take(nl) {
        let lo = get(li + q) >> sh;
        let hi = if sh == 0 {
            0
        } else {
            get(li + q + 1) << (64 - sh)
        };
        *o = lo | hi;
    }
    if rem != 0 {
        out[full] &= (1u64 << rem) - 1;
    }
    // Flip bit w - 1, then sign-extend from it.
    let tb = w - 1;
    let (tl, tbit) = ((tb / 64) as usize, (tb % 64) as u32);
    out[tl] ^= 1u64 << tbit;
    let neg = (out[tl] >> tbit) & 1 == 1;
    if neg {
        if tbit < 63 {
            out[tl] |= !0u64 << (tbit + 1);
        }
        for o in out[tl + 1..].iter_mut() {
            *o = !0;
        }
    } else {
        for o in out[nl.max(tl + 1)..].iter_mut() {
            *o = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keygen::bigpoly;

    struct Mix(u64);
    impl Mix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn big(&mut self, bits: u64) -> BigInt {
            let limbs: Vec<u64> = (0..bits.div_ceil(64)).map(|_| self.next()).collect();
            let m = biguint_from_limbs(&limbs) >> (limbs.len() as u64 * 64 - bits);
            let x = BigInt::from(m);
            if self.next() & 1 == 1 { -x } else { x }
        }
        fn k(&mut self, bits: u32) -> i128 {
            let v = (((self.next() as u128) << 64) | self.next() as u128) >> (128 - bits);
            if self.next() & 1 == 1 {
                -(v as i128)
            } else {
                v as i128
            }
        }
    }

    #[test]
    fn round_trip_bits_and_windows() {
        let mut r = Mix(1);
        let mut vals: Vec<BigInt> = (0..200).map(|i| r.big(1 + (i * 7) % 300)).collect();
        for e in [0u64, 1, 63, 64, 65, 127, 128, 200] {
            vals.push(-(BigInt::from(1u8) << e));
            vals.push(BigInt::from(1u8) << e);
            vals.push((BigInt::from(1u8) << e) - 1u8);
            vals.push(-((BigInt::from(1u8) << e) - 1u8));
        }
        let fl = Flat::from_big(&vals, 6);
        assert_eq!(fl.to_big(), vals);
        for (i, v) in vals.iter().enumerate() {
            assert_eq!(bits_abs(fl.coeff(i)), v.bits(), "bits of {v}");
            for s in [0u64, 1, 50, 64, 100, 190, 250, 383] {
                let q: BigInt = v >> s; // floor
                if q.bits() < 127 {
                    let expect = i128::try_from(&q).unwrap() as f64;
                    assert_eq!(fl.window(i, s), expect, "window {v} >> {s}");
                }
            }
        }
    }

    #[test]
    fn sub_shifted_matches_bigint() {
        let mut r = Mix(2);
        for &(bf, ba, la) in &[(100u64, 60u64, 2usize), (400, 200, 5), (900, 64, 3)] {
            let n = 8;
            let fv: Vec<BigInt> = (0..n).map(|_| r.big(bf)).collect();
            let av: Vec<BigInt> = (0..n).map(|_| r.big(ba)).collect();
            for s in [0u64, 1, 63, 64, 77, 200] {
                let lf = ((bf.max(ba + s) + 2) / 64 + 2) as usize;
                let mut fl = Flat::from_big(&fv, lf);
                fl.sub_shifted(&Flat::from_big(&av, la), s);
                let expect: Vec<BigInt> = fv.iter().zip(&av).map(|(x, y)| x - (y << s)).collect();
                assert_eq!(fl.to_big(), expect, "s {s}");
            }
        }
    }

    #[test]
    fn products_match_bigint() {
        let mut r = Mix(3);
        for logn in 0..=7u32 {
            let n = 1usize << logn;
            for &(kb, fb) in &[
                (1u32, 3u64),
                (53, 40),
                (64, 64),
                (65, 300),
                (106, 20),
                (126, 1000),
            ] {
                let k: Vec<i128> = (0..n).map(|_| r.k(kb)).collect();
                let f: Vec<BigInt> = (0..n).map(|_| r.big(fb)).collect();
                let expect = bigpoly::mul_small(&k, &f);
                let sm = SignMag::from_big(&f);
                let mut acc = Flat::zero(n, sm.l + 3);
                mul_small_into(&mut acc, &k, &sm);
                assert_eq!(
                    acc.to_big(),
                    expect,
                    "schoolbook logn {logn} kb {kb} fb {fb}"
                );
                let mut acc2 = Flat::zero(n, sm.l + 3);
                let mut cache = None;
                kron_small_into(&mut acc2, &k, &f, bigpoly::max_bits(&f), &mut cache);
                assert_eq!(
                    acc2.to_big(),
                    expect,
                    "kronecker logn {logn} kb {kb} fb {fb}"
                );
            }
        }
    }
}
