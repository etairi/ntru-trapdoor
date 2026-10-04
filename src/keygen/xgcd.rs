//! The extended gcd of two big integers, for the deepest level of NTRUSolve (the resultants of
//! $`f`$ and $`g`$, about 9,650 bits each at the PCS set).
//!
//! Lehmer's algorithm (Knuth, TAOCP vol. 2, 3rd ed., §4.5.2, Algorithm L), extended with the
//! cofactor of the first input only; the other cofactor is recovered by one exact division at
//! the end. The simulation runs on the top 62 bits, so that the matrix entries and every
//! intermediate value fit in `i64` [derived: the entries are bounded by the leading part, below
//! $`2^{62}`$], and is guarded by checked arithmetic anyway. A multiprecision step is one
//! linear combination with `i64` scalars, about 60 bits of progress per four scalar products,
//! instead of one long division per Euclidean step as in the generic `extended_gcd` of
//! num-integer (and of falcon-rust src/math.rs:950-963), which is quadratic with a large
//! constant at these sizes.
//!
//! Variable time (key generation only; design.md §10).

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// Bits of the leading part simulated in single precision.
const LEAD: u64 = 62;

/// $`(d,u,v)`$ with $`ua+vb=d=\gcd(a,b)\ge0`$.
pub(crate) fn xgcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    let (pa, pb) = (
        BigInt::from(a.magnitude().clone()),
        BigInt::from(b.magnitude().clone()),
    );
    let (d, s) = xgcd_nonneg(&pa, &pb);
    // d = s |a| + t |b|.
    let t = if pb.is_zero() {
        BigInt::zero()
    } else {
        let (t, r) = (&d - &s * &pa).div_rem(&pb);
        debug_assert!(r.is_zero());
        t
    };
    let u = if a.is_negative() { -s } else { s };
    let v = if b.is_negative() { -t } else { t };
    (d, u, v)
}

/// $`(d,s)`$ with $`d=\gcd(a,b)`$ and $`d\equiv sa\pmod b`$ (exactly: $`d=sa+tb`$ for an
/// integer $`t`$), for $`a,b\ge0`$.
fn xgcd_nonneg(a: &BigInt, b: &BigInt) -> (BigInt, BigInt) {
    // Invariant: r0 = s0 a + t0 b and r1 = s1 a + t1 b (the t are not tracked).
    let (mut r0, mut r1) = (a.clone(), b.clone());
    let (mut s0, mut s1) = (BigInt::one(), BigInt::zero());
    if r0 < r1 {
        core::mem::swap(&mut r0, &mut r1);
        core::mem::swap(&mut s0, &mut s1);
    }
    while !r1.is_zero() {
        let h = r0.bits();
        let mut simulated = None;
        if h > LEAD {
            let shift = h - LEAD;
            // Leading parts of the same weight; x in [2^61, 2^62).
            let x = (&r0 >> shift).to_i64().expect("62-bit leading part");
            let y = (&r1 >> shift).to_i64().expect("62-bit leading part");
            simulated = lehmer_matrix(x, y);
        }
        match simulated {
            Some((a, b, c, d)) => {
                // (r0, r1) <- (A r0 + B r1, C r0 + D r1), the same for the cofactors.
                let n0 = &r0 * a + &r1 * b;
                let n1 = &r0 * c + &r1 * d;
                let m0 = &s0 * a + &s1 * b;
                let m1 = &s0 * c + &s1 * d;
                debug_assert!(n0 > n1 && !n1.is_negative());
                (r0, r1, s0, s1) = (n0, n1, m0, m1);
            }
            None => {
                // One full-precision Euclidean step (Knuth L4 with B = 0, or small operands).
                let (q, r) = r0.div_rem(&r1);
                let s = &s0 - &q * &s1;
                (r0, r1) = (r1, r);
                (s0, s1) = (s1, s);
            }
        }
    }
    (r0, s0)
}

/// Knuth's steps L2-L3 on the leading parts $`(\hat u,\hat v)=(x,y)`$, $`x\ge y\ge0`$: the
/// matrix $`(A,B;C,D)`$ of the Euclidean steps whose quotients are certain, or `None` if not
/// even one is (then $`B=0`$ in Knuth's notation, and step L4 divides in full precision).
///
/// The quotient of the true remainders lies between $`\lfloor(\hat u+A)/(\hat v+C)\rfloor`$ and
/// $`\lfloor(\hat u+B)/(\hat v+D)\rfloor`$; a step is taken only when both denominators are
/// positive and both quotients agree (Knuth's test), and a quotient of 0 (impossible for true
/// remainders, which decrease) also stops the simulation.
fn lehmer_matrix(mut x: i64, mut y: i64) -> Option<(i64, i64, i64, i64)> {
    let (mut a, mut b, mut c, mut d) = (1i64, 0i64, 0i64, 1i64);
    while let (Some(den1), Some(den2)) = (y.checked_add(c), y.checked_add(d)) {
        if den1 <= 0 || den2 <= 0 {
            break;
        }
        let (Some(num1), Some(num2)) = (x.checked_add(a), x.checked_add(b)) else {
            break;
        };
        if num1 < 0 || num2 < 0 {
            break;
        }
        let q = num1 / den1;
        if q == 0 || q != num2 / den2 {
            break;
        }
        let next = (|| {
            let nc = a.checked_sub(q.checked_mul(c)?)?;
            let nd = b.checked_sub(q.checked_mul(d)?)?;
            let ny = x.checked_sub(q.checked_mul(y)?)?;
            Some((nc, nd, ny))
        })();
        let Some((nc, nd, ny)) = next else {
            break;
        };
        (a, c) = (c, nc);
        (b, d) = (d, nd);
        (x, y) = (y, ny);
    }
    if b == 0 { None } else { Some((a, b, c, d)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

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
            let words: Vec<u32> = (0..bits.div_ceil(32)).map(|_| self.next() as u32).collect();
            let m = BigUint::new(words) >> (bits.div_ceil(32) * 32 - bits);
            let x = BigInt::from(m);
            if self.next() & 1 == 1 { -x } else { x }
        }
    }

    fn check(a: &BigInt, b: &BigInt) {
        let (d, u, v) = xgcd(a, b);
        assert_eq!(&u * a + &v * b, d, "Bezout for {a} {b}");
        assert_eq!(d, a.gcd(b), "gcd for {a} {b}");
        assert!(!d.is_negative());
    }

    #[test]
    fn small_and_edge_cases() {
        let z = BigInt::zero();
        for (a, b) in [
            (0i64, 0i64),
            (0, 5),
            (5, 0),
            (1, 1),
            (-7, 3),
            (12, -18),
            (-12, -18),
            (1 << 40, 3),
        ] {
            check(&BigInt::from(a), &BigInt::from(b));
        }
        check(&z, &BigInt::from(-9));
        // Consecutive Fibonacci numbers: the longest remainder sequence for their size.
        let (mut f0, mut f1) = (BigInt::zero(), BigInt::one());
        for _ in 0..2000 {
            let f2 = &f0 + &f1;
            (f0, f1) = (f1, f2);
        }
        check(&f1, &f0);
        check(&f0, &f1);
    }

    #[test]
    fn random_against_num_integer() {
        let mut r = Mix(7);
        for &(ba, bb) in &[
            (10u64, 10u64),
            (64, 64),
            (100, 63),
            (200, 200),
            (1000, 999),
            (3000, 50),
            (9700, 9650),
        ] {
            for _ in 0..20 {
                let a = r.big(ba);
                let b = r.big(bb);
                check(&a, &b);
                // A common factor.
                let c = r.big(40);
                check(&(&a * &c), &(&b * &c));
            }
        }
    }

    #[test]
    fn lehmer_matrix_is_unimodular() {
        let mut r = Mix(9);
        for _ in 0..10_000 {
            let x = (r.next() >> 2) as i64 | (1 << 61);
            let y = (r.next() >> 2) as i64 % x;
            if let Some((a, b, c, d)) = lehmer_matrix(x, y) {
                let det = a as i128 * d as i128 - b as i128 * c as i128;
                assert!(det == 1 || det == -1);
            }
        }
    }
}
