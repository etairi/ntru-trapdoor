//! Arithmetic in $`R_q=\mathbb Z_q[x]/(x^d+1)`$ for any $`3\le q<2^{25}`$ (the parameter sets
//! use a prime $`q`$).
//!
//! Design (design.md §5.2):
//! - **Multiplication** (`ntt`): exact negacyclic convolution over $`\mathbb Z`$ through the CRT
//!   over two NTT primes $`p_1=2147473409`$, $`p_2=2147389441`$ (both $`\equiv1\bmod2048`$;
//!   fn-dsa-kgen/src/mp31.rs:343-347, `PRIMES[0..2]`), then reduction modulo $`q`$. Inputs in
//!   $`[0,q)`$ give $`|c_k|<d(q-1)^2\le2^{50}`$ at the PCS set, and $`p_1p_2>2^{61.99}`$.
//!   Port of fn-dsa-kgen (The Unlicense) mp31.rs:45-132 (`mp_add`, `mp_sub`, `mp_half`,
//!   `mp_mmul`, ...) and poly.rs:22-138 (`mp_mkgmigm`, `mp_NTT`, `mp_iNTT`). The tables serve
//!   every degree $`2^k\le1024`$ (the norm tower needs all of them).
//! - **Prepared products**: [`RqPoly::prepare`] keeps the NTT residues of a fixed operand, so
//!   that [`PreparedRqPoly::mul`] needs four NTTs instead of six (Verify uses it for $`h`$).
//! - **Inversion and invertibility** (`inv`): the norm tower
//!   $`a^{-1}(x)=a(-x)\,N(a)^{-1}(x^2)`$ with $`N(a)(y)=a_e(y)^2-y\,a_o(y)^2`$, down to a scalar
//!   inverted by Fermat ($`a^{q-2}`$, with an extended-Euclid fallback that makes it exact for a
//!   composite $`q`$ too); $`a`$ is a unit iff the final scalar (its norm to $`\mathbb Z_q`$) is a
//!   unit. About two multiplications' worth of NTTs (the NTTs of $`a_e`$, $`a_o`$ are computed
//!   once per level and reused on the way back up).
//! - **Encoding** (`encode`): canonical little-endian bit packing, `q_bits` bits per
//!   coefficient (20 at the PCS set: 2560 B), as fn-dsa's `modq_encode` at $`q=12289`$
//!   (fn-dsa-comm/src/codec.rs:73-150). Decoding rejects values $`\ge q`$ and nonzero padding.
//!
//! Side channels: addition, subtraction, negation, multiplication and the norm tower use masks
//! instead of branches; their running time depends on the degree only. `inverse` returns
//! `None` for a non-unit, which is the only value-dependent outcome.
//!
//! Used by key generation and the tests: the inherent methods and the operator
//! impls below.

mod encode;
mod inv;
pub(crate) mod ntt;

use core::ops::{Add, Mul, Neg, Sub};

use crate::{Error, Params, PreparedRqPoly, RqPoly};

impl RqPoly {
    #[inline]
    fn check_same_ring(&self, other: &RqPoly) {
        assert!(
            self.q == other.q && self.c.len() == other.c.len(),
            "RqPoly: modulus or degree mismatch"
        );
    }

    #[inline]
    fn logn(&self) -> u32 {
        debug_assert!(self.c.len().is_power_of_two() && self.c.len() >= 2);
        self.c.len().trailing_zeros()
    }

    /// `self + other`. Panics if the modulus or the degree differ.
    pub fn add(&self, other: &RqPoly) -> RqPoly {
        self.check_same_ring(other);
        let q = self.q;
        let c = self
            .c
            .iter()
            .zip(other.c.iter())
            .map(|(&a, &b)| ntt::mp_add(a, b, q))
            .collect();
        RqPoly::from_raw(q, c)
    }

    /// `self - other`. Panics if the modulus or the degree differ.
    pub fn sub(&self, other: &RqPoly) -> RqPoly {
        self.check_same_ring(other);
        let q = self.q;
        let c = self
            .c
            .iter()
            .zip(other.c.iter())
            .map(|(&a, &b)| ntt::mp_sub(a, b, q))
            .collect();
        RqPoly::from_raw(q, c)
    }

    /// `-self`.
    pub fn neg(&self) -> RqPoly {
        let q = self.q;
        let c = self.c.iter().map(|&a| ntt::mp_sub(0, a, q)).collect();
        RqPoly::from_raw(q, c)
    }

    /// `self * other` in $`R_q`$, exactly. Panics if the modulus or the degree differ.
    pub fn mul(&self, other: &RqPoly) -> RqPoly {
        self.check_same_ring(other);
        let mut c = vec![0u32; self.c.len()];
        ntt::mul_mod_q(self.logn(), self.q, &self.c, &other.c, &mut c);
        RqPoly::from_raw(self.q, c)
    }

    /// `self` prepared for repeated products ([`PreparedRqPoly::mul`]): two NTTs per prime.
    pub fn prepare(&self) -> PreparedRqPoly {
        let logn = self.logn();
        let (r1, r2) = ntt::prepare(logn, &self.c);
        PreparedRqPoly {
            q: self.q,
            logn,
            r1,
            r2,
        }
    }

    /// The inverse in $`R_q`$, or `None` if `self` is not a unit. Exact for every modulus of the
    /// ring checks, prime or not; for a prime $`q`$ only the `None`/`Some` outcome depends on the
    /// value in its sequence of operations.
    pub fn inverse(&self) -> Option<RqPoly> {
        inv::invert(self.logn(), self.q, &self.c).map(|c| RqPoly::from_raw(self.q, c))
    }

    /// `true` iff `self` is a unit of $`R_q`$ (its norm to $`\mathbb Z_q`$ is a unit).
    pub fn is_invertible(&self) -> bool {
        inv::is_invertible(self.logn(), self.q, &self.c)
    }

    /// The canonical encoding: `params.rq_bytes()` bytes, `q_bits` bits per coefficient,
    /// little-endian bit packing, zero padding.
    pub fn to_bytes(&self) -> Vec<u8> {
        encode::encode(self.q, &self.c)
    }

    /// Decodes the canonical encoding; rejects a wrong length, a coefficient $`\ge q`$ or a
    /// nonzero padding bit. Errors: [`Error::InvalidParams`], [`Error::InvalidLength`],
    /// [`Error::InvalidEncoding`].
    pub fn from_bytes(params: &Params, bytes: &[u8]) -> Result<RqPoly, Error> {
        params.check_ring()?;
        let c = encode::decode(params.q, params.n(), bytes)?;
        Ok(RqPoly::from_raw(params.q, c))
    }
}

impl PreparedRqPoly {
    /// The modulus $`q`$.
    pub fn q(&self) -> u32 {
        self.q
    }

    /// The degree $`d`$.
    pub fn n(&self) -> usize {
        1usize << self.logn
    }

    /// `true` if this element belongs to the ring of `params` (same modulus and degree).
    pub fn matches(&self, params: &Params) -> bool {
        self.q == params.q && self.logn == params.logn
    }

    /// `a * other` in $`R_q`$ for the prepared `a`, exactly; equal to [`RqPoly::mul`]. Panics if
    /// the modulus or the degree differ.
    pub fn mul(&self, other: &RqPoly) -> RqPoly {
        assert!(
            self.q == other.q && self.n() == other.c.len(),
            "PreparedRqPoly: modulus or degree mismatch"
        );
        let mut c = vec![0u32; other.c.len()];
        ntt::mul_prepared_mod_q(self.logn, self.q, &self.r1, &self.r2, &other.c, &mut c);
        RqPoly::from_raw(self.q, c)
    }
}

impl Add for &RqPoly {
    type Output = RqPoly;
    fn add(self, other: &RqPoly) -> RqPoly {
        RqPoly::add(self, other)
    }
}

impl Sub for &RqPoly {
    type Output = RqPoly;
    fn sub(self, other: &RqPoly) -> RqPoly {
        RqPoly::sub(self, other)
    }
}

impl Mul for &RqPoly {
    type Output = RqPoly;
    fn mul(self, other: &RqPoly) -> RqPoly {
        RqPoly::mul(self, other)
    }
}

impl Neg for &RqPoly {
    type Output = RqPoly;
    fn neg(self) -> RqPoly {
        RqPoly::neg(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prng::{Prng, Shake256Prng};

    fn rand_poly(rng: &mut Shake256Prng, q: u32, n: usize) -> RqPoly {
        let c = (0..n).map(|_| (rng.next_u64() % q as u64) as u32).collect();
        RqPoly::from_raw(q, c)
    }

    fn schoolbook(a: &RqPoly, b: &RqPoly) -> RqPoly {
        let n = a.c.len();
        let q = a.q as i128;
        let mut acc = vec![0i128; n];
        for i in 0..n {
            for j in 0..n {
                let p = a.c[i] as i128 * b.c[j] as i128;
                if i + j < n {
                    acc[i + j] += p;
                } else {
                    acc[i + j - n] -= p;
                }
            }
        }
        RqPoly::from_raw(a.q, acc.iter().map(|&x| x.rem_euclid(q) as u32).collect())
    }

    /// The moduli of the tests: tiny, Falcon, PCS, and the largest prime below 2^25.
    const QS: [u32; 5] = [3, 17, 12289, 1048573, 33554393];

    /// S-RING-1: `mul` against the i128 schoolbook product, logn 1..10, every test modulus;
    /// plus the extreme operands (all coefficients q - 1), the CRT's worst case.
    #[test]
    fn mul_matches_schoolbook() {
        for &q in QS.iter() {
            for logn in 1..=10u32 {
                let n = 1usize << logn;
                let mut rng = Shake256Prng::from_parts(&[b"mul", &q.to_le_bytes(), &[logn as u8]]);
                let reps = if logn >= 9 { 1 } else { 3 };
                for _ in 0..reps {
                    let a = rand_poly(&mut rng, q, n);
                    let b = rand_poly(&mut rng, q, n);
                    assert_eq!(a.mul(&b), schoolbook(&a, &b), "q {q} logn {logn}");
                }
                let m = RqPoly::from_raw(q, vec![q - 1; n]);
                assert_eq!(m.mul(&m), schoolbook(&m, &m), "extreme, q {q} logn {logn}");
            }
        }
    }

    /// The prepared product equals the plain one, at every degree and test modulus.
    #[test]
    fn prepared_mul_matches_mul() {
        for &q in QS.iter() {
            for logn in 1..=10u32 {
                let n = 1usize << logn;
                let mut rng = Shake256Prng::from_parts(&[b"prep", &q.to_le_bytes(), &[logn as u8]]);
                let a = rand_poly(&mut rng, q, n);
                let ap = a.prepare();
                assert_eq!(ap.n(), n);
                for _ in 0..2 {
                    let b = rand_poly(&mut rng, q, n);
                    assert_eq!(ap.mul(&b), a.mul(&b), "q {q} logn {logn}");
                }
                let m = RqPoly::from_raw(q, vec![q - 1; n]);
                assert_eq!(
                    m.prepare().mul(&m),
                    schoolbook(&m, &m),
                    "extreme, q {q} logn {logn}"
                );
            }
        }
    }

    /// Composite moduli (which `Params::validate` rejects, but the ring checks allow): the
    /// inverse is exact, and `is_invertible` agrees with an independent unit test (a is a unit
    /// iff it is a unit modulo every prime factor of q).
    #[test]
    fn inverse_for_composite_moduli() {
        // (q, its prime factors).
        let cases: [(u32, &[u32]); 4] = [
            (15, &[3, 5]),
            (12289 * 17, &[12289, 17]),
            (1 << 20, &[2]),
            (3 * 3 * 1048573, &[3, 1048573]),
        ];
        let mut units = 0;
        let mut non_units = 0;
        for (q, factors) in cases {
            for logn in [1u32, 2, 5, 9] {
                let n = 1usize << logn;
                let mut rng = Shake256Prng::from_parts(&[b"comp", &q.to_le_bytes(), &[logn as u8]]);
                for _ in 0..4 {
                    let a = rand_poly(&mut rng, q, n);
                    let unit = factors.iter().all(|&p| {
                        let ap = RqPoly::from_raw(p, a.c.iter().map(|&x| x % p).collect());
                        coprime_to_xn1(&ap)
                    });
                    assert_eq!(a.is_invertible(), unit, "q {q} logn {logn}");
                    match a.inverse() {
                        Some(ai) => {
                            assert!(unit);
                            assert_eq!(a.mul(&ai), one(q, n), "q {q} logn {logn}");
                            units += 1;
                        }
                        None => {
                            assert!(!unit, "q {q} logn {logn}");
                            non_units += 1;
                        }
                    }
                }
            }
        }
        assert!(
            units > 0 && non_units > 0,
            "{units} units, {non_units} non-units"
        );
    }

    #[test]
    fn add_sub_neg() {
        for &q in QS.iter() {
            let mut rng = Shake256Prng::from_parts(&[b"add", &q.to_le_bytes()]);
            let a = rand_poly(&mut rng, q, 64);
            let b = rand_poly(&mut rng, q, 64);
            let s = &a + &b;
            let d = &a - &b;
            let z = &a + &(-&a);
            for i in 0..64 {
                assert_eq!(s.c[i], (a.c[i] + b.c[i]) % q);
                assert_eq!(d.c[i], (a.c[i] + q - b.c[i]) % q);
                assert_eq!(z.c[i], 0);
            }
            assert_eq!(&d + &b, a);
        }
    }

    #[test]
    #[should_panic(expected = "mismatch")]
    fn mismatch_panics() {
        let a = RqPoly::from_raw(12289, vec![0; 8]);
        let b = RqPoly::from_raw(1048573, vec![0; 8]);
        let _ = a.mul(&b);
    }

    fn one(q: u32, n: usize) -> RqPoly {
        let mut c = vec![0u32; n];
        c[0] = 1;
        RqPoly::from_raw(q, c)
    }

    /// `gcd(a, x^n + 1)` over GF(q) is 1 (Euclid's algorithm, O(n^2)): the independent unit
    /// test, since a is a unit of GF(q)[x]/(x^n+1) iff it is coprime to x^n + 1.
    fn coprime_to_xn1(a: &RqPoly) -> bool {
        let q = a.q as u64;
        let n = a.c.len();
        let inv = |x: u64| {
            let (mut r, mut b, mut e) = (1u64, x % q, q - 2);
            while e > 0 {
                if e & 1 == 1 {
                    r = r * b % q;
                }
                b = b * b % q;
                e >>= 1;
            }
            r
        };
        let trim = |v: &mut Vec<u64>| {
            while v.last() == Some(&0) {
                v.pop();
            }
        };
        let mut r0: Vec<u64> = vec![0; n + 1];
        r0[0] = 1;
        r0[n] = 1;
        let mut r1: Vec<u64> = a.c.iter().map(|&x| x as u64).collect();
        trim(&mut r1);
        while !r1.is_empty() {
            // r0 <- r0 mod r1
            let lead_inv = inv(*r1.last().unwrap());
            while r0.len() >= r1.len() {
                let f = r0.last().unwrap() * lead_inv % q;
                let shift = r0.len() - r1.len();
                for (i, &c) in r1.iter().enumerate() {
                    r0[shift + i] = (r0[shift + i] + q - f * c % q) % q;
                }
                trim(&mut r0);
            }
            core::mem::swap(&mut r0, &mut r1);
        }
        r0.len() == 1
    }

    /// S-RING-2: a * a^-1 = 1 for random a, at every degree and test modulus; every `None` is
    /// confirmed by an independent gcd computation, and `is_invertible` agrees.
    #[test]
    fn inverse_of_random_elements() {
        let mut nones = 0;
        for &q in QS.iter() {
            for logn in 1..=10u32 {
                let n = 1usize << logn;
                let mut rng = Shake256Prng::from_parts(&[b"inv", &q.to_le_bytes(), &[logn as u8]]);
                for _ in 0..3 {
                    let a = rand_poly(&mut rng, q, n);
                    let unit = coprime_to_xn1(&a);
                    assert_eq!(a.is_invertible(), unit, "q {q} logn {logn}");
                    match a.inverse() {
                        Some(ai) => {
                            assert!(unit);
                            assert_eq!(a.mul(&ai), one(q, n), "q {q} logn {logn}");
                        }
                        None => {
                            assert!(!unit, "q {q} logn {logn}");
                            nones += 1;
                        }
                    }
                }
            }
        }
        // The small moduli split x^n + 1 into many factors: some non-units must occur.
        assert!(nones > 0);
    }

    /// S-RING-2: non-units. At the PCS set, x^512 - 365259 (365259 = sqrt(-1) mod q) divides
    /// x^1024 + 1; at q = 12289, x - zeta for a root zeta of x^1024 + 1; zero; and a product with
    /// a non-unit. Constants are units.
    #[test]
    fn non_units() {
        let q = 1048573u32;
        assert_eq!((365259u64 * 365259) % q as u64, q as u64 - 1);
        let mut c = vec![0u32; 1024];
        c[512] = 1;
        c[0] = q - 365259;
        let f = RqPoly::from_raw(q, c);
        assert!(!f.is_invertible());
        assert!(f.inverse().is_none());
        let mut rng = Shake256Prng::new(b"non-units");
        let r = rand_poly(&mut rng, q, 1024);
        assert!(!f.mul(&r).is_invertible());
        // The other factor, x^512 + 365259.
        let mut c = vec![0u32; 1024];
        c[512] = 1;
        c[0] = 365259;
        assert!(RqPoly::from_raw(q, c).inverse().is_none());
        // Zero, and a nonzero constant.
        assert!(RqPoly::from_raw(q, vec![0; 1024]).inverse().is_none());
        let mut c = vec![0u32; 1024];
        c[0] = 5;
        let five = RqPoly::from_raw(q, c);
        let fi = five.inverse().unwrap();
        assert_eq!(fi.c[0] as u64 * 5 % q as u64, 1);
        assert!(fi.c[1..].iter().all(|&x| x == 0));

        // q = 12289: zeta of order 2048 (zeta^1024 = -1), then x - zeta.
        let q = 12289u32;
        let pw = |b: u64, e: u64| {
            let (mut r, mut b, mut e) = (1u64, b % q as u64, e);
            while e > 0 {
                if e & 1 == 1 {
                    r = r * b % q as u64;
                }
                b = b * b % q as u64;
                e >>= 1;
            }
            r
        };
        let zeta = (2..q as u64)
            .map(|b| pw(b, (q as u64 - 1) / 2048))
            .find(|&z| pw(z, 1024) == q as u64 - 1)
            .unwrap();
        let mut c = vec![0u32; 1024];
        c[1] = 1;
        c[0] = q - zeta as u32;
        let g = RqPoly::from_raw(q, c);
        assert!(!g.is_invertible());
        assert!(g.inverse().is_none());
    }

    /// S-RING-3 (part): encoding round trip, sizes, and the rejections.
    #[test]
    fn encoding() {
        for params in [Params::PCS, Params::FALCON_1024] {
            let mut rng = Shake256Prng::new(params.name.as_bytes());
            let a = rand_poly(&mut rng, params.q, params.n());
            let enc = a.to_bytes();
            assert_eq!(enc.len(), params.rq_bytes());
            assert_eq!(RqPoly::from_bytes(&params, &enc).unwrap(), a);
            // Wrong lengths.
            assert_eq!(
                RqPoly::from_bytes(&params, &enc[1..]),
                Err(Error::InvalidLength)
            );
            let mut long = enc.clone();
            long.push(0);
            assert_eq!(
                RqPoly::from_bytes(&params, &long),
                Err(Error::InvalidLength)
            );
            // A coefficient equal to q (the first one: set its w bits to q).
            let mut b = a.clone();
            b.c[0] = 0;
            let mut bad = b.to_bytes();
            let w = params.q_bits();
            let qv = params.q as u64;
            let mut acc = 0u64;
            for (k, byte) in bad.iter().take(4).enumerate() {
                acc |= (*byte as u64) << (8 * k);
            }
            acc |= qv & ((1u64 << w) - 1);
            for (k, byte) in bad.iter_mut().take(4).enumerate() {
                *byte = (acc >> (8 * k)) as u8;
            }
            assert_eq!(
                RqPoly::from_bytes(&params, &bad),
                Err(Error::InvalidEncoding)
            );
            // All-ones coefficients.
            let ff = vec![0xFFu8; params.rq_bytes()];
            assert_eq!(
                RqPoly::from_bytes(&params, &ff),
                Err(Error::InvalidEncoding)
            );
        }
        // Nonzero padding: q = 17 (5 bits), n = 4 -> 20 bits, 3 bytes, 4 padding bits.
        let toy = Params {
            name: "toy",
            logn: 2,
            q: 17,
            ..Params::FALCON_1024
        };
        let a = RqPoly::from_raw(17, vec![16, 0, 5, 9]);
        let enc = a.to_bytes();
        assert_eq!(enc.len(), 3);
        assert_eq!(RqPoly::from_bytes(&toy, &enc).unwrap(), a);
        let mut bad = enc.clone();
        bad[2] |= 0x80;
        assert_eq!(RqPoly::from_bytes(&toy, &bad), Err(Error::InvalidEncoding));
    }
}
