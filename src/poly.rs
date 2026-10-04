//! Shared polynomial types: elements of $`R_q`$ and SampPre's output.
//!
//! Shared by the signing and key-generation sides; the arithmetic and the encoding of
//! [`RqPoly`] live in `ring`.

use core::fmt;

use zeroize::{DefaultIsZeroes, Zeroize, ZeroizeOnDrop};

use crate::{Error, Params};

/// Wipes the elements and the spare capacity of `v` (each byte once, volatile), then
/// truncates it.
///
/// The wipes of this crate use this instead of `Vec::zeroize` (and of the `Zeroize` derive on
/// structs with `Vec` fields), which in zeroize 1.9 wipes the elements one at a time (a
/// volatile write and an optimisation barrier each), then the whole capacity again byte by
/// byte: about ten times the cost [measured: 9.4 µs against 0.9 µs for 4096
/// `f64`].
pub(crate) fn wipe_vec<Z: DefaultIsZeroes>(v: &mut Vec<Z>) {
    v.as_mut_slice().zeroize();
    v.spare_capacity_mut().zeroize();
    v.clear();
}

/// An element of $`R_q=\mathbb Z_q[x]/(x^d+1)`$, with canonical coefficients in $`[0,q)`$,
/// coefficient $`i`$ of $`x^i`$ at index $`i`$.
///
/// Arithmetic (module `ring`): [`RqPoly::add`], [`RqPoly::sub`], [`RqPoly::neg`],
/// [`RqPoly::mul`] (exact, two-prime NTT and CRT), [`RqPoly::prepare`] (for repeated products
/// with one operand), [`RqPoly::inverse`] and [`RqPoly::is_invertible`] (norm tower),
/// [`RqPoly::to_bytes`] and [`RqPoly::from_bytes`] (canonical bit packing). Operations on two
/// elements panic if their modulus or degree differ.
///
/// Every constructor checks the ring of its [`Params`] ($`1\le\mathrm{logn}\le10`$,
/// $`3\le q<2^{25}`$) and returns [`Error::InvalidParams`] otherwise, so the arithmetic is exact
/// for every element that exists; primality of $`q`$ is not needed (the inverse is exact for a
/// composite $`q`$ too).
///
/// Not wiped on drop (most elements are public); [`Zeroize`] is implemented for secret ones,
/// for example through [`zeroize::Zeroizing`]. `Debug` prints the modulus and the degree only.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RqPoly {
    /// The modulus $`q`$.
    pub(crate) q: u32,
    /// The coefficients, each in $`[0,q)`$; the degree is `c.len()`.
    pub(crate) c: Vec<u32>,
}

impl Zeroize for RqPoly {
    /// Wipes the coefficients (and the modulus) and empties the element.
    fn zeroize(&mut self) {
        self.q.zeroize();
        wipe_vec(&mut self.c);
    }
}

impl RqPoly {
    /// The zero element of $`R_q`$ for `params`. Errors: [`Error::InvalidParams`].
    pub fn zero(params: &Params) -> Result<RqPoly, Error> {
        params.check_ring()?;
        Ok(RqPoly {
            q: params.q,
            c: vec![0; params.n()],
        })
    }

    /// From canonical coefficients: `c.len()` must be $`d`$ and every entry below $`q`$.
    /// Errors: [`Error::InvalidParams`], [`Error::InvalidLength`], [`Error::InvalidEncoding`].
    pub fn from_coeffs(params: &Params, c: &[u32]) -> Result<RqPoly, Error> {
        params.check_ring()?;
        if c.len() != params.n() {
            return Err(Error::InvalidLength);
        }
        if c.iter().any(|&x| x >= params.q) {
            return Err(Error::InvalidEncoding);
        }
        Ok(RqPoly {
            q: params.q,
            c: c.to_vec(),
        })
    }

    /// From arbitrary integers, reduced modulo $`q`$; `c.len()` must be $`d`$. Errors:
    /// [`Error::InvalidParams`], [`Error::InvalidLength`].
    pub fn from_i64(params: &Params, c: &[i64]) -> Result<RqPoly, Error> {
        params.check_ring()?;
        if c.len() != params.n() {
            return Err(Error::InvalidLength);
        }
        let q = params.q as i64;
        Ok(RqPoly {
            q: params.q,
            c: c.iter().map(|&x| x.rem_euclid(q) as u32).collect(),
        })
    }

    /// From `i32` integers, reduced modulo $`q`$; `c.len()` must be $`d`$. Errors:
    /// [`Error::InvalidParams`], [`Error::InvalidLength`].
    pub fn from_i32(params: &Params, c: &[i32]) -> Result<RqPoly, Error> {
        params.check_ring()?;
        if c.len() != params.n() {
            return Err(Error::InvalidLength);
        }
        let q = params.q as i64;
        Ok(RqPoly {
            q: params.q,
            c: c.iter().map(|&x| (x as i64).rem_euclid(q) as u32).collect(),
        })
    }

    /// From `i16` integers (for example $`f`$, $`g`$, $`F`$, $`G`$), reduced modulo $`q`$;
    /// `c.len()` must be $`d`$. Errors: [`Error::InvalidParams`], [`Error::InvalidLength`].
    pub fn from_i16(params: &Params, c: &[i16]) -> Result<RqPoly, Error> {
        params.check_ring()?;
        if c.len() != params.n() {
            return Err(Error::InvalidLength);
        }
        let q = params.q as i64;
        Ok(RqPoly {
            q: params.q,
            c: c.iter().map(|&x| (x as i64).rem_euclid(q) as u32).collect(),
        })
    }

    /// Crate-internal constructor without checks: `c[i] < q` and a power-of-two length are the
    /// caller's responsibility.
    pub(crate) fn from_raw(q: u32, c: Vec<u32>) -> RqPoly {
        debug_assert!(c.len().is_power_of_two() && c.iter().all(|&x| x < q));
        RqPoly { q, c }
    }

    /// The modulus $`q`$.
    pub fn q(&self) -> u32 {
        self.q
    }

    /// The degree $`d`$ (number of coefficients).
    pub fn n(&self) -> usize {
        self.c.len()
    }

    /// The canonical coefficients, each in $`[0,q)`$.
    pub fn coeffs(&self) -> &[u32] {
        &self.c
    }

    /// The centred representatives, in $`(-q/2,q/2]`$.
    pub fn to_centered(&self) -> Vec<i32> {
        let q = self.q as i64;
        self.c
            .iter()
            .map(|&x| {
                let x = x as i64;
                (if 2 * x > q { x - q } else { x }) as i32
            })
            .collect()
    }

    /// `true` if this element belongs to the ring of `params` (same modulus and degree).
    pub fn matches(&self, params: &Params) -> bool {
        self.q == params.q && self.c.len() == params.n()
    }
}

impl fmt::Debug for RqPoly {
    /// The modulus and the degree only: an element may be secret (for example $`f^{-1}`$).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RqPoly {{ q: {}, n: {}, .. }}", self.q, self.c.len())
    }
}

/// An element $`a\in R_q`$ prepared for repeated products: its residues modulo the two NTT
/// primes of the multiplication, in NTT form ([`RqPoly::prepare`]). [`PreparedRqPoly::mul`]
/// equals [`RqPoly::mul`] and computes four NTTs instead of six. For fixed operands such as
/// the public key $`h`$ ([`crate::PublicKey::h_prepared`]) or the PCS's public parameters.
///
/// Not wiped on drop (prepared operands are normally public); [`Zeroize`] is implemented.
/// `Debug` prints the modulus and the degree only.
#[derive(Clone, PartialEq, Eq)]
pub struct PreparedRqPoly {
    /// The modulus $`q`$.
    pub(crate) q: u32,
    /// $`\log_2 d`$.
    pub(crate) logn: u32,
    /// NTT of $`a`$ modulo $`p_1`$ ($`d`$ values).
    pub(crate) r1: Vec<u32>,
    /// NTT of $`a`$ modulo $`p_2`$ ($`d`$ values).
    pub(crate) r2: Vec<u32>,
}

impl Zeroize for PreparedRqPoly {
    /// Wipes the residues and empties them.
    fn zeroize(&mut self) {
        wipe_vec(&mut self.r1);
        wipe_vec(&mut self.r2);
    }
}

impl fmt::Debug for PreparedRqPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PreparedRqPoly {{ q: {}, n: {}, .. }}",
            self.q,
            1usize << self.logn
        )
    }
}

/// SampPre's output $`s=(s_0,s_1)\in R^2`$ with $`s_0+h\,s_1=t\bmod q`$
/// ($`A=[1\mid h]`$), as signed integer coefficients.
///
/// In the PCS the preimage is the credential holder's secret witness, so it is wiped on drop
/// (elements and spare capacity, one volatile pass).
#[derive(Clone, PartialEq, Eq)]
pub struct Preimage {
    /// $`s_0`$, the coefficient vector multiplying $`1`$.
    pub s0: Vec<i32>,
    /// $`s_1`$, the coefficient vector multiplying $`h`$.
    pub s1: Vec<i32>,
}

impl Zeroize for Preimage {
    /// Wipes both halves and empties them.
    fn zeroize(&mut self) {
        wipe_vec(&mut self.s0);
        wipe_vec(&mut self.s1);
    }
}

impl Drop for Preimage {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl ZeroizeOnDrop for Preimage {}

impl Preimage {
    /// $`\|s\|^2=\|s_0\|^2+\|s_1\|^2`$, saturating at `u64::MAX`.
    pub fn norm_sq(&self) -> u64 {
        let sum: u128 = self
            .s0
            .iter()
            .chain(self.s1.iter())
            .map(|&x| {
                let x = x.unsigned_abs() as u128;
                x * x
            })
            .sum();
        u64::try_from(sum).unwrap_or(u64::MAX)
    }

    /// The degree $`d`$ (the length of each half).
    pub fn n(&self) -> usize {
        self.s0.len()
    }
}

impl fmt::Debug for Preimage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Preimage {{ n: {}, norm_sq: {}, .. }}",
            self.s0.len(),
            self.norm_sq()
        )
    }
}
