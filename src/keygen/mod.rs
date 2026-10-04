//! NTRU TrapGen (DLP14/Falcon convention).
//!
//! Design (design.md §5.9):
//! 1. $`f,g\leftarrow D_{\mathbb Z,\sigma_{fg}}^d`$ from the exact 128-bit CDT `params.fg_table`
//!    ([`sample_fg`]; sign bit plus half-Gaussian, rejecting $`(+,0)`$; binary search).
//! 2. Falcon's two Gram–Schmidt norms ([`gram_schmidt_sq_norms`]) against
//!    $`(1.17\sqrt q)^2`$: $`N_1=\|(g,-f)\|^2`$ exactly ($`10000N_1\le13689q`$), and
//!    $`N_2=\frac{2q^2}{d}\sum_{j<d/2}(|\hat f_j|^2+|\hat g_j|^2)^{-1}`$ in `f64` through
//!    `crate::fft` (fn-dsa-kgen/src/lib.rs:292-302 and ntru.rs:18-43 compute the same two
//!    norms at $`q=12289`$; falcon-rust src/math.rs:1536-1568 in floating point).
//! 3. $`f`$ invertible modulo $`q`$ and $`h=g\,f^{-1}`$ ([`RqPoly::inverse`]).
//! 4. NTRUSolve ([`ntru_solve`], big integers, stage 1): $`fG-gF=q`$, Babai-reduced.
//! 5. $`F,G`$ fit `sk_big_fg_bits`; exact check $`fG-gF=q`$ over $`\mathbb Z[x]/(x^d+1)`$ (the
//!    two-prime NTT; the big-integer version in `bigpoly` serves as the tests' oracle).
//! 6. The leaf check of [`crate::ExpandedKey::new`]: every leaf width in the base sampler's
//!    range. [`trapgen`] requires [`Params::check_leaf_window`], under which this check never
//!    fires except through `f64` rounding at the boundary (the leaf range is implied by the two
//!    norm tests; `params` module documentation).
//!
//! Randomness: one stream `Shake256Prng::from_parts(["ntru-trapdoor/v1/keygen", LE32(q),
//! [logn], seed])`; candidates are drawn from it in order (f, then g). Variable time: the
//! binary-search CDT, the rejection loop and the big-integer solver leak timing (design.md §10).
//!
//! Submodules: `gauss` ($`f,g`$ sampler), `gs` (the norms), `bigpoly` (Kronecker products,
//! field norm, lift, exact NTRU equation), `flat` (the allocation-free limb arithmetic of the
//! Babai reduction), `xgcd` (Lehmer), `solve` (NTRUSolve), `trapgen` (the candidate loop,
//! `public_key`, `validate`).

mod bigpoly;
mod flat;
mod gauss;
mod gs;
mod solve;
mod trapgen;
mod xgcd;

#[cfg(test)]
mod tests;

pub(crate) use trapgen::fits_width;

use crate::prng::Prng;
use crate::{Error, Params, PublicKey, SecretKey};

#[cfg(doc)]
use crate::RqPoly;

/// Why [`trapgen_from_fg`] rejected a pair $`(f,g)`$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KeygenReject {
    /// The input is malformed (wrong length, invalid parameter set).
    Input,
    /// $`\|(g,-f)\|^2>(1.17\sqrt q)^2`$.
    NormFg,
    /// The second Gram–Schmidt norm exceeds $`(1.17\sqrt q)^2`$.
    NormOrtho,
    /// $`f`$ is not invertible modulo $`q`$.
    NotInvertible,
    /// NTRUSolve failed (the resultants of $`f`$ and $`g`$ are not coprime, or the reduction did
    /// not converge).
    Solve,
    /// A coefficient of $`f,g,F,G`$ does not fit the secret-key encoding widths.
    Size,
    /// A leaf width of the ffLDL tree lies outside the base sampler's range.
    Leaf,
}

/// Counters of one key generation: candidates drawn and why they were rejected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct KeygenStats {
    /// Candidate pairs $`(f,g)`$ drawn (the accepted one included).
    pub candidates: u32,
    /// Rejected by $`\|(g,-f)\|^2`$.
    pub rejected_norm_fg: u32,
    /// Rejected by the second Gram–Schmidt norm.
    pub rejected_norm_ortho: u32,
    /// Rejected because $`f`$ is not invertible modulo $`q`$.
    pub rejected_not_invertible: u32,
    /// Rejected by NTRUSolve.
    pub rejected_solve: u32,
    /// Rejected by the encoding widths.
    pub rejected_size: u32,
    /// Rejected by the leaf-width check.
    pub rejected_leaf: u32,
}

/// TrapGen: a key pair for `params` from a 32-byte seed (deterministic). The seed must be
/// secret and uniform (the PCS draws it from its RNG).
///
/// Errors: [`Error::InvalidParams`] (from [`Params::validate`] or
/// [`Params::check_leaf_window`]); [`Error::KeygenExhausted`] after
/// `params.keygen_max_attempts` candidates.
pub fn trapgen(params: &Params, seed: &[u8; 32]) -> Result<(PublicKey, SecretKey), Error> {
    trapgen::trapgen_with_stats(params, seed).map(|(pk, sk, _)| (pk, sk))
}

/// [`trapgen`], also returning the rejection counters (for tests and benchmarks).
pub fn trapgen_with_stats(
    params: &Params,
    seed: &[u8; 32],
) -> Result<(PublicKey, SecretKey, KeygenStats), Error> {
    trapgen::trapgen_with_stats(params, seed)
}

/// The deterministic part of TrapGen for a given pair $`(f,g)`$: the checks of the module
/// documentation, cheapest first (widths of $`f,g`$; $`N_1`$; $`N_2`$; invertibility of $`f`$;
/// NTRUSolve; widths of $`F,G`$ and the exact equation; the leaf check). For tests, the Sage
/// cross-checks and toy parameter sets: unlike [`trapgen`], it requires only
/// [`Params::validate`], not [`Params::check_leaf_window`], so that the leaf rejection can be
/// exercised.
pub fn trapgen_from_fg(
    params: &Params,
    f: &[i16],
    g: &[i16],
) -> Result<(PublicKey, SecretKey), KeygenReject> {
    trapgen::trapgen_from_fg(params, f, g)
}

/// Falcon's two squared Gram–Schmidt norms of $`(f,g)`$: $`N_1=\|(g,-f)\|^2`$ (exact) and
/// $`N_2=\|(qf^*/(ff^*+gg^*),\,qg^*/(ff^*+gg^*))\|^2`$ (`f64`; infinite if $`f`$ and $`g`$
/// have a common root of $`x^d+1`$). Panics unless both lengths are `params.n()`.
pub fn gram_schmidt_sq_norms(params: &Params, f: &[i16], g: &[i16]) -> (u64, f64) {
    gs::gram_schmidt_sq_norms(params, f, g)
}

/// Fills `out` with independent samples of $`D_{\mathbb Z,\sigma_{fg}}`$ from
/// `params.fg_table`: per coefficient $`K=\#\{j:V<T_j\}`$ for a uniform 128-bit $`V`$ and a
/// sign bit $`b`$; return $`-K`$ if $`b=0`$, $`K`$ if $`b=1`$ and $`K\ne0`$, and draw again if
/// $`b=1,K=0`$ (exact: $`\Pr[z]\propto\rho_{\sigma_{fg}}(z)`$). Drawn lazily: one `next_u64`
/// gives $`b`$ (its low bit) and the top 63 bits of $`V`$; only when those tie with the top 63
/// bits of a table entry are the low 65 bits read (`next_u64`, then the low bit of `next_u8`).
pub fn sample_fg<P: Prng>(params: &Params, prng: &mut P, out: &mut [i16]) {
    gauss::sample_fg(params, prng, out)
}

/// NTRUSolve over big integers (stage 1): $`(F,G)`$ with $`fG-gF=q`$ in
/// $`\mathbb Z[x]/(x^d+1)`$, Babai-reduced against $`(f,g)`$, or `None` if the resultants of $`f`$
/// and $`g`$ are not coprime (or a reduced coefficient does not fit `i32`). $`d=f\text{.len()}`$,
/// a power of two (panics otherwise, or if `g.len() != d`).
pub fn ntru_solve(q: u32, f: &[i16], g: &[i16]) -> Option<(Vec<i32>, Vec<i32>)> {
    solve::ntru_solve(q, f, g)
}

impl SecretKey {
    /// The public key $`h=g\,f^{-1}\bmod q`$. Errors: [`Error::InvalidKey`] if $`f`$ is not
    /// invertible.
    pub fn public_key(&self) -> Result<PublicKey, Error> {
        trapgen::public_key(self)
    }

    /// Checks $`fG-gF=q`$ exactly (two-prime NTT, about 40 µs at $`d=1024`$), that every
    /// coefficient fits the encoding widths (in particular none equals $`-2^{w-1}`$), and that
    /// $`f`$ is invertible modulo $`q`$ (so that [`SecretKey::public_key`] succeeds). Errors:
    /// [`Error::InvalidKey`].
    pub fn validate(&self) -> Result<(), Error> {
        trapgen::validate(self)
    }
}
