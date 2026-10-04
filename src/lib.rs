//! # ntru-trapdoor
//!
//! NTRU lattice trapdoors in the DLP14 / Falcon convention: the trapdoor machinery of the
//! lattice instantiation of the predicate credential system (PCS), the credential base Σ-vSIS
//! (building-blocks.tex, sec:inst-base-vsis), as a standalone library. The PCS-specific parts
//! (the public parameters $`v,b_u,b_\phi,B_r`$, the signing randomness $`x_1`$, the credential)
//! are not here.
//!
//! **Status.** Implemented, reviewed and tested end to end at both presets.
//!
//! ## Example
//!
//! ```
//! use ntru_trapdoor::{ExpandedKey, Params, RqPoly, samp_pre, trapgen, verify};
//!
//! let params = Params::PCS;
//! // Seeds must be fresh, secret and uniform (from the caller's RNG); fixed here for brevity.
//! let (pk, sk) = trapgen(&params, &[7u8; 32])?;
//! let ek = ExpandedKey::new(&sk)?; // the ffLDL tree, once per key
//! let t = RqPoly::from_coeffs(&params, &vec![12345; params.n()])?; // any element of R_q
//! let s = samp_pre(&ek, &t, &[9u8; 32])?; // a fresh seed per call
//! assert!(verify(&pk, &t, &s, params.beta_sq));
//! # Ok::<(), ntru_trapdoor::Error>(())
//! ```
//!
//! ## What the crate provides
//!
//! - [`Params`]: parameter sets, generic over $`(d,q,\text{widths})`$, with the presets
//!   [`Params::PCS`] ($`d=1024`$, $`q=1048573\equiv5\bmod8`$) and [`Params::FALCON_1024`]
//!   (fn-dsa's constants, for bit-exact parity tests).
//! - [`RqPoly`]: $`R_q=\mathbb Z_q[x]/(x^d+1)`$ for any $`q<2^{25}`$, including
//!   $`q\equiv5\bmod8`$ where there is no full NTT: exact multiplication (two-prime NTT and
//!   CRT), prepared products for fixed operands ([`PreparedRqPoly`]), inversion and
//!   invertibility (norm tower), canonical bit-packed encoding (20 bits per coefficient, 2560 B,
//!   at the PCS set).
//! - [`trapgen`]: TrapGen. $`f,g`$ from an exact 128-bit CDT of
//!   $`D_{\mathbb Z,\sigma_{fg}}`$, $`\sigma_{fg}=1.17\sqrt{q/(2d)}`$; rejection unless $`f`$ is
//!   invertible, both Falcon Gram–Schmidt norms are at most $`1.17\sqrt q`$, NTRUSolve finds
//!   $`fG-gF=q`$ exactly, and every ffSampling leaf width lies in the base sampler's range.
//!   The public key is $`h=g\,f^{-1}\bmod q`$ and $`A=[1\mid h]`$.
//! - [`ExpandedKey`]: the per-key ffLDL tree and FFT basis (Falcon's expanded key), built from a
//!   validated key.
//! - [`samp_pre`]: SampPre for an arbitrary target $`t\in R_q`$ (no hash-to-point): Falcon's
//!   fast-Fourier sampling in native `f64`, $`s=(s_0,s_1)`$ with $`s_0+h\,s_1=t\bmod q`$ and
//!   $`\|s\|^2\le\lfloor\beta^2\rfloor`$.
//! - [`verify`]: the norm bound and the linear relation (a product with the prepared $`h`$).
//! - Key encodings: [`PublicKey::to_bytes`], [`SecretKey::to_bytes`] and their inverses.
//!
//! ## Conventions
//!
//! The paper's width $`\varsigma`$ is the GPV s-parameter, $`\rho_s(x)=e^{-\pi\|x\|^2/s^2}`$;
//! Falcon's and this crate's widths are standard-deviation parameters,
//! $`\rho(x)=e^{-x^2/(2\sigma^2)}`$, so $`\sigma=\varsigma/\sqrt{2\pi}`$. At the PCS set
//! $`\varsigma=6658.66`$ and $`\sigma=2656.42`$ (module `params`).
//!
//! ## Randomness
//!
//! The core functions take a 32-byte seed; internally they expand it with SHAKE256 (fn-dsa's
//! PRNG construction, so that the sampler can be compared bit for bit with fn-dsa). The caller
//! supplies fresh uniform seeds from a cryptographic RNG.
//!
//! ## Distribution
//!
//! The base sampler, the f, g sampler and every rejection step are exact up to their tables
//! (the base table is rounded at 192 bits) and FACCT's exponential (about $`2^{-51}`$ relative);
//! unlike fn-dsa, the Bernoulli step never over-weights the far tail (module `sampler`). The
//! ffSampling centres are computed in native `f64`: within $`2^{-35.4}\sigma'`$ of the exact
//! values on 1064 replayed calls at the PCS set. So a per-call statistical distance of
//! $`2^{-128}`$ is not claimed; the README gives the measured Rényi budget (order 256) per
//! number of calls.
//!
//! ## Side channels
//!
//! Performance comes first. SampPre keeps fn-dsa's isochronous sampler (fixed-length table
//! scans, a Bernoulli test scaled by $`\sigma_{\min}/\sigma'`$), but native `f64` arithmetic,
//! the rejection loops and key generation (binary-search CDT, big-integer NTRUSolve) are
//! variable time. The ring arithmetic uses masks; the inverse reveals only whether an element is
//! a unit. Secret keys, expanded keys, PRNG states and preimages are wiped on drop; NTRUSolve's
//! big integers are not (num-bigint cannot wipe its buffers).
//!
//! ## Credits
//!
//! Ported from fn-dsa (Thomas Pornin, The Unlicense), falcon-rust (Alan Szepieniec, MIT) and
//! Jali (MIT); see `NOTICE`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

// Shared types.
mod error;
mod keys;
mod params;
mod poly;
pub mod tables;

// R_q, FFT, PRNG, sampler, expanded key, SampPre and Verify.
mod ffsamp;
mod fft;
mod prng;
mod ring;
mod sampler;
mod sign;

// TrapGen and the key encodings.
mod codec;
mod keygen;

pub use error::Error;
pub use ffsamp::ExpandedKey;
pub use keygen::{KeygenReject, KeygenStats, trapgen, trapgen_from_fg, trapgen_with_stats};
pub use keys::{PublicKey, SecretKey};
pub use params::{BaseSampler, Params};
pub use poly::{Preimage, PreparedRqPoly, RqPoly};
pub use sign::{samp_pre, verify};

/// Low-level building blocks for tests, benchmarks and the fn-dsa parity harness.
///
/// No stability promise; misuse (for example reusing a PRNG stream across calls) voids the
/// distribution guarantees of [`crate::samp_pre`].
pub mod hazmat {
    pub use crate::fft::{fft, ifft};
    pub use crate::keygen::{gram_schmidt_sq_norms, ntru_solve, sample_fg};
    pub use crate::prng::{Prng, Shake256Prng, shake256};
    pub use crate::sampler::{LeafSampler, SamplerZ};
    pub use crate::sign::{samp_pre_attempt, samp_pre_with};
}
