//! Key types.
//!
//! Key generation, [`SecretKey::public_key`], [`SecretKey::validate`] and the encodings live
//! in `keygen` and `codec`; the expanded key lives in `ffsamp`.

use core::fmt;
use std::sync::OnceLock;

use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::poly::wipe_vec;
use crate::{Error, Params, PreparedRqPoly, RqPoly};

/// The public key $`h=g\,f^{-1}\bmod q`$, so that $`A=[1\mid h]`$ (the paper's normal form).
///
/// Keeps $`h`$ in prepared form ([`PublicKey::h_prepared`]) once it is first needed, so that
/// [`crate::verify`] computes four NTTs per call instead of six. `PartialEq` compares the
/// parameter sets and $`h`$.
#[derive(Clone)]
pub struct PublicKey {
    pub(crate) params: Params,
    pub(crate) h: RqPoly,
    /// $`h`$ prepared for products, computed on first use.
    h_prepared: OnceLock<PreparedRqPoly>,
}

impl PublicKey {
    /// Wraps an element $`h\in R_q`$ of the ring of `params`. Every element is accepted (the
    /// PCS's $`V_{pp}`$ accepts every canonically encoded $`h`$).
    pub fn from_h(params: &Params, h: RqPoly) -> Result<PublicKey, Error> {
        params.validate()?;
        if !h.matches(params) {
            return Err(Error::Mismatch);
        }
        Ok(PublicKey::from_valid(params, h))
    }

    /// For crate code whose `params` passed [`Params::validate`] and whose `h` has their ring.
    pub(crate) fn from_valid(params: &Params, h: RqPoly) -> PublicKey {
        debug_assert!(h.matches(params));
        PublicKey {
            params: *params,
            h,
            h_prepared: OnceLock::new(),
        }
    }

    /// The parameter set.
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// $`h`$.
    pub fn h(&self) -> &RqPoly {
        &self.h
    }

    /// $`h`$ prepared for repeated products ([`crate::PreparedRqPoly::mul`]): computed on the
    /// first call (two NTTs per prime, about 9 µs at $`d=1024`$) and kept.
    pub fn h_prepared(&self) -> &PreparedRqPoly {
        self.h_prepared.get_or_init(|| self.h.prepare())
    }
}

impl PartialEq for PublicKey {
    fn eq(&self, other: &PublicKey) -> bool {
        self.params == other.params && self.h == other.h
    }
}

impl fmt::Debug for PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PublicKey")
            .field("params", &self.params.name)
            .field("h", &self.h)
            .finish()
    }
}

/// The NTRU trapdoor $`(f,g,F,G)`$ with $`fG-gF=q`$ in $`\mathbb Z[x]/(x^d+1)`$; the lattice
/// basis is $`B=\begin{pmatrix}g&-f\\G&-F\end{pmatrix}`$ (Falcon's convention). Wiped on drop.
#[derive(Clone)]
pub struct SecretKey {
    pub(crate) params: Params,
    pub(crate) f: Vec<i16>,
    pub(crate) g: Vec<i16>,
    pub(crate) big_f: Vec<i16>,
    pub(crate) big_g: Vec<i16>,
    /// `true` once the key passed [`SecretKey::validate`]: set by key generation and by
    /// [`SecretKey::from_bytes`], cleared by `zeroize`. [`crate::ExpandedKey::new`] validates a
    /// key without it. The key's fields never change after construction (no mutator but
    /// `zeroize`), so the flag stays true to the key.
    pub(crate) validated: bool,
}

impl Zeroize for SecretKey {
    /// Wipes $`f,g,F,G`$ and empties them (the parameter set is public and kept).
    fn zeroize(&mut self) {
        wipe_vec(&mut self.f);
        wipe_vec(&mut self.g);
        wipe_vec(&mut self.big_f);
        wipe_vec(&mut self.big_g);
        self.validated = false;
    }
}

impl Drop for SecretKey {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl ZeroizeOnDrop for SecretKey {}

impl SecretKey {
    /// Assembles a secret key from its four polynomials. Checks the parameter set and the
    /// lengths only; [`SecretKey::validate`] checks $`fG-gF=q`$, the encoding widths and the
    /// invertibility of $`f`$, and [`crate::ExpandedKey::new`] runs it before using the key.
    pub fn from_parts(
        params: &Params,
        f: Vec<i16>,
        g: Vec<i16>,
        big_f: Vec<i16>,
        big_g: Vec<i16>,
    ) -> Result<SecretKey, Error> {
        params.validate()?;
        let n = params.n();
        if f.len() != n || g.len() != n || big_f.len() != n || big_g.len() != n {
            return Err(Error::InvalidLength);
        }
        Ok(SecretKey {
            params: *params,
            f,
            g,
            big_f,
            big_g,
            validated: false,
        })
    }

    /// The parameter set.
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// $`f`$.
    pub fn f(&self) -> &[i16] {
        &self.f
    }

    /// $`g`$.
    pub fn g(&self) -> &[i16] {
        &self.g
    }

    /// $`F`$.
    pub fn big_f(&self) -> &[i16] {
        &self.big_f
    }

    /// $`G`$.
    pub fn big_g(&self) -> &[i16] {
        &self.big_g
    }
}

impl fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecretKey {{ params: {:?}, .. }}", self.params.name)
    }
}
