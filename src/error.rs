//! The crate's error type.
//!
//! Shared by the signing and key-generation sides.

use core::fmt;

/// Errors of the public API.
///
/// Rejections inside key generation are not errors: [`crate::trapgen`] resamples, and
/// [`crate::trapgen_from_fg`] reports them as [`crate::KeygenReject`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    /// A [`crate::Params`] value fails [`crate::Params::validate`]; the string names the check.
    InvalidParams(&'static str),
    /// Operands, keys or targets belong to different parameter sets (modulus or degree).
    Mismatch,
    /// A slice or an encoding has the wrong length.
    InvalidLength,
    /// A byte string is not the canonical encoding of the expected object.
    InvalidEncoding,
    /// The element of `R_q` has no inverse.
    NotInvertible,
    /// A secret key is inconsistent: `f G - g F != q`, `f` is not invertible modulo `q`, or a
    /// coefficient does not fit the encoding widths of the parameter set.
    InvalidKey,
    /// A leaf width of the ffLDL tree lies outside the base sampler's range
    /// `[sigma_min, sigma0)`; the key cannot be used with this parameter set.
    LeafOutOfRange,
    /// Key generation drew [`crate::Params::keygen_max_attempts`] candidates without success.
    KeygenExhausted,
    /// SampPre made [`crate::Params::samppre_max_attempts`] attempts without a short enough
    /// preimage.
    SampPreExhausted,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidParams(what) => write!(f, "invalid parameter set: {what}"),
            Error::Mismatch => f.write_str("parameter-set mismatch (modulus or degree)"),
            Error::InvalidLength => f.write_str("invalid length"),
            Error::InvalidEncoding => f.write_str("invalid or non-canonical encoding"),
            Error::NotInvertible => f.write_str("element of R_q is not invertible"),
            Error::InvalidKey => f.write_str("inconsistent secret key"),
            Error::LeafOutOfRange => {
                f.write_str("ffLDL leaf width outside the base sampler's range")
            }
            Error::KeygenExhausted => f.write_str("key generation exhausted its attempts"),
            Error::SampPreExhausted => f.write_str("SampPre exhausted its attempts"),
        }
    }
}

impl std::error::Error for Error {}
