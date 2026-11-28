//! The wiping of secrets that happens by itself.
//!
//! The drops of the secret types and the scratch buffers of an operation wipe through these
//! helpers, so that the crate has one place that decides it. They always wipe.

use core::ops::{Deref, DerefMut};

use zeroize::{DefaultIsZeroes, Zeroize};

/// Whether secrets are wiped by themselves: on drop and after an operation. Always.
pub(crate) const ENABLED: bool = true;

/// Wipes `s` (one volatile pass).
#[inline]
pub(crate) fn slice<Z: DefaultIsZeroes>(s: &mut [Z]) {
    if ENABLED {
        s.zeroize();
    }
}

/// Wipes the elements and the spare capacity of `v` and empties it ([`crate::poly::wipe_vec`]).
#[inline]
pub(crate) fn vec<Z: DefaultIsZeroes>(v: &mut Vec<Z>) {
    if ENABLED {
        crate::poly::wipe_vec(v);
    }
}

/// A value wiped on drop by its own `Zeroize`: [`zeroize::Zeroizing`] for the temporaries of an
/// operation.
pub(crate) struct Secret<T: Zeroize>(T);

impl<T: Zeroize> Secret<T> {
    /// Wraps `value`.
    #[inline]
    pub(crate) fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: Zeroize> Deref for Secret<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Zeroize> DerefMut for Secret<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: Zeroize> Drop for Secret<T> {
    #[inline]
    fn drop(&mut self) {
        if ENABLED {
            self.0.zeroize();
        }
    }
}

impl<T: Zeroize> core::fmt::Debug for Secret<T> {
    /// Nothing of the value.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Secret(..)")
    }
}
