//! Byte encodings of the keys.
//!
//! Design (design.md §5.10):
//! - public key: the canonical encoding of $`h`$ ([`RqPoly::to_bytes`]), no
//!   header: `params.rq_bytes()` bytes (2560 at the PCS set). Every canonical $`h`$ is accepted.
//! - secret key: one header byte `0xA0 | logn`, then $`f`$, $`g`$ at `sk_fg_bits` and $`F`$,
//!   $`G`$ at `sk_big_fg_bits` bits per coefficient, two's complement, little-endian bit packing
//!   as fn-dsa's `trim_i8_encode` (fn-dsa-comm/src/codec.rs:7-77), each polynomial padded to a
//!   whole byte with zeros. The most negative value $`-2^{w-1}`$ is rejected (as fn-dsa does),
//!   and decoding runs [`SecretKey::validate`] ($`fG-gF=q`$, $`f`$ invertible). PCS: 1 + 5888 = 5889 bytes;
//!   Falcon-1024: 1 + 4096 = 4097 bytes. The encoding is returned in a [`Zeroizing`] buffer.
//!
//! The bit packing is fn-dsa's `trim_i8_encode` / `trim_i8_decode` (fn-dsa-comm/src/codec.rs:
//! 7-71, Pornin, The Unlicense) generalised from `i8` to `i16` coefficients and widths up to 16
//! bits, with a zero-padded last byte per polynomial (fn-dsa requires whole bytes) and a check
//! that the padding is zero, so that every key has exactly one encoding.

use zeroize::{Zeroize, Zeroizing};

use crate::{Error, Params, PublicKey, RqPoly, SecretKey};

/// The secret-key header byte: `0xA0 | logn`.
fn sk_header(params: &Params) -> u8 {
    0xA0 | params.logn as u8
}

/// Bytes of one polynomial of `n` coefficients at `w` bits each.
fn poly_bytes(n: usize, w: u32) -> usize {
    (n * w as usize).div_ceil(8)
}

/// The length of a secret-key encoding (5889 at the PCS set).
pub(crate) fn sk_bytes(params: &Params) -> usize {
    let n = params.n();
    1 + 2 * poly_bytes(n, params.sk_fg_bits) + 2 * poly_bytes(n, params.sk_big_fg_bits)
}

/// Appends `x` at `w` bits per coefficient (two's complement, little-endian bit packing, zero
/// padding to a byte); fn-dsa `trim_i8_encode` (fn-dsa-comm/src/codec.rs:7-24). Every value
/// must satisfy $`|x_i|<2^{w-1}`$; `2 <= w <= 16`.
fn encode_poly(x: &[i16], w: u32, out: &mut Vec<u8>) {
    let mask = (1u32 << w) - 1;
    let mut acc = 0u32;
    let mut acc_len = 0u32;
    for &v in x {
        acc |= ((v as i32 as u32) & mask) << acc_len;
        acc_len += w;
        while acc_len >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            acc_len -= 8;
        }
    }
    if acc_len > 0 {
        out.push(acc as u8);
    }
    acc.zeroize();
}

/// Decodes `n` coefficients at `w` bits from exactly `poly_bytes(n, w)` bytes; fn-dsa
/// `trim_i8_decode` (fn-dsa-comm/src/codec.rs:39-71). Rejects $`-2^{w-1}`$ and nonzero padding
/// bits ([`Error::InvalidEncoding`]).
fn decode_poly(bytes: &[u8], n: usize, w: u32) -> Result<Zeroizing<Vec<i16>>, Error> {
    debug_assert_eq!(bytes.len(), poly_bytes(n, w));
    let mask = (1u32 << w) - 1;
    let sign = 1u32 << (w - 1);
    let mut out = Zeroizing::new(Vec::with_capacity(n));
    let mut acc = 0u32;
    let mut acc_len = 0u32;
    let mut it = bytes.iter();
    for _ in 0..n {
        while acc_len < w {
            acc |= (*it.next().ok_or(Error::InvalidLength)? as u32) << acc_len;
            acc_len += 8;
        }
        let v = acc & mask;
        acc >>= w;
        acc_len -= w;
        if v == sign {
            return Err(Error::InvalidEncoding);
        }
        // Sign extension of a w-bit two's-complement value.
        out.push((v ^ sign).wrapping_sub(sign) as i32 as i16);
    }
    if acc != 0 || it.next().is_some() {
        return Err(Error::InvalidEncoding);
    }
    Ok(out)
}

impl PublicKey {
    /// The canonical encoding of $`h`$ (`params.rq_bytes()` bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        self.h.to_bytes()
    }

    /// Decodes a public key of `params`. Errors: [`Error::InvalidParams`],
    /// [`Error::InvalidLength`], [`Error::InvalidEncoding`].
    pub fn from_bytes(params: &Params, bytes: &[u8]) -> Result<PublicKey, Error> {
        params.validate()?;
        if bytes.len() != params.rq_bytes() {
            return Err(Error::InvalidLength);
        }
        let h = RqPoly::from_bytes(params, bytes)?;
        // `params` passed `validate` and `h` has its ring: `PublicKey::from_h` would only
        // repeat the checks.
        Ok(PublicKey::from_valid(params, h))
    }
}

impl SecretKey {
    /// The secret-key encoding, in a buffer wiped on drop.
    ///
    /// Panics if a coefficient does not fit the encoding widths: a key built with
    /// [`SecretKey::from_parts`] and not checked with [`SecretKey::validate`] (keys from
    /// [`crate::trapgen`] and [`SecretKey::from_bytes`] always fit).
    pub fn to_bytes(&self) -> Zeroizing<Vec<u8>> {
        let p = &self.params;
        let (wfg, wbig) = (p.sk_fg_bits, p.sk_big_fg_bits);
        assert!(
            crate::keygen::fits_width(&self.f, wfg)
                && crate::keygen::fits_width(&self.g, wfg)
                && crate::keygen::fits_width(&self.big_f, wbig)
                && crate::keygen::fits_width(&self.big_g, wbig),
            "SecretKey::to_bytes: a coefficient does not fit the encoding widths"
        );
        // Exact capacity: no reallocation leaves copies of the key behind.
        let mut out = Zeroizing::new(Vec::with_capacity(sk_bytes(p)));
        out.push(sk_header(p));
        encode_poly(&self.f, wfg, &mut out);
        encode_poly(&self.g, wfg, &mut out);
        encode_poly(&self.big_f, wbig, &mut out);
        encode_poly(&self.big_g, wbig, &mut out);
        debug_assert_eq!(out.len(), sk_bytes(p));
        out
    }

    /// Decodes and validates a secret key of `params` ([`SecretKey::validate`]). Errors:
    /// [`Error::InvalidParams`], [`Error::InvalidLength`], [`Error::InvalidEncoding`],
    /// [`Error::InvalidKey`].
    pub fn from_bytes(params: &Params, bytes: &[u8]) -> Result<SecretKey, Error> {
        params.validate()?;
        if bytes.len() != sk_bytes(params) {
            return Err(Error::InvalidLength);
        }
        if bytes[0] != sk_header(params) {
            return Err(Error::InvalidEncoding);
        }
        let n = params.n();
        let (wfg, wbig) = (params.sk_fg_bits, params.sk_big_fg_bits);
        let (lfg, lbig) = (poly_bytes(n, wfg), poly_bytes(n, wbig));
        let body = &bytes[1..];
        let (bf, rest) = body.split_at(lfg);
        let (bg, rest) = rest.split_at(lfg);
        let (bbf, bbg) = rest.split_at(lbig);
        let mut f = decode_poly(bf, n, wfg)?;
        let mut g = decode_poly(bg, n, wfg)?;
        let mut big_f = decode_poly(bbf, n, wbig)?;
        let mut big_g = decode_poly(bbg, n, wbig)?;
        // `params` passed `validate` and the lengths are n by construction: `from_parts` would
        // only repeat the checks.
        let mut sk = SecretKey {
            params: *params,
            f: core::mem::take(&mut *f),
            g: core::mem::take(&mut *g),
            big_f: core::mem::take(&mut *big_f),
            big_g: core::mem::take(&mut *big_g),
            validated: false,
        };
        sk.validate()?;
        sk.validated = true;
        Ok(sk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poly_round_trip_all_widths() {
        for w in 2..=16u32 {
            let lim = (1i32 << (w - 1)) - 1;
            for n in [1usize, 2, 3, 4, 7, 8, 64, 1024] {
                let x: Vec<i16> = (0..n)
                    .map(|i| {
                        let v = ((i as i32 * 7919 + 13) % (2 * lim + 1)) - lim;
                        v as i16
                    })
                    .collect();
                let mut out = Vec::new();
                encode_poly(&x, w, &mut out);
                assert_eq!(out.len(), poly_bytes(n, w));
                let y = decode_poly(&out, n, w).unwrap();
                assert_eq!(&*y, &x, "w {w} n {n}");
            }
        }
    }

    #[test]
    fn poly_matches_fn_dsa_trim_i8() {
        // fn-dsa's trim_i8_encode on whole-byte lengths, for w in 2..=8.
        for w in 2..=8u32 {
            let lim = (1i32 << (w - 1)) - 1;
            let x: Vec<i8> = (0..1024)
                .map(|i| ((i * 37 + 5) % (2 * lim + 1) - lim) as i8)
                .collect();
            let mut theirs = vec![0u8; 1024 * w as usize / 8];
            let k = fn_dsa_comm::codec::trim_i8_encode(&x, w, &mut theirs);
            assert_eq!(k, theirs.len());
            let mut ours = Vec::new();
            let x16: Vec<i16> = x.iter().map(|&v| v as i16).collect();
            encode_poly(&x16, w, &mut ours);
            assert_eq!(ours, theirs, "w {w}");
        }
    }

    #[test]
    fn poly_rejections() {
        // -2^(w-1) is rejected.
        for w in 2..=16u32 {
            let mut out = Vec::new();
            encode_poly(&[0, 1, -1, 0], w, &mut out);
            // Overwrite coefficient 0 with the pattern 100..0 (w bits).
            let mut bits: u64 = 0;
            for (i, &b) in out.iter().enumerate() {
                bits |= (b as u64) << (8 * i);
            }
            bits = (bits & !((1u64 << w) - 1)) | (1u64 << (w - 1));
            let bad: Vec<u8> = (0..out.len()).map(|i| (bits >> (8 * i)) as u8).collect();
            assert_eq!(
                decode_poly(&bad, 4, w).unwrap_err(),
                Error::InvalidEncoding,
                "w {w}"
            );
        }
        // Nonzero padding: n = 1, w = 10 leaves 6 padding bits.
        let mut out = Vec::new();
        encode_poly(&[5], 10, &mut out);
        assert_eq!(out.len(), 2);
        assert!(decode_poly(&out, 1, 10).is_ok());
        out[1] |= 0x80;
        assert_eq!(
            decode_poly(&out, 1, 10).unwrap_err(),
            Error::InvalidEncoding
        );
    }

    #[test]
    fn sizes() {
        assert_eq!(sk_bytes(&Params::PCS), 5889);
        assert_eq!(sk_bytes(&Params::FALCON_1024), 4097);
    }
}
