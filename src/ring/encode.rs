//! The canonical encoding of $`R_q`$ elements: little-endian bit packing.
//!
//! Coefficient $`i`$ occupies bits $`[iw,(i+1)w)`$ of the bit string, $`w=\mathrm{bitlen}(q-1)`$, and
//! byte $`k`$ holds bits $`[8k,8k+8)`$; the last byte is zero-padded. At $`q=12289`$ ($`w=14`$) this
//! is fn-dsa's `modq_encode` layout (fn-dsa-comm/src/codec.rs:73-100: four coefficients
//! `x3<<42 | x2<<28 | x1<<14 | x0` in 7 little-endian bytes) [read; checked against fn-dsa-comm in
//! `tests/sign_ring.rs`]. At the PCS set ($`w=20`$) an element takes 2560 bytes.

use crate::Error;

/// Bits per coefficient: the bit length of $`q-1`$ (as `Params::q_bits`).
pub(crate) const fn q_bits(q: u32) -> u32 {
    u32::BITS - (q - 1).leading_zeros()
}

/// Encodes coefficients in $`[0,q)`$. The bit string leaves the accumulator 32 bits at a time:
/// before a coefficient is added fewer than 32 bits are pending, and $`w\le32`$, so the
/// accumulator never holds more than 63 bits.
pub(crate) fn encode(q: u32, c: &[u32]) -> Vec<u8> {
    let w = q_bits(q);
    let mut out = Vec::with_capacity((c.len() * w as usize).div_ceil(8));
    let mut acc = 0u64;
    let mut nb = 0u32;
    for &x in c {
        debug_assert!(x < q);
        acc |= (x as u64) << nb;
        nb += w;
        if nb >= 32 {
            out.extend_from_slice(&(acc as u32).to_le_bytes());
            acc >>= 32;
            nb -= 32;
        }
    }
    out.extend_from_slice(&acc.to_le_bytes()[..nb.div_ceil(8) as usize]);
    out
}

/// Decodes `n` coefficients; rejects a wrong length, a coefficient $`\ge q`$ and a nonzero
/// padding bit.
pub(crate) fn decode(q: u32, n: usize, bytes: &[u8]) -> Result<Vec<u32>, Error> {
    let w = q_bits(q);
    if bytes.len() != (n * w as usize).div_ceil(8) {
        return Err(Error::InvalidLength);
    }
    let mask = (1u64 << w) - 1;
    let mut out = Vec::with_capacity(n);
    let mut acc = 0u64;
    let mut nb = 0u32;
    let mut bad = false;
    let mut src = bytes.iter();
    for _ in 0..n {
        while nb < w {
            // The length check guarantees enough bytes.
            acc |= (*src.next().ok_or(Error::InvalidLength)? as u64) << nb;
            nb += 8;
        }
        let x = (acc & mask) as u32;
        acc >>= w;
        nb -= w;
        bad |= x >= q;
        out.push(x);
    }
    // Padding: the remaining nb < 8 bits of the last byte, and nothing after it.
    bad |= acc != 0 || src.next().is_some();
    if bad {
        return Err(Error::InvalidEncoding);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The byte-at-a-time encoder of 0.1.2.
    fn encode_bytewise(q: u32, c: &[u32]) -> Vec<u8> {
        let w = q_bits(q);
        let mut out = Vec::new();
        let (mut acc, mut nb) = (0u64, 0u32);
        for &x in c {
            acc |= (x as u64) << nb;
            nb += w;
            while nb >= 8 {
                out.push(acc as u8);
                acc >>= 8;
                nb -= 8;
            }
        }
        if nb > 0 {
            out.push(acc as u8);
        }
        out
    }

    /// `encode` equals the byte-at-a-time encoder at every width from 1 to 32 bits (the
    /// smallest and the largest modulus of each width), for lengths 0 to 9 and 1024, and
    /// `decode` inverts it.
    #[test]
    fn encode_equals_bytewise() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for w in 1..=32u32 {
            let q_max = if w == 32 { u32::MAX } else { 1u32 << w };
            for q in [(1u32 << (w - 1)) + 1, q_max] {
                assert_eq!(q_bits(q), w);
                for n in (0..=9).chain([1024]) {
                    let mut c: Vec<u32> = (0..n).map(|_| (next() % q as u64) as u32).collect();
                    if let Some(c0) = c.first_mut() {
                        *c0 = q - 1;
                    }
                    let enc = encode(q, &c);
                    assert_eq!(enc, encode_bytewise(q, &c), "w = {w}, q = {q}, n = {n}");
                    assert_eq!(decode(q, n, &enc), Ok(c), "w = {w}, q = {q}, n = {n}");
                }
            }
        }
    }
}
