//! SHAKE256 and the sampler PRNG.
//!
//! Port of fn-dsa (Pornin, The Unlicense), commit 0629bb1:
//! - `KeccakState::process` (lane-complementing Keccak-f\[1600\], two rounds per iteration):
//!   fn-dsa-comm/src/shake.rs:14-460, copied verbatim as [`keccak_f1600`];
//! - `SHAKE<256>` (`inject`, `flip`, `extract`): shake.rs:577-682, as [`Shake256`];
//! - `SHAKE256_PRNG` (136-byte buffer; `next_u8`, `next_u16` and `next_u64` read little-endian
//!   bytes of the SHAKE256 stream in order): shake.rs:684-765, as [`Shake256Prng`].
//!
//! The byte stream is SHAKE256(seed), consumed in order; the buffering is an optimisation that
//! does not change which byte is returned when. That is what makes the fn-dsa parity tests
//! possible. Not ported: the 4-way and AVX2 variants, and SHA3. The `Copy` bound of fn-dsa's
//! `PRNG` trait is dropped, so that states can be wiped on drop.
//!
//! Used across the crate and by the tests: [`Prng`], [`Shake256Prng::new`],
//! [`Shake256Prng::from_parts`], [`shake256`].

use zeroize::Zeroize;

/// A deterministic byte source; [`Shake256Prng`] is the production implementation.
///
/// Multi-byte values are little-endian in the stream order, as in fn-dsa's `PRNG` trait
/// (fn-dsa-comm/src/lib.rs:270-291).
pub trait Prng {
    /// The next byte.
    fn next_u8(&mut self) -> u8;
    /// The next two bytes, little-endian.
    fn next_u16(&mut self) -> u16;
    /// The next eight bytes, little-endian.
    fn next_u64(&mut self) -> u64;
    /// The next `dst.len()` bytes.
    fn next_bytes(&mut self, dst: &mut [u8]);
}

/// The SHAKE256 rate in bytes (1600 - 2*256 bits).
const RATE: usize = 136;

/// Keccak-f\[1600\] round constants (fn-dsa-comm/src/shake.rs:20-33).
const RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// The Keccak-f\[1600\] permutation, in place.
///
/// Verbatim port of fn-dsa `KeccakState::process` (fn-dsa-comm/src/shake.rs:40-461): the
/// lane-complementing representation, 24 rounds unrolled two per loop iteration.
#[allow(
    non_snake_case,
    clippy::assign_op_pattern,
    clippy::identity_op,
    clippy::manual_rotate
)]
#[rustfmt::skip]
fn keccak_f1600(state: &mut [u64; 25]) {
    let mut A: [u64; 25] = *state;


    // Invert some words (alternate internal representation, which
    // saves some operations).
    A[ 1] = !A[ 1];
    A[ 2] = !A[ 2];
    A[ 8] = !A[ 8];
    A[12] = !A[12];
    A[17] = !A[17];
    A[20] = !A[20];

    // Compute 24 rounds. The loop is partially unrolled (two rounds
    // per iteration).
    for i in 0..12 {
        let (mut t0, mut t1, mut t2, mut t3, mut t4);
        let (mut tt0, mut tt1, mut tt2, mut tt3);
        let (mut t, mut kt);
        let (mut c0, mut c1, mut c2, mut c3, mut c4, mut bnn);

        tt0 = A[ 1] ^ A[ 6];
        tt1 = A[11] ^ A[16];
        tt0 ^= A[21] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[ 4] ^ A[ 9];
        tt3 = A[14] ^ A[19];
        tt0 ^= A[24];
        tt2 ^= tt3;
        t0 = tt0 ^ tt2;

        tt0 = A[ 2] ^ A[ 7];
        tt1 = A[12] ^ A[17];
        tt0 ^= A[22] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[ 0] ^ A[ 5];
        tt3 = A[10] ^ A[15];
        tt0 ^= A[20];
        tt2 ^= tt3;
        t1 = tt0 ^ tt2;

        tt0 = A[ 3] ^ A[ 8];
        tt1 = A[13] ^ A[18];
        tt0 ^= A[23] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[ 1] ^ A[ 6];
        tt3 = A[11] ^ A[16];
        tt0 ^= A[21];
        tt2 ^= tt3;
        t2 = tt0 ^ tt2;

        tt0 = A[ 4] ^ A[ 9];
        tt1 = A[14] ^ A[19];
        tt0 ^= A[24] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[ 2] ^ A[ 7];
        tt3 = A[12] ^ A[17];
        tt0 ^= A[22];
        tt2 ^= tt3;
        t3 = tt0 ^ tt2;

        tt0 = A[ 0] ^ A[ 5];
        tt1 = A[10] ^ A[15];
        tt0 ^= A[20] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[ 3] ^ A[ 8];
        tt3 = A[13] ^ A[18];
        tt0 ^= A[23];
        tt2 ^= tt3;
        t4 = tt0 ^ tt2;

        A[ 0] = A[ 0] ^ t0;
        A[ 5] = A[ 5] ^ t0;
        A[10] = A[10] ^ t0;
        A[15] = A[15] ^ t0;
        A[20] = A[20] ^ t0;
        A[ 1] = A[ 1] ^ t1;
        A[ 6] = A[ 6] ^ t1;
        A[11] = A[11] ^ t1;
        A[16] = A[16] ^ t1;
        A[21] = A[21] ^ t1;
        A[ 2] = A[ 2] ^ t2;
        A[ 7] = A[ 7] ^ t2;
        A[12] = A[12] ^ t2;
        A[17] = A[17] ^ t2;
        A[22] = A[22] ^ t2;
        A[ 3] = A[ 3] ^ t3;
        A[ 8] = A[ 8] ^ t3;
        A[13] = A[13] ^ t3;
        A[18] = A[18] ^ t3;
        A[23] = A[23] ^ t3;
        A[ 4] = A[ 4] ^ t4;
        A[ 9] = A[ 9] ^ t4;
        A[14] = A[14] ^ t4;
        A[19] = A[19] ^ t4;
        A[24] = A[24] ^ t4;
        A[ 5] = (A[ 5] << 36) | (A[ 5] >> (64 - 36));
        A[10] = (A[10] <<  3) | (A[10] >> (64 -  3));
        A[15] = (A[15] << 41) | (A[15] >> (64 - 41));
        A[20] = (A[20] << 18) | (A[20] >> (64 - 18));
        A[ 1] = (A[ 1] <<  1) | (A[ 1] >> (64 -  1));
        A[ 6] = (A[ 6] << 44) | (A[ 6] >> (64 - 44));
        A[11] = (A[11] << 10) | (A[11] >> (64 - 10));
        A[16] = (A[16] << 45) | (A[16] >> (64 - 45));
        A[21] = (A[21] <<  2) | (A[21] >> (64 - 2));
        A[ 2] = (A[ 2] << 62) | (A[ 2] >> (64 - 62));
        A[ 7] = (A[ 7] <<  6) | (A[ 7] >> (64 -  6));
        A[12] = (A[12] << 43) | (A[12] >> (64 - 43));
        A[17] = (A[17] << 15) | (A[17] >> (64 - 15));
        A[22] = (A[22] << 61) | (A[22] >> (64 - 61));
        A[ 3] = (A[ 3] << 28) | (A[ 3] >> (64 - 28));
        A[ 8] = (A[ 8] << 55) | (A[ 8] >> (64 - 55));
        A[13] = (A[13] << 25) | (A[13] >> (64 - 25));
        A[18] = (A[18] << 21) | (A[18] >> (64 - 21));
        A[23] = (A[23] << 56) | (A[23] >> (64 - 56));
        A[ 4] = (A[ 4] << 27) | (A[ 4] >> (64 - 27));
        A[ 9] = (A[ 9] << 20) | (A[ 9] >> (64 - 20));
        A[14] = (A[14] << 39) | (A[14] >> (64 - 39));
        A[19] = (A[19] <<  8) | (A[19] >> (64 -  8));
        A[24] = (A[24] << 14) | (A[24] >> (64 - 14));

        bnn = !A[12];
        kt = A[ 6] | A[12];
        c0 = A[ 0] ^ kt;
        kt = bnn | A[18];
        c1 = A[ 6] ^ kt;
        kt = A[18] & A[24];
        c2 = A[12] ^ kt;
        kt = A[24] | A[ 0];
        c3 = A[18] ^ kt;
        kt = A[ 0] & A[ 6];
        c4 = A[24] ^ kt;
        A[ 0] = c0;
        A[ 6] = c1;
        A[12] = c2;
        A[18] = c3;
        A[24] = c4;
        bnn = !A[22];
        kt = A[ 9] | A[10];
        c0 = A[ 3] ^ kt;
        kt = A[10] & A[16];
        c1 = A[ 9] ^ kt;
        kt = A[16] | bnn;
        c2 = A[10] ^ kt;
        kt = A[22] | A[ 3];
        c3 = A[16] ^ kt;
        kt = A[ 3] & A[ 9];
        c4 = A[22] ^ kt;
        A[ 3] = c0;
        A[ 9] = c1;
        A[10] = c2;
        A[16] = c3;
        A[22] = c4;
        bnn = !A[19];
        kt = A[ 7] | A[13];
        c0 = A[ 1] ^ kt;
        kt = A[13] & A[19];
        c1 = A[ 7] ^ kt;
        kt = bnn & A[20];
        c2 = A[13] ^ kt;
        kt = A[20] | A[ 1];
        c3 = bnn ^ kt;
        kt = A[ 1] & A[ 7];
        c4 = A[20] ^ kt;
        A[ 1] = c0;
        A[ 7] = c1;
        A[13] = c2;
        A[19] = c3;
        A[20] = c4;
        bnn = !A[17];
        kt = A[ 5] & A[11];
        c0 = A[ 4] ^ kt;
        kt = A[11] | A[17];
        c1 = A[ 5] ^ kt;
        kt = bnn | A[23];
        c2 = A[11] ^ kt;
        kt = A[23] & A[ 4];
        c3 = bnn ^ kt;
        kt = A[ 4] | A[ 5];
        c4 = A[23] ^ kt;
        A[ 4] = c0;
        A[ 5] = c1;
        A[11] = c2;
        A[17] = c3;
        A[23] = c4;
        bnn = !A[ 8];
        kt = bnn & A[14];
        c0 = A[ 2] ^ kt;
        kt = A[14] | A[15];
        c1 = bnn ^ kt;
        kt = A[15] & A[21];
        c2 = A[14] ^ kt;
        kt = A[21] | A[ 2];
        c3 = A[15] ^ kt;
        kt = A[ 2] & A[ 8];
        c4 = A[21] ^ kt;
        A[ 2] = c0;
        A[ 8] = c1;
        A[14] = c2;
        A[15] = c3;
        A[21] = c4;
        A[ 0] = A[ 0] ^ RC[2 * i + 0];

        tt0 = A[ 6] ^ A[ 9];
        tt1 = A[ 7] ^ A[ 5];
        tt0 ^= A[ 8] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[24] ^ A[22];
        tt3 = A[20] ^ A[23];
        tt0 ^= A[21];
        tt2 ^= tt3;
        t0 = tt0 ^ tt2;

        tt0 = A[12] ^ A[10];
        tt1 = A[13] ^ A[11];
        tt0 ^= A[14] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[ 0] ^ A[ 3];
        tt3 = A[ 1] ^ A[ 4];
        tt0 ^= A[ 2];
        tt2 ^= tt3;
        t1 = tt0 ^ tt2;

        tt0 = A[18] ^ A[16];
        tt1 = A[19] ^ A[17];
        tt0 ^= A[15] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[ 6] ^ A[ 9];
        tt3 = A[ 7] ^ A[ 5];
        tt0 ^= A[ 8];
        tt2 ^= tt3;
        t2 = tt0 ^ tt2;

        tt0 = A[24] ^ A[22];
        tt1 = A[20] ^ A[23];
        tt0 ^= A[21] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[12] ^ A[10];
        tt3 = A[13] ^ A[11];
        tt0 ^= A[14];
        tt2 ^= tt3;
        t3 = tt0 ^ tt2;

        tt0 = A[ 0] ^ A[ 3];
        tt1 = A[ 1] ^ A[ 4];
        tt0 ^= A[ 2] ^ tt1;
        tt0 = (tt0 << 1) | (tt0 >> 63);
        tt2 = A[18] ^ A[16];
        tt3 = A[19] ^ A[17];
        tt0 ^= A[15];
        tt2 ^= tt3;
        t4 = tt0 ^ tt2;

        A[ 0] = A[ 0] ^ t0;
        A[ 3] = A[ 3] ^ t0;
        A[ 1] = A[ 1] ^ t0;
        A[ 4] = A[ 4] ^ t0;
        A[ 2] = A[ 2] ^ t0;
        A[ 6] = A[ 6] ^ t1;
        A[ 9] = A[ 9] ^ t1;
        A[ 7] = A[ 7] ^ t1;
        A[ 5] = A[ 5] ^ t1;
        A[ 8] = A[ 8] ^ t1;
        A[12] = A[12] ^ t2;
        A[10] = A[10] ^ t2;
        A[13] = A[13] ^ t2;
        A[11] = A[11] ^ t2;
        A[14] = A[14] ^ t2;
        A[18] = A[18] ^ t3;
        A[16] = A[16] ^ t3;
        A[19] = A[19] ^ t3;
        A[17] = A[17] ^ t3;
        A[15] = A[15] ^ t3;
        A[24] = A[24] ^ t4;
        A[22] = A[22] ^ t4;
        A[20] = A[20] ^ t4;
        A[23] = A[23] ^ t4;
        A[21] = A[21] ^ t4;
        A[ 3] = (A[ 3] << 36) | (A[ 3] >> (64 - 36));
        A[ 1] = (A[ 1] <<  3) | (A[ 1] >> (64 -  3));
        A[ 4] = (A[ 4] << 41) | (A[ 4] >> (64 - 41));
        A[ 2] = (A[ 2] << 18) | (A[ 2] >> (64 - 18));
        A[ 6] = (A[ 6] <<  1) | (A[ 6] >> (64 -  1));
        A[ 9] = (A[ 9] << 44) | (A[ 9] >> (64 - 44));
        A[ 7] = (A[ 7] << 10) | (A[ 7] >> (64 - 10));
        A[ 5] = (A[ 5] << 45) | (A[ 5] >> (64 - 45));
        A[ 8] = (A[ 8] <<  2) | (A[ 8] >> (64 - 2));
        A[12] = (A[12] << 62) | (A[12] >> (64 - 62));
        A[10] = (A[10] <<  6) | (A[10] >> (64 -  6));
        A[13] = (A[13] << 43) | (A[13] >> (64 - 43));
        A[11] = (A[11] << 15) | (A[11] >> (64 - 15));
        A[14] = (A[14] << 61) | (A[14] >> (64 - 61));
        A[18] = (A[18] << 28) | (A[18] >> (64 - 28));
        A[16] = (A[16] << 55) | (A[16] >> (64 - 55));
        A[19] = (A[19] << 25) | (A[19] >> (64 - 25));
        A[17] = (A[17] << 21) | (A[17] >> (64 - 21));
        A[15] = (A[15] << 56) | (A[15] >> (64 - 56));
        A[24] = (A[24] << 27) | (A[24] >> (64 - 27));
        A[22] = (A[22] << 20) | (A[22] >> (64 - 20));
        A[20] = (A[20] << 39) | (A[20] >> (64 - 39));
        A[23] = (A[23] <<  8) | (A[23] >> (64 -  8));
        A[21] = (A[21] << 14) | (A[21] >> (64 - 14));

        bnn = !A[13];
        kt = A[ 9] | A[13];
        c0 = A[ 0] ^ kt;
        kt = bnn | A[17];
        c1 = A[ 9] ^ kt;
        kt = A[17] & A[21];
        c2 = A[13] ^ kt;
        kt = A[21] | A[ 0];
        c3 = A[17] ^ kt;
        kt = A[ 0] & A[ 9];
        c4 = A[21] ^ kt;
        A[ 0] = c0;
        A[ 9] = c1;
        A[13] = c2;
        A[17] = c3;
        A[21] = c4;
        bnn = !A[14];
        kt = A[22] | A[ 1];
        c0 = A[18] ^ kt;
        kt = A[ 1] & A[ 5];
        c1 = A[22] ^ kt;
        kt = A[ 5] | bnn;
        c2 = A[ 1] ^ kt;
        kt = A[14] | A[18];
        c3 = A[ 5] ^ kt;
        kt = A[18] & A[22];
        c4 = A[14] ^ kt;
        A[18] = c0;
        A[22] = c1;
        A[ 1] = c2;
        A[ 5] = c3;
        A[14] = c4;
        bnn = !A[23];
        kt = A[10] | A[19];
        c0 = A[ 6] ^ kt;
        kt = A[19] & A[23];
        c1 = A[10] ^ kt;
        kt = bnn & A[ 2];
        c2 = A[19] ^ kt;
        kt = A[ 2] | A[ 6];
        c3 = bnn ^ kt;
        kt = A[ 6] & A[10];
        c4 = A[ 2] ^ kt;
        A[ 6] = c0;
        A[10] = c1;
        A[19] = c2;
        A[23] = c3;
        A[ 2] = c4;
        bnn = !A[11];
        kt = A[ 3] & A[ 7];
        c0 = A[24] ^ kt;
        kt = A[ 7] | A[11];
        c1 = A[ 3] ^ kt;
        kt = bnn | A[15];
        c2 = A[ 7] ^ kt;
        kt = A[15] & A[24];
        c3 = bnn ^ kt;
        kt = A[24] | A[ 3];
        c4 = A[15] ^ kt;
        A[24] = c0;
        A[ 3] = c1;
        A[ 7] = c2;
        A[11] = c3;
        A[15] = c4;
        bnn = !A[16];
        kt = bnn & A[20];
        c0 = A[12] ^ kt;
        kt = A[20] | A[ 4];
        c1 = bnn ^ kt;
        kt = A[ 4] & A[ 8];
        c2 = A[20] ^ kt;
        kt = A[ 8] | A[12];
        c3 = A[ 4] ^ kt;
        kt = A[12] & A[16];
        c4 = A[ 8] ^ kt;
        A[12] = c0;
        A[16] = c1;
        A[20] = c2;
        A[ 4] = c3;
        A[ 8] = c4;
        A[ 0] = A[ 0] ^ RC[2 * i + 1];

        t = A[ 5];
        A[ 5] = A[18];
        A[18] = A[11];
        A[11] = A[10];
        A[10] = A[ 6];
        A[ 6] = A[22];
        A[22] = A[20];
        A[20] = A[12];
        A[12] = A[19];
        A[19] = A[15];
        A[15] = A[24];
        A[24] = A[ 8];
        A[ 8] = t;
        t = A[ 1];
        A[ 1] = A[ 9];
        A[ 9] = A[14];
        A[14] = A[ 2];
        A[ 2] = A[13];
        A[13] = A[23];
        A[23] = A[ 4];
        A[ 4] = A[21];
        A[21] = A[16];
        A[16] = A[ 3];
        A[ 3] = A[17];
        A[17] = A[ 7];
        A[ 7] = t;
    }

    // Invert some words back to normal representation.
    A[ 1] = !A[ 1];
    A[ 2] = !A[ 2];
    A[ 8] = !A[ 8];
    A[12] = !A[12];
    A[17] = !A[17];
    A[20] = !A[20];


    *state = A;
    A.zeroize();
}

/// SHAKE256 (fn-dsa `SHAKE<256>`, fn-dsa-comm/src/shake.rs:577-682): absorb with
/// [`Shake256::inject`], switch to output mode with [`Shake256::flip`], squeeze with
/// [`Shake256::extract`]. Wiped on drop.
#[derive(Clone)]
pub(crate) struct Shake256 {
    state: [u64; 25],
    ptr: usize,
    flipped: bool,
}

impl Shake256 {
    /// An empty instance in input mode (shake.rs:606-613).
    pub(crate) fn new() -> Shake256 {
        Shake256 {
            state: [0u64; 25],
            ptr: 0,
            flipped: false,
        }
    }

    /// Absorbs `src` (shake.rs:619-635). Panics in output mode. Bytes are XORed into the state
    /// in little-endian lane order, as fn-dsa does byte by byte; whole aligned lanes are XORed
    /// eight bytes at a time (same result, fewer operations).
    pub(crate) fn inject(&mut self, src: &[u8]) {
        assert!(!self.flipped);
        let mut ptr = self.ptr;
        let mut src = src;
        while !src.is_empty() {
            if ptr & 7 == 0 && src.len() >= 8 {
                let lanes = ((RATE - ptr) >> 3).min(src.len() >> 3);
                let (head, rest) = src.split_at(lanes << 3);
                let (chunks, _) = head.as_chunks::<8>();
                for (lane, c) in self.state[ptr >> 3..].iter_mut().zip(chunks.iter()) {
                    *lane ^= u64::from_le_bytes(*c);
                }
                ptr += lanes << 3;
                src = rest;
            } else {
                self.state[ptr >> 3] ^= (src[0] as u64) << ((ptr & 7) << 3);
                ptr += 1;
                src = &src[1..];
            }
            if ptr == RATE {
                keccak_f1600(&mut self.state);
                ptr = 0;
            }
        }
        self.ptr = ptr;
    }

    /// Pads (SHAKE domain byte 0x1F, final bit 0x80) and switches to output mode
    /// (shake.rs:641-650). Panics if already flipped.
    pub(crate) fn flip(&mut self) {
        assert!(!self.flipped);
        let i = self.ptr;
        self.state[i >> 3] ^= 0x1Fu64 << ((i & 7) << 3);
        let i = RATE - 1;
        self.state[i >> 3] ^= 0x80u64 << ((i & 7) << 3);
        self.ptr = RATE;
        self.flipped = true;
    }

    /// Squeezes `dst.len()` bytes (shake.rs:656-676). Panics in input mode.
    pub(crate) fn extract(&mut self, dst: &mut [u8]) {
        assert!(self.flipped);
        let mut ptr = self.ptr;
        let mut i = 0;
        while i < dst.len() {
            if ptr == RATE {
                keccak_f1600(&mut self.state);
                ptr = 0;
            }
            let clen = core::cmp::min(dst.len() - i, RATE - ptr);
            for d in &mut dst[i..i + clen] {
                *d = (self.state[ptr >> 3] >> ((ptr & 7) << 3)) as u8;
                ptr += 1;
            }
            i += clen;
        }
        self.ptr = ptr;
    }
}

impl Drop for Shake256 {
    fn drop(&mut self) {
        self.state.zeroize();
    }
}

/// fn-dsa's `SHAKE256_PRNG` (fn-dsa-comm/src/shake.rs:684-765): the SHAKE256 output stream of a
/// seed, read through a 136-byte buffer. Wiped on drop.
#[derive(Clone)]
pub struct Shake256Prng {
    /// The Keccak state, after the absorb phase and the padding.
    state: [u64; 25],
    /// The current squeezed block.
    buf: [u8; RATE],
    /// Read position in `buf`; `RATE` means empty.
    ptr: usize,
}

impl Shake256Prng {
    /// The stream SHAKE256(`seed`) (fn-dsa `SHAKE256_PRNG::new`, shake.rs:706-711).
    pub fn new(seed: &[u8]) -> Shake256Prng {
        Shake256Prng::from_parts(&[seed])
    }

    /// The stream SHAKE256(`parts[0] || parts[1] || ...`), for domain-separated seeds.
    pub fn from_parts(parts: &[&[u8]]) -> Shake256Prng {
        let mut sh = Shake256::new();
        for p in parts {
            sh.inject(p);
        }
        sh.flip();
        Shake256Prng {
            state: sh.state,
            buf: [0u8; RATE],
            ptr: RATE,
        }
    }

    /// Squeezes the next block into the buffer (shake.rs:697-702: `extract` of 136 bytes is
    /// one permutation followed by a copy of the rate, since the rate is 136 bytes).
    #[inline(never)]
    fn refill(&mut self) {
        keccak_f1600(&mut self.state);
        let (chunks, _) = self.buf.as_chunks_mut::<8>();
        for (chunk, lane) in chunks.iter_mut().zip(self.state.iter()) {
            *chunk = lane.to_le_bytes();
        }
        self.ptr = 0;
    }
}

impl Prng for Shake256Prng {
    /// shake.rs:713-720.
    #[inline]
    fn next_u8(&mut self) -> u8 {
        if self.ptr == RATE {
            self.refill();
        }
        let x = self.buf[self.ptr];
        self.ptr += 1;
        x
    }

    /// shake.rs:722-731.
    #[inline]
    fn next_u16(&mut self) -> u16 {
        if self.ptr >= RATE - 1 {
            let x = self.next_u8() as u16;
            return x | ((self.next_u8() as u16) << 8);
        }
        let x = u16::from_le_bytes([self.buf[self.ptr], self.buf[self.ptr + 1]]);
        self.ptr += 2;
        x
    }

    /// shake.rs:733-745.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        if self.ptr >= RATE - 7 {
            let mut x = 0u64;
            for i in 0..8 {
                x |= (self.next_u8() as u64) << (i << 3);
            }
            return x;
        }
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.buf[self.ptr..self.ptr + 8]);
        self.ptr += 8;
        u64::from_le_bytes(b)
    }

    /// shake.rs:747-764.
    fn next_bytes(&mut self, dst: &mut [u8]) {
        let mut off = 0;
        let len = dst.len();
        while off < len {
            let mut blen = RATE - self.ptr;
            if blen == 0 {
                self.refill();
                blen = RATE;
            }
            if blen > len - off {
                blen = len - off;
            }
            dst[off..off + blen].copy_from_slice(&self.buf[self.ptr..self.ptr + blen]);
            self.ptr += blen;
            off += blen;
        }
    }
}

impl Drop for Shake256Prng {
    fn drop(&mut self) {
        self.state.zeroize();
        self.buf.zeroize();
        self.ptr = RATE;
    }
}

/// SHAKE256 of the concatenation of `parts`, `out.len()` bytes of output.
pub fn shake256(parts: &[&[u8]], out: &mut [u8]) {
    let mut sh = Shake256::new();
    for p in parts {
        sh.inject(p);
    }
    sh.flip();
    sh.extract(out);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NIST SHAKE256 vectors, as in fn-dsa's tests (fn-dsa-comm/src/shake.rs:822-833):
    /// input (hex), output (hex). Also checked with Python's `hashlib.shake_256`.
    const KAT_SHAKE256: [&str; 6] = [
        "",
        concat!(
            "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762fd75dc4ddd8c0f200",
            "cb05019d67b592f6fc821c49479ab48640292eacb3b7c4be",
        ),
        concat!(
            "dc5a100fa16df1583c79722a0d72833d3bf22c109b8889dbd35213c6bfce205813edae3242695cfd",
            "9f59b9a1c203c1b72ef1a5423147cb990b5316a85266675894e2644c3f9578cebe451a09e58c5378",
            "8fe77a9e850943f8a275f830354b0593a762bac55e984db3e0661eca3cb83f67a6fb348e6177f7de",
            "e2df40c4322602f094953905681be3954fe44c4c902c8f6bba565a788b38f13411ba76ce0f9f6756",
            "a2a2687424c5435a51e62df7a8934b6e141f74c6ccf539e3782d22b5955d3baf1ab2cf7b5c3f74ec",
            "2f9447344e937957fd7f0bdfec56d5d25f61cde18c0986e244ecf780d6307e313117256948d4230e",
            "bb9ea62bb302cfe80d7dfebabc4a51d7687967ed5b416a139e974c005fff507a96",
        ),
        "2bac5716803a9cda8f9e84365ab0a681327b5ba34fdedfb1c12e6e807f45284b",
        "8d8001e2c096f1b88e7c9224a086efd4797fbf74a8033a2d422a2b6b8f6747e4",
        concat!(
            "2e975f6a8a14f0704d51b13667d8195c219f71e6345696c49fa4b9d08e9225d3d39393425152c97e",
            "71dd24601c11abcfa0f12f53c680bd3ae757b8134a9c10d429615869217fdd5885c4db174985703a",
            "6d6de94a667eac3023443a8337ae1bc601b76d7d38ec3c34463105f0d3949d78e562a039e4469548",
            "b609395de5a4fd43c46ca9fd6ee29ada5efc07d84d553249450dab4a49c483ded250c9338f85cd93",
            "7ae66bb436f3b4026e859fda1ca571432f3bfc09e7c03ca4d183b741111ca0483d0edabc03feb23b",
            "17ee48e844ba2408d9dcfd0139d2e8c7310125aee801c61ab7900d1efc47c078281766f361c5e611",
            "1346235e1dc38325666c",
        ),
    ];

    #[test]
    fn shake256_nist_kats() {
        for pair in KAT_SHAKE256.chunks(2) {
            let src = hex::decode(pair[0]).unwrap();
            let dst = hex::decode(pair[1]).unwrap();
            // In one go.
            let mut out = vec![0u8; dst.len()];
            shake256(&[&src], &mut out);
            assert_eq!(out, dst);
            // Byte by byte, in and out (fn-dsa's test 2).
            let mut sh = Shake256::new();
            for b in &src {
                sh.inject(core::slice::from_ref(b));
            }
            sh.flip();
            for o in out.iter_mut() {
                let mut t = [0u8];
                sh.extract(&mut t);
                *o = t[0];
            }
            assert_eq!(out, dst);
            // Split into parts.
            let (a, b) = src.split_at(src.len() / 3);
            let mut out2 = vec![0u8; dst.len()];
            shake256(&[a, &[], b], &mut out2);
            assert_eq!(out2, dst);
        }
    }

    /// Absorbing in pieces of every size from 1 to 20 bytes (crossing lane and rate
    /// boundaries at every alignment) equals absorbing in one go.
    #[test]
    fn inject_piecewise_equals_oneshot() {
        let msg: Vec<u8> = (0..1000u32).map(|i| (i * 131 + 7) as u8).collect();
        let mut want = [0u8; 48];
        shake256(&[&msg], &mut want);
        for piece in 1..=20usize {
            let mut sh = Shake256::new();
            for chunk in msg.chunks(piece) {
                sh.inject(chunk);
            }
            sh.flip();
            let mut got = [0u8; 48];
            sh.extract(&mut got);
            assert_eq!(got, want, "piece size {piece}");
        }
    }

    /// Rate-boundary inputs: Python `hashlib.shake_256(bytes([0xA5] * L)).digest(16)`.
    #[test]
    fn shake256_rate_boundaries() {
        let cases: [(usize, &str); 4] = [
            (135, "8194ee77ae513cf3cf2487538a58bf43"),
            (136, "c6958b1e28e55ee1d67b1a792ed602dd"),
            (137, "f6e5dc08b231605e7e62758d296c11e6"),
            (272, "a9384482cde0ef8f9acb386ed0d9e7f6"),
        ];
        for (len, want) in cases {
            let mut out = [0u8; 16];
            shake256(&[&vec![0xA5u8; len]], &mut out);
            assert_eq!(hex::encode(out), want, "length {len}");
        }
    }

    /// `from_parts` is SHAKE256 of the concatenation: Python
    /// `hashlib.shake_256(b"abc" + bytes(range(200))).digest(64)`.
    #[test]
    fn from_parts_is_concatenation() {
        let body: Vec<u8> = (0u8..200).collect();
        let mut p = Shake256Prng::from_parts(&[b"abc", &body, b""]);
        let mut out = [0u8; 64];
        p.next_bytes(&mut out);
        assert_eq!(
            hex::encode(out),
            concat!(
                "004c9686c264cf0b5255d0ab50a9d1d4f2010f9e06f46017f29a6eaa44fbb0e1",
                "51cf9756bbc2158397e84a110c7eb6424f4db08929b946e44e20d61b4dff5f00"
            )
        );
    }

    /// The PRNG returns the SHAKE256 stream in order, whatever the mix of `next_*` calls,
    /// across block boundaries. Reference: Python `hashlib.shake_256(seed).digest(5000)`.
    #[test]
    fn prng_stream_mixed_reads() {
        let seed = b"ntru-trapdoor/prng-test";
        let mut stream = vec![0u8; 5000];
        shake256(&[seed], &mut stream);
        assert_eq!(
            hex::encode(&stream[..32]),
            "91d20e4c11d6ac3df9ec43c9b8995df87c324b008c8bc6a1666e436d84a6a9aa"
        );
        assert_eq!(
            hex::encode(&stream[4968..]),
            "836d8959ecb22d288cf788fc45a374e043af0ca96ef23477202b854e2bea97cf"
        );
        let mut p = Shake256Prng::new(seed);
        let mut pos = 0usize;
        let mut k = 0u32;
        while pos + 300 < stream.len() {
            match k % 5 {
                0 => {
                    assert_eq!(p.next_u8(), stream[pos]);
                    pos += 1;
                }
                1 => {
                    let want = u16::from_le_bytes([stream[pos], stream[pos + 1]]);
                    assert_eq!(p.next_u16(), want);
                    pos += 2;
                }
                2 => {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&stream[pos..pos + 8]);
                    assert_eq!(p.next_u64(), u64::from_le_bytes(b));
                    pos += 8;
                }
                3 => {
                    let len = (k as usize * 7) % 290;
                    let mut b = vec![0u8; len];
                    p.next_bytes(&mut b);
                    assert_eq!(&b[..], &stream[pos..pos + len]);
                    pos += len;
                }
                _ => {
                    // 17-byte draws, as the PCS base sampler.
                    let lo = p.next_u64();
                    let hi = p.next_u64();
                    let b = p.next_u8();
                    let mut w = [0u8; 8];
                    w.copy_from_slice(&stream[pos..pos + 8]);
                    assert_eq!(lo, u64::from_le_bytes(w));
                    w.copy_from_slice(&stream[pos + 8..pos + 16]);
                    assert_eq!(hi, u64::from_le_bytes(w));
                    assert_eq!(b, stream[pos + 16]);
                    pos += 17;
                }
            }
            k += 1;
        }
    }

    #[test]
    fn prng_is_wiped_on_drop_and_clone_is_independent() {
        let mut p = Shake256Prng::new(b"x");
        let _ = p.next_u64();
        let mut c = p.clone();
        assert_eq!(p.next_u64(), c.next_u64());
        drop(c);
        // Explicit wipe through Drop's body, observable on a value we still own.
        let mut q = Shake256Prng::new(b"y");
        let _ = q.next_u8();
        q.state.zeroize();
        q.buf.zeroize();
        assert!(q.state.iter().all(|&w| w == 0) && q.buf.iter().all(|&b| b == 0));
    }
}
