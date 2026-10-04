//! Statistics tool: the far-tail behaviour of
//! SamplerZ's Bernoulli step on the crate's own code, with crafted PRNG streams (see
//! `docs/precision.md`).
//!
//! Usage: `cargo run --release --example stats_tail`
//!
//! fn-dsa's `ber_exp` saturates its shift at 63: for x >= 64 ln 2 it accepts with probability
//! 2^-64 (threshold 1) or 0 (threshold 0) instead of ccs exp(-x). The crate now uses the
//! threshold 0 whenever the shift reaches 64 (src/sampler.rs, "Far tail"). The streams below force
//! the base draw V = 0 (z0 = 41, the last table row) with sign b = 1, i.e. z = 42, at the
//! narrowest leaf sigma' = sigma_min and centre 0, where x = 42^2/(2 sigma_min^2) -
//! 41^2/(2 * 3.1^2) = 91.9: the exact acceptance probability is ccs exp(-91.9) = 2^-132.6, so the
//! exact 64-bit threshold floor(2^64 ccs exp(-x)) is 0. Then:
//! - with the Bernoulli bytes 00..00 (W = 0) the crate now REJECTS (fn-dsa's saturated threshold
//!   1 would accept, giving z = 42 probability 2^-192.3 instead of 2^-261.3: a ratio of 2^69);
//! - with the bytes 00..01 (W = 1) it rejects too.
//!
//! In both cases a second, ordinary trial follows and returns 0.

use ntru_trapdoor::Params;
use ntru_trapdoor::hazmat::{LeafSampler, Prng, SamplerZ};

/// Replays fixed bytes; panics when they run out.
struct Replay {
    bytes: Vec<u8>,
    pos: usize,
}

impl Prng for Replay {
    fn next_u8(&mut self) -> u8 {
        let b = self.bytes[self.pos];
        self.pos += 1;
        b
    }
    fn next_u16(&mut self) -> u16 {
        u16::from(self.next_u8()) | (u16::from(self.next_u8()) << 8)
    }
    fn next_u64(&mut self) -> u64 {
        (0..8).fold(0u64, |a, i| a | (u64::from(self.next_u8()) << (8 * i)))
    }
    fn next_bytes(&mut self, dst: &mut [u8]) {
        for d in dst.iter_mut() {
            *d = self.next_u8();
        }
    }
}

/// Bytes of one base draw with V = 0 and sign b: w = b (top 63 bits H = 0, which ties with the
/// table entries whose top bits are 0, so the fallback reads two zero u64 and one byte whose
/// low bit is 0: the low 129 bits of V).
fn base_v0(b: u8) -> Vec<u8> {
    let mut v = vec![b, 0, 0, 0, 0, 0, 0, 0];
    v.extend_from_slice(&[0u8; 16]);
    v.push(0);
    v
}

/// Bytes of one ordinary trial that returns 0: H = 2^63 - 1 (z0 = 0), sign 0 (z = 0), x = 0,
/// and a Bernoulli byte 0, which accepts at once.
fn ordinary_zero() -> Vec<u8> {
    vec![0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0]
}

fn main() {
    let params = Params::PCS;
    let mut isigma = 1.0 / params.sigma_min;
    while isigma * params.sigma_min > 1.0 {
        isigma = f64::from_bits(isigma.to_bits() - 1);
    }
    // The sampler's own binary64 quantities for (z0, b) = (41, 1), centre 0.
    let dss = (isigma * isigma) * 0.5;
    let x = (42.0f64 * 42.0) * dss
        - 1681.0 * ntru_trapdoor::BaseSampler::HalfGauss3_1.inv_2sqr_sigma0();
    let ccs = isigma * params.sigma_min;
    let log2_exact_acc = (ccs.ln() - x) / std::f64::consts::LN_2;
    println!(
        "sigma' = {:.10}, centre 0, (z0, b) = (41, 1): x = {x:.4} (64 ln 2 = {:.4}); exact acceptance \
         ccs exp(-x) = 2^{log2_exact_acc:.2}, so the exact 64-bit threshold is 0",
        1.0 / isigma,
        64.0 * std::f64::consts::LN_2
    );

    // Case 1: Bernoulli bytes 00..00 (W = 0): rejected (fn-dsa: accepted, z = 42).
    for (case, last_byte) in [(1, 0u8), (2, 1u8)] {
        let mut bytes = base_v0(1);
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, last_byte]);
        bytes.extend_from_slice(&ordinary_zero());
        let mut prng = Replay { bytes, pos: 0 };
        let (z, trials) = {
            let mut samp = SamplerZ::new(&params, &mut prng);
            let z = samp.sample(0.0, isigma);
            (z, samp.trials())
        };
        println!(
            "case {case}, W = {last_byte}: the far-tail trial is rejected; the next trial returns {z} \
             (trials {trials}, {} bytes read)",
            prng.pos
        );
        assert_eq!((z, trials), (0, 2));
    }
    println!(
        "With fn-dsa's saturated shift, case 1 would accept: P_impl(z = 42) = 2^-128 (V = 0) * 1/2 \
         (b) * 2^-64 / 0.6337 = 2^-192.3 against the ideal D_(Z, sigma_min, 0)(42) = 2^-261.3, a \
         ratio of 2^69 at this output. Now no BerExp threshold exceeds the exact one by more than \
         FACCT's relative error (about 2^-51)."
    );
}
