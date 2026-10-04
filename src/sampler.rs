//! SamplerZ: the integer Gaussian sampler at the leaves of ffSampling.
//!
//! Port of fn-dsa-sign/src/sampler.rs (Pornin, The Unlicense), commit 0629bb1,
//! portable branches only: `next` :131-192, `gaussian0` :241-260, `ber_exp` :262-328, the
//! constants `LOG2` and `INV_LOG2` :100-103, and `FLR::expm_p63` with `EXPM_COEFFS`
//! (fn-dsa-sign/src/flr_native.rs:672-757, the 64-bit branch :689-705).
//!
//! One leaf draw at centre $`\mu`$ and inverse width $`1/\sigma'`$ (`isigma`):
//! split $`\mu=s+r`$ ($`s=\lfloor\mu\rfloor`$), set $`dss=\texttt{isigma}^2/2`$ and
//! $`ccs=\sigma_{\min}\cdot\texttt{isigma}`$, then loop: draw $`(z_0,b)`$ from the base
//! half-Gaussian, $`z=b+(2b-1)z_0`$, $`x=(z-r)^2\,dss-z_0^2/(2\sigma_0^2)`$, accept with
//! probability $`ccs\cdot e^{-x}`$ (`ber_exp`), return $`s+z`$.
//!
//! Base samplers ([`crate::BaseSampler`]):
//! - `HalfGauss3_1` (new): $`z_0=\#\{j:V<T_j\}`$ for $`V`$ uniform on $`[0,2^{192})`$ over
//!   the 41 rows of [`crate::tables::BASE_3_1_192`], drawn lazily. One `next_u64` $`w`$ gives
//!   the sign $`b=w\bmod2`$ and the top 63 bits $`H=\lfloor w/2\rfloor`$ of $`V`$. A branch-free
//!   scan of fixed length compares $`H`$ with the top 63 bits $`H_j=\lfloor T_j/2^{129}\rfloor`$
//!   of the 28 entries where they are nonzero (the other 13 entries have $`H_j=0`$: they can
//!   only tie, when $`H=0`$, and never count). If $`H`$ equals no $`H_j`$, $`z_0`$ is decided
//!   ($`V<T_j\iff H<H_j`$). Otherwise (probability at most $`29\cdot2^{-63}=2^{-58.1}`$ per
//!   draw) the low 129 bits of $`V`$ are drawn (two `next_u64`, then the low bit of `next_u8`)
//!   and the full 192-bit comparison decides. The distribution is exactly that of the 192-bit
//!   table; a draw takes 8 PRNG bytes (25 in the rare fallback). The fallback's branch leaks
//!   only that rare event.
//! - `Falcon1_8205`: fn-dsa's `gaussian0` verbatim (one `next_u64`, one `next_u16`, the
//!   31+24+24-bit limb comparison over [`crate::tables::GAUSS0_1_8205_79`], which equals fn-dsa's
//!   `GAUSS0`).
//!
//! The base is dispatched once per leaf draw (one perfectly predictable branch), into a loop
//! that is monomorphised per base: there is no branch per trial.
//!
//! **Far tail of `ber_exp`** (a deliberate difference from fn-dsa). With
//! $`x=s\ln2+r`$, fn-dsa saturates $`s`$ at 63, so for $`x\ge64\ln2`$ it accepts with
//! probability $`2^{-64}`$ (or 0) where the exact value $`ccs\,e^{-x}`$ is smaller, down to
//! $`2^{-133}`$ at the PCS leaves: far-tail outputs are over-weighted by up to $`2^{69}`$. That is
//! harmless for a statistical distance, but it makes the Rényi divergence of order 256 per
//! SampPre call about $`2^{14}`$ nats (finding F1 in docs/precision.md). Here the threshold is 0 (always
//! reject) when $`s\ge64`$, through a mask, so that every threshold is at most the exact one
//! (truncation) apart from FACCT's relative error of about $`2^{-51}`$. The per-trial acceptance
//! changes by less than $`P(s\ge64)\cdot2^{-64}\le2^{-60.9-64}`$ ($`s\ge64`$ needs
//! $`z_0\ge28`$ at $`\sigma_0=3.1`$); the byte consumption is unchanged. At Falcon's parameters
//! the outcome differs from fn-dsa's only when $`s\ge64`$ and the 8 Bernoulli bytes are all zero
//! (about $`2^{-96}`$ per trial), which the parity tests cannot reach. fn-dsa's reduction never
//! yields $`r<0`$ for $`s\le63`$ (design.md §0, finding 1), so it is ported without a clamp.
//!
//! **Preconditions** (enforced for every leaf by [`crate::ExpandedKey::new`]):
//! $`\sigma_{\min}\le\sigma'`$ (so $`ccs\le1`$) and $`\sigma'<\sigma_0`$ with a $`2^{-40}`$ rounding
//! margin (so $`x\ge0`$). Outside them the output distribution is wrong; the sampler does not
//! check them itself (as fn-dsa).
//!
//! **Side channels** (as fn-dsa): full table scans; the acceptance probability per trial is
//! independent of $`\sigma'`$ and $`\mu`$ (0.633687 for the PCS base, 0.584958 for Falcon's
//! [computed, design.md §2]); the lazy byte comparison in `ber_exp` leaks only accept or reject.
//! Native `f64` arithmetic may be variable-time on some CPUs.
//!
//! Used by the tests through `hazmat`: [`LeafSampler`], [`SamplerZ`].

use crate::params::BaseSampler;
use crate::prng::Prng;
use crate::{Params, tables};

/// The leaf sampler of ffSampling: an integer near `mu` at inverse width `isigma`.
///
/// The production implementation is [`SamplerZ`]; the precision tests replay recorded
/// decisions through this trait.
pub trait LeafSampler {
    /// One integer from $`D_{\mathbb Z,\sigma',\mu}`$ with $`\sigma'=1/\texttt{isigma}`$.
    fn sample(&mut self, mu: f64, isigma: f64) -> i32;
}

/// fn-dsa's SamplerZ over a borrowed PRNG, with the base and $`\sigma_{\min}`$ of a parameter
/// set.
pub struct SamplerZ<'a, P: Prng> {
    prng: &'a mut P,
    params: Params,
    trials: u64,
}

impl<'a, P: Prng> SamplerZ<'a, P> {
    /// A sampler for `params` (its `base` and `sigma_min`) drawing from `prng`.
    pub fn new(params: &Params, prng: &'a mut P) -> SamplerZ<'a, P> {
        SamplerZ {
            prng,
            params: *params,
            trials: 0,
        }
    }

    /// The number of base draws (trials) so far; acceptance rate = samples / trials.
    pub fn trials(&self) -> u64 {
        self.trials
    }

    /// fn-dsa `Sampler::next` (sampler.rs:131-192), with the base `B`.
    #[inline]
    fn next<B: BaseGauss>(&mut self, mu: f64, isigma: f64) -> i32 {
        // Centre mu = s + r, s integer, 0 <= r < 1 (FLR::floor, flr_native.rs:437-480).
        let s = mu.floor() as i64;
        let r = mu - s as f64;
        let s = s as i32;

        // dss = 1/(2 sigma^2) = 0.5 * isigma^2.
        let dss = (isigma * isigma) * 0.5;

        // ccs = sigma_min / sigma = sigma_min * isigma.
        let ccs = isigma * self.params.sigma_min;

        loop {
            // Bimodal proposal: z0 from the half-Gaussian of width sigma0, b a random bit;
            // z = z0 + 1 if b = 1, z = -z0 if b = 0.
            let (z0, b) = B::draw(self.prng);
            self.trials += 1;
            let z = b + ((b << 1) - 1) * z0;

            // Accept with probability ccs * exp(-x),
            // x = (z - r)^2 / (2 sigma^2) - z0^2 / (2 sigma0^2) >= 0.
            let t = z as f64 - r;
            let mut x = (t * t) * dss;
            x -= ((z0 * z0) as f64) * B::INV_2SQRSIGMA0;
            if ber_exp(self.prng, x, ccs) {
                return s + z;
            }
        }
    }
}

impl<P: Prng> LeafSampler for SamplerZ<'_, P> {
    #[inline]
    fn sample(&mut self, mu: f64, isigma: f64) -> i32 {
        match self.params.base {
            BaseSampler::HalfGauss3_1 => self.next::<HalfGauss31>(mu, isigma),
            BaseSampler::Falcon1_8205 => self.next::<Falcon18205>(mu, isigma),
        }
    }
}

/// A base half-Gaussian: a nonnegative $`z_0`$ and a sign bit $`b`$.
trait BaseGauss {
    /// $`1/(2\sigma_0^2)`$.
    const INV_2SQRSIGMA0: f64;
    /// Draws $`(z_0, b)`$.
    fn draw<P: Prng>(prng: &mut P) -> (i32, i32);
}

/// Width 3.1, the 192-bit table [`tables::BASE_3_1_192`], drawn lazily (module documentation).
struct HalfGauss31;

/// The base table: 41 entries of 192 bits, `[high, middle, low]` 64-bit limbs.
const BASE: &[[u64; 3]; 41] = &tables::BASE_3_1_192;

/// The number of entries whose top 63 bits $`H_j=\lfloor T_j/2^{129}\rfloor`$ are nonzero: a
/// prefix of the table, which is non-increasing.
const BASE_NZ: usize = count_top63_nonzero(BASE);

/// $`H_j`$ for those entries.
const BASE_HI63: [u64; BASE_NZ] = top63_prefix(BASE);

/// Whether some entry has $`H_j=0`$ (it ties with $`H=0`$).
const BASE_HAS_ZERO_TOP: bool = BASE_NZ < BASE.len();

const fn count_top63_nonzero(t: &[[u64; 3]]) -> usize {
    let mut k = 0;
    while k < t.len() && (t[k][0] >> 1) != 0 {
        k += 1;
    }
    // The table is non-increasing, so no later entry has nonzero top bits.
    let mut i = k;
    while i < t.len() {
        assert!(t[i][0] >> 1 == 0);
        i += 1;
    }
    k
}

const fn top63_prefix<const N: usize>(t: &[[u64; 3]]) -> [u64; N] {
    let mut out = [0u64; N];
    let mut i = 0;
    while i < N {
        out[i] = t[i][0] >> 1;
        i += 1;
    }
    out
}

/// `a < b` for 192-bit values given as `[high, middle, low]` limbs.
#[inline(always)]
fn lt192(a: [u64; 3], b: [u64; 3]) -> bool {
    let al = ((a[1] as u128) << 64) | a[2] as u128;
    let bl = ((b[1] as u128) << 64) | b[2] as u128;
    (a[0] < b[0]) | ((a[0] == b[0]) & (al < bl))
}

impl BaseGauss for HalfGauss31 {
    const INV_2SQRSIGMA0: f64 = BaseSampler::HalfGauss3_1.inv_2sqr_sigma0();

    #[inline(always)]
    fn draw<P: Prng>(prng: &mut P) -> (i32, i32) {
        // V = H * 2^129 + L, H = the top 63 bits (from one u64, whose low bit is the sign).
        let w = prng.next_u64();
        let b = (w & 1) as i32;
        let h = w >> 1;
        // Fixed-length scan on the top bits: z0 = #{j : H < H_j} (the entries with H_j = 0 never
        // count), and whether H ties with some H_j (with H_j = 0 when H = 0).
        let mut z = 0i32;
        let mut eq = (h == 0) & BASE_HAS_ZERO_TOP;
        for &t in BASE_HI63.iter() {
            z += (h < t) as i32;
            eq |= h == t;
        }
        if eq {
            // Probability <= 29 * 2^-63: the low 129 bits decide (full 192-bit comparison).
            // L = m * 2^65 + l * 2 + last.
            let m = prng.next_u64();
            let l = prng.next_u64();
            let last = (prng.next_u8() & 1) as u64;
            let v = [(h << 1) | (m >> 63), (m << 1) | (l >> 63), (l << 1) | last];
            z = 0;
            for &t in BASE.iter() {
                z += lt192(v, t) as i32;
            }
        }
        (z, b)
    }
}

/// Width 1.8205, fn-dsa's 79-bit `GAUSS0` ([`tables::GAUSS0_1_8205_79`]).
struct Falcon18205;

impl BaseGauss for Falcon18205 {
    const INV_2SQRSIGMA0: f64 = BaseSampler::Falcon1_8205.inv_2sqr_sigma0();

    /// fn-dsa `gaussian0` (sampler.rs:241-260), verbatim.
    #[inline(always)]
    fn draw<P: Prng>(prng: &mut P) -> (i32, i32) {
        // 80 random bits: the low bit is the sign, then three limbs of 24 bits.
        let lo = prng.next_u64();
        let hi = prng.next_u16();
        let b = (lo as i32) & 1;
        let v0 = ((lo as u32) >> 1) & 0x00FF_FFFF;
        let v1 = ((lo >> 25) as u32) & 0x00FF_FFFF;
        let v2 = ((lo >> 49) as u32) | ((hi as u32) << 15);

        // z = number of table rows above (v2, v1, v0).
        let mut z = 0i32;
        for row in tables::GAUSS0_1_8205_79.iter() {
            let cc = v0.wrapping_sub(row[2]) >> 31;
            let cc = v1.wrapping_sub(row[1]).wrapping_sub(cc) >> 31;
            let cc = v2.wrapping_sub(row[0]).wrapping_sub(cc) >> 31;
            z += cc as i32;
        }
        (z, b)
    }
}

/// $`\ln 2`$, fn-dsa `LOG2` = `FLR::scaled(6243314768165359, -53)` (sampler.rs:100).
const LOG2: f64 = f64::from_bits(0x3fe6_2e42_fefa_39ef);
/// $`1/\ln 2`$, fn-dsa `INV_LOG2` = `FLR::scaled(6497320848556798, -52)` (sampler.rs:103).
const INV_LOG2: f64 = f64::from_bits(0x3ff7_1547_652b_82fe);
/// $`2^{63}`$ (fn-dsa `FLR::mul2p63`, flr_native.rs:354-357).
const TWO_P63: f64 = 9_223_372_036_854_775_808.0;

/// A bit that is 1 with probability $`ccs\cdot e^{-x}`$, for $`x\ge0`$ (fn-dsa `ber_exp`,
/// sampler.rs:262-328, 64-bit branch).
#[inline(always)]
fn ber_exp<P: Prng>(prng: &mut P, x: f64, ccs: f64) -> bool {
    // x = s*log(2) + r with s an integer and 0 <= r < log(2) (trunc: x >= 0).
    let s = (x * INV_LOG2) as i64;
    let r = x - (s as f64) * LOG2;

    // ccs * exp(-x) = ccs * exp(-r) / 2^s, scaled by 2^64 (minus 1 so that it fits). fn-dsa
    // saturates s at 63 here (sampler.rs:271-281); instead, the threshold is 0 when s >= 64
    // (module documentation, "Far tail"): `keep` is all ones iff s < 64, and the shift equals s
    // whenever it matters. The cast saturates, so s >= 0 for x >= 0.
    let keep = (s.wrapping_sub(64) >> 63) as u64;
    let z = ((expm_p63(r, ccs) << 1).wrapping_sub(1) >> ((s as u32) & 63)) & keep;

    // Lazy comparison with a uniform 64-bit value, high byte first; it leaks only the outcome.
    for i in 0..8 {
        let w = prng.next_u8();
        let bz = (z >> (56 - (i << 3))) as u8;
        if w != bz {
            return w < bz;
        }
    }
    false
}

/// FACCT's polynomial for $`2^{63}\cdot ccs\cdot e^{-x}`$, $`0\le x<\ln2`$, $`0<ccs\le1`$ (fn-dsa
/// `FLR::expm_p63`, flr_native.rs:672-705, the 64-bit branch, and `EXPM_COEFFS` :743-757).
#[inline(always)]
fn expm_p63(x: f64, ccs: f64) -> u64 {
    const EXPM_COEFFS: [u64; 13] = [
        0x0000_0004_7411_83A3,
        0x0000_0036_548C_FC06,
        0x0000_024F_DCBF_140A,
        0x0000_171D_939D_E045,
        0x0000_D00C_F58F_6F84,
        0x0006_8068_1CF7_96E3,
        0x002D_82D8_305B_0FEA,
        0x0111_1111_0E06_6FD0,
        0x0555_5555_5507_0F00,
        0x1555_5555_5581_FF00,
        0x4000_0000_0002_B400,
        0x7FFF_FFFF_FFFF_4800,
        0x8000_0000_0000_0000,
    ];
    let mut y = EXPM_COEFFS[0];
    // FLR::trunc is a saturating `as i64` (flr_native.rs:482-485).
    let z = (((x * TWO_P63) as i64) as u64) << 1;
    for &c in EXPM_COEFFS[1..].iter() {
        // z * y over 128 bits, top 64 bits.
        let yy = (z as u128) * (y as u128);
        y = c.wrapping_sub((yy >> 64) as u64);
    }
    let z = (((ccs * TWO_P63) as i64) as u64) << 1;
    (((z as u128) * (y as u128)) >> 64) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prng::Shake256Prng;

    /// A deterministic byte source for unit tests of `ber_exp`: replays given bytes.
    struct Bytes(Vec<u8>, usize);
    impl Prng for Bytes {
        fn next_u8(&mut self) -> u8 {
            let b = self.0[self.1];
            self.1 += 1;
            b
        }
        fn next_u16(&mut self) -> u16 {
            self.next_u8() as u16 | (self.next_u8() as u16) << 8
        }
        fn next_u64(&mut self) -> u64 {
            (0..8).fold(0u64, |a, i| a | (self.next_u8() as u64) << (8 * i))
        }
        fn next_bytes(&mut self, dst: &mut [u8]) {
            for d in dst.iter_mut() {
                *d = self.next_u8();
            }
        }
    }

    /// expm_p63 approximates 2^63 * ccs * exp(-x) to about 50 bits.
    #[test]
    fn expm_p63_accuracy() {
        let mut worst = 0.0f64;
        for i in 0..=1000 {
            let x = LOG2 * i as f64 / 1001.0;
            for ccs in [1.0, 0.75, 0.5, 0.1] {
                let got = expm_p63(x, ccs) as f64;
                let want = TWO_P63 * ccs * (-x).exp();
                let rel = ((got - want) / want).abs();
                worst = worst.max(rel);
            }
        }
        assert!(worst < 2f64.powi(-45), "worst relative error {worst:e}");
    }

    /// ber_exp compares the 64-bit threshold with the stream, high byte first, lazily.
    #[test]
    fn ber_exp_lazy_comparison() {
        // x = 0, ccs = 1: expm_p63 = 2^63 - 1 (ccs * 2^63 saturates), threshold
        // 2^64 - 3 = FF..FF FD.
        assert_eq!(expm_p63(0.0, 1.0), (1u64 << 63) - 1);
        let mut p = Bytes(vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD], 0);
        assert!(!ber_exp(&mut p, 0.0, 1.0)); // equal to the threshold: reject
        assert_eq!(p.1, 8);
        let mut p = Bytes(vec![0x00], 0);
        assert!(ber_exp(&mut p, 0.0, 1.0)); // first byte smaller: accept after one byte
        assert_eq!(p.1, 1);
        // x large: the threshold is 0, so every nonzero byte rejects at once.
        let mut p = Bytes(vec![0x01], 0);
        assert!(!ber_exp(&mut p, 1000.0, 1.0));
        assert_eq!(p.1, 1);
    }

    /// The far tail (finding F1 in docs/precision.md): for s = floor(x / ln 2) >= 64 the exact
    /// acceptance probability ccs exp(-x) is below 2^-64, and the threshold is 0, so even the
    /// all-zero Bernoulli bytes reject (fn-dsa's saturated shift gives the threshold 1 there and
    /// accepts them). For s = 63 the exact probability lies in (2^-64, 2^-63] and the threshold is
    /// floor(2^64 ccs exp(-x)) = 1, as in fn-dsa: the all-zero bytes accept.
    #[test]
    fn ber_exp_far_tail() {
        let zeros = || Bytes(vec![0u8; 8], 0);
        for (x, accept) in [
            (63.5 * LOG2, true),  // s = 63
            (64.0 * LOG2, false), // s = 64
            (64.5 * LOG2, false),
            (91.95, false), // the crafted case of examples/stats_tail.rs
            (1000.0, false),
        ] {
            let s = (x * INV_LOG2) as i64;
            let mut p = zeros();
            assert_eq!(ber_exp(&mut p, x, 1.0), accept, "x = {x}, s = {s}");
            assert_eq!(p.1, 8, "x = {x}");
        }
        // fn-dsa's threshold at s >= 64, for comparison: (2 expm_p63(r, 1) - 1) >> 63 = 1.
        let x = 64.5 * LOG2;
        let s = (x * INV_LOG2) as i64;
        let r = x - (s as f64) * LOG2;
        assert_eq!((expm_p63(r, 1.0) << 1).wrapping_sub(1) >> 63, 1);
    }

    /// The base samplers' byte consumption, as `BaseSampler::bytes_per_draw` states: 8 bytes per
    /// draw for the PCS base (25 in the rare fallback) and 10 for Falcon's, as fn-dsa.
    #[test]
    fn base_draw_consumption() {
        let mut p = Bytes((0..=255u8).cycle().take(4096).collect(), 0);
        let _ = HalfGauss31::draw(&mut p);
        assert_eq!(p.1, 8);
        assert_eq!(p.1, BaseSampler::HalfGauss3_1.bytes_per_draw());
        let mut p = Bytes((0..=255u8).cycle().take(4096).collect(), 0);
        let _ = Falcon18205::draw(&mut p);
        assert_eq!(p.1, BaseSampler::Falcon1_8205.bytes_per_draw());
    }

    /// The trimmed scan's constants: the entries with nonzero top bits form a prefix of 28, and
    /// the remaining 13 have zero top bits.
    #[test]
    fn base_table_prefix() {
        assert_eq!(BASE.len(), 41);
        assert_eq!(BASE_NZ, 28);
        const { assert!(BASE_HAS_ZERO_TOP) };
        for (j, t) in BASE.iter().enumerate() {
            assert_eq!((t[0] >> 1) != 0, j < BASE_NZ, "entry {j}");
        }
        for w in BASE.windows(2) {
            assert!(!lt192(w[0], w[1]), "the table must be non-increasing");
        }
    }

    /// The 192-bit value as `[high, middle, low]` limbs.
    fn limbs(v_hi: u64, l: u128) -> [u64; 3] {
        // V = v_hi * 2^128 + l, l < 2^128.
        [v_hi, (l >> 64) as u64, l as u64]
    }

    /// The reference: z0 = #{j : V < T_j} with the full 192-bit V.
    fn z_full(v: [u64; 3]) -> i32 {
        BASE.iter().map(|&t| lt192(v, t) as i32).sum()
    }

    /// Bytes for a draw with top bits `h`, sign `b`, and (if the fallback runs) the low 129 bits
    /// L = m * 2^65 + l * 2 + last.
    fn draw_bytes(h: u64, b: u64, m: u64, l: u64, last: u8) -> Vec<u8> {
        let mut v = ((h << 1) | b).to_le_bytes().to_vec();
        v.extend_from_slice(&m.to_le_bytes());
        v.extend_from_slice(&l.to_le_bytes());
        v.push(last);
        v
    }

    /// V from the draw's parts: H * 2^129 + m * 2^65 + l * 2 + last.
    fn v_of(h: u64, m: u64, l: u64, last: u64) -> [u64; 3] {
        [(h << 1) | (m >> 63), (m << 1) | (l >> 63), (l << 1) | last]
    }

    /// The width-3.1 draw at the extremes: V = 0 gives the largest z0 (41) through the
    /// fallback (H = 0 ties with the entries whose top bits are 0); V = max gives 0 directly.
    #[test]
    fn half_gauss_extremes() {
        let mut p = Bytes(draw_bytes(0, 1, 0, 0, 0), 0);
        assert_eq!(HalfGauss31::draw(&mut p), (41, 1));
        assert_eq!(p.1, 25);
        let mut p = Bytes(draw_bytes(u64::MAX >> 1, 0, 0, 0, 0), 0);
        assert_eq!(HalfGauss31::draw(&mut p), (0, 0));
        assert_eq!(p.1, 8);
    }

    /// The lazy draw equals the full 192-bit comparison, in particular on the fallback path: H
    /// equal to the top bits of every entry, with low bits just below, at and above the entry's;
    /// and H = 0 against the small entries.
    #[test]
    fn half_gauss_lazy_equals_full() {
        let mut rng = Shake256Prng::new(b"lazy");
        for &t in BASE.iter() {
            let h = t[0] >> 1;
            // The entry's low 129 bits: (t[0] & 1) * 2^128 + t[1] * 2^64 + t[2].
            let low = ((t[1] as u128) << 64) | t[2] as u128;
            let top_bit = t[0] & 1;
            for delta in [-2i128, -1, 0, 1, 2] {
                // Shift the 129-bit low part by delta (clamped to [0, 2^129)), as (bit128, u128).
                let (mut bit, mut l128) = (top_bit, low);
                if delta < 0 {
                    let d = delta.unsigned_abs();
                    if l128 >= d {
                        l128 -= d;
                    } else if bit == 1 {
                        bit = 0;
                        l128 = l128.wrapping_sub(d);
                    } else {
                        l128 = 0;
                    }
                } else {
                    let d = delta as u128;
                    match l128.checked_add(d) {
                        Some(x) => l128 = x,
                        None if bit == 0 => {
                            bit = 1;
                            l128 = l128.wrapping_add(d);
                        }
                        None => l128 = u128::MAX,
                    }
                }
                // L = m * 2^65 + l * 2 + last with L = bit * 2^128 + l128.
                let m = ((bit as u128) << 63 | (l128 >> 65)) as u64;
                let l = (l128 >> 1) as u64;
                let last = (l128 & 1) as u8;
                let v = v_of(h, m, l, last as u64);
                assert_eq!(v, limbs((h << 1) | bit, l128));
                let mut p = Bytes(draw_bytes(h, 1, m, l, last), 0);
                assert_eq!(HalfGauss31::draw(&mut p), (z_full(v), 1));
                assert_eq!(p.1, 25);
            }
        }
        for _ in 0..100_000 {
            let w = rng.next_u64();
            let (m, l, last) = (rng.next_u64(), rng.next_u64(), rng.next_u8() & 1);
            let mut p = Bytes(draw_bytes(w >> 1, w & 1, m, l, last), 0);
            let (z, b) = HalfGauss31::draw(&mut p);
            let v = v_of(w >> 1, m, l, last as u64);
            assert_eq!((z, b), (z_full(v), (w & 1) as i32));
        }
    }

    /// Chi-square (bins merged to expected counts >= 10) and its Wilson–Hilferty z-score.
    fn chi2_z(hist: &[u64], probs: &[f64], draws: u64) -> (f64, usize, f64) {
        let mut bins: Vec<(f64, f64)> = Vec::new();
        let (mut e, mut o) = (0.0, 0.0);
        for (i, &p) in probs.iter().enumerate() {
            e += p * draws as f64;
            o += hist[i] as f64;
            if e >= 10.0 {
                bins.push((e, o));
                e = 0.0;
                o = 0.0;
            }
        }
        let last = bins.last_mut().unwrap();
        last.0 += e;
        last.1 += o;
        let x2: f64 = bins.iter().map(|(e, o)| (o - e) * (o - e) / e).sum();
        let k = (bins.len() - 1) as f64;
        let z = ((x2 / k).cbrt() - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
        (x2, bins.len() - 1, z)
    }

    /// Reverse-CDT probabilities P[K = i] = (T_{i-1} - T_i) / 2^bits, T_{-1} = 2^bits.
    fn rcdt_probs(t: &[u128], bits: u32) -> Vec<f64> {
        let scale = 2f64.powi(bits as i32);
        let mut out = Vec::with_capacity(t.len() + 1);
        for i in 0..=t.len() {
            let hi = if i == 0 {
                // 2^bits - T_0, without overflow at bits = 128.
                (u128::MAX >> (128 - bits)) - t[0] + 1
            } else {
                t[i - 1] - if i < t.len() { t[i] } else { 0 }
            };
            out.push(hi as f64 / scale);
        }
        out
    }

    /// The same for the 192-bit table: the differences are exact (192-bit subtraction), then
    /// scaled.
    fn rcdt_probs_192(t: &[[u64; 3]]) -> Vec<f64> {
        let to_f64 = |v: [u64; 3]| {
            (v[0] as f64) * 2f64.powi(128) + (v[1] as f64) * 2f64.powi(64) + v[2] as f64
        };
        let sub = |a: [u64; 3], b: [u64; 3]| {
            let (lo, c1) = a[2].overflowing_sub(b[2]);
            let (mid, c2a) = a[1].overflowing_sub(b[1]);
            let (mid, c2b) = mid.overflowing_sub(c1 as u64);
            let hi = a[0] - b[0] - (c2a | c2b) as u64;
            [hi, mid, lo]
        };
        let scale = 2f64.powi(192);
        let mut out = Vec::with_capacity(t.len() + 1);
        for i in 0..=t.len() {
            let d = if i == 0 {
                // 2^192 - T_0 = (2^192 - 1 - T_0) + 1.
                let c = [!t[0][0], !t[0][1], !t[0][2]];
                to_f64(c) + 1.0
            } else {
                to_f64(sub(t[i - 1], if i < t.len() { t[i] } else { [0; 3] }))
            };
            out.push(d / scale);
        }
        out
    }

    /// S-SAMP-3: each base half-Gaussian against its exact table probabilities, 10^7 draws, and
    /// the sign bit (binomial, 6 standard errors).
    #[test]
    fn base_samplers_chi2() {
        let draws = 10_000_000u64;
        // Width 3.1, 192-bit table.
        let probs = rcdt_probs_192(BASE);
        let mut hist = vec![0u64; probs.len()];
        let mut ones = 0u64;
        let mut prng = Shake256Prng::new(b"base 3.1");
        for _ in 0..draws {
            let (z, b) = HalfGauss31::draw(&mut prng);
            hist[z as usize] += 1;
            ones += b as u64;
        }
        let (x2, df, z) = chi2_z(&hist, &probs, draws);
        assert!(z < 4.75, "width 3.1: chi2 {x2:.1} on {df} df, z = {z:.2}");
        let half = draws as f64 / 2.0;
        assert!((ones as f64 - half).abs() < 6.0 * (draws as f64 / 4.0).sqrt());
        // Width 1.8205, fn-dsa's 79-bit GAUSS0.
        let t79: Vec<u128> = tables::GAUSS0_1_8205_79
            .iter()
            .map(|r| ((r[0] as u128) << 48) | ((r[1] as u128) << 24) | r[2] as u128)
            .collect();
        let probs = rcdt_probs(&t79, 79);
        let mut hist = vec![0u64; probs.len()];
        let mut ones = 0u64;
        let mut prng = Shake256Prng::new(b"base 1.8205");
        for _ in 0..draws {
            let (z, b) = Falcon18205::draw(&mut prng);
            hist[z as usize] += 1;
            ones += b as u64;
        }
        let (x2, df, z) = chi2_z(&hist, &probs, draws);
        assert!(
            z < 4.75,
            "width 1.8205: chi2 {x2:.1} on {df} df, z = {z:.2}"
        );
        assert!((ones as f64 - half).abs() < 6.0 * (draws as f64 / 4.0).sqrt());
    }

    /// Acceptance per trial is ~0.6337 at the PCS base (design.md §2) at every width and
    /// centre: a quick check (the 10^7-draw version is in tests/sign_sampler.rs).
    #[test]
    fn acceptance_rate_quick() {
        let params = Params::PCS;
        for (k, isig) in [1.0 / 2.2173, 1.0 / 2.6, 1.0 / 3.0351]
            .into_iter()
            .enumerate()
        {
            let mut prng = Shake256Prng::new(&[b'a', k as u8]);
            let mut s = SamplerZ::new(&params, &mut prng);
            let draws = 200_000;
            for i in 0..draws {
                let _ = s.sample(0.37 + i as f64, isig);
            }
            let acc = draws as f64 / s.trials() as f64;
            // 6 standard errors of a binomial proportion over ~3.2e5 trials.
            assert!((acc - 0.633687).abs() < 0.006, "acceptance {acc}");
        }
    }
}
