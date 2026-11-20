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
//!   only tie, when $`H=0`$, and never count). The scan counts the entries above $`H`$, and a
//!   second sum over every entry tells whether $`H`$ equals one of them (`scan_top`). If it
//!   equals none and $`H\ne0`$, $`z_0`$ is decided ($`V<T_j\iff H<H_j`$). Otherwise
//!   (probability at most $`29\cdot2^{-63}=2^{-58.1}`$ per draw) the low 129 bits of $`V`$ are
//!   drawn (two `next_u64`, then the low bit of `next_u8`) and the full 192-bit comparison
//!   decides. The distribution is exactly that of the 192-bit table. A draw takes 8 PRNG bytes
//!   (25 in the rare fallback). The fallback's branch leaks only that rare event. The tie is not
//!   tested by comparing $`H`$ with each $`H_j`$ for equality: compilers turn that test into a
//!   tree of branches on $`H`$ (`tools/sampler_branches.py` checks the compiled code).
//! - `Falcon1_8205`: fn-dsa's `gaussian0` (one `next_u64`, one `next_u16`, and the 31+24+24-bit
//!   limb comparison with every row of [`crate::tables::GAUSS0_1_8205_79`], which equals fn-dsa's
//!   `GAUSS0`). The rows are read through a copy of the table by columns, padded with zero rows
//!   that never count, so that the compiler vectorizes the scan: the same comparisons, so the
//!   same count.
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
//! **First-byte shortcut.** `ber_exp` compares a uniform 64-bit value with the threshold
//! $`z\approx2^{64}ccs\,e^{-x}`$ from the top byte down, and the first byte decides in all but
//! 1 of 256 trials. The sampler computes FACCT's polynomial only when that first byte could be
//! decided otherwise: an `f64` estimate $`f`$ of $`z/2^{56}`$ (a degree-5 Taylor polynomial of
//! $`e^{-u}`$ around $`\ln2/2`$) gives $`m=\min(254,\max(0,\lfloor f-1/32\rfloor))`$, a first
//! byte below $`m`$ accepts, one above $`m+1`$ rejects, and $`m`$ or $`m+1`$ (2 of 256 trials)
//! run the exact comparison from that byte on (the comparisons with $`m`$ are taken on
//! $`\max(f-1/32,0)`$ in floating point, which gives the same three outcomes without a
//! conversion to an integer). For $`x\ge0`$ and $`1/2\le ccs\le1`$,
//! $`|f-z/2^{56}|\le6.2\cdot10^{-4}`$, so $`\lfloor z/2^{56}\rfloor\in\{m,m+1\}`$, and the
//! result and the bytes read are those of `ber_exp` (lemma of the 2026-11-18 study: a paper
//! proof whose numerical steps are certified with interval arithmetic, the window logic checked
//! in Lean, and 6.4e9 inputs compared in Rust). Leaves outside that range of `ccs`, and
//! negative or NaN `x`, take `ber_exp` itself.
//!
//! **Preconditions** (enforced for every leaf by [`crate::ExpandedKey::new`]):
//! $`\sigma_{\min}\le\sigma'`$ (so $`ccs\le1`$) and $`\sigma'<\sigma_0`$ with a $`2^{-40}`$ rounding
//! margin (so $`x\ge0`$). Outside them the output distribution is wrong; the sampler does not
//! check them itself (as fn-dsa).
//!
//! **Side channels** (as fn-dsa): full table scans; the acceptance probability per trial is
//! independent of $`\sigma'`$ and $`\mu`$ (0.633687 for the PCS base, 0.584958 for Falcon's
//! [computed, design.md §2]); the lazy byte comparison in `ber_exp` leaks only accept or reject.
//! The first-byte shortcut computes FACCT's polynomial in 2 of 256 trials, whatever the inputs.
//! Jointly with the trial's outcome that event depends on the inputs, as the lazy comparison's
//! second byte already does. Native `f64` arithmetic may be variable-time on some CPUs.
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

    /// fn-dsa `Sampler::next` (sampler.rs:131-192), with the base `B`. `ROUND` says that the code
    /// is compiled for a processor that rounds to an integer in one instruction (aarch64, and the
    /// x86-64-v3 copy): then the integer parts below are taken by rounding (`floor`, `trunc`)
    /// rather than through a conversion to an integer and back, with the same values.
    #[inline(always)]
    fn next<B: BaseGauss, const ROUND: bool>(&mut self, mu: f64, isigma: f64) -> i32 {
        // Centre mu = s + r, s integer, 0 <= r < 1 (FLR::floor, flr_native.rs:437-480).
        let (s, r) = centre::<ROUND>(mu);

        // dss = 1/(2 sigma^2) = 0.5 * isigma^2.
        let dss = (isigma * isigma) * 0.5;

        // ccs = sigma_min / sigma = sigma_min * isigma.
        let ccs = isigma * self.params.sigma_min;
        // The first-byte shortcut of the Bernoulli step holds for 1/2 <= ccs <= 1 (module
        // documentation), which every leaf of a valid key satisfies. Other leaves (reachable
        // only through `hazmat`) take the exact step on every trial.
        let shortcut = (0.5..=1.0).contains(&ccs);

        loop {
            // Bimodal proposal: z0 from the half-Gaussian of width sigma0, b a random bit;
            // z = z0 + 1 if b = 1, z = -z0 if b = 0.
            let (z0, b) = B::draw::<P, ROUND>(self.prng);
            self.trials += 1;
            let z = b + ((b << 1) - 1) * z0;

            // Accept with probability ccs * exp(-x),
            // x = (z - r)^2 / (2 sigma^2) - z0^2 / (2 sigma0^2) >= 0.
            let t = z as f64 - r;
            let mut x = (t * t) * dss;
            x -= ((z0 * z0) as f64) * B::INV_2SQRSIGMA0;
            let accept = if shortcut {
                ber_exp_shortcut::<P, ROUND>(self.prng, x, ccs)
            } else {
                ber_exp(self.prng, x, ccs)
            };
            if accept {
                return s + z;
            }
        }
    }
}

/// Whether the portable code rounds to an integer in one instruction (`frintm`, `frintz` on
/// aarch64). On x86-64 without SSE4.1 `floor` and `trunc` are library calls, and the conversions
/// to an integer and back are faster.
const NATIVE_ROUND: bool = cfg!(target_arch = "aarch64");

impl<P: Prng> LeafSampler for SamplerZ<'_, P> {
    #[inline]
    fn sample(&mut self, mu: f64, isigma: f64) -> i32 {
        match self.params.base {
            BaseSampler::HalfGauss3_1 => self.next::<HalfGauss31, NATIVE_ROUND>(mu, isigma),
            BaseSampler::Falcon1_8205 => self.next::<Falcon18205, NATIVE_ROUND>(mu, isigma),
        }
    }
}

/// A centre $`\mu=s+r`$ with $`s=\lfloor\mu\rfloor`$ (as `i32`, wrapping like fn-dsa's cast)
/// and $`r=\mu-s`$. fn-dsa computes `r = mu - (floor(mu) as i64) as f64`. With `ROUND`, `r` is
/// `mu - floor(mu)` instead, one conversion shorter: the same value whenever
/// `|floor(mu)| < 2^62` (the integer is then exact both ways), except for `mu = -0.0`, where it
/// is `+0.0` against fn-dsa's `-0.0`, which no difference `z - r` of the sampler distinguishes
/// (test `centre_forms_agree`). Larger centres take fn-dsa's form.
#[inline(always)]
fn centre<const ROUND: bool>(mu: f64) -> (i32, f64) {
    let fl = mu.floor();
    if ROUND {
        if fl.abs() >= 4_611_686_018_427_387_904.0 {
            return centre_far(mu);
        }
        return (fl as i64 as i32, mu - fl);
    }
    let s = fl as i64;
    (s as i32, mu - s as f64)
}

/// [`centre`] for `|floor(mu)| >= 2^62` (or NaN): fn-dsa's form.
#[cold]
#[inline(never)]
fn centre_far(mu: f64) -> (i32, f64) {
    let s = mu.floor() as i64;
    (s as i32, mu - s as f64)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl<P: Prng> SamplerZ<'_, P> {
    /// [`LeafSampler::sample`] compiled for x86-64-v3 (AVX2, BMI1, BMI2, LZCNT, POPCNT, no FMA),
    /// with the trial loop, the base draws and the PRNG reads inside it: the same operations, so
    /// the same results and the same bytes. SampPre calls it on processors that have the
    /// instructions (sign.rs).
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
    pub(crate) fn sample_v3(&mut self, mu: f64, isigma: f64) -> i32 {
        match self.params.base {
            BaseSampler::HalfGauss3_1 => self.next::<HalfGauss31, true>(mu, isigma),
            BaseSampler::Falcon1_8205 => self.next::<Falcon18205, true>(mu, isigma),
        }
    }
}

/// A leaf sampler from a closure: lets a function compiled with more instructions pass a call
/// of its own compilation (`SamplerZ::sample_v3`) to the generic ffSampling.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(crate) struct FnSampler<F>(pub(crate) F);

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl<F: FnMut(f64, f64) -> i32> LeafSampler for FnSampler<F> {
    #[inline(always)]
    fn sample(&mut self, mu: f64, isigma: f64) -> i32 {
        (self.0)(mu, isigma)
    }
}

/// A base half-Gaussian: a nonnegative $`z_0`$ and a sign bit $`b`$.
trait BaseGauss {
    /// $`1/(2\sigma_0^2)`$.
    const INV_2SQRSIGMA0: f64;
    /// Draws $`(z_0, b)`$, in code for 64-bit vector comparisons if `WIDE` ([`scan_top`]).
    fn draw<P: Prng, const WIDE: bool>(prng: &mut P) -> (i32, i32);
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

/// The fixed-length scan on the top bits $`H`$ of a draw: $`z_0=\#\{j:H<H_j\}`$ (the entries
/// with $`H_j=0`$ never count), and whether $`H`$ ties with some $`H_j`$ (with an entry
/// $`H_j=0`$ when $`H=0`$). The tie comes from the same sums over every entry, with no branch
/// and no index that depends on $`H`$, in one of two forms of equal value, whichever compiles to
/// the faster code. With `WIDE` (code with 64-bit vector comparisons: aarch64, x86-64-v3), one
/// sum $`d=\#\{j:H<H_j\}-\#\{j:H>H_j\}`$: the nonzero $`H_j`$ are distinct, so at most one
/// equals $`H`$, and $`d+28=2z_0+t`$ with $`t\in\{0,1\}`$ the number of entries equal to $`H`$.
/// Without it, two counts, $`z_0`$ and $`\#\{j:H\le H_j\}`$, which exceeds $`z_0`$ by $`t`$. The
/// direct form, an OR of the equalities $`H=H_j`$, is the same value too, but compilers turn that
/// OR into a tree of branches on $`H`$, that is, on the secret draw (test
/// `tie_counts_equal_scan`, and `tools/sampler_branches.py` on the compiled code).
#[inline(always)]
fn scan_top<const WIDE: bool>(h: u64) -> (i32, bool) {
    let zero_tie = (h == 0) & BASE_HAS_ZERO_TOP;
    if WIDE {
        let mut d = 0i32;
        for &t in BASE_HI63.iter() {
            d += (h < t) as i32 - (h > t) as i32;
        }
        let s = d + BASE_NZ as i32;
        (s >> 1, ((s & 1) != 0) | zero_tie)
    } else {
        let mut lt = 0i32;
        let mut le = 0i32;
        for &t in BASE_HI63.iter() {
            lt += (h < t) as i32;
            le += (h <= t) as i32;
        }
        (lt, (le != lt) | zero_tie)
    }
}

/// The nonzero $`H_j`$ decrease strictly, as the one-sum form of [`scan_top`] needs.
const _: () = {
    let mut j = 1;
    while j < BASE_NZ {
        assert!(BASE_HI63[j] < BASE_HI63[j - 1]);
        j += 1;
    }
};

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
    fn draw<P: Prng, const WIDE: bool>(prng: &mut P) -> (i32, i32) {
        // V = H * 2^129 + L, H = the top 63 bits (from one u64, whose low bit is the sign).
        let w = prng.next_u64();
        let b = (w & 1) as i32;
        let h = w >> 1;
        let (mut z, eq) = scan_top::<WIDE>(h);
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

/// The rows of [`tables::GAUSS0_1_8205_79`] padded with zero rows to a multiple of 8.
const GAUSS0_ROWS: usize = tables::GAUSS0_1_8205_79.len().next_multiple_of(8);

/// The columns of [`tables::GAUSS0_1_8205_79`] (the 31-bit high limbs, then the two 24-bit
/// limbs), padded with zero rows to [`GAUSS0_ROWS`].
const GAUSS0_COLS: [[u32; GAUSS0_ROWS]; 3] = {
    let t = &tables::GAUSS0_1_8205_79;
    let mut c = [[0u32; GAUSS0_ROWS]; 3];
    let mut i = 0;
    while i < t.len() {
        c[0][i] = t[i][0];
        c[1][i] = t[i][1];
        c[2][i] = t[i][2];
        i += 1;
    }
    c
};

impl BaseGauss for Falcon18205 {
    const INV_2SQRSIGMA0: f64 = BaseSampler::Falcon1_8205.inv_2sqr_sigma0();

    /// fn-dsa `gaussian0` (sampler.rs:241-260).
    #[inline(always)]
    fn draw<P: Prng, const WIDE: bool>(prng: &mut P) -> (i32, i32) {
        let lo = prng.next_u64();
        let hi = prng.next_u16();
        Self::limbs(lo, hi)
    }
}

impl Falcon18205 {
    /// The table comparison of fn-dsa's `gaussian0` on the 80 bits `lo`, `hi`.
    #[inline(always)]
    fn limbs(lo: u64, hi: u16) -> (i32, i32) {
        // 80 random bits: the low bit is the sign, then three limbs of 24 bits.
        let b = (lo as i32) & 1;
        let v0 = ((lo as u32) >> 1) & 0x00FF_FFFF;
        let v1 = ((lo >> 25) as u32) & 0x00FF_FFFF;
        let v2 = ((lo >> 49) as u32) | ((hi as u32) << 15);

        // z = number of table rows above (v2, v1, v0): fn-dsa's comparison on each row, over the
        // columns of the table (GAUSS0_COLS), which the compiler can vectorize. The padding rows
        // are zero and never count: v0, v1 < 2^24 and v2 < 2^31.
        let mut z = 0i32;
        let [c0, c1, c2] = &GAUSS0_COLS;
        for ((&t0, &t1), &t2) in c0.iter().zip(c1.iter()).zip(c2.iter()) {
            let cc = v0.wrapping_sub(t2) >> 31;
            let cc = v1.wrapping_sub(t1).wrapping_sub(cc) >> 31;
            let cc = v2.wrapping_sub(t0).wrapping_sub(cc) >> 31;
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

/// Half-width of the first-byte window of [`ber_exp_shortcut`], in units of the top byte.
const DELTA: f64 = 1.0 / 32.0;
/// The expansion point of the estimate: $`\ln2/2`$ (bit for bit `LOG2 / 2`).
const R0: f64 = 0.346_573_590_279_972_65;
/// $`2^8e^{-R_0}`$, rounded (one unit in the last place below the nearest `f64`, which the
/// error bound includes).
const E0: f64 = 0.707_106_781_186_547_5 * 256.0;

/// [`ber_exp`] with the first byte decided from an `f64` estimate of the threshold's top byte
/// (module documentation, "First-byte shortcut"): the same result and the same PRNG bytes as
/// [`ber_exp`] for every `x >= 0` and `1/2 <= ccs <= 1`. FACCT's polynomial runs only when the
/// first byte is `m` or `m + 1`, `m = clamp(floor(f - 1/32), 0, 254)`: in 2 of 256 trials,
/// whatever the inputs. Other `x` (negative, NaN) take [`ber_exp`].
#[inline(always)]
fn ber_exp_shortcut<P: Prng, const ROUND: bool>(prng: &mut P, x: f64, ccs: f64) -> bool {
    if x.is_nan() || x < 0.0 {
        return ber_exp(prng, x, ccs);
    }
    // s = trunc(x / ln 2), r = x - s ln 2. With ROUND the float s is the rounding itself, not
    // the integer converted back: the same value for s < 2^63, and for larger s (x >= 2^63 ln 2)
    // f below is 0 either way.
    let xs = x * INV_LOG2;
    let (s, sf) = if ROUND {
        let sf = xs.trunc();
        (sf as i64, sf)
    } else {
        let s = xs as i64;
        (s, s as f64)
    };
    let r = x - sf * LOG2;
    // f ~ ccs e^-r 2^(8 - s) = z / 2^56, within 6.2e-4 for 0 <= s <= 7. For s >= 8 the
    // threshold's top byte is 0 (s >= 64: the threshold is 0), and f = 0 gives m = 0.
    let f = if s < 8 {
        let u = r - R0;
        let u2 = u * u;
        let p = (1.0 - u) + u2 * ((0.5 - u * (1.0 / 6.0)) + u2 * (1.0 / 24.0 - u * (1.0 / 120.0)));
        (ccs * f64::from_bits((1023 - s as u64) << 52)) * (E0 * p)
    } else {
        0.0
    };
    let w0 = prng.next_u8();
    let (accept, reject) = first_byte_decision(f, w0);
    if accept {
        return true;
    }
    if reject {
        return false;
    }
    ber_exp_from(prng, w0, x, ccs)
}

/// The decision of the first Bernoulli byte `w` against the window
/// $`m=\min(254,\max(0,\lfloor f-1/32\rfloor))`$ of the estimate $`0\le f<260`$: accept when
/// $`w<m`$, reject when $`w>m+1`$, and neither (the exact comparison) otherwise. Computed
/// without forming $`m`$, so that neither a conversion to an integer nor the subtraction of
/// $`1/32`$ sits in the chain of dependent operations: $`f-1/32`$ is exact for $`f\ge1/32`$
/// (both are multiples of the ulp of $`f`$), so for integers $`w`$, $`w<\lfloor f-1/32\rfloor`$
/// is $`f\ge w+1+1/32`$, and $`w>\min(254,\max(0,\lfloor f-1/32\rfloor))+1`$ is
/// $`w\ge2`$ and $`f<w-1+1/32`$ (both sides false when $`\lfloor f-1/32\rfloor\ge254`$, since
/// $`w\le255`$). The thresholds are exact sums of small integers and $`1/32`$ (test
/// `first_byte_decision_equals_window`).
#[inline(always)]
fn first_byte_decision(f: f64, w: u8) -> (bool, bool) {
    let wf = w as f64;
    (
        (w < 254) & (f >= (wf + 1.0) + DELTA),
        (w >= 2) & (f < (wf - 1.0) + DELTA),
    )
}

/// [`ber_exp`] from an already read first byte `w0`: the exact threshold, then the lazy
/// comparison from byte 0 on.
#[cold]
#[inline(never)]
fn ber_exp_from<P: Prng>(prng: &mut P, w0: u8, x: f64, ccs: f64) -> bool {
    let s = (x * INV_LOG2) as i64;
    let r = x - (s as f64) * LOG2;
    let keep = (s.wrapping_sub(64) >> 63) as u64;
    let z = ((expm_p63(r, ccs) << 1).wrapping_sub(1) >> ((s as u32) & 63)) & keep;
    let bz = (z >> 56) as u8;
    if w0 != bz {
        return w0 < bz;
    }
    for i in 1..8 {
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
        let _ = HalfGauss31::draw::<_, false>(&mut p);
        assert_eq!(p.1, 8);
        assert_eq!(p.1, BaseSampler::HalfGauss3_1.bytes_per_draw());
        let mut p = Bytes((0..=255u8).cycle().take(4096).collect(), 0);
        let _ = Falcon18205::draw::<_, false>(&mut p);
        assert_eq!(p.1, BaseSampler::Falcon1_8205.bytes_per_draw());
    }

    /// The scan of fn-dsa's table by columns (with zero padding rows) gives the count of
    /// fn-dsa's row-by-row loop (sampler.rs:241-260, verbatim below), on random 80-bit draws and
    /// on draws just below, at and above every row.
    #[test]
    fn gauss0_columns_equal_rows() {
        let rows = |lo: u64, hi: u16| -> (i32, i32) {
            let b = (lo as i32) & 1;
            let v0 = ((lo as u32) >> 1) & 0x00FF_FFFF;
            let v1 = ((lo >> 25) as u32) & 0x00FF_FFFF;
            let v2 = ((lo >> 49) as u32) | ((hi as u32) << 15);
            let mut z = 0i32;
            for row in tables::GAUSS0_1_8205_79.iter() {
                let cc = v0.wrapping_sub(row[2]) >> 31;
                let cc = v1.wrapping_sub(row[1]).wrapping_sub(cc) >> 31;
                let cc = v2.wrapping_sub(row[0]).wrapping_sub(cc) >> 31;
                z += cc as i32;
            }
            (z, b)
        };
        let mut rng = Shake256Prng::new(b"gauss0-columns");
        for _ in 0..200_000 {
            let (lo, hi) = (rng.next_u64(), rng.next_u16());
            assert_eq!(Falcon18205::limbs(lo, hi), rows(lo, hi));
        }
        // The 79-bit value V = v2 * 2^48 + v1 * 2^24 + v0 next to every row.
        for row in tables::GAUSS0_1_8205_79.iter() {
            let t = ((row[0] as u128) << 48) | ((row[1] as u128) << 24) | row[2] as u128;
            for d in [-2i128, -1, 0, 1, 2] {
                let v = (t as i128 + d).clamp(0, (1i128 << 79) - 1) as u128;
                for sign in 0..2u64 {
                    let lo = ((v as u64) << 1) | sign;
                    let hi = (v >> 63) as u16;
                    assert_eq!(
                        Falcon18205::limbs(lo, hi),
                        rows(lo, hi),
                        "row {row:?} d {d}"
                    );
                }
            }
        }
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

    /// `scan_top` (two counts) gives the count and the tie of the direct scan (an OR of
    /// equalities), for top bits at, next to and between all entries, at both ends of their
    /// range, and at random values of every magnitude.
    #[test]
    fn tie_counts_equal_scan() {
        let mut hs: Vec<u64> = vec![0, 1, 2, (1 << 63) - 1, (1 << 63) - 2, 1 << 62];
        for &t in BASE_HI63.iter() {
            hs.extend([t, t - 1, t + 1, t / 2, (t + t / 3).min((1 << 63) - 1)]);
        }
        for w in BASE_HI63.windows(2) {
            hs.push(w[1] + (w[0] - w[1]) / 2);
        }
        let mut rng = Shake256Prng::new(b"tie-lookup");
        for _ in 0..100_000 {
            let e = rng.next_u8() % 63;
            hs.push((rng.next_u64() >> 1) >> e);
        }
        for &h in &hs {
            assert!(h < 1 << 63);
            let mut z = 0i32;
            let mut eq = (h == 0) & BASE_HAS_ZERO_TOP;
            for &t in BASE_HI63.iter() {
                z += (h < t) as i32;
                eq |= h == t;
            }
            assert_eq!(scan_top::<true>(h), (z, eq), "h {h:#x}");
            assert_eq!(scan_top::<false>(h), (z, eq), "h {h:#x}");
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

    /// The width-3.1 draw, with both forms of the scan.
    fn draw31(p: &mut Bytes, wide: bool) -> (i32, i32) {
        if wide {
            HalfGauss31::draw::<_, true>(p)
        } else {
            HalfGauss31::draw::<_, false>(p)
        }
    }

    /// The width-3.1 draw at the extremes: V = 0 gives the largest z0 (41) through the
    /// fallback (H = 0 ties with the entries whose top bits are 0); V = max gives 0 directly.
    #[test]
    fn half_gauss_extremes() {
        for wide in [false, true] {
            let mut p = Bytes(draw_bytes(0, 1, 0, 0, 0), 0);
            assert_eq!(draw31(&mut p, wide), (41, 1));
            assert_eq!(p.1, 25);
            let mut p = Bytes(draw_bytes(u64::MAX >> 1, 0, 0, 0, 0), 0);
            assert_eq!(draw31(&mut p, wide), (0, 0));
            assert_eq!(p.1, 8);
        }
    }

    /// The lazy draw equals the full 192-bit comparison, in particular on the fallback path: H
    /// equal to the top bits of every entry, with low bits just below, at and above the entry's;
    /// and H = 0 against the small entries. Both forms of the scan.
    #[test]
    fn half_gauss_lazy_equals_full() {
        for wide in [false, true] {
            lazy_equals_full(wide);
        }
    }

    fn lazy_equals_full(wide: bool) {
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
                assert_eq!(draw31(&mut p, wide), (z_full(v), 1));
                assert_eq!(p.1, 25);
            }
        }
        for _ in 0..100_000 {
            let w = rng.next_u64();
            let (m, l, last) = (rng.next_u64(), rng.next_u64(), rng.next_u8() & 1);
            let mut p = Bytes(draw_bytes(w >> 1, w & 1, m, l, last), 0);
            let (z, b) = draw31(&mut p, wide);
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
            let (z, b) = HalfGauss31::draw::<_, NATIVE_ROUND>(&mut prng);
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
            let (z, b) = Falcon18205::draw::<_, NATIVE_ROUND>(&mut prng);
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

    /// The exact threshold `z` of `ber_exp` (for building byte strings that reach deep bytes).
    fn threshold(x: f64, ccs: f64) -> u64 {
        let s = (x * INV_LOG2) as i64;
        let r = x - (s as f64) * LOG2;
        let keep = (s.wrapping_sub(64) >> 63) as u64;
        ((expm_p63(r, ccs) << 1).wrapping_sub(1) >> ((s as u32) & 63)) & keep
    }

    /// The decision of `first_byte_decision` is the window's: `w < m` accepts and `w > m + 1`
    /// rejects, for `m = clamp(floor(f - 1/32), 0, 254)` computed as fn-dsa-style integer code
    /// would (the form that it replaces), for every byte and for `f` at, just below and just
    /// above every integer and every integer plus 1/32 from 0 to 258, at 0, 1/32 and the
    /// smallest positive value, and at random values in [0, 260).
    #[test]
    fn first_byte_decision_equals_window() {
        let mut fs: Vec<f64> = vec![0.0, -0.0, 5e-324, DELTA, DELTA.next_down(), DELTA.next_up()];
        for k in 0..=258i32 {
            for v in [k as f64, k as f64 + DELTA, k as f64 - DELTA] {
                fs.extend([v, v.next_down(), v.next_up(), v - 1e-9, v + 1e-9, v + 0.5]);
            }
        }
        let mut rng = Shake256Prng::new(b"first-byte-decision");
        for _ in 0..20_000 {
            let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            fs.push(u * 260.0);
        }
        for &f in &fs {
            if f.is_nan() || f < 0.0 {
                continue;
            }
            let fl = f - DELTA;
            let m = if fl < 0.0 { 0 } else { (fl as u32).min(254) };
            for w in 0..=255u8 {
                let want = ((w as u32) < m, (w as u32) > m + 1);
                assert_eq!(first_byte_decision(f, w), want, "f {f:e} w {w}");
            }
        }
    }

    /// `centre::<true>` (one rounding) and `centre::<false>` (fn-dsa's conversion and back) give
    /// the same integer and offset, and the same differences `z - r` for the proposals `z` of the
    /// sampler: around every integer, at the 2^62 switch, at the edges of `i64` and beyond, and
    /// on random centres of every magnitude. The offsets differ only in the sign of a zero (for
    /// `mu = -0.0`), which `z - r` does not see.
    #[test]
    fn centre_forms_agree() {
        let mut mus: Vec<f64> = vec![
            0.0,
            -0.0,
            0.5,
            -0.5,
            1e-300,
            -1e-300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1e300,
            -1e300,
        ];
        for k in [1.0f64, 2.0, 1e3, 4_611_686_018_427_387_904.0, 9.3e18, 1e19] {
            for v in [k, -k] {
                mus.extend([v, v.next_down(), v.next_up(), v + 0.5, v - 0.5]);
            }
        }
        let mut rng = Shake256Prng::new(b"centre-forms");
        for _ in 0..100_000 {
            let e = (rng.next_u8() % 70) as i32 - 10;
            let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            mus.push(u * 2f64.powi(e));
        }
        for &mu in &mus {
            let (sa, ra) = centre::<true>(mu);
            let (sb, rb) = centre::<false>(mu);
            assert_eq!(sa, sb, "mu {mu:e}");
            assert!(
                ra == rb || (ra.is_nan() && rb.is_nan()),
                "mu {mu:e}: {ra} {rb}"
            );
            if ra.to_bits() != rb.to_bits() && !ra.is_nan() {
                assert_eq!((ra, mu.to_bits()), (0.0, (-0.0f64).to_bits()), "mu {mu:e}");
            }
            for z in -45..=45i32 {
                let (ta, tb) = (z as f64 - ra, z as f64 - rb);
                assert!(ta.to_bits() == tb.to_bits() || (ta.is_nan() && tb.is_nan()));
            }
        }
    }

    /// `ber_exp_shortcut` returns what `ber_exp` returns and reads the same bytes: random inputs
    /// over the sampler's range, the boundaries of the lemma (x at and next to every k ln 2,
    /// s = 7 / 8 and 63 / 64, ccs at 1/2 and 1, the clamp edges of the window), and byte strings
    /// aimed at the window (the first byte equal to the threshold's top byte or next to it) and
    /// at the deeper bytes. Negative, NaN and infinite x go to `ber_exp` by construction.
    #[test]
    fn ber_exp_shortcut_equals_ber_exp() {
        let mut rng = Shake256Prng::new(b"ber-shortcut");
        let mut xs: Vec<f64> = Vec::new();
        for k in 0..=70u32 {
            let base = k as f64 * LOG2;
            let mut v = base;
            for _ in 0..4 {
                xs.push(v);
                v = v.next_down();
            }
            let mut v = base;
            for _ in 0..4 {
                v = v.next_up();
                xs.push(v);
            }
        }
        xs.extend([0.0, -0.0, 5e-324, f64::INFINITY, -1.0, f64::NAN, 1e300]);
        for _ in 0..20_000 {
            let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            xs.push(u * 12.0);
            xs.push(u * 100.0);
            xs.push(u * 1e-6);
        }
        let mut ccss = vec![
            0.5,
            0.5f64.next_up(),
            1.0,
            1.0f64.next_down(),
            0.7152373,
            0.7131449,
        ];
        for _ in 0..8 {
            ccss.push(0.5 + 0.5 * (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64);
        }
        let mut cases = 0u64;
        for &x in &xs {
            for &ccs in &ccss {
                let z = if x >= 0.0 { threshold(x, ccs) } else { 0 };
                let top = (z >> 56) as u8;
                let mut firsts = vec![
                    top,
                    top.wrapping_add(1),
                    top.wrapping_sub(1),
                    0,
                    1,
                    254,
                    255,
                ];
                firsts.push(rng.next_u8());
                for &w0 in &firsts {
                    // Bytes equal to the threshold's after the first, then random: the deep path.
                    for deep in [false, true] {
                        let mut bytes = vec![w0];
                        for i in 1..8 {
                            bytes.push(if deep {
                                (z >> (56 - 8 * i)) as u8
                            } else {
                                rng.next_u8()
                            });
                        }
                        bytes.extend((0..8).map(|_| rng.next_u8()));
                        let mut a = Bytes(bytes.clone(), 0);
                        let mut b = Bytes(bytes.clone(), 0);
                        let mut c = Bytes(bytes, 0);
                        let ra = ber_exp(&mut a, x, ccs);
                        let rb = ber_exp_shortcut::<_, false>(&mut b, x, ccs);
                        let rc = ber_exp_shortcut::<_, true>(&mut c, x, ccs);
                        assert_eq!(
                            ((ra, a.1), (ra, a.1)),
                            ((rb, b.1), (rc, c.1)),
                            "x {x:e} ({:#x}) ccs {ccs} w0 {w0}",
                            x.to_bits()
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert!(cases > 2_000_000, "{cases}");
    }

    /// A leaf sampler with the shortcut reads the same bytes and returns the same integers as
    /// one with the exact step, over long streams at both bases (the known-answer tests check
    /// the same through SampPre).
    #[test]
    fn sampler_with_shortcut_equals_exact() {
        struct Exact<'a>(&'a mut Shake256Prng, Params);
        for params in [Params::PCS, Params::FALCON_1024] {
            let (lo, hi) = params.leaf_window();
            let mut a = Shake256Prng::new(b"leaf-stream");
            let mut b = Shake256Prng::new(b"leaf-stream");
            let mut pick = Shake256Prng::new(b"leaf-centres");
            let mut sa = SamplerZ::new(&params, &mut a);
            let mut exact = Exact(&mut b, params);
            for i in 0..200_000u32 {
                let u = (pick.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                let mu = (u - 0.5) * 4000.0;
                let sigma = lo + (hi - lo) * ((i % 97) as f64 / 96.0);
                let isigma = 1.0 / sigma;
                let got = sa.sample(mu, isigma);
                let want = exact_next(&mut exact, mu, isigma);
                assert_eq!(got, want, "params {:?} leaf {i}", params.base);
            }
            fn exact_next(e: &mut Exact<'_>, mu: f64, isigma: f64) -> i32 {
                let s = mu.floor() as i64;
                let r = mu - s as f64;
                let dss = (isigma * isigma) * 0.5;
                let ccs = isigma * e.1.sigma_min;
                loop {
                    let (z0, b, inv) = match e.1.base {
                        BaseSampler::HalfGauss3_1 => {
                            let (z0, b) = HalfGauss31::draw::<_, false>(e.0);
                            (z0, b, HalfGauss31::INV_2SQRSIGMA0)
                        }
                        BaseSampler::Falcon1_8205 => {
                            let (z0, b) = Falcon18205::draw::<_, false>(e.0);
                            (z0, b, Falcon18205::INV_2SQRSIGMA0)
                        }
                    };
                    let z = b + ((b << 1) - 1) * z0;
                    let t = z as f64 - r;
                    let mut x = (t * t) * dss;
                    x -= ((z0 * z0) as f64) * inv;
                    if ber_exp(e.0, x, ccs) {
                        return s as i32 + z;
                    }
                }
            }
            assert_eq!(a.next_u64(), b.next_u64(), "stream positions");
        }
    }
}
