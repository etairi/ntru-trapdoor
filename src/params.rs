//! Parameter sets.
//!
//! Shared by the signing and key-generation sides.
//!
//! # Conventions
//!
//! Two Gaussian conventions meet in this crate, and every width is labelled with its own:
//!
//! - The GPV **s-parameter** (DKLW25 p. 5, the paper's $`\varsigma`$):
//!   $`\rho_s(x)=\exp(-\pi\|x\|^2/s^2)`$. The smoothing bound
//!   $`\eta_\varepsilon(\mathbb Z^m)\le\sqrt{\ln(2m(1+1/\varepsilon))/\pi}`$ is in this
//!   convention.
//! - The **standard-deviation parameter** of Falcon and fn-dsa:
//!   $`\rho(x)=\exp(-x^2/(2\sigma^2))`$, so $`\sigma=s/\sqrt{2\pi}`$. Every `f64` width in
//!   [`Params`] and every table in [`crate::tables`] uses this one.
//!
//! With $`d=2^{\mathrm{logn}}`$ and the smoothing parameter $`\varepsilon`$:
//! $`\sigma_{\min}=\eta_\varepsilon(\mathbb Z^{2d})/\sqrt{2\pi}`$,
//! $`\sigma=1.17\sqrt q\,\sigma_{\min}`$, $`\varsigma=\sigma\sqrt{2\pi}`$,
//! $`\sigma_{fg}=1.17\sqrt{q/(2d)}`$.
//!
//! **Leaf widths.** The ffSampling leaf widths are $`\sigma'_i=\sigma/\sqrt{d_i}`$, where the
//! $`d_i`$ are the leaf values of the ffLDL tree (the squared Gram–Schmidt norms in the order
//! the sampler uses). Every $`d_i`$ lies in $`[q^2/M,M]`$ with $`M=\max(N_1,N_2)`$, Falcon's two
//! squared Gram–Schmidt norms, and both ends are attained [proved, and checked on 600 keys]: each split of the tree
//! produces a child whose values are arithmetic means and a child whose values are harmonic
//! means of pairs of its parent's values, and at the root $`d_{00}d_{11}=q^2`$ pointwise. So a
//! key that passes both norm tests ($`M\le(1.17\sqrt q)^2`$) has every
//! $`\sigma'_i\in[\sigma/(1.17\sqrt q),\,1.17\sigma/\sqrt q]=[\sigma_{\min},1.17^2\sigma_{\min}]`$
//! ([`Params::check_leaf_window`]).
//!
//! The `f64` constants below are the nearest binary64 values of the exact quantities, computed
//! at 256 bits by `tools/constants.py`; for
//! [`Params::FALCON_1024`] they equal fn-dsa's constants bit for bit (same script).

use crate::{Error, tables};

/// The base half-Gaussian of `SamplerZ`, the bimodal proposal of Falcon's leaf sampler.
///
/// SamplerZ is exact only for leaf widths $`\sigma'\le\sigma_0`$ (otherwise the Bernoulli
/// exponent goes negative) and needs $`\sigma'\ge\sigma_{\min}`$ (so that
/// $`\sigma_{\min}/\sigma'\le1`$); [`crate::ExpandedKey::new`] enforces both on every leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BaseSampler {
    /// Width $`\sigma_0=3.1`$ ($`\Pr[K=i]\propto e^{-50i^2/961}`$): the 41 rows of the reverse
    /// CDT, each rounded at 192 bits ([`tables::BASE_3_1_192`]), drawn lazily: 8 PRNG bytes per
    /// draw (one `u64` holding the sign bit and the top 63 bits of the 192-bit uniform value),
    /// and 25 in the rare case, of probability at most $`29\cdot2^{-63}=2^{-58.1}`$, where the
    /// top bits tie with a table entry. The distribution is exactly that of the table. Covers
    /// the PCS leaves, $`[2.21724, 3.03517]`$.
    HalfGauss3_1,
    /// Width $`\sigma_0=1.8205`$: fn-dsa's 79-bit `GAUSS0` ([`tables::GAUSS0_1_8205_79`],
    /// 18 rows), consumed exactly as fn-dsa does (10 PRNG bytes per draw), for bit-exact parity
    /// at Falcon's parameters. Covers Falcon-1024's leaves, $`[1.29828, 1.77722]`$.
    Falcon1_8205,
}

impl BaseSampler {
    /// The width $`\sigma_0`$ (standard-deviation convention).
    pub const fn sigma0(self) -> f64 {
        match self {
            BaseSampler::HalfGauss3_1 => 3.1,
            BaseSampler::Falcon1_8205 => 1.8205,
        }
    }

    /// $`1/(2\sigma_0^2)`$, nearest binary64: 50/961 for width 3.1; fn-dsa's
    /// `INV_2SQRSIGMA0` (sampler.rs:30) for width 1.8205.
    pub const fn inv_2sqr_sigma0(self) -> f64 {
        match self {
            // 0.052029136316337148803 = 50/961
            BaseSampler::HalfGauss3_1 => f64::from_bits(0x3faa_a390_1dd5_e917),
            // 0.15086504887537272153 = 2000000/13256881 (fn-dsa FLR::scaled(5435486223186882, -55))
            BaseSampler::Falcon1_8205 => f64::from_bits(0x3fc3_4f8b_c183_bbc2),
        }
    }

    /// PRNG bytes consumed per base draw (the sign bit included), in the typical case: 8 for
    /// the width-3.1 base (25 with probability at most $`2^{-58.1}`$), 10 for fn-dsa's.
    pub const fn bytes_per_draw(self) -> usize {
        match self {
            BaseSampler::HalfGauss3_1 => 8,
            BaseSampler::Falcon1_8205 => 10,
        }
    }
}

/// A parameter set: ring degree, modulus, Gaussian widths, bounds and encoding widths.
///
/// Generic over $`(d, q,\text{widths})`$; the presets are [`Params::PCS`] (the main instance)
/// and [`Params::FALCON_1024`] (fn-dsa's constants, for the bit-exact parity tests). Custom sets
/// must pass [`Params::validate`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// A human-readable name.
    pub name: &'static str,
    /// $`\log_2 d`$, in `1..=10`; the ring is $`\mathbb Z_q[x]/(x^d+1)`$.
    pub logn: u32,
    /// The prime modulus $`q`$, $`3\le q<2^{25}`$ (the two-prime CRT of the ring
    /// multiplication is exact up to $`q\approx2^{25.5}`$).
    pub q: u32,
    /// $`\sigma=\varsigma/\sqrt{2\pi}`$: the standard deviation of each coefficient of SampPre's
    /// output (documentation and statistical tests; the sampler uses `inv_sigma`).
    pub sigma: f64,
    /// $`1/\sigma`$ as the sampler uses it (fn-dsa's `INV_SIGMA[logn]`); leaf $`i`$ of the tree
    /// stores $`1/\sigma'_i=\sqrt{d_i}\cdot\texttt{inv\_sigma}`$.
    pub inv_sigma: f64,
    /// $`\sigma_{\min}=\eta_\varepsilon(\mathbb Z^{2d})/\sqrt{2\pi}`$ (fn-dsa's
    /// `SIGMA_MIN[logn]`): the smallest admissible leaf width, and the numerator of the
    /// Bernoulli scaling $`\sigma_{\min}/\sigma'`$.
    pub sigma_min: f64,
    /// The base half-Gaussian of SamplerZ.
    pub base: BaseSampler,
    /// $`1/q`$, nearest binary64 (fn-dsa's `INV_Q`), for the target transform.
    pub inv_q: f64,
    /// $`\lfloor\beta^2\rfloor`$: SampPre resamples above it, and it is the bound that
    /// [`crate::verify`] is normally called with.
    pub beta_sq: u64,
    /// $`\sigma_{fg}=1.17\sqrt{q/(2d)}`$, the width of $`f,g`$ (documentation and tests; the
    /// sampler uses `fg_table`).
    pub sigma_fg: f64,
    /// The 128-bit half-Gaussian reverse CDT of width $`\sigma_{fg}`$ (see [`crate::tables`]).
    pub fg_table: &'static [u128],
    /// Numerator of the Gram–Schmidt bound $`(1.17\sqrt q)^2=13689\,q/10000`$; a custom set
    /// may make the bound smaller, never larger ([`Params::validate`]).
    pub gs_bound_num: u64,
    /// Denominator of the Gram–Schmidt bound (10000).
    pub gs_bound_den: u64,
    /// Width in bits (two's complement) of each coefficient of $`f,g`$ in the secret-key
    /// encoding; key generation rejects keys that do not fit.
    pub sk_fg_bits: u32,
    /// Width in bits of each coefficient of $`F,G`$ in the secret-key encoding.
    pub sk_big_fg_bits: u32,
    /// Attempts of SampPre before [`Error::SampPreExhausted`] (fn-dsa: 27), in `1..=256` (the
    /// attempt counter of the seed derivation is one byte).
    pub samppre_max_attempts: u32,
    /// Candidate pairs $`(f,g)`$ of key generation before [`Error::KeygenExhausted`].
    pub keygen_max_attempts: u32,
}

impl Params {
    /// The PCS set (Σ-vSIS, building-blocks.tex sec:inst-base-vsis; parameter sheet §2):
    /// $`d=1024`$, $`q=1048573=2^{20}-3`$ (prime, $`\equiv5\bmod8`$, so $`x^{1024}+1`$ has two
    /// irreducible factors of degree 512), $`\varepsilon=2^{-128}`$,
    /// $`\varsigma=6658.662596`$, $`\sigma=2656.422041`$, $`\sigma_{\min}=2.217236`$, leaves in
    /// $`[2.21724, 3.03517]`$ under the base width 3.1,
    /// $`\lfloor\beta_s^2\rfloor=\lfloor\varsigma^2\cdot2d\rfloor=90{,}803{,}788{,}942`$,
    /// $`\sigma_{fg}=26.474040`$.
    pub const PCS: Params = Params {
        name: "PCS-1024 (q = 1048573)",
        logn: 10,
        q: 1_048_573,
        // 2656.4220405436989764
        sigma: f64::from_bits(0x40a4_c0d8_15b2_b98c),
        // 0.00037644620649033862965
        inv_sigma: f64::from_bits(0x3f38_abb8_2544_aed0),
        // 2.2172357777394359348
        sigma_min: f64::from_bits(0x4001_bce6_1c87_4bc8),
        base: BaseSampler::HalfGauss3_1,
        // 9.5367704489816159676e-7
        inv_q: f64::from_bits(0x3eb0_0003_0000_9000),
        // floor(90803788942.8769) = floor(varsigma^2 * 2d)
        beta_sq: 90_803_788_942,
        // 26.474040016125053538
        sigma_fg: f64::from_bits(0x403a_795a_afbe_409e),
        fg_table: &tables::FG_PCS_128,
        // 13689 * 1048573 / 10000 = 1435391.5797
        gs_bound_num: 14_353_915_797,
        gs_bound_den: 10_000,
        // |f_i|, |g_i| <= 348 (the table length) < 2^9
        sk_fg_bits: 10,
        // |F_i|, |G_i| < 2^12; the Sage oracle measured at most 1263 over 300 keys
        sk_big_fg_bits: 13,
        samppre_max_attempts: 27,
        keygen_max_attempts: 4096,
    };

    /// Falcon-1024's signing parameters, with fn-dsa's constants bit for bit
    /// (fn-dsa-sign/src/sampler.rs:59, :72, :30; lib.rs:480; fn-dsa-comm/src/mq.rs:83):
    /// $`d=1024`$, $`q=12289`$, $`\varepsilon=1/\sqrt{256\cdot2^{64}}=2^{-36}`$,
    /// $`\sigma=168.388571`$, $`\sigma_{\min}=1.298280`$, base width 1.8205, and Falcon's
    /// tail-cut $`\lfloor\beta^2\rfloor=\lfloor(1.1\sigma\sqrt{2d})^2\rfloor=70{,}265{,}242`$.
    /// Key generation uses this crate's own sampler and solver (not fn-dsa's).
    pub const FALCON_1024: Params = Params {
        name: "Falcon-1024 (q = 12289)",
        logn: 10,
        q: 12_289,
        // 168.38857144654393943
        sigma: f64::from_bits(0x4065_0c6f_2d62_e21a),
        // fn-dsa INV_SIGMA[10] = FLR::scaled(6846791885593314, -60)
        inv_sigma: f64::from_bits(0x3f78_531e_f631_1ae2),
        // fn-dsa SIGMA_MIN[10] = FLR::scaled(5846934829975396, -52)
        sigma_min: f64::from_bits(0x3ff4_c5c1_9990_c764),
        base: BaseSampler::Falcon1_8205,
        // fn-dsa INV_Q = FLR::scaled(6004310871091074, -66)
        inv_q: f64::from_bits(0x3f15_54e3_9097_a782),
        // fn-dsa SQBETA[10]
        beta_sq: 70_265_242,
        // 2.8660196105754623744
        sigma_fg: f64::from_bits(0x4006_ed9b_b088_ee1d),
        fg_table: &tables::FG_FALCON1024_128,
        // 13689 * 12289 / 10000 = 16822.4121
        gs_bound_num: 168_224_121,
        gs_bound_den: 10_000,
        // |f_i|, |g_i| <= 38 (the table length) < 2^6
        sk_fg_bits: 7,
        // |F_i|, |G_i| < 2^8; the Sage oracle measured at most 126 over 300 keys
        sk_big_fg_bits: 9,
        samppre_max_attempts: 27,
        keygen_max_attempts: 4096,
    };

    /// The ring degree $`d=2^{\mathrm{logn}}`$.
    pub const fn n(&self) -> usize {
        1usize << self.logn
    }

    /// Bits per coefficient of the canonical $`R_q`$ encoding: the bit length of $`q-1`$
    /// (20 at the PCS set, 14 at Falcon-1024).
    pub const fn q_bits(&self) -> u32 {
        u32::BITS - (self.q - 1).leading_zeros()
    }

    /// Bytes of the canonical encoding of an element of $`R_q`$:
    /// $`\lceil d\cdot\texttt{q\_bits}/8\rceil`$ (2560 at the PCS set).
    pub const fn rq_bytes(&self) -> usize {
        (self.n() * self.q_bits() as usize).div_ceil(8)
    }

    /// The Gram–Schmidt bound $`(1.17\sqrt q)^2`$ as the nearest binary64.
    pub fn gs_bound(&self) -> f64 {
        self.gs_bound_num as f64 / self.gs_bound_den as f64
    }

    /// The ring checks that every [`crate::RqPoly`] constructor runs (O(1)): $`1\le\mathrm{logn}\le10`$
    /// (the NTT tables) and $`3\le q<2^{25}`$ (the two-prime CRT of the multiplication is exact
    /// below $`q\approx2^{25.5}`$). Primality is not needed for the ring operations.
    pub(crate) fn check_ring(&self) -> Result<(), Error> {
        if !(1..=10).contains(&self.logn) {
            return Err(Error::InvalidParams("logn must lie in 1..=10"));
        }
        if self.q < 3 || self.q >= 1 << 25 {
            return Err(Error::InvalidParams("q must lie in [3, 2^25)"));
        }
        Ok(())
    }

    /// Structural checks of a parameter set, and one security floor: the Gram–Schmidt bound must
    /// be at most Falcon's $`(1.17\sqrt q)^2=13689\,q/10000`$ (exact integer comparison). Under
    /// that bound, every key that key generation accepts is a trapdoor basis with Gram–Schmidt
    /// norms at most $`1.17\sqrt q`$ (up to the `f64` evaluation of the second norm, relative
    /// error about $`2^{-43}`$), so every rank-$`d`$ sublattice of the lattice
    /// $`\{(s_0,s_1):s_0+hs_1=0\bmod q\}`$ has volume at least $`(\sqrt q/1.17)^d`$: none is
    /// denser, per dimension, than the lattice itself by more than a factor 1.17, unlike the
    /// sublattice $`R\cdot(g,-f)`$ that overstretched NTRU attacks exploit (docs/security.md, "Key
    /// distribution"). Tighter bounds are accepted. Key generation also requires
    /// [`Params::check_leaf_window`].
    pub fn validate(&self) -> Result<(), Error> {
        self.check_ring()?;
        if !is_prime_u32(self.q) {
            return Err(Error::InvalidParams("q must be a prime in [3, 2^25)"));
        }
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if !positive(self.sigma) || !positive(self.inv_sigma) || !positive(self.sigma_min) {
            return Err(Error::InvalidParams(
                "sigma, inv_sigma and sigma_min must be positive",
            ));
        }
        if (self.sigma * self.inv_sigma - 1.0).abs() > 1e-12 {
            return Err(Error::InvalidParams("inv_sigma must be 1/sigma"));
        }
        if !positive(self.inv_q) || (self.inv_q * self.q as f64 - 1.0).abs() > 1e-12 {
            return Err(Error::InvalidParams("inv_q must be 1/q"));
        }
        if self.sigma_min >= self.base.sigma0() {
            return Err(Error::InvalidParams(
                "sigma_min must be below the base width sigma0",
            ));
        }
        if self.beta_sq == 0 {
            return Err(Error::InvalidParams("beta_sq must be positive"));
        }
        // A reverse CDT is non-increasing, and equal neighbours are legitimate: the value
        // between them has probability 0 after the 128-bit rounding (FG_PCS_128 ends
        // `.., 2, 1, 1`). A zero first entry would make every draw 0.
        if self.fg_table.first().is_none_or(|&t0| t0 == 0)
            || self.fg_table.windows(2).any(|w| w[0] < w[1])
        {
            return Err(Error::InvalidParams(
                "fg_table must be non-empty, non-increasing and start above 0",
            ));
        }
        if !positive(self.sigma_fg) || self.gs_bound_num == 0 || self.gs_bound_den == 0 {
            return Err(Error::InvalidParams(
                "sigma_fg and the GS bound must be positive",
            ));
        }
        // gs_bound_num / gs_bound_den <= 13689 q / 10000, in u128 (both sides below 2^103).
        if u128::from(self.gs_bound_num) * 10_000
            > 13_689 * u128::from(self.q) * u128::from(self.gs_bound_den)
        {
            return Err(Error::InvalidParams(
                "the GS bound must be at most (1.17 sqrt q)^2 = 13689 q / 10000",
            ));
        }
        if !(2..=16).contains(&self.sk_fg_bits) || !(2..=16).contains(&self.sk_big_fg_bits) {
            return Err(Error::InvalidParams(
                "secret-key widths must lie in 2..=16 bits",
            ));
        }
        if (self.fg_table.len() as u64) >= 1u64 << (self.sk_fg_bits - 1) {
            return Err(Error::InvalidParams(
                "sk_fg_bits cannot hold every value of fg_table",
            ));
        }
        if self.samppre_max_attempts == 0 || self.keygen_max_attempts == 0 {
            return Err(Error::InvalidParams("attempt caps must be positive"));
        }
        if self.samppre_max_attempts > 256 {
            return Err(Error::InvalidParams(
                "samppre_max_attempts must be at most 256 (one-byte attempt counter)",
            ));
        }
        Ok(())
    }

    /// The window of leaf widths that the keys accepted by key generation can have:
    /// $`[\sigma/\sqrt B,\ \sigma\sqrt B/q]`$ with $`B`$ = [`Params::gs_bound`]. A key that
    /// passes both of Falcon's Gram–Schmidt norm tests has every leaf width in this window
    /// (module documentation, "Leaf widths").
    pub fn leaf_window(&self) -> (f64, f64) {
        let r = self.gs_bound().sqrt();
        (self.sigma / r, self.sigma * r / self.q as f64)
    }

    /// Checks that [`Params::leaf_window`] lies inside the base sampler's range
    /// $`[\sigma_{\min},\sigma_0)`$, up to the rounding of the constants: then every key that
    /// passes Falcon's two norm tests also passes the leaf check of
    /// [`crate::ExpandedKey::new`] (except through `f64` rounding at the very boundary), so key
    /// generation never rejects a key for its leaves. [`crate::trapgen`] requires it; without it
    /// a custom set can make key generation reject most or all candidates. The presets give
    /// $`[2.21724,3.03517]\subset[2.21724,3.1)`$ (PCS; the lower ends agree, since
    /// $`\sigma=1.17\sqrt q\,\sigma_{\min}`$) and $`[1.29828,1.77722]\subset[1.29828,1.8205)`$
    /// (Falcon-1024). Errors: [`Error::InvalidParams`].
    pub fn check_leaf_window(&self) -> Result<(), Error> {
        let (lo, hi) = self.leaf_window();
        // Relative slack for the binary64 rounding of sigma, sigma_min and the bound (each
        // within 2^-52), far below 2^-40.
        let slack = 1.0 / (1u64 << 40) as f64;
        if lo.is_nan() || lo < self.sigma_min * (1.0 - slack) {
            return Err(Error::InvalidParams(
                "leaf window: sigma / sqrt(gs_bound) is below sigma_min",
            ));
        }
        // The leaf check needs sigma' <= sigma0 / sqrt(1 + 2^-40); keep a margin of 2^-38.
        if hi.is_nan() || hi > self.base.sigma0() * (1.0 - 4.0 * slack) {
            return Err(Error::InvalidParams(
                "leaf window: sigma sqrt(gs_bound) / q reaches the base width sigma0",
            ));
        }
        Ok(())
    }
}

/// Trial division; `n < 2^32`.
fn is_prime_u32(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2u32;
    while (d as u64) * (d as u64) <= n as u64 {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 1;
    }
    true
}
