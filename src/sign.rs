//! SampPre and Verify.
//!
//! SampPre follows fn-dsa-sign `sign_inner` (lib.rs:482-744; Pornin, The
//! Unlicense, commit 0629bb1) with the target given instead of hashed, and the Gram matrix and
//! LDL tree taken from the [`ExpandedKey`]:
//! 1. target in FFT form: `t0 = FFT(t) * b11 * inv_q`, `t1 = FFT(t) * b01 * (-inv_q)`
//!    (lib.rs:621-640), i.e. $`(t,0)\,B^{-1}=(-tF/q,\,tf/q)`$;
//! 2. ffSampling over the tree (`ffsamp`) with [`crate::hazmat::SamplerZ`];
//! 3. lattice point `v = z B` in FFT form, iFFT (lib.rs:673-689);
//! 4. `s0 = t - rint(v0)`, `s1 = -rint(v1)` (lib.rs:691-722), $`\|s\|^2`$ in `u64`; accept iff
//!    $`\|s\|^2\le\lfloor\beta^2\rfloor`$ (`params.beta_sq`), else the next attempt.
//!
//! Since $`z`$ is integral, $`v=zB`$ is a lattice point and $`A\,s=t-A\,v\equiv t\pmod q`$ exactly,
//! provided the rounding of step 4 recovers $`v`$ exactly. The `f64` lattice point is within
//! $`2^{-26.9}`$ of the exact integer vector at the PCS set ($`2^{-33.1}`$ at Falcon-1024,
//! $`2^{-22.2}`$ at $`q\approx2^{25}`$; 100 calls per set [measured]), so the
//! rounding has a margin of about $`2^{26}`$; the stats stage found $`s`$ exact on all 1064 replayed
//! calls. A guard enforces a margin: an attempt whose lattice point lies farther than $`2^{-10}`$
//! from an integer in some coordinate (which a valid key never produces) is discarded like a
//! too-long one. The tests check $`A\,s=t`$ on every output.
//!
//! Seeds (design.md §5.1): `seed* = SHAKE256("ntru-trapdoor/v1/samp-pre" || key_id ||
//! encode(t) || seed)` (64 bytes), and attempt `k` reads `Shake256Prng::from_parts([seed*,
//! [k]])`; a repeated seed with another target or key gives unrelated randomness (hedging, as
//! fn-dsa's lib.rs:285-295, with the target in place of the message representative).
//!
//! **Versioning of the coins** (LTYZ25, cited by the paper). The same coins on the same target
//! under slightly different floating-point arithmetic can give two different lattice points,
//! whose difference leaks the key. Within one version the map (key, target, seed) to $`s`$ is a
//! function (the same output for the same inputs), which is safe. `tests/kat.rs` pins it at both
//! presets: any change of the sampler's arithmetic, tables or stream use changes that known
//! answer, and must then come with a new label (`v2`, ...) so that no two versions ever use the
//! same coins.
//!
//! **Copies and batches (the `simd` feature).** On x86-64 processors with AVX2, BMI1, BMI2,
//! LZCNT and POPCNT, [`samp_pre_attempt`] runs a copy of these steps compiled for those
//! instructions (`samp_pre_attempt_v3`: the same source through the macros `samp_pre_with_fn`
//! and `ffsampling_fn`, the leaf sampler `SamplerZ::sample_v3`, and AVX2 kernels for the FFT
//! passes), with the same operations in the same order and without FMA. [`samp_pre_batch`] derives
//! the seeds of a group of calls together and squeezes the first blocks of their streams
//! together, with the multi-state Keccak kernels. Both give the outputs of the portable code
//! (tests `samp_pre_attempt_v3_equals_portable` and `samp_pre_batch_groups_equal_samp_pre`).
//!
//! Verify: $`\|s\|^2\le\beta^2`$ (exact, `u128` accumulation) and $`s_0+h\,s_1\equiv t`$ (one
//! product with the prepared $`h`$: four NTTs).
//!
//! Scratch buffers are allocated per call and wiped before they are released.

use crate::ffsamp::ffsampling;
use crate::fft::{fft, ifft, poly_lattice_point_fft, poly_target_fft};
use crate::prng::{Prng, Shake256Prng, shake256};
use crate::ring::ntt::{self, ModQ};
use crate::sampler::{LeafSampler, NATIVE_ROUND, SamplerZ, rint};
use crate::wipe::{self, Secret};
use crate::{Error, ExpandedKey, Preimage, PublicKey, RqPoly};

/// Domain-separation label of the per-call seed (design.md §5.1). Bump the version whenever
/// `tests/kat.rs` changes (module documentation, "Versioning of the coins").
const SAMP_PRE_LABEL: &[u8] = b"ntru-trapdoor/v1/samp-pre";

/// The rounding guard of step 4: the largest accepted distance between a coordinate of the
/// `f64` lattice point and its nearest integer.
const ROUND_GUARD: f64 = 1.0 / 1024.0;

/// SampPre: a preimage $`s=(s_0,s_1)`$ of `t` under $`A=[1\mid h]`$ with
/// $`\|s\|^2\le\lfloor\beta^2\rfloor`$, distributed (up to the sampler's precision) as the
/// discrete Gaussian of width $`\sigma`$ on the coset $`\{s: As=t\}`$.
///
/// `seed` must be fresh and uniform for every call (the PCS draws it from its RNG); a repeated
/// seed with a different `t` still gives independent randomness (hedging). A seed must never be
/// reused across versions of this crate, even for the same `t` (module documentation,
/// "Versioning of the coins").
///
/// Errors: [`Error::Mismatch`] if `t` is not in the key's ring; [`Error::SampPreExhausted`]
/// after `params.samppre_max_attempts` attempts (never observed at the PCS set: the
/// probability per attempt is below $`2^{-5000}`$).
pub fn samp_pre(ek: &ExpandedKey, t: &RqPoly, seed: &[u8; 32]) -> Result<Preimage, Error> {
    if !t.matches(&ek.params) {
        return Err(Error::Mismatch);
    }
    let mut seed_star = Secret::new([0u8; 64]);
    shake256(
        &[SAMP_PRE_LABEL, &ek.key_id, &t.to_bytes(), seed],
        &mut seed_star[..],
    );
    // The attempt counter is one byte (design.md §5.1): `Params::validate` caps the attempts
    // at 256 (and the `min` keeps the counter in range regardless).
    let attempts = ek.params.samppre_max_attempts.min(256);
    for k in 0..attempts {
        let mut prng = Shake256Prng::from_parts(&[&seed_star[..], &[k as u8]]);
        if let Some(s) = samp_pre_attempt(ek, t, &mut prng) {
            return Ok(s);
        }
    }
    Err(Error::SampPreExhausted)
}

/// [`samp_pre`] for many targets: entry `i` is `samp_pre(ek, items[i].0, items[i].1)`, the same
/// bytes, in the same order. Each call has its own seed and therefore its own coins, so the
/// calls are independent. With the `parallel` feature they run on rayon's thread pool,
/// otherwise one after the other.
///
/// With the `simd` feature, on processors with a multi-state Keccak kernel (8 states with
/// AVX-512, 4 with AVX2, 3 with the SHA3 extension of aarch64, 2 on Apple's processors), the
/// calls go in groups: their seed derivations run together, and so do the first blocks of their
/// PRNG streams (about as many blocks as an attempt reads, `presqueeze_blocks`), squeezed before
/// the sampling starts. Each call reads the bytes of its own stream, so the outputs are those of
/// `samp_pre`.
pub fn samp_pre_batch(
    ek: &ExpandedKey,
    items: &[(&RqPoly, &[u8; 32])],
) -> Vec<Result<Preimage, Error>> {
    let group = crate::prng::keccak_lanes();
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        // Groups no larger than the calls per thread, so that every thread has work: a power of
        // two from 1 to `per_thread` when that is below the kernel's width (which may be 3).
        let per_thread = (items.len() / rayon::current_num_threads()).max(1);
        let group = if per_thread >= group {
            group
        } else {
            1 << per_thread.ilog2()
        };
        items
            .par_chunks(group)
            .flat_map_iter(|g| samp_pre_group(ek, g))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        items
            .chunks(group)
            .flat_map(|g| samp_pre_group(ek, g))
            .collect()
    }
}

/// [`samp_pre`] on each call of a group, the seed derivations and the first blocks of the
/// streams of attempt 0 computed together ([`samp_pre_batch`]). A group of one, or one with a
/// target from another ring, runs call by call (and so would a parameter set without attempts,
/// which `Params::validate` refuses).
fn samp_pre_group(
    ek: &ExpandedKey,
    group: &[(&RqPoly, &[u8; 32])],
) -> Vec<Result<Preimage, Error>> {
    if group.len() < 2
        || ek.params.samppre_max_attempts == 0
        || group.iter().any(|(t, _)| !t.matches(&ek.params))
    {
        return group
            .iter()
            .map(|&(t, seed)| samp_pre(ek, t, seed))
            .collect();
    }
    // seed* of each call (samp_pre): the messages have the same length.
    let mut msgs: Vec<Vec<u8>> = group
        .iter()
        .map(|(t, seed)| [SAMP_PRE_LABEL, &ek.key_id, &t.to_bytes(), &seed[..]].concat())
        .collect();
    let refs: Vec<&[u8]> = msgs.iter().map(|m| m.as_slice()).collect();
    // The seeds go straight into a buffer wiped on drop.
    let mut seeds = Secret::new(vec![[0u8; 64]; group.len()]);
    crate::prng::shake256_parallel(&refs, &mut seeds);
    drop(refs);
    for m in msgs.iter_mut() {
        wipe::slice(m);
    }
    // The streams of attempt 0, `Shake256Prng::from_parts(&[seed*, &[0]])`.
    let mut inputs: Vec<[u8; 65]> = seeds
        .iter()
        .map(|s| core::array::from_fn(|i| if i < 64 { s[i] } else { 0 }))
        .collect();
    let refs: Vec<&[u8]> = inputs.iter().map(|m| m.as_slice()).collect();
    let prngs = crate::prng::squeeze_streams(&refs, presqueeze_blocks(&ek.params));
    drop(refs);
    for x in inputs.iter_mut() {
        wipe::slice(x);
    }
    group
        .iter()
        .zip(seeds.iter())
        .zip(prngs)
        .map(|((&(t, _), seed_star), mut prng)| {
            if let Some(s) = samp_pre_attempt(ek, t, &mut prng) {
                return Ok(s);
            }
            drop(prng);
            // Attempts 1 and on, as in samp_pre (never observed at `Params::PCS`).
            let attempts = ek.params.samppre_max_attempts.min(256);
            for k in 1..attempts {
                let mut prng = Shake256Prng::from_parts(&[&seed_star[..], &[k as u8]]);
                if let Some(s) = samp_pre_attempt(ek, t, &mut prng) {
                    return Ok(s);
                }
            }
            Err(Error::SampPreExhausted)
        })
        .collect()
}

/// The PRNG blocks to squeeze ahead for one attempt: about as many as an attempt reads, which
/// is $`2d\,(b+1)/(136\,a)`$ for a base draw of $`b`$ bytes and an acceptance rate $`a`$ per
/// trial (one Bernoulli byte per trial), plus one. At the presets, 215 and 285 against 214.3 and
/// 283.5 read on average (400 attempts each, standard deviations 2.8 and 3.9). Any value gives
/// the same outputs: blocks squeezed ahead and not read are dropped, and an attempt that reads
/// more squeezes the next ones itself. Not inlined: its rounding up takes a call into the C
/// library on x86-64 without SSE4.1, on public values, and `tools/sampler_branches.py` refuses
/// such calls in the functions of SampPre.
#[inline(never)]
fn presqueeze_blocks(params: &crate::Params) -> usize {
    let acceptance = match params.base {
        crate::params::BaseSampler::HalfGauss3_1 => 0.633687,
        crate::params::BaseSampler::Falcon1_8205 => 0.584958,
    };
    let bytes = (2 * params.n()) as f64 * (params.base.bytes_per_draw() + 1) as f64 / acceptance;
    (bytes / 136.0).ceil() as usize + 1
}

/// One SampPre attempt with an explicit PRNG: `None` if $`\|s\|^2>\lfloor\beta^2\rfloor`$ (or
/// if the rounding guard trips, which a valid key never does).
///
/// Consumes the PRNG exactly as one iteration of fn-dsa's `sign_inner` after the nonce
/// (lib.rs:509-744): this is the entry point of the fn-dsa parity tests. Panics if `t` is not
/// in the key's ring.
pub fn samp_pre_attempt<P: Prng>(ek: &ExpandedKey, t: &RqPoly, prng: &mut P) -> Option<Preimage> {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(s) = crate::simd::samp_pre_attempt_v3(ek, t, prng) {
        return s;
    }
    let mut samp = SamplerZ::new(&ek.params, prng);
    let s = samp_pre_with(ek, t, &mut samp)?;
    if s.norm_sq() <= ek.params.beta_sq {
        Some(s)
    } else {
        None
    }
}

/// Defines `samp_pre_with` (below) under the name `$name`, with the given FFT functions and
/// ffSampling: once portable and, with the `simd` feature on x86-64, once compiled for
/// x86-64-v3 with their copies.
macro_rules! samp_pre_with_fn {
    (
        $(#[$m:meta])*
        $vis:vis fn $name:ident;
        $fft:path, $target:path, $ffsampling:path, $point:path, $ifft:path, $round:expr
    ) => {
        $(#[$m])*
        $vis fn $name<S: LeafSampler>(
            ek: &ExpandedKey,
            t: &RqPoly,
            sampler: &mut S,
        ) -> Option<Preimage> {
            let params = &ek.params;
            assert!(t.matches(params), "samp_pre: target not in the key's ring");
            let logn = params.logn;
            let n = params.n();
            let (b00, b01, b10, b11) = ek.basis_parts();

            // Scratch: t0, t1, then 2n values for ffSampling.
            let mut buf = vec![0.0f64; 4 * n];
            let (t0, rest) = buf.split_at_mut(n);
            let (t1, tmp) = rest.split_at_mut(n);

            // Target (fn-dsa lib.rs:621-640): t0 = FFT(t) b11 / q, t1 = -FFT(t) b01 / q.
            for (x, &c) in t0.iter_mut().zip(t.c.iter()) {
                *x = c as f64;
            }
            $fft(logn, t0);
            $target(logn, t0, t1, b01, b11, params.inv_q);

            // Sampling (lib.rs:651).
            $ffsampling(sampler, logn, t0, t1, &ek.tree, tmp);

            // Lattice point v = z B (lib.rs:673-689): v0 = z0 b00 + z1 b10, v1 = z0 b01 + z1 b11.
            $point(logn, t0, t1, (b00, b01, b10, b11));
            $ifft(logn, t0);
            $ifft(logn, t1);

            // s0 = t - rint(v0), s1 = -rint(v1) (lib.rs:691-722). FLR::rint is round-half-even
            // (aarch64 vcvtnd, x86-64 cvtsd2si, flr_native.rs:373-435), and so is `rint`, which
            // takes no library call where rounding has no instruction (the lattice point depends
            // on the key and the samples). The casts saturate, and out-of-range values (which
            // cannot occur for a valid key) saturate in i32 so that the norm check rejects them.
            // The guard flags a coordinate farther than 2^-10 from an integer (or NaN).
            let sat = |x: i64| x.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            let mut off = false;
            let s0: Vec<i32> = t0
                .iter()
                .zip(t.c.iter())
                .map(|(&v, &c)| {
                    let r = rint::<{ $round }>(v);
                    let d = (v - r).abs();
                    off |= (d > ROUND_GUARD) | d.is_nan();
                    sat((c as i64).saturating_sub(r as i64))
                })
                .collect();
            let s1: Vec<i32> = t1
                .iter()
                .map(|&v| {
                    let r = rint::<{ $round }>(v);
                    let d = (v - r).abs();
                    off |= (d > ROUND_GUARD) | d.is_nan();
                    sat(0i64.saturating_sub(r as i64))
                })
                .collect();
            // A slice wipe (one volatile pass). `Vec::zeroize` would also wipe the capacity byte by
            // byte, about 9 us here [measured, sign/README.md].
            wipe::slice(&mut buf);
            let s = Preimage { s0, s1 };
            // On `None` the preimage is wiped when dropped.
            (!off).then_some(s)
        }
    };
}

samp_pre_with_fn! {
    /// ffSampling with any leaf sampler, without the norm check (for the precision tests, which
    /// replay recorded leaf decisions). `None` if the rounding guard trips: some coordinate of the
    /// `f64` lattice point lies farther than $`2^{-10}`$ from an integer (module documentation).
    /// Panics if `t` is not in the key's ring.
    pub fn samp_pre_with;
    fft, poly_target_fft, ffsampling, poly_lattice_point_fft, ifft, NATIVE_ROUND
}

samp_pre_with_fn! {
    /// [`samp_pre_with`] compiled for x86-64-v3 (AVX2, BMI1, BMI2, LZCNT, POPCNT, no FMA), with
    /// the copies of the FFT passes and of ffSampling.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
    fn samp_pre_with_v3;
    crate::fft::fft_v3,
    crate::fft::poly_target_fft_v3,
    crate::ffsamp::ffsampling_v3,
    crate::fft::poly_lattice_point_fft_v3,
    crate::fft::ifft_v3,
    true
}

/// [`samp_pre_attempt`] compiled for x86-64-v3: the leaf sampler ([`SamplerZ::sample_v3`]),
/// ffSampling and the FFT passes all run in their x86-64-v3 copies. `simd::samp_pre_attempt_v3`
/// calls it on processors that have the instructions.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
pub(crate) fn samp_pre_attempt_v3<P: Prng>(
    ek: &ExpandedKey,
    t: &RqPoly,
    prng: &mut P,
) -> Option<Preimage> {
    let mut samp = SamplerZ::new(&ek.params, prng);
    let mut leaf = crate::sampler::FnSampler(|mu, isigma| samp.sample_v3(mu, isigma));
    let s = samp_pre_with_v3(ek, t, &mut leaf)?;
    if s.norm_sq() <= ek.params.beta_sq {
        Some(s)
    } else {
        None
    }
}

/// Verify: `true` iff `s` has the key's degree, $`\|s\|^2\le\texttt{beta\_sq}`$ and
/// $`s_0+h\,s_1\equiv t\pmod q`$. The standard bound is `pk.params().beta_sq`; `u64::MAX`
/// disables the norm check.
///
/// As the paper's Verify, it accepts $`s=0`$ for $`t=0`$ (the PCS handles that case by a union
/// bound over its own target distribution).
#[must_use]
pub fn verify(pk: &PublicKey, t: &RqPoly, s: &Preimage, beta_sq: u64) -> bool {
    let params = &pk.params;
    let n = params.n();
    if s.s0.len() != n || s.s1.len() != n || !t.matches(params) || !pk.h.matches(params) {
        return false;
    }
    if norm_sq_u128(&s.s0, &s.s1) > beta_sq as u128 {
        return false;
    }
    // s0 + h s1 = t (mod q), with the prepared h: s1 mod q, four NTTs, then one pass that
    // compares s0 + (h s1) with t. The temporaries depend on the (secret) preimage: wiped.
    let q = params.q;
    let mq = ModQ::new(q);
    let mut s1q = vec![0u32; n];
    ntt::reduce_i32_slice(&mq, &s.s1, &mut s1q);
    let mut prod = vec![0u32; n];
    let hp = pk.h_prepared();
    ntt::mul_prepared_mod_q(params.logn, q, &hp.r1, &hp.r2, &s1q, &mut prod);
    let ok = ntt::add_i32_equals(&mq, &s.s0, &prod, &t.c);
    wipe::slice(&mut s1q);
    wipe::slice(&mut prod);
    ok
}

/// $`\|s_0\|^2+\|s_1\|^2`$ exactly (in `u128`), with a kernel when the `simd` feature has one.
fn norm_sq_u128(s0: &[i32], s1: &[i32]) -> u128 {
    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    if let Some(v) = crate::simd::norm_sq(s0, s1) {
        return v;
    }
    s0.iter()
        .chain(s1.iter())
        .map(|&x| {
            let a = x.unsigned_abs() as u128;
            a * a
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use zeroize::Zeroize;

    use super::*;
    use crate::Preimage;

    /// S-ZERO-1: a preimage (the PCS holder's witness) is wiped by `zeroize()` (and on drop).
    #[test]
    fn preimage_zeroize() {
        let mut s = Preimage {
            s0: vec![3, -4, 5],
            s1: vec![-6, 7, 8],
        };
        assert_eq!(s.norm_sq(), 9 + 16 + 25 + 36 + 49 + 64);
        s.zeroize();
        assert!(s.s0.is_empty() && s.s1.is_empty());
        assert_eq!(s.norm_sq(), 0);
    }

    /// The rounding guard: with the FFT basis shifted by a quarter in one point, the `f64`
    /// lattice point is no longer near an integer vector, and the attempt is discarded (`None`)
    /// instead of returning a wrong lattice point. The unmodified key passes.
    #[test]
    fn rounding_guard_trips_on_a_corrupted_basis() {
        let params = crate::Params::FALCON_1024;
        let (pk, sk) = crate::trapgen(&params, &[0x61; 32]).unwrap();
        let ek = ExpandedKey::new(&sk).unwrap();
        let t = RqPoly::from_coeffs(&params, &vec![1234; params.n()]).unwrap();
        let mut prng = Shake256Prng::new(b"guard");
        let s = samp_pre_attempt(&ek, &t, &mut prng).expect("a valid key passes");
        assert!(verify(&pk, &t, &s, params.beta_sq));
        let mut bad = ExpandedKey::new(&sk).unwrap();
        bad.basis[0] += 0.25;
        let mut prng = Shake256Prng::new(b"guard");
        let mut samp = SamplerZ::new(&params, &mut prng);
        assert!(samp_pre_with(&bad, &t, &mut samp).is_none());
    }

    /// Verify's prepared product and signed reduction agree with the plain formula
    /// s0 + h s1 = t on random preimage-shaped inputs, including wrong ones.
    #[test]
    fn verify_matches_plain_formula() {
        let params = crate::Params::PCS;
        let n = params.n();
        let mut rng = Shake256Prng::new(b"verify-plain");
        let h = RqPoly::from_i64(
            &params,
            &(0..n).map(|_| rng.next_u64() as i64).collect::<Vec<i64>>(),
        )
        .unwrap();
        let pk = PublicKey::from_h(&params, h.clone()).unwrap();
        for k in 0..20 {
            let small = |rng: &mut Shake256Prng| -> Vec<i32> {
                (0..n)
                    .map(|_| (rng.next_u64() % 20001) as i32 - 10000)
                    .collect()
            };
            let s = Preimage {
                s0: small(&mut rng),
                s1: small(&mut rng),
            };
            let s0 = RqPoly::from_i32(&params, &s.s0).unwrap();
            let s1 = RqPoly::from_i32(&params, &s.s1).unwrap();
            let good = s0.add(&h.mul(&s1));
            assert!(verify(&pk, &good, &s, u64::MAX), "case {k}");
            let mut c = good.coeffs().to_vec();
            c[k * 37 % n] = (c[k * 37 % n] + 1) % params.q;
            let wrong = RqPoly::from_coeffs(&params, &c).unwrap();
            assert!(!verify(&pk, &wrong, &s, u64::MAX), "case {k}");
        }
    }

    /// `samp_pre_batch` on enough calls for several groups (the seed derivations and the first
    /// blocks of the streams computed together with the `simd` feature) returns, call by call,
    /// what `samp_pre` returns, at both presets.
    #[test]
    fn samp_pre_batch_groups_equal_samp_pre() {
        for params in [crate::Params::PCS, crate::Params::FALCON_1024] {
            let (_, sk) = crate::trapgen(&params, &[0x63; 32]).unwrap();
            let ek = ExpandedKey::new(&sk).unwrap();
            let mut rng = Shake256Prng::new(b"batch-groups");
            let targets: Vec<RqPoly> = (0..21)
                .map(|_| {
                    let c: Vec<u32> = (0..params.n())
                        .map(|_| (rng.next_u64() % params.q as u64) as u32)
                        .collect();
                    RqPoly::from_coeffs(&params, &c).unwrap()
                })
                .collect();
            let seeds: Vec<[u8; 32]> = (0..21u8).map(|i| [i.wrapping_mul(37); 32]).collect();
            let items: Vec<(&RqPoly, &[u8; 32])> = targets.iter().zip(seeds.iter()).collect();
            let batch = samp_pre_batch(&ek, &items);
            for (i, (got, &(t, seed))) in batch.iter().zip(items.iter()).enumerate() {
                assert_eq!(got, &samp_pre(&ek, t, seed), "{} item {i}", params.name);
            }
        }
    }

    /// `samp_pre_batch` returns, entry by entry, what `samp_pre` returns for the same target and
    /// seed (with or without the `parallel` feature), errors included and in place.
    #[test]
    fn samp_pre_batch_equals_samp_pre() {
        let params = crate::Params::PCS;
        let (_, sk) = crate::trapgen(&params, &[0x62; 32]).unwrap();
        let ek = ExpandedKey::new(&sk).unwrap();
        let mut targets: Vec<RqPoly> = (0..9u32)
            .map(|i| {
                let c: Vec<u32> = (0..params.n() as u32)
                    .map(|j| (j * 7919 + i * 104729) % params.q)
                    .collect();
                RqPoly::from_coeffs(&params, &c).unwrap()
            })
            .collect();
        // An element of another ring: Error::Mismatch, in its slot.
        targets[4] = RqPoly::from_coeffs(&crate::Params::FALCON_1024, &[1u32; 1024]).unwrap();
        let seeds: Vec<[u8; 32]> = (0..9u8).map(|i| [i ^ 0xa5; 32]).collect();
        let items: Vec<(&RqPoly, &[u8; 32])> = targets.iter().zip(seeds.iter()).collect();
        let batch = samp_pre_batch(&ek, &items);
        assert_eq!(batch.len(), items.len());
        for (i, (got, &(t, seed))) in batch.iter().zip(items.iter()).enumerate() {
            assert_eq!(got, &samp_pre(&ek, t, seed), "item {i}");
        }
        assert_eq!(batch[4], Err(Error::Mismatch));
    }
}
