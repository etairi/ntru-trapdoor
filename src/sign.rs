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
//! Verify: $`\|s\|^2\le\beta^2`$ (exact, `u128` accumulation) and $`s_0+h\,s_1\equiv t`$ (one
//! product with the prepared $`h`$: four NTTs).
//!
//! Scratch buffers are allocated per call and wiped before they are released.

use zeroize::{Zeroize, Zeroizing};

use crate::ffsamp::ffsampling;
use crate::fft::{fft, ifft, poly_add, poly_mul_fft, poly_mulconst};
use crate::prng::{Prng, Shake256Prng, shake256};
use crate::ring::ntt::{self, ModQ};
use crate::sampler::{LeafSampler, SamplerZ};
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
    let mut seed_star = Zeroizing::new([0u8; 64]);
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

/// One SampPre attempt with an explicit PRNG: `None` if $`\|s\|^2>\lfloor\beta^2\rfloor`$ (or
/// if the rounding guard trips, which a valid key never does).
///
/// Consumes the PRNG exactly as one iteration of fn-dsa's `sign_inner` after the nonce
/// (lib.rs:509-744): this is the entry point of the fn-dsa parity tests. Panics if `t` is not
/// in the key's ring.
pub fn samp_pre_attempt<P: Prng>(ek: &ExpandedKey, t: &RqPoly, prng: &mut P) -> Option<Preimage> {
    let mut samp = SamplerZ::new(&ek.params, prng);
    let s = samp_pre_with(ek, t, &mut samp)?;
    if s.norm_sq() <= ek.params.beta_sq {
        Some(s)
    } else {
        None
    }
}

/// ffSampling with any leaf sampler, without the norm check (for the precision tests, which
/// replay recorded leaf decisions). `None` if the rounding guard trips: some coordinate of the
/// `f64` lattice point lies farther than $`2^{-10}`$ from an integer (module documentation).
/// Panics if `t` is not in the key's ring.
pub fn samp_pre_with<S: LeafSampler>(
    ek: &ExpandedKey,
    t: &RqPoly,
    sampler: &mut S,
) -> Option<Preimage> {
    let params = &ek.params;
    assert!(t.matches(params), "samp_pre: target not in the key's ring");
    let logn = params.logn;
    let n = params.n();
    let (b00, b01, b10, b11) = ek.basis_parts();

    // Scratch: t0, t1, then 2n values for ffSampling (later tx, ty).
    let mut buf = vec![0.0f64; 4 * n];
    let (t0, rest) = buf.split_at_mut(n);
    let (t1, tmp) = rest.split_at_mut(n);

    // Target (fn-dsa lib.rs:621-640): t0 = FFT(t) b11 / q, t1 = -FFT(t) b01 / q.
    for (x, &c) in t0.iter_mut().zip(t.c.iter()) {
        *x = c as f64;
    }
    fft(logn, t0);
    t1.copy_from_slice(t0);
    poly_mul_fft(logn, t1, b01);
    poly_mulconst(logn, t1, -params.inv_q);
    poly_mul_fft(logn, t0, b11);
    poly_mulconst(logn, t0, params.inv_q);

    // Sampling (lib.rs:651).
    ffsampling(sampler, logn, t0, t1, &ek.tree, tmp);

    // Lattice point v = z B (lib.rs:673-689): v0 = z0 b00 + z1 b10, v1 = z0 b01 + z1 b11.
    {
        let (tx, ty) = tmp.split_at_mut(n);
        tx.copy_from_slice(t0);
        ty.copy_from_slice(t1);
        poly_mul_fft(logn, tx, b00);
        poly_mul_fft(logn, ty, b10);
        poly_add(logn, tx, ty);
        ty.copy_from_slice(t0);
        poly_mul_fft(logn, ty, b01);
        t0.copy_from_slice(tx);
        poly_mul_fft(logn, t1, b11);
        poly_add(logn, t1, ty);
    }
    ifft(logn, t0);
    ifft(logn, t1);

    // s0 = t - rint(v0), s1 = -rint(v1) (lib.rs:691-722). FLR::rint is round-half-even
    // (aarch64 vcvtnd, x86-64 cvtsd2si; flr_native.rs:373-435); the casts saturate, and
    // out-of-range values (which cannot occur for a valid key) saturate in i32 so that the
    // norm check rejects them. The guard flags a coordinate farther than 2^-10 from an integer
    // (or NaN).
    let sat = |x: i64| x.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let mut off = false;
    let s0: Vec<i32> = t0
        .iter()
        .zip(t.c.iter())
        .map(|(&v, &c)| {
            let r = v.round_ties_even();
            let d = (v - r).abs();
            off |= (d > ROUND_GUARD) | d.is_nan();
            sat((c as i64).saturating_sub(r as i64))
        })
        .collect();
    let s1: Vec<i32> = t1
        .iter()
        .map(|&v| {
            let r = v.round_ties_even();
            let d = (v - r).abs();
            off |= (d > ROUND_GUARD) | d.is_nan();
            sat(0i64.saturating_sub(r as i64))
        })
        .collect();
    // A slice wipe (one volatile pass); `Vec::zeroize` would also wipe the capacity byte by byte,
    // about 9 us here [measured, sign/README.md].
    buf.as_mut_slice().zeroize();
    let s = Preimage { s0, s1 };
    // On `None` the preimage is wiped when dropped.
    (!off).then_some(s)
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
    let norm: u128 =
        s.s0.iter()
            .chain(s.s1.iter())
            .map(|&x| {
                let a = x.unsigned_abs() as u128;
                a * a
            })
            .sum();
    if norm > beta_sq as u128 {
        return false;
    }
    // s0 + h s1 = t (mod q), with the prepared h: s1 mod q, four NTTs, then one pass that
    // compares s0 + (h s1) with t. The temporaries depend on the (secret) preimage: wiped.
    let q = params.q;
    let mq = ModQ::new(q);
    let mut s1q: Vec<u32> = s.s1.iter().map(|&x| mq.reduce_i32(x)).collect();
    let mut prod = vec![0u32; n];
    let hp = pk.h_prepared();
    ntt::mul_prepared_mod_q(params.logn, q, &hp.r1, &hp.r2, &s1q, &mut prod);
    let mut diff = 0u32;
    for ((&a, &b), &c) in s.s0.iter().zip(prod.iter()).zip(t.c.iter()) {
        diff |= ntt::mp_add(mq.reduce_i32(a), b, q) ^ c;
    }
    s1q.as_mut_slice().zeroize();
    prod.as_mut_slice().zeroize();
    diff == 0
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
}
