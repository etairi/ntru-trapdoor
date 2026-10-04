//! Differential parity of SampPre with Pornin's fn-dsa 0.4.0.
//!
//! # What is compared
//!
//! fn-dsa's signer (fn-dsa-sign/src/lib.rs:268-319 and 482-744) decomposes as
//! 1. hedge: `seed' = SHAKE256(H(f,g) || mu || seed40)`, 40 bytes;
//! 2. per attempt `k = 0..27`: `P_k = SHAKE256_PRNG(seed' || k)`; the first 40 bytes of `P_k`
//!    are the nonce, the target is `c_k = hash_to_point(nonce, mu)`;
//! 3. the **core**: ffSampling of `(c_k, 0)` over the key, with the rest of `P_k`;
//!    `(s1, s2) = (c_k - rint(v0), -rint(v1))`; reject if `||s||^2 > SQBETA`;
//! 4. signature-format checks: reject if some `|s1_i| > 840` or `comp_encode(s2)` does not fit.
//!
//! This crate's `samp_pre(ek, t, seed32)` decomposes as
//! 1. `seed* = SHAKE256("ntru-trapdoor/v1/samp-pre" || key_id || encode(t) || seed32)`, 64 bytes;
//! 2. per attempt `k`: `P*_k = SHAKE256(seed* || k)`, the target `t` is fixed;
//! 3. the same core, `hazmat::samp_pre_attempt(ek, t, P*_k)`, returning `(s0, s1)`.
//!
//! The cores are meant to be identical bit for bit (same IEEE operations in the same order,
//! with the ffLDL tree precomputed instead of recomputed per attempt). The **streams** cannot be:
//! `P*_k` starts at byte 0 of a SHAKE256 output, fn-dsa's core starts at byte 40 of another, so
//! no 32-byte seed of the public API reproduces an fn-dsa signature. The smallest difference is
//! therefore the seed-to-stream map (and, on retries, fn-dsa's fresh nonce and target per
//! attempt, plus its two format checks). The tests make everything else exact:
//!
//! - **P-PRNG**: this crate's `Shake256Prng` and `shake256` against fn-dsa-comm's `SHAKE256_PRNG`
//!   and `SHAKE256`, under mixed reads.
//! - **P-1024** (main): 32 fn-dsa Falcon-1024 key pairs × 64 messages = 2048 messages. fn-dsa's
//!   steps 1, 2 and 4 are replayed with fn-dsa-comm's code (its SHAKE256, its PRNG, its
//!   `hash_to_point`, its codec) around this crate's core. Checked per message: the signature,
//!   byte for byte; the same attempt index; **the same s = (s0, s1)**, with fn-dsa's s0 rebuilt by
//!   fn-dsa's verifier arithmetic (`c - s2 h mod q`, fn-dsa-vrfy/src/lib.rs:243-293) and its s1
//!   decoded from the signature; the same result with this crate's own PRNG; fn-dsa verifies the
//!   replayed signature, and this crate's `verify` accepts `(t, s)`.
//! - **P-OURKEYS**: the same at Falcon-1024 on keys from this crate's `trapgen`, exported to
//!   fn-dsa's key format (when f, g fit fn-dsa's 5 bits and F, G its 8 bits).
//! - **P-TOY**: fn-dsa's toy and standard degrees `logn = 2..9`, where fn-dsa's norm check
//!   rejects often: the core's reject-and-resample branch runs in lockstep with fn-dsa's.
//! - **P-PUB**: the public `samp_pre` equals the core on the documented stream `P*_k` (an
//!   independent re-derivation of `seed*` with fn-dsa's SHAKE256): the only difference to fn-dsa
//!   is pinned down exactly.
//! - **P-VER**: fn-dsa verifies signatures made with the public `samp_pre` (this crate's seeds),
//!   on fn-dsa keys and on `trapgen` keys; mutated signatures fail in both verifiers.
//! - **P-VKAT**: fn-dsa's own verification KATs (512 and 1024) pass this crate's `verify`.
//! - **P-LONG** (ignored): 128 keys × 512 messages = 65,536 messages at Falcon-1024.
//! - **P-SENS** (ignored): the resolution of output parity. Constants perturbed by 1 ulp to
//!   2^-10 (relative) in the replay only, counting the fn-dsa signatures that change: what a
//!   matching s does and does not certify about the floating-point intermediates.
//!
//! Run: `cargo test --test parity_fndsa_diff -- --nocapture` (statistics on stderr, about 5 s);
//! `cargo test --test parity_fndsa_diff -- --ignored --nocapture` for P-LONG (about 40 s) and
//! P-SENS (about 2 s). The test profile is optimised (`opt-level = 3`).
//!
//! Platform note: on aarch64 fn-dsa runs its NEON code paths and this crate its portable code;
//! on x86-64 with AVX2 fn-dsa would run its AVX2 paths instead.

use fn_dsa::{
    CryptoRng, DOMAIN_NONE, DomainContext, HASH_ID_RAW, KeyPairGenerator, KeyPairGenerator512,
    KeyPairGenerator1024, KeyPairGeneratorWeak, RngCore, RngError, SigningKey, SigningKey512,
    SigningKey1024, SigningKeyWeak, VerifyingKey, VerifyingKey512, VerifyingKey1024,
    VerifyingKeyWeak, compute_mu, hashed_vrfykey_from_vrfykey, sign_key_size, signature_size,
    vrfy_key_size,
};
use fn_dsa_comm::PRNG as FnDsaPrngTrait;
use fn_dsa_comm::codec::{
    B_INF, comp_decode, comp_encode, modq_decode, modq_encode, trim_i8_decode, trim_i8_encode,
};
use fn_dsa_comm::hash_to_point;
use fn_dsa_comm::mq::{
    SQBETA, mqpoly_NTT_to_int, mqpoly_div_ntt, mqpoly_ext_to_int, mqpoly_int_to_NTT,
    mqpoly_int_to_ext, mqpoly_int_to_small, mqpoly_mul_ntt, mqpoly_signed_to_ext,
    mqpoly_small_to_int, mqpoly_sub_int,
};
use fn_dsa_comm::shake::{SHAKE256, SHAKE256_PRNG};
use ntru_trapdoor::hazmat::{Prng, Shake256Prng, samp_pre_attempt, shake256};
use ntru_trapdoor::{
    BaseSampler, ExpandedKey, Params, Preimage, PublicKey, RqPoly, SecretKey, samp_pre, trapgen,
    verify,
};

// ---------------------------------------------------------------------------------------------
// fn-dsa's constants and parameter sets
// ---------------------------------------------------------------------------------------------

/// `FLR::scaled(x, sc)` = x * 2^sc, exact for |x| < 2^53.
fn scaled(x: i64, sc: i32) -> f64 {
    (x as f64) * f64::from_bits(((1023 + sc) as u64) << 52)
}

/// fn-dsa's `INV_SIGMA[logn]` mantissas, `FLR::scaled(x, -60)` (fn-dsa-sign/src/sampler.rs:48-60).
const FNDSA_INV_SIGMA: [i64; 11] = [
    0,
    7961475618707097,
    7851656902127320,
    7746260754658859,
    7595833604889141,
    7453842886538220,
    7319528409832599,
    7192222552237877,
    7071336252758509,
    6956347512113097,
    6846791885593314,
];

/// fn-dsa's `SIGMA_MIN[logn]` mantissas, `FLR::scaled(x, -52)` (fn-dsa-sign/src/sampler.rs:61-73).
const FNDSA_SIGMA_MIN: [i64; 11] = [
    0,
    5028307297130123,
    5098636688852518,
    5168009084304506,
    5270355833453349,
    5370752584786614,
    5469306724145091,
    5566116128735780,
    5661270305715104,
    5754851361258101,
    5846934829975396,
];

const NAMES: [&str; 11] = [
    "",
    "fn-dsa logn 1",
    "fn-dsa logn 2 (n = 4)",
    "fn-dsa logn 3 (n = 8)",
    "fn-dsa logn 4 (n = 16)",
    "fn-dsa logn 5 (n = 32)",
    "fn-dsa logn 6 (n = 64)",
    "fn-dsa logn 7 (n = 128)",
    "fn-dsa logn 8 (n = 256)",
    "Falcon-512 (fn-dsa constants)",
    "Falcon-1024 (q = 12289)",
];

/// fn-dsa's signing parameters at `logn`: q = 12289, `INV_SIGMA[logn]`, `SIGMA_MIN[logn]`, the
/// 1.8205 base, `SQBETA[logn]`. At `logn = 10` this is the crate's preset, after checking that it
/// carries exactly these constants. Elsewhere the encoding widths are fn-dsa's i8 range (8 bits),
/// so that `SecretKey::validate` accepts fn-dsa's toy keys.
fn fndsa_params(logn: u32) -> Params {
    let l = logn as usize;
    let inv_sigma = scaled(FNDSA_INV_SIGMA[l], -60);
    let sigma_min = scaled(FNDSA_SIGMA_MIN[l], -52);
    if logn == 10 {
        let p = Params::FALCON_1024;
        assert_eq!(p.inv_sigma.to_bits(), inv_sigma.to_bits());
        assert_eq!(p.sigma_min.to_bits(), sigma_min.to_bits());
        assert_eq!(p.beta_sq, SQBETA[10] as u64);
        assert_eq!(p.base, BaseSampler::Falcon1_8205);
        assert_eq!(p.samppre_max_attempts, 27);
        return p;
    }
    let p = Params {
        name: NAMES[l],
        logn,
        sigma: 1.0 / inv_sigma,
        inv_sigma,
        sigma_min,
        base: BaseSampler::Falcon1_8205,
        beta_sq: SQBETA[l] as u64,
        sk_fg_bits: 8,
        sk_big_fg_bits: 8,
        ..Params::FALCON_1024
    };
    p.validate().unwrap();
    p
}

/// Bits per coefficient of f and g in fn-dsa's signing key (fn-dsa-comm/src/lib.rs:36-41).
fn nbits_fg(logn: u32) -> u32 {
    match logn {
        2..=5 => 8,
        6..=7 => 7,
        8..=9 => 6,
        _ => 5,
    }
}

// ---------------------------------------------------------------------------------------------
// Randomness and hashing on fn-dsa's side
// ---------------------------------------------------------------------------------------------

/// SHAKE256 of the concatenation of `parts`, with fn-dsa-comm's implementation.
fn fn_shake(parts: &[&[u8]], out: &mut [u8]) {
    let mut sh = SHAKE256::new();
    for p in parts {
        sh.inject(p);
    }
    sh.flip();
    sh.extract(out);
}

/// Deterministic test bytes.
fn derive<const N: usize>(parts: &[&[u8]]) -> [u8; N] {
    let mut out = [0u8; N];
    fn_shake(parts, &mut out);
    out
}

/// A `rand_core` 0.6 RNG that hands out one fixed byte string with one `fill_bytes` call, and
/// panics on any other use. fn-dsa's key generation draws exactly 32 bytes this way
/// (fn-dsa-kgen/src/lib.rs:207-208), its signer exactly 40 (fn-dsa-sign/src/lib.rs:286-287).
struct OneShotRng<'a> {
    bytes: &'a [u8],
    used: bool,
}

impl<'a> OneShotRng<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        OneShotRng { bytes, used: false }
    }
}

impl CryptoRng for OneShotRng<'_> {}

impl RngCore for OneShotRng<'_> {
    fn next_u32(&mut self) -> u32 {
        panic!("fn-dsa called next_u32")
    }
    fn next_u64(&mut self) -> u64 {
        panic!("fn-dsa called next_u64")
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        assert!(!self.used, "fn-dsa drew randomness twice");
        assert_eq!(dest.len(), self.bytes.len(), "unexpected draw length");
        dest.copy_from_slice(self.bytes);
        self.used = true;
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RngError> {
        self.fill_bytes(dest);
        Ok(())
    }
}

/// fn-dsa-comm's `SHAKE256_PRNG` behind this crate's `Prng` trait: the core then reads fn-dsa's
/// stream through fn-dsa's own code.
struct FnPrng(SHAKE256_PRNG);

impl FnPrng {
    fn new(seed: &[u8]) -> Self {
        FnPrng(<SHAKE256_PRNG as FnDsaPrngTrait>::new(seed))
    }
}

impl Prng for FnPrng {
    fn next_u8(&mut self) -> u8 {
        FnDsaPrngTrait::next_u8(&mut self.0)
    }
    fn next_u16(&mut self) -> u16 {
        FnDsaPrngTrait::next_u16(&mut self.0)
    }
    fn next_u64(&mut self) -> u64 {
        FnDsaPrngTrait::next_u64(&mut self.0)
    }
    fn next_bytes(&mut self, dst: &mut [u8]) {
        FnDsaPrngTrait::next_bytes(&mut self.0, dst)
    }
}

/// Counts the reads of a PRNG. With fn-dsa's base sampler a base draw is one `next_u64` and
/// one `next_u16`, and BerExp reads single bytes; so `u64s` counts the sampler's trials.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Reads {
    u8s: u64,
    u16s: u64,
    u64s: u64,
    bytes: u64,
}

impl Reads {
    fn total_bytes(&self) -> u64 {
        self.u8s + 2 * self.u16s + 8 * self.u64s + self.bytes
    }
    fn add(&mut self, o: &Reads) {
        self.u8s += o.u8s;
        self.u16s += o.u16s;
        self.u64s += o.u64s;
        self.bytes += o.bytes;
    }
}

struct Counting<P: Prng> {
    inner: P,
    reads: Reads,
}

impl<P: Prng> Prng for Counting<P> {
    fn next_u8(&mut self) -> u8 {
        self.reads.u8s += 1;
        self.inner.next_u8()
    }
    fn next_u16(&mut self) -> u16 {
        self.reads.u16s += 1;
        self.inner.next_u16()
    }
    fn next_u64(&mut self) -> u64 {
        self.reads.u64s += 1;
        self.inner.next_u64()
    }
    fn next_bytes(&mut self, dst: &mut [u8]) {
        self.reads.bytes += dst.len() as u64;
        self.inner.next_bytes(dst)
    }
}

// ---------------------------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------------------------

/// An fn-dsa key pair, imported.
struct FnKey {
    params: Params,
    sk_enc: Vec<u8>,
    vk_enc: Vec<u8>,
    sk: SecretKey,
    /// h = g / f, computed by this crate's ring (`SecretKey::public_key`).
    pk: PublicKey,
    /// h in fn-dsa's NTT representation (internal range), as fn-dsa's verifier holds it.
    h_ntt: Vec<u16>,
    /// SHAKE256 of the (f, g) encoding, fn-dsa's hedging key (fn-dsa-sign/src/lib.rs:385-389).
    hashed_sign_key: [u8; 40],
    /// SHAKE256 of the verifying key, for mu.
    hashed_vk: [u8; 64],
}

/// fn-dsa's key generation with a fixed 32-byte seed.
fn fndsa_keypair<KG: KeyPairGenerator>(logn: u32, seed32: &[u8; 32]) -> (Vec<u8>, Vec<u8>) {
    let mut kg = KG::default();
    let mut sk = vec![0u8; sign_key_size(logn)];
    let mut vk = vec![0u8; vrfy_key_size(logn)];
    kg.keygen(logn, &mut OneShotRng::new(seed32), &mut sk, &mut vk);
    (sk, vk)
}

/// f G - g F in Z[x]/(x^n + 1), exactly (schoolbook, i64).
fn fg_minus_gf(f: &[i16], g: &[i16], big_f: &[i16], big_g: &[i16]) -> Vec<i64> {
    let n = f.len();
    let mut e = vec![0i64; n];
    for i in 0..n {
        for k in 0..n {
            let v = f[i] as i64 * big_g[k] as i64 - g[i] as i64 * big_f[k] as i64;
            if i + k < n {
                e[i + k] += v;
            } else {
                e[i + k - n] -= v;
            }
        }
    }
    e
}

/// Imports an fn-dsa key pair, following fn-dsa's own decoder (fn-dsa-sign/src/lib.rs:343-454)
/// with fn-dsa-comm's arithmetic: f, g, F from the signing key, G = g F / f mod q, h = g / f in
/// NTT form. Checks: fG - gF = q exactly; fn-dsa's verifying key is `logn || modq_encode(NTT(h))`
/// and its hash ends the signing key; this crate's h = g / f (through its own ring) equals
/// fn-dsa's h after fn-dsa's inverse NTT; `SecretKey::validate` passes.
fn import_fndsa_key(params: &Params, sk_enc: &[u8], vk_enc: &[u8]) -> FnKey {
    let logn = params.logn;
    let n = params.n();
    assert_eq!(sk_enc[0], 0x50 + logn as u8);
    assert_eq!(sk_enc.len(), sign_key_size(logn));
    assert_eq!(vk_enc.len(), vrfy_key_size(logn));
    let nb = nbits_fg(logn);
    let (mut f, mut g, mut big_f) = (vec![0i8; n], vec![0i8; n], vec![0i8; n]);
    let j = 1 + trim_i8_decode(&sk_enc[1..], &mut f, nb).unwrap();
    let j = j + trim_i8_decode(&sk_enc[j..], &mut g, nb).unwrap();
    let mut hashed_sign_key = [0u8; 40];
    fn_shake(&[&sk_enc[1..j]], &mut hashed_sign_key);
    let j = j + trim_i8_decode(&sk_enc[j..], &mut big_f, 8).unwrap();
    assert_eq!(j + 64, sk_enc.len());
    let hashed_vk = hashed_vrfykey_from_vrfykey(vk_enc);
    assert_eq!(&sk_enc[j..], &hashed_vk[..]);

    // h = g / f (NTT), G = h F (fn-dsa-sign/src/lib.rs:404-451).
    let (mut w0, mut w1) = (vec![0u16; n], vec![0u16; n]);
    mqpoly_small_to_int(logn, &g, &mut w0);
    mqpoly_small_to_int(logn, &f, &mut w1);
    mqpoly_int_to_NTT(logn, &mut w0);
    mqpoly_int_to_NTT(logn, &mut w1);
    assert!(mqpoly_div_ntt(logn, &mut w0, &w1), "f not invertible");
    let h_ntt = w0.clone();
    mqpoly_small_to_int(logn, &big_f, &mut w1);
    mqpoly_int_to_NTT(logn, &mut w1);
    mqpoly_mul_ntt(logn, &mut w1, &w0);
    mqpoly_NTT_to_int(logn, &mut w1);
    let mut big_g = vec![0i8; n];
    assert!(mqpoly_int_to_small(logn, &w1, &mut big_g), "G out of range");

    // The verifying key is header || modq_encode(NTT(h)) (fn-dsa-sign/src/lib.rs:423-427).
    let mut hx = h_ntt.clone();
    mqpoly_int_to_ext(logn, &mut hx);
    let mut vk = vec![0u8; vrfy_key_size(logn)];
    vk[0] = logn as u8;
    modq_encode(&hx, &mut vk[1..]);
    assert_eq!(vk, vk_enc, "fn-dsa's verifying key layout");

    let to16 = |v: &[i8]| v.iter().map(|&x| x as i16).collect::<Vec<i16>>();
    let (f, g, big_f, big_g) = (to16(&f), to16(&g), to16(&big_f), to16(&big_g));
    let e = fg_minus_gf(&f, &g, &big_f, &big_g);
    assert_eq!(e[0], params.q as i64);
    assert!(e[1..].iter().all(|&x| x == 0));
    let sk = SecretKey::from_parts(params, f, g, big_f, big_g).unwrap();
    sk.validate().unwrap();
    let pk = sk.public_key().unwrap();
    let mut hc = h_ntt.clone();
    mqpoly_NTT_to_int(logn, &mut hc);
    mqpoly_int_to_ext(logn, &mut hc);
    let hc: Vec<u32> = hc.iter().map(|&x| x as u32).collect();
    assert_eq!(
        pk.h().coeffs(),
        &hc[..],
        "h = g / f: this crate against fn-dsa"
    );
    FnKey {
        params: *params,
        sk_enc: sk_enc.to_vec(),
        vk_enc: vk_enc.to_vec(),
        sk,
        pk,
        h_ntt,
        hashed_sign_key,
        hashed_vk,
    }
}

/// This crate's key pair in fn-dsa's formats, if f, g fit fn-dsa's `nbits_fg` bits and F, G its
/// 8 bits (fn-dsa's G is recomputed from f, g, F and must lie in [-127, 127]).
fn export_to_fndsa(sk: &SecretKey, pk: &PublicKey) -> Option<(Vec<u8>, Vec<u8>)> {
    let p = sk.params();
    let logn = p.logn;
    let lim_fg = (1i16 << (nbits_fg(logn) - 1)) - 1;
    let fits = |v: &[i16], lim: i16| v.iter().all(|&x| x.abs() <= lim);
    if !fits(sk.f(), lim_fg) || !fits(sk.g(), lim_fg) {
        return None;
    }
    if !fits(sk.big_f(), 127) || !fits(sk.big_g(), 127) {
        return None;
    }
    // Verifying key: logn || modq_encode(NTT(h)).
    let mut h: Vec<u16> = pk.h().coeffs().iter().map(|&x| x as u16).collect();
    mqpoly_ext_to_int(logn, &mut h);
    mqpoly_int_to_NTT(logn, &mut h);
    mqpoly_int_to_ext(logn, &mut h);
    let mut vk = vec![0u8; vrfy_key_size(logn)];
    vk[0] = logn as u8;
    modq_encode(&h, &mut vk[1..]);
    // Signing key: 0x50 + logn || f || g || F || SHAKE256(vk) (fn-dsa-kgen/src/lib.rs:243-253).
    let to8 = |v: &[i16]| v.iter().map(|&x| x as i8).collect::<Vec<i8>>();
    let mut skb = vec![0u8; sign_key_size(logn)];
    skb[0] = 0x50 + logn as u8;
    let nb = nbits_fg(logn);
    let j = 1 + trim_i8_encode(&to8(sk.f()), nb, &mut skb[1..]);
    let j = j + trim_i8_encode(&to8(sk.g()), nb, &mut skb[j..]);
    let j = j + trim_i8_encode(&to8(sk.big_f()), 8, &mut skb[j..]);
    skb[j..].copy_from_slice(&hashed_vrfykey_from_vrfykey(&vk));
    Some((skb, vk))
}

// ---------------------------------------------------------------------------------------------
// fn-dsa's signer, replayed around this crate's core
// ---------------------------------------------------------------------------------------------

/// Why an attempt was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reject {
    /// `samp_pre_attempt` returned `None`: ||s||^2 > floor(beta^2) (fn-dsa: `sqn > SQBETA`).
    Norm,
    /// Some |s0_i| > 840 (fn-dsa folds this into its norm flag, lib.rs:697-731).
    Binf0,
    /// Some |s1_i| > 840: `comp_encode(s1)` refuses it (fn-dsa-comm/src/codec.rs:176-183).
    Binf1,
    /// `comp_encode(s1)` does not fit the signature (codec.rs:196-214; fn-dsa-sign/src/lib.rs:733-742).
    Size,
}

/// One replayed signature.
struct Replay {
    sig: Vec<u8>,
    counter: u8,
    t: RqPoly,
    s: Preimage,
    rejects: Vec<Reject>,
    /// PRNG reads of the core in the accepted attempt (the nonce excluded).
    reads: Reads,
}

/// fn-dsa's `sign_inner` loop (fn-dsa-sign/src/lib.rs:503-743) with `samp_pre_attempt` as the
/// core: for counter k, `P_k = new_prng(seed' || k)`, the nonce is its first 40 bytes, the target
/// `hash_to_point(nonce, mu)`, then the core on the rest of `P_k`, then fn-dsa's format checks.
/// `Err(rejects)` after 27 failed attempts (fn-dsa's `sign` then returns `None`).
fn replay_sign<P: Prng>(
    ek: &ExpandedKey,
    seed40: &[u8; 40],
    mu: &[u8; 64],
    new_prng: impl Fn(&[u8]) -> P,
) -> Result<Replay, Vec<Reject>> {
    let params = ek.params();
    let logn = params.logn;
    let n = params.n();
    let mut rejects = Vec::new();
    for counter in 0..27u8 {
        let mut seed2 = [0u8; 41];
        seed2[..40].copy_from_slice(seed40);
        seed2[40] = counter;
        let mut prng = Counting {
            inner: new_prng(&seed2),
            reads: Reads::default(),
        };
        let mut nonce = [0u8; 40];
        prng.next_bytes(&mut nonce);
        prng.reads = Reads::default();
        let mut hm = vec![0u16; n];
        hash_to_point(&nonce, mu, &mut hm);
        let hm32: Vec<u32> = hm.iter().map(|&x| x as u32).collect();
        let t = RqPoly::from_coeffs(params, &hm32).unwrap();
        let Some(s) = samp_pre_attempt(ek, &t, &mut prng) else {
            rejects.push(Reject::Norm);
            continue;
        };
        if s.s0.iter().any(|&z| z.abs() > B_INF) {
            rejects.push(Reject::Binf0);
            continue;
        }
        // fn-dsa makes one call, comp_encode, for the last two checks; split here only to count
        // the two causes separately (the accept/reject outcome is the same).
        if s.s1.iter().any(|&z| z.abs() > B_INF) {
            rejects.push(Reject::Binf1);
            continue;
        }
        let s2: Vec<i16> = s.s1.iter().map(|&z| z as i16).collect();
        let mut sig = vec![0u8; signature_size(logn)];
        if !comp_encode(&s2, &mut sig[41..]) {
            rejects.push(Reject::Size);
            continue;
        }
        sig[0] = 0x30 + logn as u8;
        sig[1..41].copy_from_slice(&nonce);
        return Ok(Replay {
            sig,
            counter,
            t,
            s,
            rejects,
            reads: prng.reads,
        });
    }
    Err(rejects)
}

/// fn-dsa's (s1, s2), i.e. this crate's (s0, s1), and the target c, from an fn-dsa signature,
/// by fn-dsa's verifier arithmetic (fn-dsa-vrfy/src/lib.rs:243-293): s2 = comp_decode(sig),
/// s1 = c - s2 h mod q, centred. Exact: fn-dsa's signer only accepts |s1_i| <= 840 < q/2.
fn fndsa_preimage(
    logn: u32,
    h_ntt: &[u16],
    mu: &[u8],
    sig: &[u8],
) -> (Vec<u32>, Vec<i32>, Vec<i32>) {
    let n = 1usize << logn;
    assert_eq!(sig.len(), signature_size(logn));
    assert_eq!(sig[0], 0x30 + logn as u8);
    let mut s2 = vec![0i16; n];
    assert!(
        comp_decode(&sig[41..], &mut s2),
        "fn-dsa's signature decodes"
    );
    let mut c = vec![0u16; n];
    hash_to_point(&sig[1..41], mu, &mut c);
    let mut t1 = c.clone();
    mqpoly_ext_to_int(logn, &mut t1);
    let mut t2 = vec![0u16; n];
    mqpoly_signed_to_ext(logn, &s2, &mut t2);
    mqpoly_ext_to_int(logn, &mut t2);
    mqpoly_int_to_NTT(logn, &mut t2);
    mqpoly_mul_ntt(logn, &mut t2, h_ntt);
    mqpoly_NTT_to_int(logn, &mut t2);
    mqpoly_sub_int(logn, &mut t1, &t2);
    mqpoly_int_to_ext(logn, &mut t1);
    let q = 12289i32;
    let s0: Vec<i32> = t1
        .iter()
        .map(|&x| {
            if x as i32 > q / 2 {
                x as i32 - q
            } else {
                x as i32
            }
        })
        .collect();
    let s1: Vec<i32> = s2.iter().map(|&x| x as i32).collect();
    (c.iter().map(|&x| x as u32).collect(), s0, s1)
}

/// The first index where two coefficient vectors differ, for readable failures.
fn first_diff(a: &[i32], b: &[i32]) -> Option<(usize, i32, i32)> {
    if a.len() != b.len() {
        return Some((usize::MAX, a.len() as i32, b.len() as i32));
    }
    a.iter()
        .zip(b.iter())
        .enumerate()
        .find(|(_, (x, y))| x != y)
        .map(|(i, (x, y))| (i, *x, *y))
}

// ---------------------------------------------------------------------------------------------
// Tallies
// ---------------------------------------------------------------------------------------------

struct Tally {
    label: String,
    keys: u64,
    refused_keys: u64,
    messages: u64,
    attempts: u64,
    norm: u64,
    binf0: u64,
    binf1: u64,
    size: u64,
    retried: u64,
    max_counter: u8,
    fndsa_failures: u64,
    leaf_min: f64,
    leaf_max: f64,
    reads: Reads,
    leaf_samples: u64,
}

impl Tally {
    fn new(label: impl Into<String>) -> Self {
        Tally {
            label: label.into(),
            keys: 0,
            refused_keys: 0,
            messages: 0,
            attempts: 0,
            norm: 0,
            binf0: 0,
            binf1: 0,
            size: 0,
            retried: 0,
            max_counter: 0,
            fndsa_failures: 0,
            leaf_min: f64::INFINITY,
            leaf_max: 0.0,
            reads: Reads::default(),
            leaf_samples: 0,
        }
    }

    fn key(&mut self, ek: &ExpandedKey) {
        let (lo, hi) = ek.leaf_sigma_range();
        self.leaf_min = self.leaf_min.min(lo);
        self.leaf_max = self.leaf_max.max(hi);
        self.keys += 1;
    }

    fn rejects(&mut self, rejects: &[Reject]) {
        self.attempts += rejects.len() as u64;
        for r in rejects {
            match r {
                Reject::Norm => self.norm += 1,
                Reject::Binf0 => self.binf0 += 1,
                Reject::Binf1 => self.binf1 += 1,
                Reject::Size => self.size += 1,
            }
        }
    }

    fn replay(&mut self, r: &Replay, n: usize) {
        self.messages += 1;
        self.attempts += 1;
        self.rejects(&r.rejects);
        if r.counter > 0 {
            self.retried += 1;
        }
        self.max_counter = self.max_counter.max(r.counter);
        self.reads.add(&r.reads);
        self.leaf_samples += 2 * n as u64;
    }

    fn report(&self) {
        let trials = self.reads.u64s as f64;
        eprintln!(
            "[{}] keys {} (refused {}), messages {}, attempts {}, rejected: norm {} / \
             |s0|>840 {} / |s1|>840 {} / size {}; messages retried {} (max counter {}); fn-dsa \
             failures {}; leaf sigma' in [{:.6}, {:.6}]; accepted attempts: {:.1} PRNG bytes each, \
             acceptance per trial {:.6} ({} leaf samples, {} trials), BerExp bytes per trial {:.5}",
            self.label,
            self.keys,
            self.refused_keys,
            self.messages,
            self.attempts,
            self.norm,
            self.binf0,
            self.binf1,
            self.size,
            self.retried,
            self.max_counter,
            self.fndsa_failures,
            self.leaf_min,
            self.leaf_max,
            self.reads.total_bytes() as f64 / self.messages.max(1) as f64,
            self.leaf_samples as f64 / trials,
            self.leaf_samples,
            self.reads.u64s,
            self.reads.u8s as f64 / trials,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The per-message check
// ---------------------------------------------------------------------------------------------

/// Which PRNG(s) drive the core in the replay.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Streams {
    /// fn-dsa-comm's `SHAKE256_PRNG` and this crate's `Shake256Prng`, both, compared.
    Both,
    /// This crate's `Shake256Prng` only (the production PRNG).
    Ours,
}

fn message(logn: u32, key_ix: u32, msg_ix: u32) -> Vec<u8> {
    let mut m = vec![0u8; (msg_ix % 67) as usize];
    fn_shake(
        &[
            b"parity/message",
            &[logn as u8],
            &key_ix.to_le_bytes(),
            &msg_ix.to_le_bytes(),
        ],
        &mut m,
    );
    m
}

/// fn-dsa signs `msg` with the raw seed `raw40`; the replay must give the same signature, the
/// same attempt index and the same s, bit for bit; both verifiers accept.
#[allow(clippy::too_many_arguments)]
fn check_message<S: SigningKey, V: VerifyingKey>(
    key: &FnKey,
    ek: &ExpandedKey,
    signer: &mut S,
    verifier: &V,
    msg: &[u8],
    raw40: &[u8; 40],
    streams: Streams,
    tally: &mut Tally,
    what: &str,
) {
    let params = &key.params;
    let logn = params.logn;
    let mu = compute_mu(&key.hashed_vk, &DOMAIN_NONE, &HASH_ID_RAW, msg);
    let mut sig = vec![0u8; signature_size(logn)];
    let fn_ok = signer
        .sign(
            &mut OneShotRng::new(raw40),
            &DOMAIN_NONE,
            &HASH_ID_RAW,
            msg,
            &mut sig,
        )
        .is_some();
    // Hedged seed (fn-dsa-sign/src/lib.rs:289-295).
    let mut seed40 = [0u8; 40];
    fn_shake(&[&key.hashed_sign_key, &mu, raw40], &mut seed40);

    let ours = replay_sign(ek, &seed40, &mu, Shake256Prng::new);
    let rep = match streams {
        Streams::Both => {
            let theirs = replay_sign(ek, &seed40, &mu, FnPrng::new);
            match (&theirs, &ours) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a.sig, b.sig, "{what}: the two PRNGs disagree");
                    assert_eq!((a.counter, &a.rejects), (b.counter, &b.rejects));
                    assert!(a.s == b.s && a.t == b.t && a.reads == b.reads);
                }
                (Err(a), Err(b)) => assert_eq!(a, b),
                _ => panic!("{what}: the two PRNGs disagree on exhaustion"),
            }
            theirs
        }
        Streams::Ours => ours,
    };
    let rep = match rep {
        Ok(r) => r,
        Err(rejects) => {
            assert!(
                !fn_ok,
                "{what}: fn-dsa signed, the replay exhausted 27 attempts"
            );
            tally.fndsa_failures += 1;
            tally.rejects(&rejects);
            return;
        }
    };
    assert!(fn_ok, "{what}: the replay signed, fn-dsa failed");

    // (1) The same signature, byte for byte.
    assert!(rep.sig == sig, "{what}: signatures differ");
    // (2) The same attempt: fn-dsa's nonce is the first 40 bytes of P_k.
    let fn_counter = (0..27u8)
        .find(|&k| {
            let mut s2 = [0u8; 41];
            s2[..40].copy_from_slice(&seed40);
            s2[40] = k;
            let mut nonce = [0u8; 40];
            FnPrng::new(&s2).next_bytes(&mut nonce);
            nonce[..] == sig[1..41]
        })
        .expect("fn-dsa's nonce comes from one of its attempts");
    assert_eq!(fn_counter, rep.counter, "{what}: attempt index");
    // (3) The same s = (s0, s1), bit for bit, and the same target.
    let (c, s0_fn, s1_fn) = fndsa_preimage(logn, &key.h_ntt, &mu, &sig);
    assert_eq!(rep.t.coeffs(), &c[..], "{what}: target");
    if let Some(d) = first_diff(&rep.s.s0, &s0_fn) {
        panic!("{what}: s0 differs at {d:?} (ours, fn-dsa)");
    }
    if let Some(d) = first_diff(&rep.s.s1, &s1_fn) {
        panic!("{what}: s1 differs at {d:?} (ours, fn-dsa)");
    }
    // (4) Both verifiers accept the common (t, s).
    assert!(verifier.verify(&rep.sig, &DOMAIN_NONE, &HASH_ID_RAW, msg));
    assert!(verify(&key.pk, &rep.t, &rep.s, params.beta_sq));
    tally.replay(&rep, params.n());
}

/// fn-dsa key pairs at `params.logn`, imported; `msgs` messages each.
fn run_fndsa_keys<KG: KeyPairGenerator, S: SigningKey, V: VerifyingKey>(
    params: &Params,
    keys: u32,
    msgs: u32,
    streams: Streams,
    label: &str,
) -> Tally {
    let logn = params.logn;
    let mut tally = Tally::new(label);
    for key_ix in 0..keys {
        let seed32: [u8; 32] = derive(&[b"parity/keygen", &[logn as u8], &key_ix.to_le_bytes()]);
        let (sk_enc, vk_enc) = fndsa_keypair::<KG>(logn, &seed32);
        let key = import_fndsa_key(params, &sk_enc, &vk_enc);
        let ek = match ExpandedKey::new(&key.sk) {
            Ok(ek) => ek,
            Err(e) => {
                eprintln!("[{label}] key {key_ix}: ExpandedKey::new refused it: {e}");
                tally.refused_keys += 1;
                continue;
            }
        };
        tally.key(&ek);
        let mut signer = S::decode(&key.sk_enc).expect("fn-dsa decodes its key");
        let verifier = V::decode(&key.vk_enc).expect("fn-dsa decodes its key");
        for msg_ix in 0..msgs {
            let msg = message(logn, key_ix, msg_ix);
            let raw40: [u8; 40] = derive(&[
                b"parity/sign",
                &[logn as u8],
                &key_ix.to_le_bytes(),
                &msg_ix.to_le_bytes(),
            ]);
            let what = format!("{label} key {key_ix} message {msg_ix}");
            check_message(
                &key,
                &ek,
                &mut signer,
                &verifier,
                &msg,
                &raw40,
                streams,
                &mut tally,
                &what,
            );
        }
    }
    tally.report();
    tally
}

// ---------------------------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------------------------

/// P-PRNG: this crate's PRNG and SHAKE256 against fn-dsa-comm's, under mixed reads across block
/// boundaries, for seeds of 0 to 272 bytes (41 bytes is the signer's `seed' || k`).
#[test]
fn p_prng_matches_fndsa() {
    let mut ops = Shake256Prng::new(b"parity/prng-ops");
    let mut reads = 0u64;
    for len in [0usize, 1, 31, 32, 40, 41, 64, 65, 135, 136, 137, 272] {
        for v in 0..4u8 {
            let mut seed = vec![0u8; len];
            fn_shake(
                &[b"parity/prng-seed", &[v], &(len as u32).to_le_bytes()],
                &mut seed,
            );
            // Seeded in one piece, and in two (`from_parts` is the concatenation, as SampPre's
            // `seed* || k`).
            let (a, b) = seed.split_at(len / 2);
            for mut ours in [Shake256Prng::new(&seed), Shake256Prng::from_parts(&[a, b])] {
                let mut theirs = FnPrng::new(&seed);
                for _ in 0..2500 {
                    match ops.next_u8() % 4 {
                        0 => assert_eq!(ours.next_u8(), theirs.next_u8()),
                        1 => assert_eq!(ours.next_u16(), theirs.next_u16()),
                        2 => assert_eq!(ours.next_u64(), theirs.next_u64()),
                        _ => {
                            let l = (ops.next_u16() % 300) as usize;
                            let (mut x, mut y) = (vec![0u8; l], vec![0u8; l]);
                            ours.next_bytes(&mut x);
                            theirs.next_bytes(&mut y);
                            assert_eq!(x, y);
                        }
                    }
                    reads += 1;
                }
            }
            // One-shot SHAKE256 with parts (the hedge, the key identifier, seed*).
            let mut o1 = vec![0u8; 200];
            let mut o2 = vec![0u8; 200];
            shake256(&[b"x", &seed, b"", &[v]], &mut o1);
            fn_shake(&[b"x", &seed, &[v]], &mut o2);
            assert_eq!(o1, o2);
        }
    }
    eprintln!("[P-PRNG] {reads} mixed reads, 48 seeds: identical streams");
}

/// P-1024: fn-dsa key pairs at Falcon-1024, 32 keys x 64 messages = 2048 messages; s bit for bit,
/// with fn-dsa's PRNG and with this crate's.
#[test]
fn p_1024_fndsa_keys() {
    let params = fndsa_params(10);
    let t = run_fndsa_keys::<KeyPairGenerator1024, SigningKey1024, VerifyingKey1024>(
        &params,
        32,
        64,
        Streams::Both,
        "P-1024",
    );
    assert_eq!(t.keys, 32);
    assert_eq!(t.messages, 2048);
    assert_eq!(t.fndsa_failures, 0);
}

/// P-OURKEYS: this crate's TrapGen keys at Falcon-1024, exported to fn-dsa's formats; fn-dsa
/// decodes them (recomputing G and the verifying key), signs, and the replay matches.
#[test]
fn p_1024_trapgen_keys() {
    let params = fndsa_params(10);
    let mut tally = Tally::new("P-OURKEYS");
    let mut skipped = 0;
    for key_ix in 0..16u32 {
        let seed: [u8; 32] = derive(&[b"parity/trapgen", &key_ix.to_le_bytes()]);
        let (pk, sk) = trapgen(&params, &seed).unwrap();
        let Some((sk_enc, vk_enc)) = export_to_fndsa(&sk, &pk) else {
            eprintln!("[P-OURKEYS] key {key_ix}: outside fn-dsa's encoding ranges, skipped");
            skipped += 1;
            continue;
        };
        let mut signer = SigningKey1024::decode(&sk_enc).expect("fn-dsa decodes our key");
        let mut vk_back = vec![0u8; vrfy_key_size(10)];
        signer.to_verifying_key(&mut vk_back);
        assert_eq!(vk_back, vk_enc, "fn-dsa recomputes our verifying key");
        let verifier = VerifyingKey1024::decode(&vk_enc).unwrap();
        let key = import_fndsa_key(&params, &sk_enc, &vk_enc);
        assert!(key.sk.f() == sk.f() && key.sk.g() == sk.g());
        assert!(key.sk.big_f() == sk.big_f() && key.sk.big_g() == sk.big_g());
        assert_eq!(key.pk, pk);
        let ek = ExpandedKey::new(&sk).unwrap();
        tally.key(&ek);
        for msg_ix in 0..32u32 {
            let msg = message(10, 1000 + key_ix, msg_ix);
            let raw40: [u8; 40] = derive(&[
                b"parity/sign-ours",
                &key_ix.to_le_bytes(),
                &msg_ix.to_le_bytes(),
            ]);
            let what = format!("P-OURKEYS key {key_ix} message {msg_ix}");
            check_message(
                &key,
                &ek,
                &mut signer,
                &verifier,
                &msg,
                &raw40,
                Streams::Both,
                &mut tally,
                &what,
            );
        }
    }
    tally.report();
    eprintln!("[P-OURKEYS] {skipped} of 16 keys outside fn-dsa's encoding ranges");
    assert!(tally.keys >= 12, "too few exportable keys: {}", tally.keys);
}

/// P-TOY: fn-dsa's degrees 4 to 512 (`logn = 2..9`); at small degrees fn-dsa's norm check rejects
/// often, so the core's reject-and-resample branch runs in lockstep with fn-dsa's.
#[test]
fn p_toy_degrees() {
    let mut norm_rejects = 0;
    for logn in 2..=9u32 {
        let params = fndsa_params(logn);
        let label = format!("P-TOY logn {logn}");
        let t = if logn <= 8 {
            run_fndsa_keys::<KeyPairGeneratorWeak, SigningKeyWeak, VerifyingKeyWeak>(
                &params,
                8,
                64,
                Streams::Both,
                &label,
            )
        } else {
            run_fndsa_keys::<KeyPairGenerator512, SigningKey512, VerifyingKey512>(
                &params,
                8,
                64,
                Streams::Both,
                &label,
            )
        };
        assert!(t.keys >= 6, "{label}: too few usable keys");
        norm_rejects += t.norm;
    }
    assert!(
        norm_rejects > 100,
        "the norm-reject branch was barely exercised"
    );
}

/// The key identifier of design.md §5.1, re-derived with fn-dsa's SHAKE256:
/// SHAKE256("ntru-trapdoor/v1/key-id" || q (LE32) || logn || f, g, F, G as i16 LE), 32 bytes.
fn key_id(sk: &SecretKey) -> [u8; 32] {
    let p = sk.params();
    let mut enc = Vec::new();
    for poly in [sk.f(), sk.g(), sk.big_f(), sk.big_g()] {
        for &c in poly {
            enc.extend_from_slice(&c.to_le_bytes());
        }
    }
    derive(&[
        b"ntru-trapdoor/v1/key-id",
        &p.q.to_le_bytes(),
        &[p.logn as u8],
        &enc,
    ])
}

/// The public `samp_pre` by its specification (src/sign.rs, design.md §5.1), on fn-dsa's PRNG:
/// seed* = SHAKE256("ntru-trapdoor/v1/samp-pre" || key_id || encode(t) || seed), 64 bytes, and
/// attempt k runs the core on SHAKE256(seed* || k). `encode(t)` is fn-dsa's `modq_encode` here.
fn samp_pre_by_spec(
    ek: &ExpandedKey,
    kid: &[u8; 32],
    t: &RqPoly,
    seed: &[u8; 32],
) -> (Preimage, u32) {
    let p = ek.params();
    let tc: Vec<u16> = t.coeffs().iter().map(|&x| x as u16).collect();
    let mut enc = vec![0u8; (p.n() * 14).div_ceil(8)];
    modq_encode(&tc, &mut enc);
    assert_eq!(enc, t.to_bytes(), "encode(t) is modq_encode at q = 12289");
    let mut seed_star = [0u8; 64];
    fn_shake(
        &[b"ntru-trapdoor/v1/samp-pre", kid, &enc, seed],
        &mut seed_star,
    );
    for k in 0..p.samppre_max_attempts.min(256) {
        let mut sk = seed_star.to_vec();
        sk.push(k as u8);
        if let Some(s) = samp_pre_attempt(ek, t, &mut FnPrng::new(&sk)) {
            return (s, k);
        }
    }
    panic!("samp_pre_by_spec: attempts exhausted");
}

/// P-PUB: the public `samp_pre` is the core on the documented stream, attempt by attempt, at
/// Falcon-1024 and at toy degrees (where its attempt loop is exercised).
#[test]
fn p_public_samp_pre_is_core_on_documented_stream() {
    let mut retried = 0;
    let mut calls = 0;
    for logn in [3u32, 4, 5, 6, 10] {
        let params = fndsa_params(logn);
        let n = params.n();
        for key_ix in 0..3u32 {
            let seed32: [u8; 32] =
                derive(&[b"parity/pub-keygen", &[logn as u8], &key_ix.to_le_bytes()]);
            let (sk_enc, vk_enc) = if logn == 10 {
                fndsa_keypair::<KeyPairGenerator1024>(logn, &seed32)
            } else {
                fndsa_keypair::<KeyPairGeneratorWeak>(logn, &seed32)
            };
            let key = import_fndsa_key(&params, &sk_enc, &vk_enc);
            let Ok(ek) = ExpandedKey::new(&key.sk) else {
                continue;
            };
            let kid = key_id(&key.sk);
            let mut targets = vec![
                RqPoly::zero(&params).unwrap(),
                RqPoly::from_coeffs(&params, &vec![params.q - 1; n]).unwrap(),
            ];
            let mut tp = Shake256Prng::new(&[b'T', logn as u8, key_ix as u8]);
            for _ in 0..30 {
                let c: Vec<u32> = (0..n)
                    .map(|_| (tp.next_u64() % params.q as u64) as u32)
                    .collect();
                targets.push(RqPoly::from_coeffs(&params, &c).unwrap());
            }
            for (j, t) in targets.iter().enumerate() {
                let seed: [u8; 32] =
                    derive(&[b"parity/pub-seed", &[logn as u8, key_ix as u8, j as u8]]);
                let s = samp_pre(&ek, t, &seed).unwrap();
                let (s_spec, k) = samp_pre_by_spec(&ek, &kid, t, &seed);
                assert!(s == s_spec, "logn {logn} key {key_ix} target {j}");
                assert!(verify(&key.pk, t, &s, params.beta_sq));
                retried += (k > 0) as u32;
                calls += 1;
            }
        }
    }
    eprintln!("[P-PUB] {calls} samp_pre calls equal the specified stream; {retried} needed k > 0");
    assert!(retried > 10, "the public attempt loop was barely exercised");
}

/// Encodes this crate's preimage of `hash_to_point(nonce, mu)` as an fn-dsa signature, if fn-dsa's
/// format conditions hold (|s0_i| <= 840, compressed s1 fits).
fn as_fndsa_signature(logn: u32, nonce: &[u8; 40], s: &Preimage) -> Option<Vec<u8>> {
    if s.s0.iter().any(|&z| z.abs() > B_INF) {
        return None;
    }
    let s2: Vec<i16> = s.s1.iter().map(|&z| z as i16).collect();
    let mut sig = vec![0u8; signature_size(logn)];
    if !comp_encode(&s2, &mut sig[41..]) {
        return None;
    }
    sig[0] = 0x30 + logn as u8;
    sig[1..41].copy_from_slice(nonce);
    Some(sig)
}

/// P-VER: fn-dsa verifies signatures made with the public `samp_pre` (this crate's seeds), on
/// fn-dsa keys and on TrapGen keys; a perturbed s1 fails in both verifiers.
#[test]
fn p_fndsa_verifies_public_samp_pre() {
    let params = fndsa_params(10);
    let mut checked = 0;
    let mut format_retries = 0;
    for key_ix in 0..16u32 {
        // Keys 0..8 from fn-dsa, 8..16 from this crate's TrapGen (as fn-dsa verifying keys).
        let (pk, sk, vk_enc) = if key_ix < 8 {
            let seed32: [u8; 32] = derive(&[b"parity/ver-keygen", &key_ix.to_le_bytes()]);
            let (sk_enc, vk_enc) = fndsa_keypair::<KeyPairGenerator1024>(10, &seed32);
            let key = import_fndsa_key(&params, &sk_enc, &vk_enc);
            (key.pk.clone(), key.sk.clone(), vk_enc)
        } else {
            let seed: [u8; 32] = derive(&[b"parity/ver-trapgen", &key_ix.to_le_bytes()]);
            let (pk, sk) = trapgen(&params, &seed).unwrap();
            let mut h: Vec<u16> = pk.h().coeffs().iter().map(|&x| x as u16).collect();
            mqpoly_ext_to_int(10, &mut h);
            mqpoly_int_to_NTT(10, &mut h);
            mqpoly_int_to_ext(10, &mut h);
            let mut vk = vec![0u8; vrfy_key_size(10)];
            vk[0] = 10;
            modq_encode(&h, &mut vk[1..]);
            (pk, sk, vk)
        };
        let ek = ExpandedKey::new(&sk).unwrap();
        let verifier = VerifyingKey1024::decode(&vk_enc).unwrap();
        let hvk = hashed_vrfykey_from_vrfykey(&vk_enc);
        for msg_ix in 0..32u32 {
            let msg = message(10, 2000 + key_ix, msg_ix);
            let mu = compute_mu(&hvk, &DOMAIN_NONE, &HASH_ID_RAW, &msg);
            let mut done = false;
            for attempt in 0..16u32 {
                let nonce: [u8; 40] = derive(&[
                    b"parity/ver-nonce",
                    &key_ix.to_le_bytes(),
                    &msg_ix.to_le_bytes(),
                    &attempt.to_le_bytes(),
                ]);
                let mut hm = vec![0u16; 1024];
                hash_to_point(&nonce, &mu, &mut hm);
                let t =
                    RqPoly::from_coeffs(&params, &hm.iter().map(|&x| x as u32).collect::<Vec<_>>())
                        .unwrap();
                let seed: [u8; 32] = derive(&[b"parity/ver-seed", &nonce]);
                let s = samp_pre(&ek, &t, &seed).unwrap();
                assert!(verify(&pk, &t, &s, params.beta_sq));
                let Some(sig) = as_fndsa_signature(10, &nonce, &s) else {
                    format_retries += 1;
                    continue;
                };
                assert!(
                    verifier.verify(&sig, &DOMAIN_NONE, &HASH_ID_RAW, &msg),
                    "fn-dsa rejected key {key_ix} message {msg_ix}"
                );
                assert!(!verifier.verify(&sig, &DOMAIN_NONE, &HASH_ID_RAW, b"another message"));
                // Perturb one coefficient of s1: both verifiers reject.
                let mut bad = s.clone();
                let i = (msg_ix as usize * 37) % 1024;
                bad.s1[i] += if bad.s1[i] < 0 { 1 } else { -1 };
                assert!(!verify(&pk, &t, &bad, params.beta_sq));
                if let Some(bad_sig) = as_fndsa_signature(10, &nonce, &bad) {
                    assert!(!verifier.verify(&bad_sig, &DOMAIN_NONE, &HASH_ID_RAW, &msg));
                }
                checked += 1;
                done = true;
                break;
            }
            assert!(done, "no encodable preimage in 16 nonces");
        }
    }
    eprintln!(
        "[P-VER] fn-dsa verified {checked} signatures made with samp_pre (8 fn-dsa keys, 8 TrapGen \
         keys); {format_retries} preimages failed fn-dsa's format checks and got a new nonce"
    );
    assert_eq!(checked, 512);
}

/// P-VKAT: fn-dsa's verification KATs (fn-dsa-vrfy/src/lib.rs:389-914) pass this crate's
/// `verify` after converting the key (inverse NTT) and the signature (s1 decoded, s0 = c - s1 h).
#[test]
fn p_fndsa_verify_kats() {
    let fx = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/parity/fndsa_vrfy_kat.txt"
    ))
    .unwrap();
    let get = |k: &str| {
        let line = fx
            .lines()
            .find(|l| l.starts_with(&format!("{k} ")))
            .unwrap();
        hex::decode(line.split_whitespace().nth(1).unwrap()).unwrap()
    };
    for (logn, deg) in [(9u32, 512), (10, 1024)] {
        let params = fndsa_params(logn);
        let n = params.n();
        let vk = get(&format!("vk{deg}"));
        let mu: [u8; 64] = get(&format!("mu{deg}")).try_into().unwrap();
        let sig = get(&format!("sig{deg}"));
        // mu is the message representative of b"message" under the context b"context".
        let hvk = hashed_vrfykey_from_vrfykey(&vk);
        assert_eq!(
            compute_mu(&hvk, &DomainContext(b"context"), &HASH_ID_RAW, b"message"),
            mu
        );
        // h, in NTT (internal) and coefficient form.
        let mut h_ntt = vec![0u16; n];
        modq_decode(&vk[1..], &mut h_ntt).unwrap();
        mqpoly_ext_to_int(logn, &mut h_ntt);
        let mut hc = h_ntt.clone();
        mqpoly_NTT_to_int(logn, &mut hc);
        mqpoly_int_to_ext(logn, &mut hc);
        let pk = PublicKey::from_h(
            &params,
            RqPoly::from_coeffs(&params, &hc.iter().map(|&x| x as u32).collect::<Vec<_>>())
                .unwrap(),
        )
        .unwrap();
        let (c, s0, s1) = fndsa_preimage(logn, &h_ntt, &mu, &sig);
        let t = RqPoly::from_coeffs(&params, &c).unwrap();
        let s = Preimage { s0, s1 };
        let norm = s.norm_sq();
        assert!(verify(&pk, &t, &s, params.beta_sq), "KAT {deg}");
        assert!(verify(&pk, &t, &s, norm) && !verify(&pk, &t, &s, norm - 1));
        // fn-dsa agrees (sanity) and both reject a perturbed s1.
        let verifier_ok = if logn == 10 {
            VerifyingKey1024::decode(&vk).unwrap().verify(
                &sig,
                &DomainContext(b"context"),
                &HASH_ID_RAW,
                b"message",
            )
        } else {
            VerifyingKey512::decode(&vk).unwrap().verify(
                &sig,
                &DomainContext(b"context"),
                &HASH_ID_RAW,
                b"message",
            )
        };
        assert!(verifier_ok);
        let mut bad = s.clone();
        bad.s1[0] += 1;
        assert!(!verify(&pk, &t, &bad, params.beta_sq));
        eprintln!("[P-VKAT] fn-dsa's {deg} KAT: ||s||^2 = {norm}, accepted by verify");
    }
}

/// P-SENS: what output parity can detect. The replay runs with deliberately perturbed constants
/// (1/sigma, which scales every leaf width 1/sigma'_i, and sigma_min, which scales BerExp's ccs)
/// against unmodified fn-dsa signatures, on 4 keys x 128 messages, and counts the messages whose
/// signature changes. Perturbations go downwards so that every leaf stays in the base sampler's
/// range. A perturbation that changes no signature is invisible to P-1024 and P-LONG.
#[test]
#[ignore = "an experiment, about 2 s: cargo test --test parity_fndsa_diff -- --ignored p_sensitivity"]
fn p_sensitivity() {
    let base = fndsa_params(10);
    let with_inv_sigma = |x: f64| Params {
        inv_sigma: x,
        sigma: 1.0 / x,
        ..base
    };
    let shrink = |x: f64, e: i32| x * (1.0 - 2f64.powi(-e));
    let variants: Vec<(String, Params)> = vec![
        ("none (control)".into(), base),
        (
            "1/sigma: 1 ulp down".into(),
            with_inv_sigma(base.inv_sigma.next_down()),
        ),
        (
            "1/sigma: 2^-40 down".into(),
            with_inv_sigma(shrink(base.inv_sigma, 40)),
        ),
        (
            "1/sigma: 2^-30 down".into(),
            with_inv_sigma(shrink(base.inv_sigma, 30)),
        ),
        (
            "1/sigma: 2^-24 down".into(),
            with_inv_sigma(shrink(base.inv_sigma, 24)),
        ),
        (
            "1/sigma: 2^-20 down".into(),
            with_inv_sigma(shrink(base.inv_sigma, 20)),
        ),
        (
            "1/sigma: 2^-16 down".into(),
            with_inv_sigma(shrink(base.inv_sigma, 16)),
        ),
        (
            "1/sigma: 2^-10 down".into(),
            with_inv_sigma(shrink(base.inv_sigma, 10)),
        ),
        (
            "sigma_min: 1 ulp down".into(),
            Params {
                sigma_min: base.sigma_min.next_down(),
                ..base
            },
        ),
        (
            "sigma_min: 2^-20 down".into(),
            Params {
                sigma_min: shrink(base.sigma_min, 20),
                ..base
            },
        ),
    ];
    let mut changed = vec![0u32; variants.len()];
    let mut total = 0u32;
    for key_ix in 0..4u32 {
        let seed32: [u8; 32] = derive(&[b"parity/sens-keygen", &key_ix.to_le_bytes()]);
        let (sk_enc, vk_enc) = fndsa_keypair::<KeyPairGenerator1024>(10, &seed32);
        let key = import_fndsa_key(&base, &sk_enc, &vk_enc);
        let eks: Vec<ExpandedKey> = variants
            .iter()
            .map(|(_, p)| {
                let sk = SecretKey::from_parts(
                    p,
                    key.sk.f().to_vec(),
                    key.sk.g().to_vec(),
                    key.sk.big_f().to_vec(),
                    key.sk.big_g().to_vec(),
                )
                .unwrap();
                ExpandedKey::new(&sk).unwrap()
            })
            .collect();
        let mut signer = SigningKey1024::decode(&sk_enc).unwrap();
        for msg_ix in 0..128u32 {
            let msg = message(10, 3000 + key_ix, msg_ix);
            let raw40: [u8; 40] = derive(&[
                b"parity/sens-sign",
                &key_ix.to_le_bytes(),
                &msg_ix.to_le_bytes(),
            ]);
            let mu = compute_mu(&key.hashed_vk, &DOMAIN_NONE, &HASH_ID_RAW, &msg);
            let mut sig = vec![0u8; signature_size(10)];
            signer
                .sign(
                    &mut OneShotRng::new(&raw40),
                    &DOMAIN_NONE,
                    &HASH_ID_RAW,
                    &msg,
                    &mut sig,
                )
                .unwrap();
            let mut seed40 = [0u8; 40];
            fn_shake(&[&key.hashed_sign_key, &mu, &raw40], &mut seed40);
            for (v, ek) in eks.iter().enumerate() {
                let same = replay_sign(ek, &seed40, &mu, Shake256Prng::new)
                    .map(|r| r.sig == sig)
                    .unwrap_or(false);
                changed[v] += (!same) as u32;
            }
            total += 1;
        }
    }
    for ((name, _), c) in variants.iter().zip(changed.iter()) {
        eprintln!("[P-SENS] {name:<24} signatures changed: {c:>3} / {total}");
    }
    assert_eq!(changed[0], 0, "the control must match");
}

/// P-LONG: 128 fn-dsa keys x 512 messages = 65,536 messages at Falcon-1024, this crate's PRNG.
#[test]
#[ignore = "about 40 s: cargo test --test parity_fndsa_diff -- --ignored p_long_1024"]
fn p_long_1024() {
    let params = fndsa_params(10);
    let t = run_fndsa_keys::<KeyPairGenerator1024, SigningKey1024, VerifyingKey1024>(
        &params,
        128,
        512,
        Streams::Ours,
        "P-LONG",
    );
    assert_eq!(t.messages, 65_536);
}
