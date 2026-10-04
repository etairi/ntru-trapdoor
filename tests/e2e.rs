//! End to end (design.md §7):
//! - K-E2E-1: `trapgen` → `ExpandedKey::new` → `samp_pre` on 16 targets → `verify`, with
//!   $`s_0+h\,s_1=t`$ checked through the ring and $`\|s\|^2\le\lfloor\beta^2\rfloor`$, at both
//!   presets;
//! - K-E2E-2: Falcon-1024 across implementations. Our key becomes an fn-dsa verifying key,
//!   and our SampPre on fn-dsa's `hash_to_point(nonce, μ)` becomes an fn-dsa signature, which
//!   fn-dsa 0.4.0 verifies.

use fn_dsa::{
    DOMAIN_NONE, HASH_ID_RAW, VerifyingKey, VerifyingKey1024, compute_mu,
    hashed_vrfykey_from_vrfykey, signature_size, vrfy_key_size,
};
use fn_dsa_comm::codec::{B_INF, comp_encode, modq_encode};
use fn_dsa_comm::hash_to_point;
use fn_dsa_comm::mq::{mqpoly_ext_to_int, mqpoly_int_to_NTT, mqpoly_int_to_ext};
use ntru_trapdoor::hazmat::{Prng, Shake256Prng, samp_pre_attempt};
use ntru_trapdoor::{ExpandedKey, Params, PublicKey, RqPoly, samp_pre, trapgen, verify};

/// Deterministic test targets: 0, all q − 1, and 14 pseudorandom ones.
fn targets(p: &Params) -> Vec<RqPoly> {
    let n = p.n();
    let mut out = vec![
        RqPoly::zero(p).unwrap(),
        RqPoly::from_coeffs(p, &vec![p.q - 1; n]).unwrap(),
    ];
    for k in 0..14u8 {
        let mut prng = Shake256Prng::new(&[b't', k]);
        let c: Vec<u32> = (0..n)
            .map(|_| (prng.next_u64() % p.q as u64) as u32)
            .collect();
        out.push(RqPoly::from_coeffs(p, &c).unwrap());
    }
    out
}

fn check_preimage(pk: &PublicKey, t: &RqPoly, s: &ntru_trapdoor::Preimage) {
    let p = pk.params();
    assert!(verify(pk, t, s, p.beta_sq));
    assert!(s.norm_sq() <= p.beta_sq);
    let s0 = RqPoly::from_i32(p, &s.s0).unwrap();
    let s1 = RqPoly::from_i32(p, &s.s1).unwrap();
    assert_eq!(&s0 + &(pk.h() * &s1), *t);
}

/// K-E2E-1.
#[test]
fn trapgen_samp_pre_verify() {
    for p in [Params::PCS, Params::FALCON_1024] {
        let (pk, sk) = trapgen(&p, &[0x77; 32]).unwrap();
        let ek = ExpandedKey::new(&sk).unwrap();
        let mut ratio_sum = 0.0;
        let ts = targets(&p);
        for (k, t) in ts.iter().enumerate() {
            let s = samp_pre(&ek, t, &[k as u8; 32]).unwrap();
            check_preimage(&pk, t, &s);
            // The same seed gives the same preimage; another seed another one.
            assert_eq!(samp_pre(&ek, t, &[k as u8; 32]).unwrap(), s);
            assert_ne!(samp_pre(&ek, t, &[k as u8 ^ 0x80; 32]).unwrap(), s);
            ratio_sum += s.norm_sq() as f64 / (2.0 * p.n() as f64 * p.sigma * p.sigma);
        }
        let mean = ratio_sum / ts.len() as f64;
        eprintln!(
            "K-E2E-1 {}: mean ||s||^2 / (2 d sigma^2) = {mean:.4}",
            p.name
        );
        // E||s||^2 = 2 d sigma^2; over 16 samples the mean is within a few percent.
        assert!((mean - 1.0).abs() < 0.05, "{}: {mean}", p.name);
    }
}

/// Our h as an fn-dsa 0.4.0 verifying key: header `logn`, then `modq_encode` of h in fn-dsa's
/// NTT representation (the inverse of the conversion in tests/parity_fndsa.rs).
fn fndsa_vk(pk: &PublicKey) -> Vec<u8> {
    let p = pk.params();
    let logn = p.logn;
    let mut h: Vec<u16> = pk.h().coeffs().iter().map(|&x| x as u16).collect();
    mqpoly_ext_to_int(logn, &mut h);
    mqpoly_int_to_NTT(logn, &mut h);
    mqpoly_int_to_ext(logn, &mut h);
    let mut vk = vec![0u8; vrfy_key_size(logn)];
    vk[0] = logn as u8;
    modq_encode(&h, &mut vk[1..]);
    vk
}

/// K-E2E-2.
#[test]
fn falcon1024_keys_and_preimages_verify_in_fndsa() {
    let p = Params::FALCON_1024;
    for key_ix in 0..3u8 {
        let (pk, sk) = trapgen(&p, &[key_ix ^ 0x3c; 32]).unwrap();
        let ek = ExpandedKey::new(&sk).unwrap();
        let vk = fndsa_vk(&pk);
        let verifier = VerifyingKey1024::decode(&vk).expect("fn-dsa decodes our key");
        let hvk = hashed_vrfykey_from_vrfykey(&vk);
        for msg_ix in 0..8u8 {
            let msg = [b'e', key_ix, msg_ix];
            let mu = compute_mu(&hvk, &DOMAIN_NONE, &HASH_ID_RAW, &msg);
            let mut done = false;
            for attempt in 0..16u8 {
                let mut prng = Shake256Prng::new(&[b'n', key_ix, msg_ix, attempt]);
                let mut nonce = [0u8; 40];
                prng.next_bytes(&mut nonce);
                let mut hm = vec![0u16; p.n()];
                hash_to_point(&nonce, &mu, &mut hm);
                let hm32: Vec<u32> = hm.iter().map(|&x| x as u32).collect();
                let t = RqPoly::from_coeffs(&p, &hm32).unwrap();
                let Some(s) = samp_pre_attempt(&ek, &t, &mut prng) else {
                    continue;
                };
                check_preimage(&pk, &t, &s);
                // fn-dsa's signature-format conditions (fn-dsa-sign/src/lib.rs:697-743).
                if s.s0.iter().any(|&z| z.abs() > B_INF) {
                    continue;
                }
                let s2: Vec<i16> = s.s1.iter().map(|&z| z as i16).collect();
                let mut sig = vec![0u8; signature_size(10)];
                if !comp_encode(&s2, &mut sig[41..]) {
                    continue;
                }
                sig[0] = 0x30 + 10;
                sig[1..41].copy_from_slice(&nonce);
                assert!(
                    verifier.verify(&sig, &DOMAIN_NONE, &HASH_ID_RAW, &msg),
                    "fn-dsa rejected key {key_ix} message {msg_ix}"
                );
                // A different message must fail.
                assert!(!verifier.verify(&sig, &DOMAIN_NONE, &HASH_ID_RAW, b"other"));
                done = true;
                break;
            }
            assert!(done, "no encodable signature in 16 attempts");
        }
    }
}
