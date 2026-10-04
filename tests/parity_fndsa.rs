//! Bit-exact parity of SampPre with Pornin's fn-dsa (design.md §7: S-PAR-2, S-PAR-3).
//!
//! fn-dsa's signing loop (fn-dsa-sign/src/lib.rs:482-744) is replayed with this crate's
//! `samp_pre_attempt` in place of the ffLDL/ffSampling core: per attempt `k`, the PRNG is
//! SHAKE256(seed40 || k), the first 40 bytes are the nonce, the target is
//! `hash_to_point(nonce, mu)`, and the remaining stream feeds the sampler. fn-dsa's
//! signature-format conditions (|s0_i| <= 840, and the compressed encoding of s1 fits) are
//! applied to our output, as in lib.rs:697-743. The resulting signature must equal fn-dsa's
//! byte for byte, which exercises the FFT, the Gram matrix, the tree, the target transform,
//! ffSampling, SamplerZ (with fn-dsa's base sampler), the PRNG and the rounding.
//!
//! Note: fn-dsa 0.4.0's verifying key stores h in fn-dsa's NTT representation, not in
//! coefficient form; [`fndsa_vk_to_h`] converts it.
//!
//! Run: `cargo test --test parity_fndsa`.

use fn_dsa::{
    CryptoRng, DOMAIN_NONE, HASH_ID_RAW, KeyPairGenerator, KeyPairGenerator1024, RngCore, RngError,
    SigningKey, SigningKey1024, VerifyingKey, VerifyingKey1024, compute_mu,
    hashed_vrfykey_from_vrfykey, sign_key_size, signature_size, vrfy_key_size,
};
use fn_dsa_comm::codec::{B_INF, comp_encode, modq_decode, trim_i8_decode};
use fn_dsa_comm::hash_to_point;
use fn_dsa_comm::mq::{SQBETA, mqpoly_NTT_to_int, mqpoly_ext_to_int, mqpoly_int_to_ext};
use ntru_trapdoor::hazmat::{Prng, Shake256Prng, samp_pre_attempt, shake256};
use ntru_trapdoor::{BaseSampler, ExpandedKey, Params, RqPoly, SecretKey};

/// `FLR::scaled(x, sc)` = x * 2^sc, exact.
fn scaled(x: i64, sc: i32) -> f64 {
    (x as f64) * f64::from_bits(((1023 + sc) as u64) << 52)
}

/// Falcon-512's signing parameters with fn-dsa's constants: INV_SIGMA[9], SIGMA_MIN[9]
/// (fn-dsa-sign/src/sampler.rs:59, :72), SQBETA[9] (fn-dsa-comm/src/mq.rs:72-84).
fn falcon512() -> Params {
    let inv_sigma = scaled(6956347512113097, -60);
    Params {
        name: "Falcon-512 (fn-dsa constants)",
        logn: 9,
        sigma: 1.0 / inv_sigma,
        inv_sigma,
        sigma_min: scaled(5754851361258101, -52),
        base: BaseSampler::Falcon1_8205,
        beta_sq: SQBETA[9] as u64,
        ..Params::FALCON_1024
    }
}

/// fn-dsa's constants equal the preset's (Params::FALCON_1024 claims bit equality).
#[test]
fn falcon1024_preset_matches_fndsa_constants() {
    let p = Params::FALCON_1024;
    assert_eq!(
        p.inv_sigma.to_bits(),
        scaled(6846791885593314, -60).to_bits()
    );
    assert_eq!(
        p.sigma_min.to_bits(),
        scaled(5846934829975396, -52).to_bits()
    );
    assert_eq!(p.inv_q.to_bits(), scaled(6004310871091074, -66).to_bits());
    assert_eq!(
        BaseSampler::Falcon1_8205.inv_2sqr_sigma0().to_bits(),
        scaled(5435486223186882, -55).to_bits()
    );
    assert_eq!(p.beta_sq, SQBETA[10] as u64);
    falcon512().validate().unwrap();
}

/// A decoded fn-dsa signing key: (f, g, F, G) and the hash of the (f, g) encoding that
/// fn-dsa uses for hedging (fn-dsa-sign/src/lib.rs:343-454).
struct FnDsaKey {
    sk: SecretKey,
    hashed_sign_key: [u8; 40],
}

fn decode_fndsa_key(params: &Params, src: &[u8]) -> FnDsaKey {
    let logn = params.logn;
    assert_eq!(src[0], 0x50 + logn as u8);
    assert_eq!(src.len(), sign_key_size(logn));
    let n = params.n();
    let nbits_fg = match logn {
        2..=5 => 8,
        6..=7 => 7,
        8..=9 => 6,
        _ => 5,
    };
    let (mut f, mut g, mut big_f) = (vec![0i8; n], vec![0i8; n], vec![0i8; n]);
    let j = 1 + trim_i8_decode(&src[1..], &mut f, nbits_fg).unwrap();
    let j = j + trim_i8_decode(&src[j..], &mut g, nbits_fg).unwrap();
    let mut hashed_sign_key = [0u8; 40];
    shake256(&[&src[1..j]], &mut hashed_sign_key);
    let _ = trim_i8_decode(&src[j..], &mut big_f, 8).unwrap();
    let f: Vec<i16> = f.iter().map(|&x| x as i16).collect();
    let g: Vec<i16> = g.iter().map(|&x| x as i16).collect();
    let big_f: Vec<i16> = big_f.iter().map(|&x| x as i16).collect();
    // G = g F / f mod q (fn-dsa recomputes it the same way, lib.rs:411-450).
    let fr = RqPoly::from_i16(params, &f).unwrap();
    let gr = RqPoly::from_i16(params, &g).unwrap();
    let br = RqPoly::from_i16(params, &big_f).unwrap();
    let big_g: Vec<i16> = gr
        .mul(&br)
        .mul(&fr.inverse().expect("f invertible"))
        .to_centered()
        .iter()
        .map(|&x| x as i16)
        .collect();
    // Exact check f G - g F = q over Z[x]/(x^n + 1).
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
    assert_eq!(e[0], params.q as i64);
    assert!(e[1..].iter().all(|&x| x == 0));
    FnDsaKey {
        sk: SecretKey::from_parts(params, f, g, big_f, big_g).unwrap(),
        hashed_sign_key,
    }
}

/// fn-dsa's `sign_inner` loop with our SampPre core; returns the signature bytes.
fn replay_sign(ek: &ExpandedKey, seed40: &[u8; 40], mu: &[u8; 64]) -> Vec<u8> {
    let params = ek.params();
    let logn = params.logn;
    let n = params.n();
    for counter in 0..27u8 {
        let mut seed2 = [0u8; 41];
        seed2[..40].copy_from_slice(seed40);
        seed2[40] = counter;
        let mut prng = Shake256Prng::new(&seed2);
        let mut nonce = [0u8; 40];
        prng.next_bytes(&mut nonce);
        let mut hm = vec![0u16; n];
        hash_to_point(&nonce, mu, &mut hm);
        let hm32: Vec<u32> = hm.iter().map(|&x| x as u32).collect();
        let t = RqPoly::from_coeffs(params, &hm32).unwrap();
        let Some(s) = samp_pre_attempt(ek, &t, &mut prng) else {
            continue;
        };
        // fn-dsa's extra conditions (lib.rs:697-743).
        if s.s0.iter().any(|&z| z.abs() > B_INF) {
            continue;
        }
        let s2: Vec<i16> = s.s1.iter().map(|&z| z as i16).collect();
        let mut sig = vec![0u8; signature_size(logn)];
        if comp_encode(&s2, &mut sig[41..]) {
            sig[0] = 0x30 + logn as u8;
            sig[1..41].copy_from_slice(&nonce);
            return sig;
        }
    }
    panic!("27 attempts failed");
}

/// S-PAR-2: fn-dsa's `sign_512` KAT (fn-dsa-sign/src/lib.rs:784-1151), replayed at logn 9:
/// same key, same derived seed and message representative, same signature.
#[test]
fn fndsa_sign512_kat() {
    let fx = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sign/fndsa_sign512_kat.txt"
    ))
    .unwrap();
    let get = |k: &str| {
        let line = fx
            .lines()
            .find(|l| l.starts_with(&format!("{k} ")))
            .unwrap();
        hex::decode(line.split_whitespace().nth(1).unwrap()).unwrap()
    };
    let params = falcon512();
    let key = decode_fndsa_key(&params, &get("sk"));
    // The verifying key holds h = g / f in fn-dsa's NTT representation.
    let h = key.sk.public_key_h_for_tests(&params);
    assert_eq!(fndsa_vk_to_h(&params, &get("vk")), h);
    // Hedged seed (lib.rs:289-295) from the raw seed.
    let mu: [u8; 64] = get("mu").try_into().unwrap();
    let mut derived = [0u8; 40];
    shake256(&[&key.hashed_sign_key, &mu, &get("seed")], &mut derived);
    assert_eq!(derived.to_vec(), get("seed_derived"));
    let ek = ExpandedKey::new(&key.sk).unwrap();
    let sig = replay_sign(&ek, &derived, &mu);
    assert_eq!(hex::encode(sig), hex::encode(get("sig")));
}

/// h = g / f mod q from a secret key, through the public ring API (the keygen side's
/// `SecretKey::public_key` is not used here).
trait PublicH {
    fn public_key_h_for_tests(&self, params: &Params) -> RqPoly;
}
impl PublicH for SecretKey {
    fn public_key_h_for_tests(&self, params: &Params) -> RqPoly {
        let f = RqPoly::from_i16(params, self.f()).unwrap();
        let g = RqPoly::from_i16(params, self.g()).unwrap();
        g.mul(&f.inverse().unwrap())
    }
}

/// h in coefficient form from an fn-dsa verifying key. fn-dsa 0.4.0 stores h in its NTT
/// representation (fn-dsa-kgen `mqpoly_div_small_nttx`; fn-dsa-vrfy `decode_inner` feeds the
/// decoded key straight into `mqpoly_mul_ntt`), so the key is `header || modq_encode(NTT(h))`,
/// not `header || modq_encode(h)`; fn-dsa-comm's public mq functions undo the NTT.
fn fndsa_vk_to_h(params: &Params, vk: &[u8]) -> RqPoly {
    let logn = params.logn;
    assert_eq!(vk[0], logn as u8);
    let mut h = vec![0u16; params.n()];
    modq_decode(&vk[1..], &mut h).unwrap();
    mqpoly_ext_to_int(logn, &mut h);
    mqpoly_NTT_to_int(logn, &mut h);
    mqpoly_int_to_ext(logn, &mut h);
    let h: Vec<u32> = h.iter().map(|&x| x as u32).collect();
    RqPoly::from_coeffs(params, &h).unwrap()
}

/// A deterministic `rand_core` 0.6 RNG over this crate's SHAKE256 stream (fn-dsa's FakeRng1
/// pattern, fn-dsa/src/lib.rs:136-157).
struct TestRng(Shake256Prng);
impl CryptoRng for TestRng {}
impl RngCore for TestRng {
    fn next_u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        self.0.next_bytes(&mut b);
        u32::from_le_bytes(b)
    }
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.next_bytes(dest);
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RngError> {
        self.fill_bytes(dest);
        Ok(())
    }
}

/// S-PAR-3: live parity at logn 10. fn-dsa generates 8 keys and signs 32 messages per key
/// with deterministic RNGs; our replay produces the same signatures byte for byte. Also checks
/// fn-dsa's verifying key against our h = g / f (after fn-dsa's inverse NTT), and that fn-dsa
/// verifies our signatures.
#[test]
fn fndsa_live_parity_1024() {
    let params = Params::FALCON_1024;
    let mut kg = KeyPairGenerator1024::default();
    for key_ix in 0..8u8 {
        let mut rng = TestRng(Shake256Prng::new(&[b'k', key_ix]));
        let mut sk_enc = vec![0u8; sign_key_size(10)];
        let mut vk_enc = vec![0u8; vrfy_key_size(10)];
        kg.keygen(10, &mut rng, &mut sk_enc, &mut vk_enc);
        let key = decode_fndsa_key(&params, &sk_enc);
        let h = key.sk.public_key_h_for_tests(&params);
        assert_eq!(fndsa_vk_to_h(&params, &vk_enc), h);
        let ek = ExpandedKey::new(&key.sk).unwrap();
        let hvk = hashed_vrfykey_from_vrfykey(&vk_enc);
        let mut signer = SigningKey1024::decode(&sk_enc).unwrap();
        let verifier = VerifyingKey1024::decode(&vk_enc).unwrap();
        for msg_ix in 0..32u8 {
            let msg = [b'm', key_ix, msg_ix];
            let mu = compute_mu(&hvk, &DOMAIN_NONE, &HASH_ID_RAW, &msg);
            // fn-dsa draws a 40-byte seed from the RNG, then hedges it (lib.rs:283-295).
            let mut raw = [0u8; 40];
            Shake256Prng::new(&[b's', key_ix, msg_ix]).next_bytes(&mut raw);
            let mut fn_rng = TestRng(Shake256Prng::new(&[b's', key_ix, msg_ix]));
            let mut sig = vec![0u8; signature_size(10)];
            signer
                .sign(&mut fn_rng, &DOMAIN_NONE, &HASH_ID_RAW, &msg, &mut sig)
                .unwrap();
            let mut derived = [0u8; 40];
            shake256(&[&key.hashed_sign_key, &mu, &raw], &mut derived);
            let ours = replay_sign(&ek, &derived, &mu);
            assert_eq!(
                hex::encode(&ours),
                hex::encode(&sig),
                "key {key_ix} message {msg_ix}"
            );
            assert!(verifier.verify(&ours, &DOMAIN_NONE, &HASH_ID_RAW, &msg));
        }
    }
}
