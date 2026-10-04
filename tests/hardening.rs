//! Regression tests for the findings of the review and statistics stages, through the public
//! API:
//! - H-1 (review 1.1): an inconsistent trapdoor (`fG - gF != q`) is refused by
//!   `ExpandedKey::new`, so SampPre can no longer return "preimages" with `A s != t`;
//! - H-2 (review 1.2): `RqPoly` refuses rings outside `1 <= logn <= 10`, `3 <= q < 2^25`
//!   instead of computing wrong products or wrong inverses, or panicking;
//! - H-3 (review 1.6): the leaf window of the presets, and of sets that `trapgen` refuses;
//! - H-4 (stats F1): SamplerZ rejects a far-tail trial that fn-dsa's saturated BerExp would
//!   accept with probability 2^-64;
//! - H-5 (review 1.8): the attempt cap, the redacted `Debug` output, the prepared `h`, and
//!   Verify on `s = 0`.

use ntru_trapdoor::hazmat::{LeafSampler, Prng, SamplerZ};
use ntru_trapdoor::{
    BaseSampler, Error, ExpandedKey, Params, Preimage, PublicKey, RqPoly, SecretKey, samp_pre,
    trapgen, verify,
};
use zeroize::Zeroize;

fn key_from(sk: &SecretKey, f: Vec<i16>, g: Vec<i16>, bf: Vec<i16>, bg: Vec<i16>) -> SecretKey {
    SecretKey::from_parts(sk.params(), f, g, bf, bg).unwrap()
}

/// H-1: single-coefficient faults in F or G (the review's E3 perturbations) are refused before
/// any sampling; the unmodified key, rebuilt through `from_parts`, still works.
#[test]
fn inconsistent_trapdoors_are_refused() {
    for (params, seed) in [(Params::PCS, [11u8; 32]), (Params::FALCON_1024, [12u8; 32])] {
        let (pk, sk) = trapgen(&params, &seed).unwrap();
        let (f, g, bf, bg) = (
            sk.f().to_vec(),
            sk.g().to_vec(),
            sk.big_f().to_vec(),
            sk.big_g().to_vec(),
        );
        // The control: the same key through `from_parts` (validated by `ExpandedKey::new`).
        let same = key_from(&sk, f.clone(), g.clone(), bf.clone(), bg.clone());
        let ek = ExpandedKey::new(&same).unwrap();
        let t = RqPoly::from_coeffs(&params, &vec![params.q / 3; params.n()]).unwrap();
        let s = samp_pre(&ek, &t, &[1; 32]).unwrap();
        assert!(verify(&pk, &t, &s, params.beta_sq));
        // Faults: F[0] + 1; G[5] - 1; F[0] + 1 and G[0] + 1; f[3] + 1.
        let mut cases = Vec::new();
        let mut x = bf.clone();
        x[0] += 1;
        cases.push(key_from(&sk, f.clone(), g.clone(), x, bg.clone()));
        let mut y = bg.clone();
        y[5] -= 1;
        cases.push(key_from(&sk, f.clone(), g.clone(), bf.clone(), y));
        let (mut x, mut y) = (bf.clone(), bg.clone());
        x[0] += 1;
        y[0] += 1;
        cases.push(key_from(&sk, f.clone(), g.clone(), x, y));
        let mut ff = f.clone();
        ff[3] += 1;
        cases.push(key_from(&sk, ff, g.clone(), bf.clone(), bg.clone()));
        for (i, bad) in cases.iter().enumerate() {
            assert_eq!(bad.validate().unwrap_err(), Error::InvalidKey, "case {i}");
            assert_eq!(
                ExpandedKey::new(bad).err(),
                Some(Error::InvalidKey),
                "{} case {i}",
                params.name
            );
        }
        // A wiped key cannot be expanded either.
        let mut wiped = sk.clone();
        wiped.zeroize();
        assert_eq!(ExpandedKey::new(&wiped).err(), Some(Error::InvalidKey));
        // A decoded key is validated once, by the decoder.
        let dec = SecretKey::from_bytes(&params, &sk.to_bytes()).unwrap();
        assert!(ExpandedKey::new(&dec).is_ok());
    }
}

/// H-2: the review's E1 rings (with `inv_q` set consistently): every constructor returns
/// `InvalidParams` instead of building an element whose products or inverse would be wrong, or
/// that would panic later. The largest admitted modulus still multiplies exactly.
#[test]
fn rq_poly_refuses_invalid_rings() {
    let with_q = |q: u32| Params {
        q,
        inv_q: 1.0 / q as f64,
        ..Params::PCS
    };
    let mut bad = vec![
        Params {
            logn: 0,
            ..Params::PCS
        },
        Params {
            logn: 11,
            ..Params::PCS
        },
    ];
    for q in [
        0u32,
        1,
        2,
        1 << 25,
        47_451_991,
        268_435_399,
        (1 << 31) - 1,
        4_294_967_291,
    ] {
        bad.push(with_q(q));
    }
    for p in &bad {
        let n = 1usize << p.logn.min(11);
        let is_bad = |r: Result<RqPoly, Error>| matches!(r, Err(Error::InvalidParams(_)));
        assert!(is_bad(RqPoly::zero(p)), "q {} logn {}", p.q, p.logn);
        assert!(is_bad(RqPoly::from_coeffs(p, &vec![0; n])));
        assert!(is_bad(RqPoly::from_i64(p, &vec![1; n])));
        assert!(is_bad(RqPoly::from_i32(p, &vec![1; n])));
        assert!(is_bad(RqPoly::from_i16(p, &vec![1; n])));
        assert!(is_bad(RqPoly::from_bytes(p, &vec![0; 4096])));
    }
    // q = 33,554,393, the largest prime below 2^25: the all-(q-1) square, against i128.
    let p = with_q(33_554_393);
    p.validate().unwrap();
    let m = RqPoly::from_coeffs(&p, &vec![p.q - 1; p.n()]).unwrap();
    let sq = m.mul(&m);
    // (q-1)^2 sum over the negacyclic convolution: coefficient k is ((k+1) - (n-1-k)) (q-1)^2.
    let qq = (p.q as i128 - 1).pow(2);
    for (k, &c) in sq.coeffs().iter().enumerate() {
        let want = ((k as i128 + 1) - (p.n() as i128 - 1 - k as i128)) * qq;
        assert_eq!(c as i128, want.rem_euclid(p.q as i128), "k {k}");
    }
    assert_eq!(m.prepare().mul(&m), sq);
}

/// H-3: the leaf windows of the presets, and the sets that key generation refuses.
#[test]
fn leaf_windows() {
    // PCS: sigma = 1.17 sqrt(q) sigma_min, so the window is [sigma_min, 1.17^2 sigma_min].
    let p = Params::PCS;
    let (lo, hi) = p.leaf_window();
    assert!((lo / p.sigma_min - 1.0).abs() < 1e-12, "{lo}");
    assert!((hi / (1.3689 * p.sigma_min) - 1.0).abs() < 1e-12, "{hi}");
    assert!(hi < 3.0352 && hi > 3.0351, "{hi}");
    Params::PCS.check_leaf_window().unwrap();
    let (lo, hi) = Params::FALCON_1024.leaf_window();
    assert!(
        lo >= Params::FALCON_1024.sigma_min && hi < 1.8205,
        "{lo} {hi}"
    );
    Params::FALCON_1024.check_leaf_window().unwrap();
    // sigma_min above the window's lower end; a base width below its upper end.
    let p = Params {
        sigma_min: 2.3,
        ..Params::PCS
    };
    p.validate().unwrap();
    assert!(matches!(
        p.check_leaf_window(),
        Err(Error::InvalidParams(_))
    ));
    assert!(matches!(
        trapgen(&p, &[0; 32]),
        Err(Error::InvalidParams(_))
    ));
    let p = Params {
        base: BaseSampler::Falcon1_8205,
        sigma_min: 1.2,
        ..Params::PCS
    };
    p.validate().unwrap();
    assert!(matches!(
        p.check_leaf_window(),
        Err(Error::InvalidParams(_))
    ));
}

/// Replays fixed bytes.
struct Replay(Vec<u8>, usize);

impl Prng for Replay {
    fn next_u8(&mut self) -> u8 {
        let b = self.0[self.1];
        self.1 += 1;
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

/// H-4 (the crafted stream of `examples/stats_tail.rs`): at the narrowest PCS leaf width and
/// centre 0, a base draw V = 0 with sign 1 gives z0 = 41, z = 42, and x = 91.95, so the exact
/// acceptance probability is 2^-132.6 and the exact 64-bit threshold is 0. With the all-zero
/// Bernoulli bytes fn-dsa's saturated BerExp accepts (threshold 1); this sampler rejects, and
/// the next, ordinary trial decides.
#[test]
fn far_tail_trial_is_rejected() {
    let params = Params::PCS;
    let mut isigma = 1.0 / params.sigma_min;
    while isigma * params.sigma_min > 1.0 {
        isigma = f64::from_bits(isigma.to_bits() - 1);
    }
    // Trial 1: V = 0 (w = sign bit 1; the fallback reads two zero u64 and one zero byte), then
    // eight zero Bernoulli bytes. Trial 2: H = 2^63 - 1 (z0 = 0), sign 0 (z = 0), x = 0, and a
    // Bernoulli byte 0 that accepts at once.
    let mut bytes = vec![1u8, 0, 0, 0, 0, 0, 0, 0];
    bytes.extend_from_slice(&[0u8; 17]);
    bytes.extend_from_slice(&[0u8; 8]);
    bytes.extend_from_slice(&[0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    bytes.push(0);
    let mut prng = Replay(bytes, 0);
    let mut samp = SamplerZ::new(&params, &mut prng);
    let z = samp.sample(0.0, isigma);
    assert_eq!((z, samp.trials()), (0, 2));
    assert_eq!(prng.1, 8 + 17 + 8 + 8 + 1);
}

/// H-5: the attempt cap of the one-byte counter, the redacted `Debug` output, the prepared `h`,
/// and Verify on s = 0, t = 0 (accepted, as the paper's Verify).
#[test]
fn api_details() {
    let p = Params {
        samppre_max_attempts: 257,
        ..Params::PCS
    };
    assert!(matches!(p.validate(), Err(Error::InvalidParams(_))));
    Params {
        samppre_max_attempts: 256,
        ..Params::PCS
    }
    .validate()
    .unwrap();

    let params = Params::FALCON_1024;
    let (pk, sk) = trapgen(&params, &[21; 32]).unwrap();
    let fq = RqPoly::from_i16(&params, sk.f()).unwrap();
    let finv = fq.inverse().unwrap();
    assert_eq!(format!("{finv:?}"), "RqPoly { q: 12289, n: 1024, .. }");
    assert_eq!(
        format!("{:?}", finv.prepare()),
        "PreparedRqPoly { q: 12289, n: 1024, .. }"
    );
    assert_eq!(
        format!("{sk:?}"),
        "SecretKey { params: \"Falcon-1024 (q = 12289)\", .. }"
    );
    assert_eq!(
        format!("{pk:?}"),
        "PublicKey { params: \"Falcon-1024 (q = 12289)\", h: RqPoly { q: 12289, n: 1024, .. } }"
    );
    // Keys can be shared between threads (the prepared h is computed once, behind a OnceLock).
    fn send_sync<T: Send + Sync>() {}
    send_sync::<PublicKey>();
    send_sync::<ExpandedKey>();
    send_sync::<ntru_trapdoor::PreparedRqPoly>();
    // The prepared h: equal products, the ring, and Verify through it.
    let hp = pk.h_prepared();
    assert!(hp.matches(&params) && hp.n() == params.n() && hp.q() == params.q);
    let x = RqPoly::from_i64(&params, &(0..params.n() as i64).collect::<Vec<_>>()).unwrap();
    assert_eq!(hp.mul(&x), pk.h().mul(&x));
    let pk2 = PublicKey::from_bytes(&params, &pk.to_bytes()).unwrap();
    assert_eq!(pk2, pk);
    let zero = RqPoly::zero(&params).unwrap();
    let s0 = Preimage {
        s0: vec![0; params.n()],
        s1: vec![0; params.n()],
    };
    assert!(verify(&pk, &zero, &s0, params.beta_sq));
}
