//! SampPre, Verify and the expanded key at the PCS set, on Sage fixture keys
//! (design.md §7: S-TREE-1, S-TREE-2, S-SP-1, S-SP-2, S-SP-3, S-VER-1).
//!
//! Keys: `tests/fixtures/sign/pcs_keys.txt` (`tools/sign/precision_fixture.sage.py`): three PCS
//! keys solved by an independent exact Sage NTRUSolve, with h = g / f computed in Sage, and the
//! 200-bit leaf-width range of each key.
//!
//! Run: `cargo test --test sign_samppre` (the ignored distribution test:
//! `cargo test --test sign_samppre -- --ignored --nocapture`, about a minute).

use ntru_trapdoor::hazmat::{Prng, Shake256Prng};
use ntru_trapdoor::{
    Error, ExpandedKey, Params, Preimage, PublicKey, RqPoly, SecretKey, samp_pre, verify,
};

struct FixtureKey {
    sk: SecretKey,
    pk: PublicKey,
    leaf_min: f64,
    leaf_max: f64,
}

fn load_keys(params: &Params) -> Vec<FixtureKey> {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sign/pcs_keys.txt"
    ))
    .unwrap();
    let mut keys = Vec::new();
    let mut polys: Vec<Vec<i64>> = Vec::new();
    let mut leaf = (0.0, 0.0);
    let finish = |polys: &mut Vec<Vec<i64>>, leaf: (f64, f64), keys: &mut Vec<FixtureKey>| {
        if polys.len() == 5 {
            let i16v = |v: &Vec<i64>| v.iter().map(|&x| x as i16).collect::<Vec<i16>>();
            let sk = SecretKey::from_parts(
                params,
                i16v(&polys[0]),
                i16v(&polys[1]),
                i16v(&polys[2]),
                i16v(&polys[3]),
            )
            .unwrap();
            let h = RqPoly::from_i64(params, &polys[4]).unwrap();
            let pk = PublicKey::from_h(params, h).unwrap();
            keys.push(FixtureKey {
                sk,
                pk,
                leaf_min: leaf.0,
                leaf_max: leaf.1,
            });
        }
        polys.clear();
    };
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("key") => finish(&mut polys, leaf, &mut keys),
            Some("f" | "g" | "F" | "G" | "h") => {
                polys.push(it.map(|x| x.parse::<i64>().unwrap()).collect())
            }
            Some("leaf_min") => leaf.0 = it.next().unwrap().parse().unwrap(),
            Some("leaf_max") => leaf.1 = it.next().unwrap().parse().unwrap(),
            _ => {}
        }
    }
    finish(&mut polys, leaf, &mut keys);
    assert_eq!(keys.len(), 3);
    keys
}

fn rand_target(params: &Params, rng: &mut Shake256Prng) -> RqPoly {
    let c: Vec<u32> = (0..params.n())
        .map(|_| (rng.next_u64() % params.q as u64) as u32)
        .collect();
    RqPoly::from_coeffs(params, &c).unwrap()
}

fn seed(tag: &[u8]) -> [u8; 32] {
    let mut s = [0u8; 32];
    Shake256Prng::new(tag).next_bytes(&mut s);
    s
}

/// A s = t over the integers, independently of `verify` (schoolbook, i128).
fn a_times_s(pk: &PublicKey, s: &Preimage) -> Vec<u32> {
    let p = pk.params();
    let n = p.n();
    let q = p.q as i128;
    let h = pk.h().coeffs();
    let mut acc: Vec<i128> = s.s0.iter().map(|&x| x as i128).collect();
    for i in 0..n {
        for j in 0..n {
            let v = h[i] as i128 * s.s1[j] as i128;
            if i + j < n {
                acc[i + j] += v;
            } else {
                acc[i + j - n] -= v;
            }
        }
    }
    acc.iter().map(|&x| x.rem_euclid(q) as u32).collect()
}

/// S-TREE-1: every fixture key expands; its leaf widths lie in [sigma_min, 1.17^2 sigma_min]
/// (+1e-9) and match the 200-bit range (2^-45 relative).
#[test]
fn tree_leaf_ranges() {
    let params = Params::PCS;
    let sigma_max = 1.17 * 1.17 * params.sigma_min;
    for (k, key) in load_keys(&params).iter().enumerate() {
        let ek = ExpandedKey::new(&key.sk).unwrap();
        let (lo, hi) = ek.leaf_sigma_range();
        assert!(
            lo >= params.sigma_min && hi <= sigma_max * (1.0 + 1e-9),
            "key {k}: [{lo}, {hi}]"
        );
        assert!(hi < params.base.sigma0());
        assert!(
            ((lo - key.leaf_min) / key.leaf_min).abs() < 2f64.powi(-45),
            "key {k}"
        );
        assert!(
            ((hi - key.leaf_max) / key.leaf_max).abs() < 2f64.powi(-45),
            "key {k}"
        );
        assert_eq!(ek.params(), &params);
    }
}

/// S-TREE-2: the leaf check rejects valid keys whose widths fall outside the base sampler's
/// range: a PCS key under a sigma 10% smaller (leaves below sigma_min) or 10% larger (leaves
/// above 3.1). Keys that are not trapdoors of the parameter set (a PCS key under Falcon-1024's
/// modulus, a scaled basis) are refused earlier, by the validation of `ExpandedKey::new`.
#[test]
fn tree_leaf_check_rejects() {
    let pcs = Params::PCS;
    let key = &load_keys(&pcs)[0];
    let parts = |p: &Params, scale: i16| {
        SecretKey::from_parts(
            p,
            key.sk.f().iter().map(|&x| x / scale).collect(),
            key.sk.g().iter().map(|&x| x / scale).collect(),
            key.sk.big_f().iter().map(|&x| x / scale).collect(),
            key.sk.big_g().iter().map(|&x| x / scale).collect(),
        )
        .unwrap()
    };
    for factor in [0.9, 1.1] {
        let mut p = pcs;
        p.sigma *= factor;
        p.inv_sigma = 1.0 / p.sigma;
        p.validate().unwrap();
        assert!(matches!(
            ExpandedKey::new(&parts(&p, 1)),
            Err(Error::LeafOutOfRange)
        ));
    }
    assert!(ExpandedKey::new(&parts(&pcs, 1)).is_ok());
    // Not trapdoors for these sets: fG - gF = 1048573 at q = 12289, and a scaled basis.
    assert!(matches!(
        ExpandedKey::new(&parts(&Params::FALCON_1024, 1)),
        Err(Error::InvalidKey)
    ));
    assert!(matches!(
        ExpandedKey::new(&parts(&pcs, 9)),
        Err(Error::InvalidKey)
    ));
}

/// S-SP-1: on every fixture key and on t = 0, t = all q - 1 and random targets: `verify`
/// accepts, A s = t exactly (independent schoolbook check), ||s||^2 <= beta^2; the same seed
/// gives the same s, another seed or target another s; a target of another ring is rejected.
#[test]
fn samp_pre_outputs_are_valid_preimages() {
    let params = Params::PCS;
    let keys = load_keys(&params);
    let mut rng = Shake256Prng::new(b"targets");
    for (k, key) in keys.iter().enumerate() {
        let ek = ExpandedKey::new(&key.sk).unwrap();
        let mut targets = vec![
            RqPoly::zero(&params).unwrap(),
            RqPoly::from_coeffs(&params, &vec![params.q - 1; params.n()]).unwrap(),
        ];
        let reps = if k == 0 { 16 } else { 4 };
        for _ in 0..reps {
            targets.push(rand_target(&params, &mut rng));
        }
        for (i, t) in targets.iter().enumerate() {
            let sd = seed(&[b's', k as u8, i as u8]);
            let s = samp_pre(&ek, t, &sd).unwrap();
            assert_eq!(s.n(), params.n());
            assert!(s.norm_sq() <= params.beta_sq, "key {k} target {i}");
            assert!(verify(&key.pk, t, &s, params.beta_sq), "key {k} target {i}");
            if i < 3 {
                assert_eq!(a_times_s(&key.pk, &s), t.coeffs(), "key {k} target {i}");
            }
            let s2 = samp_pre(&ek, t, &sd).unwrap();
            assert_eq!(s, s2, "determinism");
            let s3 = samp_pre(&ek, t, &seed(&[b'x', k as u8, i as u8])).unwrap();
            assert_ne!(s, s3);
            assert!(verify(&key.pk, t, &s3, params.beta_sq));
        }
        // Hedging: the same seed on another target gives unrelated randomness (s - s' is not
        // a function of t - t' alone): check that the two outputs differ in most coordinates.
        let sd = seed(b"same");
        let a = samp_pre(&ek, &targets[2], &sd).unwrap();
        let b = samp_pre(&ek, &targets[3], &sd).unwrap();
        let same = a.s1.iter().zip(b.s1.iter()).filter(|(x, y)| x == y).count();
        assert!(same < params.n() / 10, "{same} equal coordinates");
    }
    // Wrong ring.
    let ek = ExpandedKey::new(&keys[0].sk).unwrap();
    let t = RqPoly::zero(&Params::FALCON_1024).unwrap();
    assert_eq!(samp_pre(&ek, &t, &[0u8; 32]), Err(Error::Mismatch));
}

/// S-SP-2: E ||s||^2 = 2 d sigma^2: the mean of ||s||^2 / (2 d sigma^2) over 100 calls is
/// within 1 +- 6 sqrt(2 / (2 d * 100)) (chi-square law with 2d degrees of freedom).
#[test]
fn samp_pre_norm_mean() {
    let params = Params::PCS;
    let key = &load_keys(&params)[1];
    let ek = ExpandedKey::new(&key.sk).unwrap();
    let mut rng = Shake256Prng::new(b"norm-mean");
    let calls = 100;
    let mut sum = 0.0;
    let mut max_ratio: f64 = 0.0;
    for i in 0..calls {
        let t = rand_target(&params, &mut rng);
        let s = samp_pre(&ek, &t, &seed(&[b'n', i as u8])).unwrap();
        let r = s.norm_sq() as f64 / (2.0 * params.n() as f64 * params.sigma * params.sigma);
        sum += r;
        max_ratio = max_ratio.max(r);
    }
    let mean = sum / calls as f64;
    let tol = 6.0 * (2.0 / (2.0 * params.n() as f64 * calls as f64)).sqrt();
    assert!((mean - 1.0).abs() < tol, "mean {mean}, tolerance {tol}");
    // beta^2 = 2 pi * 2 d sigma^2: the largest ratio stays far below 2 pi.
    assert!(max_ratio < 1.2);
}

/// An `ExpandedKey` is `Send + Sync`: the issuer can run SampPre calls in parallel on one key;
/// the outputs equal the sequential ones (determinism per seed).
#[test]
fn samp_pre_in_parallel() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ExpandedKey>();
    let params = Params::PCS;
    let key = &load_keys(&params)[0];
    let ek = ExpandedKey::new(&key.sk).unwrap();
    let mut rng = Shake256Prng::new(b"parallel");
    let targets: Vec<RqPoly> = (0..8).map(|_| rand_target(&params, &mut rng)).collect();
    let sequential: Vec<Preimage> = targets
        .iter()
        .enumerate()
        .map(|(i, t)| samp_pre(&ek, t, &seed(&[b'p', i as u8])).unwrap())
        .collect();
    let parallel: Vec<Preimage> = std::thread::scope(|sc| {
        let handles: Vec<_> = targets
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let ek = &ek;
                sc.spawn(move || samp_pre(ek, t, &seed(&[b'p', i as u8])).unwrap())
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(sequential, parallel);
    for (t, s) in targets.iter().zip(parallel.iter()) {
        assert!(verify(&key.pk, t, s, params.beta_sq));
    }
}

/// S-VER-1: Verify rejects a changed coefficient, a wrong target, an over-norm s (and accepts
/// a norm exactly at the bound), wrong lengths and a target of another ring.
#[test]
fn verify_rejections() {
    let params = Params::PCS;
    let key = &load_keys(&params)[0];
    let ek = ExpandedKey::new(&key.sk).unwrap();
    let mut rng = Shake256Prng::new(b"verify");
    let t = rand_target(&params, &mut rng);
    let s = samp_pre(&ek, &t, &seed(b"v")).unwrap();
    assert!(verify(&key.pk, &t, &s, params.beta_sq));
    // One coefficient changed (in s0, then in s1).
    let mut bad = s.clone();
    bad.s0[17] += 1;
    assert!(!verify(&key.pk, &t, &bad, params.beta_sq));
    let mut bad = s.clone();
    bad.s1[1023] -= 1;
    assert!(!verify(&key.pk, &t, &bad, params.beta_sq));
    // Wrong target.
    let t2 = t.add(&RqPoly::from_i64(&params, &[1; 1024]).unwrap());
    assert!(!verify(&key.pk, &t2, &s, params.beta_sq));
    // The bound is inclusive: exactly ||s||^2 passes, one less fails.
    let nrm = s.norm_sq();
    assert!(verify(&key.pk, &t, &s, nrm));
    assert!(!verify(&key.pk, &t, &s, nrm - 1));
    // An s that satisfies A s = t but is long: add q to one coordinate of s0.
    let mut long = s.clone();
    long.s0[0] += params.q as i32 * 300;
    assert!(long.norm_sq() > params.beta_sq);
    assert!(!verify(&key.pk, &t, &long, params.beta_sq));
    assert!(verify(&key.pk, &t, &long, u64::MAX));
    // Wrong lengths.
    let mut short = s.clone();
    short.s1.pop();
    assert!(!verify(&key.pk, &t, &short, params.beta_sq));
    let mut empty = s.clone();
    empty.s0.clear();
    empty.s1.clear();
    assert!(!verify(&key.pk, &t, &empty, params.beta_sq));
    // A target of another ring.
    assert!(!verify(
        &key.pk,
        &RqPoly::zero(&Params::FALCON_1024).unwrap(),
        &s,
        params.beta_sq
    ));
    // Extreme coefficients do not overflow the norm.
    let mut huge = s.clone();
    huge.s0[0] = i32::MIN;
    huge.s1[0] = i32::MIN;
    assert!(!verify(&key.pk, &t, &huge, u64::MAX));
}

/// S-SP-3 (ignored; about a minute): the empirical distribution of SampPre on one target.
/// 10^4 preimages of the same t: (a) every coordinate has mean ~ c_i (the coset's centre is
/// unknown, so we check the pooled variance of the coordinates around their sample means) and
/// variance ~ sigma^2; (b) for 16 random unit directions u and for the directions of the rows
/// of B (Gram–Schmidt directions of the first and last basis vectors), the variance of <s, u>
/// is sigma^2 within 6 standard errors. A trapdoor leak of the parallelepiped kind (a sampler
/// that follows the basis) shows up as a direction-dependent variance.
#[test]
#[ignore]
fn samp_pre_distribution() {
    let params = Params::PCS;
    let key = &load_keys(&params)[2];
    let ek = ExpandedKey::new(&key.sk).unwrap();
    let n = params.n();
    let dim = 2 * n;
    let mut rng = Shake256Prng::new(b"distribution");
    let t = rand_target(&params, &mut rng);
    // Directions: 16 random ones, and the rows of B, (g, -f) and (G, -F), in the coefficient
    // embedding (lattice vectors of A's kernel). Normalised.
    let mut dirs: Vec<Vec<f64>> = (0..16)
        .map(|_| {
            (0..dim)
                .map(|_| (rng.next_u64() >> 11) as f64 - (1u64 << 52) as f64)
                .collect()
        })
        .collect();
    let row = |a: &[i16], b: &[i16]| -> Vec<f64> {
        a.iter()
            .map(|&x| x as f64)
            .chain(b.iter().map(|&x| -(x as f64)))
            .collect()
    };
    dirs.push(row(key.sk.g(), key.sk.f()));
    dirs.push(row(key.sk.big_g(), key.sk.big_f()));
    for u in dirs.iter_mut() {
        let norm = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        u.iter_mut().for_each(|x| *x /= norm);
    }
    // Streaming sums (shifted by the first sample, for numerical stability).
    let calls = 10_000usize;
    let mut shift: Vec<f64> = Vec::new();
    let mut sum = vec![0.0f64; dim];
    let mut sumsq = vec![0.0f64; dim];
    let mut psum = vec![0.0f64; dirs.len()];
    let mut psumsq = vec![0.0f64; dirs.len()];
    for i in 0..calls {
        let s = samp_pre(&ek, &t, &seed(&(i as u32).to_le_bytes())).unwrap();
        let v: Vec<f64> = s.s0.iter().chain(s.s1.iter()).map(|&x| x as f64).collect();
        if shift.is_empty() {
            shift = v.clone();
        }
        for j in 0..dim {
            let d = v[j] - shift[j];
            sum[j] += d;
            sumsq[j] += d * d;
        }
        for (k, u) in dirs.iter().enumerate() {
            let p: f64 = v
                .iter()
                .zip(shift.iter())
                .zip(u.iter())
                .map(|((x, c), y)| (x - c) * y)
                .sum();
            psum[k] += p;
            psumsq[k] += p * p;
        }
    }
    let nn = calls as f64;
    let sigma2 = params.sigma * params.sigma;
    // (a) Pooled per-coordinate variance around the per-coordinate sample means. Coordinates
    // are correlated (they live on a coset of a lattice), so the bound is loose: 24 standard
    // errors of the independent case.
    let var: f64 = (0..dim)
        .map(|j| (sumsq[j] - sum[j] * sum[j] / nn) / (nn - 1.0))
        .sum::<f64>()
        / dim as f64;
    let se = (2.0 / (dim as f64 * (nn - 1.0))).sqrt();
    println!(
        "pooled variance / sigma^2 = {:.8} (se {:.2e})",
        var / sigma2,
        se
    );
    assert!((var / sigma2 - 1.0).abs() < 24.0 * se);
    // (b) Directional variances, 6 standard errors.
    for k in 0..dirs.len() {
        let v = (psumsq[k] - psum[k] * psum[k] / nn) / (nn - 1.0);
        let se = (2.0 / (nn - 1.0)).sqrt();
        println!(
            "direction {k}: var / sigma^2 = {:.4} (se {:.4})",
            v / sigma2,
            se
        );
        assert!((v / sigma2 - 1.0).abs() < 6.0 * se, "direction {k}");
    }
}
