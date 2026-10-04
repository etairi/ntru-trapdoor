//! TrapGen (design.md §7):
//! - K-TG-1: keys of both presets for 4 seeds: $`hf=g\bmod q`$, `validate`, `public_key`,
//!   the expanded key and its leaf range, determinism in the seed;
//! - K-TG-2 (`#[ignore]`): 64 PCS keys; the Gram–Schmidt acceptance and the solver failure rate
//!   against the Sage oracle's 22/410 and 110/410 (two-proportion z-tests at 99.9%):
//!   `cargo test --release --test keygen_trapgen -- --ignored --nocapture`;
//! - K-TG-3: every rejection reason of `trapgen_from_fg`, on crafted inputs;
//! - both presets pass `Params::validate` (`FG_PCS_128` ends with two equal entries, which a
//!   reverse CDT allows), and malformed tables are rejected.

use ntru_trapdoor::hazmat::gram_schmidt_sq_norms;
use ntru_trapdoor::{
    ExpandedKey, KeygenReject, Params, RqPoly, SecretKey, tables, trapgen, trapgen_from_fg,
    trapgen_with_stats,
};

/// Both presets validate (`FG_PCS_128` ends `.., 2, 1, 1`: a non-increasing reverse CDT), and
/// `validate` still rejects an empty, an increasing or an all-zero table.
#[test]
fn presets_validate() {
    Params::PCS.validate().unwrap();
    Params::FALCON_1024.validate().unwrap();
    let t = tables::FG_PCS_128;
    assert_eq!(t[t.len() - 2], t[t.len() - 1]);
    static INCREASING: [u128; 3] = [5, 4, 6];
    static ZERO: [u128; 2] = [0, 0];
    for bad in [&[][..], &INCREASING[..], &ZERO[..]] {
        let p = Params {
            fg_table: bad,
            ..Params::PCS
        };
        assert!(matches!(
            p.validate(),
            Err(ntru_trapdoor::Error::InvalidParams(_))
        ));
    }
}

/// K-TG-1.
#[test]
fn trapgen_keys_are_consistent() {
    for p in [Params::PCS, Params::FALCON_1024] {
        let mut seen = Vec::new();
        for s in 0..4u8 {
            let seed = [s.wrapping_mul(37).wrapping_add(1); 32];
            let (pk, sk, stats) = trapgen_with_stats(&p, &seed).unwrap();
            // h f = g mod q.
            let fq = RqPoly::from_i16(&p, sk.f()).unwrap();
            let gq = RqPoly::from_i16(&p, sk.g()).unwrap();
            assert_eq!(pk.h().mul(&fq), gq, "{} seed {s}", p.name);
            // fG - gF = q, widths, invertibility; the public key from the secret key.
            sk.validate().unwrap();
            assert_eq!(sk.public_key().unwrap(), pk);
            // Coefficient sizes (Sage oracle, 300 keys: |f|,|g| <= 132 and |F|,|G| <= 1263 at
            // the PCS set; 14 and 126 at Falcon-1024).
            let mf = sk
                .f()
                .iter()
                .chain(sk.g())
                .map(|x| x.unsigned_abs())
                .max()
                .unwrap();
            let mb = sk
                .big_f()
                .iter()
                .chain(sk.big_g())
                .map(|x| x.unsigned_abs())
                .max()
                .unwrap();
            // Both Gram-Schmidt norms pass.
            let (n1, n2) = gram_schmidt_sq_norms(&p, sk.f(), sk.g());
            assert!(n1 * p.gs_bound_den <= p.gs_bound_num && n2 <= p.gs_bound());
            // The expanded key exists and its leaves lie in [sigma_min, sigma0).
            let ek = ExpandedKey::new(&sk).unwrap();
            let (lo, hi) = ek.leaf_sigma_range();
            assert!(
                lo >= p.sigma_min && hi < p.base.sigma0(),
                "leaves [{lo}, {hi}]"
            );
            eprintln!(
                "{} seed {s}: {} candidates, max |f,g| {mf}, max |F,G| {mb}, leaves [{lo:.5}, {hi:.5}]",
                p.name, stats.candidates
            );
            // Determinism in the seed; different seeds give different keys.
            let (pk2, sk2) = trapgen(&p, &seed).unwrap();
            assert_eq!(pk2, pk);
            assert_eq!(sk2.f(), sk.f());
            assert_eq!(sk2.big_g(), sk.big_g());
            assert!(!seen.contains(&pk.h().clone()));
            seen.push(pk.h().clone());
        }
    }
}

/// Two-proportion z statistic of k1/n1 against k2/n2.
fn two_prop_z(k1: f64, n1: f64, k2: f64, n2: f64) -> f64 {
    let p = (k1 + k2) / (n1 + n2);
    (k1 / n1 - k2 / n2) / (p * (1.0 - p) * (1.0 / n1 + 1.0 / n2)).sqrt()
}

/// K-TG-2.
#[test]
#[ignore = "64 PCS keys; run in release"]
fn trapgen_statistics_pcs() {
    let p = Params::PCS;
    let keys = 64u32;
    let (mut cand, mut gs_pass, mut solve_runs, mut solve_fail, mut leaf, mut size) =
        (0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
    let t = std::time::Instant::now();
    for k in 0..keys {
        let mut seed = [0u8; 32];
        seed[..4].copy_from_slice(&k.to_le_bytes());
        seed[31] = 0x5c;
        let (_, _, s) = trapgen_with_stats(&p, &seed).unwrap();
        cand += s.candidates;
        gs_pass += s.candidates - s.rejected_norm_fg - s.rejected_norm_ortho;
        solve_runs +=
            s.candidates - s.rejected_norm_fg - s.rejected_norm_ortho - s.rejected_not_invertible;
        solve_fail += s.rejected_solve;
        leaf += s.rejected_leaf;
        size += s.rejected_size;
    }
    let dt = t.elapsed();
    let z_gs = two_prop_z(gs_pass as f64, cand as f64, 22.0, 410.0);
    let z_solve = two_prop_z(solve_fail as f64, solve_runs as f64, 110.0, 410.0);
    eprintln!(
        "K-TG-2: {keys} keys in {dt:?} ({:.1} ms/key); {cand} candidates ({:.2}/key); GS pass \
         {gs_pass}/{cand} = {:.4} (Sage 22/410 = 0.0537, z = {z_gs:.2}); solver failures \
         {solve_fail}/{solve_runs} = {:.3} (Sage 110/410 = 0.268, z = {z_solve:.2}); size {size}, \
         leaf {leaf}",
        dt.as_secs_f64() * 1e3 / keys as f64,
        cand as f64 / keys as f64,
        gs_pass as f64 / cand as f64,
        solve_fail as f64 / solve_runs as f64
    );
    assert!(z_gs.abs() < 3.29 && z_solve.abs() < 3.29);
    assert_eq!((size, leaf), (0, 0));
}

/// A toy parameter set: degree 4, q = 17 (x^4 + 1 splits into linear factors, roots 2, 8, 9,
/// 15), with Falcon-1024's 38-row table.
fn toy_params() -> Params {
    let q = 17u32;
    let sigma = 3.0f64;
    Params {
        name: "toy (n = 4, q = 17)",
        logn: 2,
        q,
        sigma,
        inv_sigma: 1.0 / sigma,
        sigma_min: 0.5,
        base: Params::FALCON_1024.base,
        inv_q: 1.0 / q as f64,
        beta_sq: 1000,
        sigma_fg: 1.0,
        fg_table: &tables::FG_FALCON1024_128,
        gs_bound_num: 13689 * q as u64,
        gs_bound_den: 10000,
        sk_fg_bits: 7,
        sk_big_fg_bits: 9,
        samppre_max_attempts: 27,
        keygen_max_attempts: 64,
    }
}

/// SplitMix64.
struct Mix(u64);
impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn small(&mut self, n: usize) -> Vec<i16> {
        (0..n).map(|_| (self.next() % 5) as i16 - 2).collect()
    }
}

/// f(r) mod q.
fn eval_mod(f: &[i16], r: i64, q: i64) -> i64 {
    f.iter()
        .rev()
        .fold(0i64, |acc, &c| (acc * r + c as i64).rem_euclid(q))
}

/// The first random toy pair (deterministic) that passes both norms and satisfies `pred`.
fn find_toy(p: &Params, seed: u64, pred: impl Fn(&[i16], &[i16]) -> bool) -> (Vec<i16>, Vec<i16>) {
    let mut r = Mix(seed);
    for _ in 0..1_000_000 {
        let (f, g) = (r.small(4), r.small(4));
        let (n1, n2) = gram_schmidt_sq_norms(p, &f, &g);
        if n1 * p.gs_bound_den <= p.gs_bound_num && n2 <= p.gs_bound() && pred(&f, &g) {
            return (f, g);
        }
    }
    panic!("no toy pair found");
}

/// K-TG-3.
#[test]
fn trapgen_from_fg_rejection_reasons() {
    let pcs = Params::PCS;
    let n = pcs.n();
    // Input: wrong lengths; an invalid parameter set.
    assert_eq!(
        trapgen_from_fg(&pcs, &[0; 3], &[0; 3]).unwrap_err(),
        KeygenReject::Input
    );
    let mut bad = pcs;
    bad.q = 1_048_575; // not prime
    assert_eq!(
        trapgen_from_fg(&bad, &vec![1; n], &vec![1; n]).unwrap_err(),
        KeygenReject::Input
    );
    // Size: f_0 = -2^9 = -512 does not fit 10 bits (the encoding rejects -2^(w-1)).
    let mut f = vec![0i16; n];
    f[0] = -512;
    assert_eq!(
        trapgen_from_fg(&pcs, &f, &vec![0; n]).unwrap_err(),
        KeygenReject::Size
    );
    // NormFg: ||(g, -f)||^2 = 2048 * 400^2 is far above 1435391.
    assert_eq!(
        trapgen_from_fg(&pcs, &vec![400; n], &vec![400; n]).unwrap_err(),
        KeygenReject::NormFg
    );
    // NormOrtho: f = g = 0 passes N1 = 0, but N2 is infinite.
    assert_eq!(
        trapgen_from_fg(&pcs, &vec![0; n], &vec![0; n]).unwrap_err(),
        KeygenReject::NormOrtho
    );
    // NotInvertible: a toy f with f(2) = 0 mod 17 (2 is a root of x^4 + 1 mod 17).
    let toy = toy_params();
    let (f, g) = find_toy(&toy, 1, |f, _| eval_mod(f, 2, 17) == 0);
    assert_eq!(
        trapgen_from_fg(&toy, &f, &g).unwrap_err(),
        KeygenReject::NotInvertible
    );
    // Solve: f invertible, but f(1) and g(1) both even, so both resultants are even.
    let (f, g) = find_toy(&toy, 2, |f, g| {
        let even = |p: &[i16]| p.iter().map(|&x| x as i64).sum::<i64>() % 2 == 0;
        [2, 8, 9, 15].iter().all(|&r| eval_mod(f, r, 17) != 0) && even(f) && even(g)
    });
    assert_eq!(
        trapgen_from_fg(&toy, &f, &g).unwrap_err(),
        KeygenReject::Solve
    );
    // Size and Leaf: a valid PCS pair under narrower widths, or a larger sigma (every leaf
    // width sigma'_i = sigma / ||b~_i|| grows by 10%, above 3.1).
    let (_, sk) = trapgen(&pcs, &[9u8; 32]).unwrap();
    let mut narrow = pcs;
    narrow.sk_big_fg_bits = 8;
    assert_eq!(
        trapgen_from_fg(&narrow, sk.f(), sk.g()).unwrap_err(),
        KeygenReject::Size
    );
    let mut wide = pcs;
    wide.sigma *= 1.1;
    wide.inv_sigma = 1.0 / wide.sigma;
    assert_eq!(
        trapgen_from_fg(&wide, sk.f(), sk.g()).unwrap_err(),
        KeygenReject::Leaf
    );
    // And the unmodified set accepts the pair, with the same key.
    let (_, sk2) = trapgen_from_fg(&pcs, sk.f(), sk.g()).unwrap();
    assert_eq!(sk2.big_f(), sk.big_f());
}

/// `SecretKey::validate` and `SecretKey::public_key` on inconsistent keys.
#[test]
fn validate_rejects_inconsistent_keys() {
    let p = Params::FALCON_1024;
    let (_, sk) = trapgen(&p, &[3u8; 32]).unwrap();
    let parts = |sk: &SecretKey| {
        (
            sk.f().to_vec(),
            sk.g().to_vec(),
            sk.big_f().to_vec(),
            sk.big_g().to_vec(),
        )
    };
    // F changed by one: fG - gF != q.
    let (f, g, mut big_f, big_g) = parts(&sk);
    big_f[7] += 1;
    let bad = SecretKey::from_parts(&p, f, g, big_f, big_g).unwrap();
    assert_eq!(
        bad.validate().unwrap_err(),
        ntru_trapdoor::Error::InvalidKey
    );
    // A coefficient at -2^(w-1) (w = 9 for F at Falcon-1024).
    let (f, g, mut big_f, big_g) = parts(&sk);
    big_f[0] = -256;
    let bad = SecretKey::from_parts(&p, f, g, big_f, big_g).unwrap();
    assert_eq!(
        bad.validate().unwrap_err(),
        ntru_trapdoor::Error::InvalidKey
    );
    // f not invertible: public_key fails (f = 0).
    let (_, g, big_f, big_g) = parts(&sk);
    let bad = SecretKey::from_parts(&p, vec![0; p.n()], g, big_f, big_g).unwrap();
    assert_eq!(
        bad.public_key().unwrap_err(),
        ntru_trapdoor::Error::InvalidKey
    );
    assert_eq!(
        bad.validate().unwrap_err(),
        ntru_trapdoor::Error::InvalidKey
    );
}
