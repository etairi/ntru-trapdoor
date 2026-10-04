//! S-SP-4 (design.md §7): SampPre against the exact coset Gaussian, at toy size.
//!
//! Toy parameter set: n = 2 (logn = 1), q = 17, sigma = 11, sigma_min = 2.0, the PCS base
//! sampler (width 3.1), no norm bound. Key (found by search over Z[i] = Z[x]/(x^2+1)):
//! f = -3 - 2x, g = -1 - x, F = 2 - 2x, G = -3 + 2x, with f G - g F = 17 and f invertible
//! modulo 17. Its two leaf widths are sigma/sqrt(15) = 2.840 (t0 side) and
//! sigma sqrt(15)/17 = 2.506 (t1 side): unequal, so a swapped leaf would show, and l10 != 0.
//!
//! With every leaf width >= 2.0 (s-parameter >= 5.0), Klein/ffSampling is exact up to a
//! statistical distance of order 1e-30 per sample [derived: the smoothing slack of Z at
//! s = 5 is about 2 exp(-pi s^2)], so SampPre's output must follow the discrete Gaussian
//! D_{c + Lambda, sigma} on the coset {s in Z^4 : s0 + h s1 = t mod 17}, which is enumerated
//! here exactly (f64 weights exp(-|s|^2 / (2 sigma^2)) over |s_i| <= 7 sigma; the mass beyond
//! is below 1e-9).
//!
//! The test bins every coset point with expected count >= 10 on its own and groups the others
//! by squared norm, then applies a chi-square test (fail if z > 4.75, p < 1e-6). It also checks
//! that every output satisfies A s = t.
//!
//! Run: `cargo test --test sign_toy_exact` (10^5 samples); the full test with 10^6 samples is
//! `cargo test --test sign_toy_exact -- --ignored --nocapture`.

use std::collections::{BTreeMap, HashMap};

use ntru_trapdoor::{
    BaseSampler, ExpandedKey, Params, PublicKey, RqPoly, SecretKey, samp_pre, tables, verify,
};

fn toy_params() -> Params {
    let sigma = 11.0;
    Params {
        name: "toy (n = 2, q = 17)",
        logn: 1,
        q: 17,
        sigma,
        inv_sigma: 1.0 / sigma,
        sigma_min: 2.0,
        base: BaseSampler::HalfGauss3_1,
        inv_q: 1.0 / 17.0,
        beta_sq: u64::MAX,
        sigma_fg: 1.0,
        fg_table: &tables::FG_FALCON1024_128,
        gs_bound_num: 13689 * 17,
        gs_bound_den: 10000,
        sk_fg_bits: 7,
        sk_big_fg_bits: 9,
        samppre_max_attempts: 1,
        keygen_max_attempts: 1,
    }
}

fn run(samples: u32, print: bool) {
    let params = toy_params();
    params.validate().unwrap();
    let (f, g, big_f, big_g) = (
        vec![-3i16, -2],
        vec![-1i16, -1],
        vec![2i16, -2],
        vec![-3i16, 2],
    );
    let sk = SecretKey::from_parts(&params, f.clone(), g.clone(), big_f, big_g).unwrap();
    let ek = ExpandedKey::new(&sk).unwrap();
    let (lo, hi) = ek.leaf_sigma_range();
    assert!((lo - 11.0 * 15f64.sqrt() / 17.0).abs() < 1e-12, "{lo}");
    assert!((hi - 11.0 / 15f64.sqrt()).abs() < 1e-12, "{hi}");
    let fr = RqPoly::from_i16(&params, &f).unwrap();
    let gr = RqPoly::from_i16(&params, &g).unwrap();
    let h = gr.mul(&fr.inverse().unwrap());
    let pk = PublicKey::from_h(&params, h.clone()).unwrap();
    let t = RqPoly::from_coeffs(&params, &[5, 11]).unwrap();

    // Samples.
    let mut hist: HashMap<[i32; 4], u64> = HashMap::new();
    for i in 0..samples {
        let mut seed = [0u8; 32];
        seed[..4].copy_from_slice(&i.to_le_bytes());
        let s = samp_pre(&ek, &t, &seed).unwrap();
        if i < 1000 {
            assert!(verify(&pk, &t, &s, u64::MAX));
        }
        *hist
            .entry([s.s0[0], s.s0[1], s.s1[0], s.s1[1]])
            .or_insert(0) += 1;
    }

    // The exact coset Gaussian: s1 = (a, b) free, s0 = t - h s1 + 17 k.
    let sigma = params.sigma;
    let r = (7.0 * sigma).ceil() as i64;
    let (h0, h1) = (h.coeffs()[0] as i64, h.coeffs()[1] as i64);
    let (t0, t1) = (t.coeffs()[0] as i64, t.coeffs()[1] as i64);
    let mut points: Vec<([i32; 4], f64)> = Vec::new();
    for a in -r..=r {
        for b in -r..=r {
            // h s1 in Z[x]/(x^2+1): (h0 a - h1 b) + (h0 b + h1 a) x.
            let c0 = (t0 - (h0 * a - h1 * b)).rem_euclid(17);
            let c1 = (t1 - (h0 * b + h1 * a)).rem_euclid(17);
            let first = |c: i64| c - 17 * ((c + r) / 17);
            let mut x = first(c0);
            while x <= r {
                if x >= -r {
                    let mut y = first(c1);
                    while y <= r {
                        if y >= -r {
                            let n2 = (x * x + y * y + a * a + b * b) as f64;
                            let w = (-n2 / (2.0 * sigma * sigma)).exp();
                            points.push(([x as i32, y as i32, a as i32, b as i32], w));
                        }
                        y += 17;
                    }
                }
                x += 17;
            }
        }
    }
    let total: f64 = points.iter().map(|p| p.1).sum();
    let n = samples as f64;
    let mut bins: Vec<(f64, f64)> = Vec::new(); // (expected, observed)
    let mut shells: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
    let mut seen = 0u64;
    for (s, w) in points.iter() {
        let e = n * w / total;
        let o = *hist.get(s).unwrap_or(&0) as f64;
        seen += o as u64;
        if e >= 10.0 {
            bins.push((e, o));
        } else {
            let n2: i64 = s.iter().map(|&v| (v as i64) * (v as i64)).sum();
            let entry = shells.entry(n2).or_insert((0.0, 0.0));
            entry.0 += e;
            entry.1 += o;
        }
    }
    assert_eq!(
        seen, samples as u64,
        "samples outside the enumerated coset ball"
    );
    let (mut e, mut o) = (0.0, 0.0);
    for (_, (de, dob)) in shells {
        e += de;
        o += dob;
        if e >= 10.0 {
            bins.push((e, o));
            e = 0.0;
            o = 0.0;
        }
    }
    if let Some(last) = bins.last_mut() {
        last.0 += e;
        last.1 += o;
    }
    let x2: f64 = bins.iter().map(|(e, o)| (o - e) * (o - e) / e).sum();
    let k = (bins.len() - 1) as f64;
    let z = ((x2 / k).cbrt() - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
    if print {
        println!(
            "toy exact test: {samples} samples, {} coset points, {} bins: chi2 {x2:.1} on {k} df, z = {z:.2}",
            points.len(),
            bins.len()
        );
    }
    assert!(z < 4.75, "chi2 {x2:.1} on {k} df, z = {z:.2}");
}

#[test]
fn toy_exact_quick() {
    run(100_000, true);
}

#[test]
#[ignore]
fn toy_exact_full() {
    run(1_000_000, true);
}
