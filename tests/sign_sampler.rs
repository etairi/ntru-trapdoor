//! SamplerZ: fn-dsa's known answers, the table digests, and chi-square tests of the leaf
//! distribution (design.md §7: S-SAMP-1, S-SAMP-2, S-SAMP-4; S-SAMP-3, the base sampler alone,
//! is a unit test in src/sampler.rs).
//!
//! Statistics: a test fails if p < 1e-6 (Wilson–Hilferty z > 4.75), bins with expected count
//! below 10 are merged; means, variances and acceptance rates must lie within 6 standard
//! errors. The exact pmf of D_{Z,sigma',mu} is computed in f64 over |z - mu| <= 20 sigma'.
//!
//! Run: `cargo test --test sign_sampler` (10^6 draws per case); the full plan (10^7 draws per
//! case, every width and centre) is `cargo test --test sign_sampler -- --ignored`.

use std::collections::BTreeMap;

use ntru_trapdoor::hazmat::{LeafSampler, Prng, SamplerZ, Shake256Prng, shake256};
use ntru_trapdoor::{BaseSampler, Params, tables};

fn scaled(x: i64, sc: i32) -> f64 {
    (x as f64) * f64::from_bits(((1023 + sc) as u64) << 52)
}

/// S-SAMP-1: fn-dsa's sampler KAT (fn-dsa-sign/src/sampler.rs:646-2816): logn 9 (SIGMA_MIN[9]),
/// the 1.8205 base; the 40-byte nonce, then 1024 draws at given centres and widths.
#[test]
fn fndsa_sampler_kat() {
    let fx = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sign/fndsa_sampler512_kat.txt"
    ))
    .unwrap();
    let mut lines = fx.lines().filter(|l| !l.starts_with('#'));
    let seed = hex::decode(lines.next().unwrap().strip_prefix("seed ").unwrap()).unwrap();
    let nonce = hex::decode(lines.next().unwrap().strip_prefix("nonce ").unwrap()).unwrap();
    let params = Params {
        logn: 9,
        sigma_min: scaled(5754851361258101, -52), // fn-dsa SIGMA_MIN[9]
        base: BaseSampler::Falcon1_8205,
        ..Params::FALCON_1024
    };
    let mut prng = Shake256Prng::new(&seed);
    let mut got = [0u8; 40];
    prng.next_bytes(&mut got);
    assert_eq!(got.to_vec(), nonce);
    let mut samp = SamplerZ::new(&params, &mut prng);
    let mut count = 0;
    for line in lines {
        let mut it = line.split_whitespace();
        let mu = f64::from_bits(u64::from_str_radix(it.next().unwrap(), 16).unwrap());
        let isigma = f64::from_bits(u64::from_str_radix(it.next().unwrap(), 16).unwrap());
        let want: i32 = it.next().unwrap().parse().unwrap();
        assert_eq!(samp.sample(mu, isigma), want, "draw {count}");
        count += 1;
    }
    assert_eq!(count, 1024);
}

/// S-SAMP-2: the table digests (`tools/gauss_tables.py --digest`): SHAKE256 of LE64(rows) ||
/// entries (24 LE bytes per 192-bit entry, 16 per 128-bit entry; three 4-byte LE limbs per
/// GAUSS0 row). The 128-bit tables' digests are those of design.md §12.
#[test]
fn table_digests() {
    let d128 = |t: &[u128]| {
        let mut data = (t.len() as u64).to_le_bytes().to_vec();
        for v in t {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let mut out = [0u8; 32];
        shake256(&[&data], &mut out);
        hex::encode(out)
    };
    // The 192-bit base table: 24 little-endian bytes per entry (low limb first).
    let mut data = (tables::BASE_3_1_192.len() as u64).to_le_bytes().to_vec();
    for row in tables::BASE_3_1_192.iter() {
        for limb in row.iter().rev() {
            data.extend_from_slice(&limb.to_le_bytes());
        }
    }
    let mut out = [0u8; 32];
    shake256(&[&data], &mut out);
    assert_eq!(
        hex::encode(out),
        "14d0b01ecc0d4af7fab36ab6510c9795f42da1d7d73b826a74d1777bd390c62f"
    );
    assert_eq!(
        d128(&tables::FG_PCS_128),
        "6ca822384c2feff9b51b8d3ce96f57f3c2ac8d5b5c58e02c080fc453f1a2bef2"
    );
    assert_eq!(
        d128(&tables::FG_FALCON1024_128),
        "0c071ff400d43a2364ac096508fd4eeac408a1e306e4a700f08f7e3c2edf7386"
    );
    let mut data = (tables::GAUSS0_1_8205_79.len() as u64)
        .to_le_bytes()
        .to_vec();
    for row in tables::GAUSS0_1_8205_79.iter() {
        for limb in row {
            data.extend_from_slice(&limb.to_le_bytes());
        }
    }
    let mut out = [0u8; 32];
    shake256(&[&data], &mut out);
    assert_eq!(
        hex::encode(out),
        "0f8b8b1a02dbcb7a1b0655e8256d7a6ae4d7ef228ee72424ff52f9821887e381"
    );
}

/// Chi-square statistic over bins merged to expected counts >= 10, and its Wilson–Hilferty
/// z-score (upper tail).
fn chi2(observed: &BTreeMap<i64, u64>, pmf: &[(i64, f64)], n: u64) -> (f64, usize, f64) {
    let total_p: f64 = pmf.iter().map(|x| x.1).sum();
    let mut bins: Vec<(f64, f64)> = Vec::new(); // (expected, observed)
    let (mut e, mut o) = (0.0, 0.0);
    for &(z, p) in pmf {
        e += n as f64 * p / total_p;
        o += *observed.get(&z).unwrap_or(&0) as f64;
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
    let outside: u64 = observed
        .iter()
        .filter(|(z, _)| !pmf.iter().any(|(y, _)| y == *z))
        .map(|(_, c)| c)
        .sum();
    assert_eq!(outside, 0, "values outside the 20-sigma support");
    let x2: f64 = bins.iter().map(|(e, o)| (o - e) * (o - e) / e).sum();
    let df = bins.len() - 1;
    let k = df as f64;
    let z = ((x2 / k).cbrt() - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
    (x2, df, z)
}

/// SamplerZ at (sigma', mu): chi-square against D_{Z,sigma',mu}, mean and variance within 6
/// standard errors, acceptance rate within 6 standard errors of `acc`.
fn check_samplerz(params: &Params, sigma: f64, mu: f64, draws: u64, acc: f64, tag: u8) {
    let mut prng =
        Shake256Prng::from_parts(&[b"samplerz", &[tag], &sigma.to_le_bytes(), &mu.to_le_bytes()]);
    let mut samp = SamplerZ::new(params, &mut prng);
    let isigma = 1.0 / sigma;
    let mut hist: BTreeMap<i64, u64> = BTreeMap::new();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for _ in 0..draws {
        let z = samp.sample(mu, isigma) as i64;
        *hist.entry(z).or_insert(0) += 1;
        let d = z as f64 - mu;
        s1 += d;
        s2 += d * d;
    }
    let trials = samp.trials();
    // Exact pmf in f64.
    let lo = (mu - 20.0 * sigma).floor() as i64;
    let hi = (mu + 20.0 * sigma).ceil() as i64;
    let pmf: Vec<(i64, f64)> = (lo..=hi)
        .map(|z| {
            let d = z as f64 - mu;
            (z, (-d * d / (2.0 * sigma * sigma)).exp())
        })
        .collect();
    let tot: f64 = pmf.iter().map(|x| x.1).sum();
    let m1: f64 = pmf.iter().map(|(z, p)| (*z as f64 - mu) * p).sum::<f64>() / tot;
    let m2: f64 = pmf
        .iter()
        .map(|(z, p)| (*z as f64 - mu) * (*z as f64 - mu) * p)
        .sum::<f64>()
        / tot;
    let m4: f64 = pmf
        .iter()
        .map(|(z, p)| (*z as f64 - mu).powi(4) * p)
        .sum::<f64>()
        / tot;
    let n = draws as f64;
    let var = m2 - m1 * m1;
    let mean_hat = s1 / n;
    let m2_hat = s2 / n;
    assert!(
        (mean_hat - m1).abs() < 6.0 * (var / n).sqrt(),
        "sigma {sigma} mu {mu}: mean {mean_hat} vs {m1}"
    );
    assert!(
        (m2_hat - m2).abs() < 6.0 * ((m4 - m2 * m2) / n).sqrt(),
        "sigma {sigma} mu {mu}: second moment {m2_hat} vs {m2}"
    );
    let (x2, df, z) = chi2(&hist, &pmf, draws);
    assert!(
        z < 4.75,
        "sigma {sigma} mu {mu}: chi2 {x2:.1} on {df} df (z = {z:.2})"
    );
    let rate = n / trials as f64;
    let se = (acc * (1.0 - acc) / trials as f64).sqrt();
    println!(
        "sigma' {sigma:.6} mu {mu:>9}: chi2 {x2:8.1} on {df:3} df (z {z:+.2}); mean err {:+.2} se; \
         acceptance {rate:.6} ({:+.2} se)",
        (mean_hat - m1) / (var / n).sqrt(),
        (rate - acc) / se
    );
    assert!(
        (rate - acc).abs() < 6.0 * se,
        "sigma {sigma} mu {mu}: acceptance {rate} vs {acc}"
    );
}

/// PCS acceptance per trial, independent of sigma' and mu [computed, design.md §2].
const ACC_PCS: f64 = 0.63368744039;
/// Falcon-1024 acceptance per trial [computed, design.md §2].
const ACC_FALCON: f64 = 0.584957917495;

/// The leaf widths of the plan: sigma_min, a middle width, and 1.17^2 sigma_min.
fn widths(params: &Params, mid: f64) -> [f64; 3] {
    let s = params.sigma_min;
    [s, mid, 1.17 * 1.17 * s]
}
const CENTRES: [f64; 6] = [0.0, 0.25, 0.5, 0.999, 13579.3, -12345.7];

/// S-SAMP-4, quick: 10^6 draws at the PCS extremes and the middle width, three centres.
#[test]
fn samplerz_pcs_quick() {
    let params = Params::PCS;
    for (i, &s) in widths(&Params::PCS, 2.6).iter().enumerate() {
        for (j, &mu) in [0.25, 0.999, -12345.7].iter().enumerate() {
            check_samplerz(&params, s, mu, 1_000_000, ACC_PCS, (3 * i + j) as u8);
        }
    }
}

/// S-SAMP-4, quick: the Falcon base at Falcon-1024's widths.
#[test]
fn samplerz_falcon_quick() {
    let params = Params::FALCON_1024;
    for (i, &s) in widths(&Params::FALCON_1024, 1.5).iter().enumerate() {
        check_samplerz(&params, s, 0.5, 1_000_000, ACC_FALCON, 100 + i as u8);
    }
}

/// S-SAMP-4, full (ignored; about a minute): 10^7 draws for every width and centre of the
/// plan, both bases.
#[test]
#[ignore]
fn samplerz_full() {
    for (i, &s) in widths(&Params::PCS, 2.6).iter().enumerate() {
        for (j, &mu) in CENTRES.iter().enumerate() {
            check_samplerz(&Params::PCS, s, mu, 10_000_000, ACC_PCS, (10 * i + j) as u8);
        }
    }
    for (i, &s) in widths(&Params::FALCON_1024, 1.5).iter().enumerate() {
        for (j, &mu) in CENTRES.iter().enumerate() {
            check_samplerz(
                &Params::FALCON_1024,
                s,
                mu,
                10_000_000,
                ACC_FALCON,
                (100 + 10 * i + j) as u8,
            );
        }
    }
}
