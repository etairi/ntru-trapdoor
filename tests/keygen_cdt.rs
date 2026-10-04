//! The f, g sampler (design.md §7):
//! - K-CDT-1: the digests of `FG_PCS_128` and `FG_FALCON1024_128`, as printed by
//!   `python3 tools/gauss_tables.py --digest` (SHAKE256-256 of LE64(rows) then each entry as 16
//!   little-endian bytes);
//! - K-CDT-2: 2·10⁶ draws per table on the key-generation stream: a χ² test against the exact
//!   pmf of D_{Z,σ_fg}, $`\Pr[z]\propto e^{-z^2/(2\sigma_{fg}^2)}`$ with
//!   $`1/(2\sigma_{fg}^2)=10000d/(13689q)`$, and a sign-symmetry test. A test fails at
//!   $`p<10^{-6}`$ (Wilson–Hilferty $`z>4.75`$), with bins of expected count below 10 merged into
//!   the tails.

use ntru_trapdoor::hazmat::{Shake256Prng, sample_fg, shake256};
use ntru_trapdoor::{Params, tables};

fn digest(entries: &[u128]) -> String {
    let mut data = (entries.len() as u64).to_le_bytes().to_vec();
    for e in entries {
        data.extend_from_slice(&e.to_le_bytes());
    }
    let mut out = [0u8; 32];
    shake256(&[&data], &mut out);
    hex::encode(out)
}

/// K-CDT-1.
#[test]
fn fg_table_digests() {
    assert_eq!(tables::FG_PCS_128.len(), 348);
    assert_eq!(
        digest(&tables::FG_PCS_128),
        "6ca822384c2feff9b51b8d3ce96f57f3c2ac8d5b5c58e02c080fc453f1a2bef2"
    );
    assert_eq!(tables::FG_FALCON1024_128.len(), 38);
    assert_eq!(
        digest(&tables::FG_FALCON1024_128),
        "0c071ff400d43a2364ac096508fd4eeac408a1e306e4a700f08f7e3c2edf7386"
    );
    // The presets point at these tables.
    assert_eq!(Params::PCS.fg_table, &tables::FG_PCS_128[..]);
    assert_eq!(Params::FALCON_1024.fg_table, &tables::FG_FALCON1024_128[..]);
}

/// Wilson–Hilferty normal deviate of a χ² statistic with `k` degrees of freedom.
fn wilson_hilferty(chi2: f64, k: f64) -> f64 {
    let c = 2.0 / (9.0 * k);
    ((chi2 / k).cbrt() - (1.0 - c)) / c.sqrt()
}

/// χ² of the counts against D_{Z,σ_fg} and the sign-symmetry deviate.
fn check_distribution(p: &Params, seed: &[u8], draws: usize) -> (f64, f64, f64) {
    let mut prng = Shake256Prng::from_parts(&[b"ntru-trapdoor/test/keygen_cdt", seed]);
    let mut out = vec![0i16; draws];
    sample_fg(p, &mut prng, &mut out);
    let zmax = p.fg_table.len() as i64;
    let mut counts = vec![0u64; (2 * zmax + 1) as usize];
    for &z in &out {
        counts[(z as i64 + zmax) as usize] += 1;
    }
    // Exact pmf: rho(z) = exp(-a z^2), a = 1/(2 sigma_fg^2) = 10000 n / (13689 q).
    let a = 10000.0 * p.n() as f64 / (13689.0 * p.q as f64);
    let rho: Vec<f64> = (-zmax..=zmax)
        .map(|z| (-(a * (z * z) as f64)).exp())
        .collect();
    let total: f64 = rho.iter().sum();
    let expect: Vec<f64> = rho.iter().map(|r| r / total * draws as f64).collect();
    // Merge the tails into the outermost bins with expected count >= 10.
    let lo = expect.iter().position(|&e| e >= 10.0).unwrap();
    let hi = expect.iter().rposition(|&e| e >= 10.0).unwrap();
    let mut bins: Vec<(f64, f64)> = Vec::new(); // (observed, expected)
    let (mut o, mut e) = (0.0, 0.0);
    for i in 0..=lo {
        o += counts[i] as f64;
        e += expect[i];
    }
    bins.push((o, e));
    for i in lo + 1..hi {
        bins.push((counts[i] as f64, expect[i]));
    }
    let (mut o, mut e) = (0.0, 0.0);
    for i in hi..counts.len() {
        o += counts[i] as f64;
        e += expect[i];
    }
    bins.push((o, e));
    let chi2: f64 = bins.iter().map(|(o, e)| (o - e) * (o - e) / e).sum();
    let k = (bins.len() - 1) as f64;
    let z_chi = wilson_hilferty(chi2, k);
    // Sign symmetry: positives against negatives, a fair binomial.
    let pos = out.iter().filter(|&&z| z > 0).count() as f64;
    let neg = out.iter().filter(|&&z| z < 0).count() as f64;
    let z_sym = (pos - neg).abs() / (pos + neg).sqrt();
    (chi2 / k, z_chi, z_sym)
}

/// K-CDT-2.
#[test]
fn sample_fg_chi_square() {
    for (p, seed) in [(Params::PCS, b"pcs"), (Params::FALCON_1024, b"fal")] {
        let (ratio, z_chi, z_sym) = check_distribution(&p, seed, 2_000_000);
        eprintln!(
            "K-CDT-2 {}: chi2/k = {ratio:.4}, Wilson-Hilferty z = {z_chi:.3}, symmetry z = {z_sym:.3}",
            p.name
        );
        assert!(z_chi < 4.75, "{}: chi-square z = {z_chi}", p.name);
        assert!(z_sym < 4.75, "{}: symmetry z = {z_sym}", p.name);
    }
}

/// K-CDT-2, long run (integrate stage, after the lazy draw): 8 seeds × 10⁷ draws per table.
/// Run: `cargo test --release --test keygen_cdt -- --ignored --nocapture` (about 15 s).
#[test]
#[ignore = "8 x 10^7 draws per table"]
fn sample_fg_chi_square_long() {
    for p in [Params::PCS, Params::FALCON_1024] {
        let mut zs = Vec::new();
        for s in 0..8u8 {
            let (ratio, z_chi, z_sym) = check_distribution(&p, &[b'L', s], 10_000_000);
            eprintln!(
                "K-CDT-2 long {} seed {s}: chi2/k = {ratio:.4}, z = {z_chi:.3}, symmetry z = {z_sym:.3}",
                p.name
            );
            assert!(z_chi < 4.75 && z_sym < 4.75);
            zs.push(z_chi);
        }
        let mean = zs.iter().sum::<f64>() / zs.len() as f64;
        eprintln!(
            "K-CDT-2 long {}: mean z over 8 seeds = {mean:.3} (expected 0, SE 0.354)",
            p.name
        );
        assert!(mean.abs() < 4.75 / 8f64.sqrt());
    }
}
