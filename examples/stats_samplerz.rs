//! Statistics tool: SamplerZ histograms over a grid of leaf widths
//! and centres at the PCS set, for the chi-square and moment tests of
//! `tools/stats/samplerz_pvalues.py`.
//!
//! Usage: `cargo run --release --example stats_samplerz -- OUT DRAWS THREADS [TAG PAIRS]`
//!
//! With TAG and PAIRS (comma-separated pair indices), only those pairs run, on fresh streams
//! (TAG is mixed into every stream; without it the streams are those of the first run).
//!
//! Grid: the leaf widths sigma_min, 2.3, 2.45, 2.6, 2.75, 2.9 and 1.17^2 sigma_min of
//! `Params::PCS` (the whole leaf range), times ten centres (fractional parts 0, 1/8, 1/4,
//! 1/2, 3/4 and 1 - 2^-40, a negative centre, and three centres of magnitude about 2^14, as at
//! the PCS leaves). Every pair gets `DRAWS` draws from `THREADS` independent SHAKE256 streams.
//!
//! The sampler is called with the leaf value `isigma` = 1/sigma' that the ffLDL tree stores
//! (nudged down by one ulp if `isigma * sigma_min > 1`, which the leaf check of
//! `ExpandedKey::new` forbids). The reference distribution is D_{Z, 1/isigma, mu} with both
//! parameters read as exact reals.
//!
//! Output, per pair: `pair <i> <isigma bits> <mu bits> <draws> <trials> <zmin>` and
//! `hist <count of zmin> <count of zmin + 1> ...` (bit patterns of f64 in hex; the histogram
//! covers floor(mu) - 64 ..= floor(mu) + 64, beyond the sampler's reach of 42).

use std::io::Write;

use ntru_trapdoor::Params;
use ntru_trapdoor::hazmat::{LeafSampler, SamplerZ, Shake256Prng};

/// Half the histogram width; `SamplerZ` returns floor(mu) + z with -41 <= z <= 42.
const HALF: i64 = 64;

fn widths(params: &Params) -> Vec<f64> {
    let s = params.sigma_min;
    vec![s, 2.3, 2.45, 2.6, 2.75, 2.9, 1.17 * 1.17 * s]
}

fn centres() -> Vec<f64> {
    vec![
        0.0,
        0.125,
        0.25,
        0.5,
        0.75,
        1.0 - 1.0 / (1u64 << 40) as f64,
        -0.3,
        13579.3,
        -12345.7,
        16383.5,
    ]
}

/// The tree's leaf value for width `sigma`, kept inside the leaf check's `v * sigma_min <= 1`.
fn leaf_value(sigma: f64, sigma_min: f64) -> f64 {
    let mut v = 1.0 / sigma;
    while v * sigma_min > 1.0 {
        v = f64::from_bits(v.to_bits() - 1);
    }
    v
}

fn run_pair(
    params: &Params,
    idx: u32,
    isigma: f64,
    mu: f64,
    draws: u64,
    threads: u64,
    tag: &[u8],
) -> (u64, Vec<u64>) {
    let per = draws / threads;
    let extra = draws % threads;
    let results: Vec<(u64, Vec<u64>)> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads)
            .map(|th| {
                let count = per + u64::from(th < extra);
                sc.spawn(move || {
                    let (i, t) = (idx.to_le_bytes(), th.to_le_bytes());
                    let mut prng = if tag.is_empty() {
                        Shake256Prng::from_parts(&[b"ntru-trapdoor/stats/samplerz", &i, &t])
                    } else {
                        Shake256Prng::from_parts(&[b"ntru-trapdoor/stats/samplerz", &i, &t, tag])
                    };
                    let mut samp = SamplerZ::new(params, &mut prng);
                    let base = mu.floor() as i64 - HALF;
                    let mut hist = vec![0u64; (2 * HALF + 1) as usize];
                    for _ in 0..count {
                        let z = samp.sample(mu, isigma) as i64;
                        hist[(z - base) as usize] += 1;
                    }
                    (samp.trials(), hist)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut trials = 0;
    let mut hist = vec![0u64; (2 * HALF + 1) as usize];
    for (t, h) in results {
        trials += t;
        for (a, b) in hist.iter_mut().zip(h) {
            *a += b;
        }
    }
    (trials, hist)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 && args.len() != 6 {
        eprintln!("usage: stats_samplerz OUT DRAWS THREADS [TAG PAIRS]");
        std::process::exit(2);
    }
    let draws: u64 = args[2].parse().expect("DRAWS");
    let threads: u64 = args[3].parse().expect("THREADS");
    let tag: Vec<u8> = args.get(4).map_or(Vec::new(), |t| t.as_bytes().to_vec());
    let only: Option<Vec<u32>> = args
        .get(5)
        .map(|p| p.split(',').map(|x| x.parse().expect("PAIRS")).collect());
    let params = Params::PCS;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[1]).expect("OUT"));
    writeln!(
        out,
        "# stats_samplerz: Params::PCS (base width 3.1), sigma_min bits {:016x}, {draws} draws per pair, {threads} threads, tag {:?}",
        params.sigma_min.to_bits(),
        String::from_utf8_lossy(&tag)
    )
    .unwrap();
    let mut idx = 0u32;
    for &sigma in widths(&params).iter() {
        let isigma = leaf_value(sigma, params.sigma_min);
        for &mu in centres().iter() {
            if only.as_ref().is_some_and(|o| !o.contains(&idx)) {
                idx += 1;
                continue;
            }
            let t0 = std::time::Instant::now();
            let (trials, hist) = run_pair(&params, idx, isigma, mu, draws, threads, &tag);
            let zmin = mu.floor() as i64 - HALF;
            writeln!(
                out,
                "pair {idx} {:016x} {:016x} {draws} {trials} {zmin}",
                isigma.to_bits(),
                mu.to_bits()
            )
            .unwrap();
            let line: Vec<String> = hist.iter().map(|c| c.to_string()).collect();
            writeln!(out, "hist {}", line.join(" ")).unwrap();
            out.flush().unwrap();
            eprintln!(
                "pair {idx}: sigma' {:.6} mu {mu}: acceptance {:.6}, {:.1} s",
                1.0 / isigma,
                draws as f64 / trials as f64,
                t0.elapsed().as_secs_f64()
            );
            idx += 1;
        }
    }
}
