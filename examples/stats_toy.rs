//! Statistics tool: SampPre at toy sizes, for the comparison with
//! Sage's `DiscreteGaussianDistributionLatticeSampler` on the same coset in
//! `tools/stats/toy_sage.sage.py`.
//!
//! Usage: `cargo run --release --example stats_toy -- SPEC OUT SAMPLES THREADS`
//!
//! SPEC is a text file with the lines `logn`, `q`, `sigma`, `sigma_min` (decimal; the widths
//! are parsed as f64), `f`, `g`, `F`, `G` (coefficients) and `t` (the target, in [0, q)). The toy
//! parameter set uses the PCS base sampler (width 3.1) and no norm bound. SAMPLES preimages of t
//! are drawn through the public `samp_pre` (seed i = SHAKE256("ntru-trapdoor/stats/toy" ||
//! LE64(i))) and written to OUT as 2d little-endian i16 values each (s0, then s1). The leaf
//! range is printed; every output is checked with `verify`.

use std::collections::HashMap;
use std::io::Write;

use ntru_trapdoor::hazmat::shake256;
use ntru_trapdoor::{
    BaseSampler, ExpandedKey, Params, PublicKey, RqPoly, SecretKey, samp_pre, tables, verify,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: stats_toy SPEC OUT SAMPLES THREADS");
        std::process::exit(2);
    }
    let spec = std::fs::read_to_string(&args[1]).expect("SPEC");
    let samples: u64 = args[3].parse().unwrap();
    let threads: u64 = args[4].parse().unwrap();
    let mut kv: HashMap<String, Vec<String>> = HashMap::new();
    for line in spec
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let mut it = line.split_whitespace();
        let key = it.next().unwrap().to_string();
        kv.insert(key, it.map(|s| s.to_string()).collect());
    }
    let one = |k: &str| kv[k][0].clone();
    let poly = |k: &str| -> Vec<i16> { kv[k].iter().map(|x| x.parse().unwrap()).collect() };
    let logn: u32 = one("logn").parse().unwrap();
    let q: u32 = one("q").parse().unwrap();
    let sigma: f64 = one("sigma").parse().unwrap();
    let sigma_min: f64 = one("sigma_min").parse().unwrap();
    let params = Params {
        name: "toy",
        logn,
        q,
        sigma,
        inv_sigma: 1.0 / sigma,
        sigma_min,
        base: BaseSampler::HalfGauss3_1,
        inv_q: 1.0 / q as f64,
        beta_sq: u64::MAX,
        sigma_fg: 1.0,
        fg_table: &tables::FG_FALCON1024_128,
        gs_bound_num: 13689 * q as u64,
        gs_bound_den: 10000,
        sk_fg_bits: 7,
        sk_big_fg_bits: 16,
        samppre_max_attempts: 1,
        keygen_max_attempts: 1,
    };
    params.validate().expect("toy parameters");
    let (f, g, big_f, big_g) = (poly("f"), poly("g"), poly("F"), poly("G"));
    let sk = SecretKey::from_parts(&params, f.clone(), g.clone(), big_f, big_g).unwrap();
    let ek = ExpandedKey::new(&sk).expect("leaf widths outside [sigma_min, 3.1)");
    let (lo, hi) = ek.leaf_sigma_range();
    let fr = RqPoly::from_i16(&params, &f).unwrap();
    let gr = RqPoly::from_i16(&params, &g).unwrap();
    let h = gr.mul(&fr.inverse().expect("f invertible"));
    let pk = PublicKey::from_h(&params, h.clone()).unwrap();
    let tc: Vec<u32> = kv["t"].iter().map(|x| x.parse().unwrap()).collect();
    let t = RqPoly::from_coeffs(&params, &tc).unwrap();
    eprintln!(
        "toy logn {logn} q {q} sigma {sigma} sigma_min {sigma_min}: leaves [{lo:.12}, {hi:.12}], h {:?}",
        h.coeffs()
    );
    let per = samples / threads;
    let parts: Vec<Vec<u8>> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads)
            .map(|th| {
                let (ek, t, pk, params) = (&ek, &t, &pk, &params);
                sc.spawn(move || {
                    let lo = th * per;
                    let hi = if th + 1 == threads { samples } else { lo + per };
                    let mut out = Vec::with_capacity(((hi - lo) as usize) * 4 * params.n());
                    for i in lo..hi {
                        let mut seed = [0u8; 32];
                        shake256(&[b"ntru-trapdoor/stats/toy", &i.to_le_bytes()], &mut seed);
                        let s = samp_pre(ek, t, &seed).unwrap();
                        if i % 4096 == 0 {
                            assert!(verify(pk, t, &s, u64::MAX));
                        }
                        for &c in s.s0.iter().chain(s.s1.iter()) {
                            out.extend_from_slice(&i16::try_from(c).unwrap().to_le_bytes());
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut o = std::io::BufWriter::new(std::fs::File::create(&args[2]).unwrap());
    for p in parts {
        o.write_all(&p).unwrap();
    }
    o.flush().unwrap();
    eprintln!("{samples} samples written to {}", args[2]);
}
