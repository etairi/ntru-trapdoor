//! Statistics tool: the output distribution of SampPre at the PCS set,
//! for `tools/stats/samppre_analysis.py`.
//!
//! Usage: `cargo run --release --example stats_samppre -- OUT_DIR KEYS CALLS THREADS DIRS FIXED API [FIRST_KEY]`
//!
//! (keys FIRST_KEY .. FIRST_KEY + KEYS - 1; default 0)
//!
//! Keys: `trapgen(&Params::PCS, seed_k)`, seed_k = SHAKE256("ntru-trapdoor/stats/samppre-key"
//! || LE32(k))[0..32]. Per key, `CALLS` preimages of fresh uniform targets, each one attempt of
//! SampPre (`samp_pre_with` over `SamplerZ` and a fresh SHAKE256 stream, without the norm
//! check, so that the rate of ||s||^2 > beta^2 can be counted). Accumulated, per key:
//! - every ||s||^2 (`norms_key<k>.bin`, u64 little-endian);
//! - per coordinate j < 2d: sum s_j and sum s_j^2; pooled sum s_j^4;
//! - for `DIRS` fixed random unit directions u (Gaussian entries, normalised): sums of
//!   <s,u>^m for m = 1..4;
//! - key-adapted directions: with FFT(s0), FFT(s1) at the d/2 evaluation points k, the
//!   projections p0_k on w0_k = (FFT(g)_k, -FFT(f)_k)/|.| (the direction of the basis row
//!   (g, -f)) and p1_k on w1_k = (conj FFT(f)_k, conj FFT(g)_k)/|.| (its orthogonal
//!   complement: the direction of the first Gram–Schmidt step, sampled first by ffSampling);
//!   sums of |p|^2 and |p|^4 per k. Under the ideal distribution (covariance sigma^2 I) every
//!   p is a complex Gaussian with E|p|^2 = d sigma^2.
//!
//! Then `FIXED` preimages of one fixed uniform target on key 0 (`fixed_key0.txt`: per
//! coordinate sums and sums of squares, for the per-coordinate means of one coset), and `API`
//! preimages through the public `samp_pre` (hedged seeds, norm check) on key 0
//! (`api_norms_key0.bin`).

use std::io::Write;

use ntru_trapdoor::hazmat::{
    Prng, SamplerZ, Shake256Prng, fft, samp_pre_attempt, samp_pre_with, shake256,
};
use ntru_trapdoor::{ExpandedKey, Params, Preimage, RqPoly, samp_pre, trapgen};

fn uniform_target(params: &Params, prng: &mut Shake256Prng) -> RqPoly {
    let bits = 32 - (params.q - 1).leading_zeros();
    let mask = (1u64 << bits) - 1;
    let c: Vec<u32> = (0..params.n())
        .map(|_| {
            loop {
                let x = prng.next_u64() & mask;
                if x < params.q as u64 {
                    break x as u32;
                }
            }
        })
        .collect();
    RqPoly::from_coeffs(params, &c).unwrap()
}

/// A standard normal variate (Box–Muller) from 2 x 53 uniform bits.
fn normal(prng: &mut Shake256Prng) -> f64 {
    let u1 = ((prng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let u2 = ((prng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Key-adapted unit directions at each of the d/2 FFT points: (w0_k, w1_k), each as
/// (re0, im0, re1, im1).
fn fft_directions(params: &Params, f: &[i16], g: &[i16]) -> Vec<([f64; 4], [f64; 4])> {
    let n = params.n();
    let hn = n / 2;
    let mut fh: Vec<f64> = f.iter().map(|&x| x as f64).collect();
    let mut gh: Vec<f64> = g.iter().map(|&x| x as f64).collect();
    fft(params.logn, &mut fh);
    fft(params.logn, &mut gh);
    (0..hn)
        .map(|k| {
            let (fr, fi, gr, gi) = (fh[k], fh[k + hn], gh[k], gh[k + hn]);
            let nk = (fr * fr + fi * fi + gr * gr + gi * gi).sqrt();
            // w0 = (g, -f) / |.|; w1 = (conj f, conj g) / |.|.
            let w0 = [gr / nk, gi / nk, -fr / nk, -fi / nk];
            let w1 = [fr / nk, -fi / nk, gr / nk, -gi / nk];
            (w0, w1)
        })
        .collect()
}

#[derive(Clone)]
struct Acc {
    calls: u64,
    exceed: u64,
    norms: Vec<u64>,
    sum: Vec<i64>,
    sumsq: Vec<u64>,
    sum4: u128,
    dir: Vec<[f64; 4]>,
    p0: Vec<[f64; 2]>,
    p1: Vec<[f64; 2]>,
}

impl Acc {
    fn new(dim: usize, dirs: usize, hn: usize) -> Acc {
        Acc {
            calls: 0,
            exceed: 0,
            norms: Vec::new(),
            sum: vec![0; dim],
            sumsq: vec![0; dim],
            sum4: 0,
            dir: vec![[0.0; 4]; dirs],
            p0: vec![[0.0; 2]; hn],
            p1: vec![[0.0; 2]; hn],
        }
    }

    fn merge(&mut self, o: &Acc) {
        self.calls += o.calls;
        self.exceed += o.exceed;
        self.norms.extend_from_slice(&o.norms);
        for (a, b) in self.sum.iter_mut().zip(o.sum.iter()) {
            *a += b;
        }
        for (a, b) in self.sumsq.iter_mut().zip(o.sumsq.iter()) {
            *a += b;
        }
        self.sum4 += o.sum4;
        for (a, b) in self.dir.iter_mut().zip(o.dir.iter()) {
            for (x, y) in a.iter_mut().zip(b.iter()) {
                *x += y;
            }
        }
        for (a, b) in self.p0.iter_mut().zip(o.p0.iter()) {
            a[0] += b[0];
            a[1] += b[1];
        }
        for (a, b) in self.p1.iter_mut().zip(o.p1.iter()) {
            a[0] += b[0];
            a[1] += b[1];
        }
    }

    fn add(
        &mut self,
        params: &Params,
        s: &Preimage,
        dirs: &[Vec<f64>],
        w: &[([f64; 4], [f64; 4])],
    ) {
        let n = params.n();
        let hn = n / 2;
        self.calls += 1;
        let nrm = s.norm_sq();
        if nrm > params.beta_sq {
            self.exceed += 1;
        }
        self.norms.push(nrm);
        let v: Vec<i64> = s.s0.iter().chain(s.s1.iter()).map(|&x| x as i64).collect();
        for (j, &x) in v.iter().enumerate() {
            self.sum[j] += x;
            self.sumsq[j] += (x * x) as u64;
            self.sum4 += ((x * x) as u128) * ((x * x) as u128);
        }
        let vf: Vec<f64> = v.iter().map(|&x| x as f64).collect();
        for (u, acc) in dirs.iter().zip(self.dir.iter_mut()) {
            let p: f64 = u.iter().zip(vf.iter()).map(|(a, b)| a * b).sum();
            let p2 = p * p;
            acc[0] += p;
            acc[1] += p2;
            acc[2] += p2 * p;
            acc[3] += p2 * p2;
        }
        let mut a = vf[..n].to_vec();
        let mut b = vf[n..].to_vec();
        fft(params.logn, &mut a);
        fft(params.logn, &mut b);
        for (k, (w0, w1)) in w.iter().enumerate() {
            let (ar, ai, br, bi) = (a[k], a[k + hn], b[k], b[k + hn]);
            // <x, w> = x0 conj(w0) + x1 conj(w1).
            let proj = |w: &[f64; 4]| -> f64 {
                let re = ar * w[0] + ai * w[1] + br * w[2] + bi * w[3];
                let im = ai * w[0] - ar * w[1] + bi * w[2] - br * w[3];
                re * re + im * im
            };
            let q0 = proj(w0);
            let q1 = proj(w1);
            self.p0[k][0] += q0;
            self.p0[k][1] += q0 * q0;
            self.p1[k][0] += q1;
            self.p1[k][1] += q1 * q1;
        }
    }
}

fn call_seed(label: &[u8], k: u32, i: u64) -> [u8; 32] {
    let mut s = [0u8; 32];
    shake256(&[label, &k.to_le_bytes(), &i.to_le_bytes()], &mut s);
    s
}

fn write_acc(path: &std::path::Path, params: &Params, acc: &Acc, extra: &str) {
    let mut o = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    writeln!(o, "# stats_samppre accumulators; {extra}").unwrap();
    writeln!(
        o,
        "n {} q {} sigma_bits {:016x} beta_sq {}",
        params.n(),
        params.q,
        params.sigma.to_bits(),
        params.beta_sq
    )
    .unwrap();
    writeln!(
        o,
        "calls {} exceed {} max_norm {}",
        acc.calls,
        acc.exceed,
        acc.norms.iter().max().copied().unwrap_or(0)
    )
    .unwrap();
    let join_i = |v: &[i64]| {
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };
    let join_u = |v: &[u64]| {
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };
    writeln!(o, "sum {}", join_i(&acc.sum)).unwrap();
    writeln!(o, "sumsq {}", join_u(&acc.sumsq)).unwrap();
    writeln!(o, "sum4 {}", acc.sum4).unwrap();
    for d in acc.dir.iter() {
        writeln!(o, "dir {:e} {:e} {:e} {:e}", d[0], d[1], d[2], d[3]).unwrap();
    }
    for (k, (a, b)) in acc.p0.iter().zip(acc.p1.iter()).enumerate() {
        writeln!(o, "fftdir {k} {:e} {:e} {:e} {:e}", a[0], a[1], b[0], b[1]).unwrap();
    }
}

fn write_norms(path: &std::path::Path, norms: &[u64]) {
    let mut o = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for &x in norms {
        o.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 8 && args.len() != 9 {
        eprintln!("usage: stats_samppre OUT_DIR KEYS CALLS THREADS DIRS FIXED API [FIRST_KEY]");
        std::process::exit(2);
    }
    let dir = std::path::PathBuf::from(&args[1]);
    std::fs::create_dir_all(&dir).unwrap();
    let keys: u32 = args[2].parse().unwrap();
    let calls: u64 = args[3].parse().unwrap();
    let threads: u64 = args[4].parse().unwrap();
    let ndirs: usize = args[5].parse().unwrap();
    let fixed: u64 = args[6].parse().unwrap();
    let api: u64 = args[7].parse().unwrap();
    let first: u32 = args.get(8).map_or(0, |a| a.parse().unwrap());
    let params = Params::PCS;
    let n = params.n();
    let hn = n / 2;

    // Random unit directions in R^{2d}, shared by all keys.
    let mut dprng = Shake256Prng::new(b"ntru-trapdoor/stats/samppre-directions");
    let dirs: Vec<Vec<f64>> = (0..ndirs)
        .map(|_| {
            let mut u: Vec<f64> = (0..2 * n).map(|_| normal(&mut dprng)).collect();
            let nu = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            u.iter_mut().for_each(|x| *x /= nu);
            u
        })
        .collect();

    for k in first..first + keys {
        let t0 = std::time::Instant::now();
        let mut seed = [0u8; 32];
        shake256(
            &[b"ntru-trapdoor/stats/samppre-key", &k.to_le_bytes()],
            &mut seed,
        );
        let (pk, sk) = trapgen(&params, &seed).expect("trapgen");
        let ek = ExpandedKey::new(&sk).expect("expand");
        let w = fft_directions(&params, sk.f(), sk.g());
        // Parseval check of the FFT normalisation on one vector: sum x^2 = (2/d) sum |x_k|^2.
        {
            let x: Vec<f64> = sk.big_f().iter().map(|&v| v as f64).collect();
            let mut y = x.clone();
            fft(params.logn, &mut y);
            let lhs: f64 = x.iter().map(|v| v * v).sum();
            let rhs: f64 = (0..hn)
                .map(|j| y[j] * y[j] + y[j + hn] * y[j + hn])
                .sum::<f64>()
                * 2.0
                / n as f64;
            assert!((lhs / rhs - 1.0).abs() < 1e-12, "Parseval {lhs} {rhs}");
        }
        let per = calls / threads;
        let accs: Vec<Acc> = std::thread::scope(|sc| {
            let handles: Vec<_> = (0..threads)
                .map(|th| {
                    let (ek, dirs, w, params) = (&ek, &dirs, &w, &params);
                    sc.spawn(move || {
                        let mut acc = Acc::new(2 * n, dirs.len(), hn);
                        let lo = th * per;
                        let hi = if th + 1 == threads { calls } else { lo + per };
                        for i in lo..hi {
                            let cs = call_seed(b"ntru-trapdoor/stats/samppre-call", k, i);
                            let mut tp = Shake256Prng::from_parts(&[b"target", &cs]);
                            let t = uniform_target(params, &mut tp);
                            let mut prng = Shake256Prng::from_parts(&[b"sampler", &cs]);
                            let mut samp = SamplerZ::new(params, &mut prng);
                            let s = samp_pre_with(ek, &t, &mut samp).expect("rounding guard");
                            acc.add(params, &s, dirs, w);
                        }
                        acc
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut acc = Acc::new(2 * n, dirs.len(), hn);
        for a in accs.iter() {
            acc.merge(a);
        }
        // Sanity: samp_pre_attempt (with the norm check) gives the same preimage as
        // samp_pre_with for the same stream, on the first 4 calls.
        for i in 0..4u64 {
            let cs = call_seed(b"ntru-trapdoor/stats/samppre-call", k, i);
            let mut tp = Shake256Prng::from_parts(&[b"target", &cs]);
            let t = uniform_target(&params, &mut tp);
            let mut p1 = Shake256Prng::from_parts(&[b"sampler", &cs]);
            let a = samp_pre_with(&ek, &t, &mut SamplerZ::new(&params, &mut p1))
                .expect("rounding guard");
            let mut p2 = Shake256Prng::from_parts(&[b"sampler", &cs]);
            let b = samp_pre_attempt(&ek, &t, &mut p2).expect("attempt");
            assert_eq!(a, b);
            assert!(ntru_trapdoor::verify(&pk, &t, &a, params.beta_sq));
        }
        let (lo, hi) = ek.leaf_sigma_range();
        let extra = format!(
            "key {k} seed {} leaves [{lo:.9}, {hi:.9}] dirs {ndirs}",
            seed.iter().map(|b| format!("{b:02x}")).collect::<String>()
        );
        write_acc(
            &dir.join(format!("samppre_key{k}.txt")),
            &params,
            &acc,
            &extra,
        );
        write_norms(&dir.join(format!("norms_key{k}.bin")), &acc.norms);
        let mean = acc.norms.iter().map(|&x| x as f64).sum::<f64>()
            / acc.calls as f64
            / (2.0 * n as f64 * params.sigma * params.sigma);
        eprintln!(
            "key {k}: {} calls, exceed {}, mean ||s||^2/(2d sigma^2) {mean:.6}, {:.1} s",
            acc.calls,
            acc.exceed,
            t0.elapsed().as_secs_f64()
        );

        if k == 0 && fixed > 0 {
            // One coset: per-coordinate means and variances.
            let mut tp = Shake256Prng::new(b"ntru-trapdoor/stats/samppre-fixed-target");
            let t = uniform_target(&params, &mut tp);
            let per = fixed / threads;
            let accs: Vec<Acc> = std::thread::scope(|sc| {
                let handles: Vec<_> = (0..threads)
                    .map(|th| {
                        let (ek, t, params) = (&ek, &t, &params);
                        sc.spawn(move || {
                            let mut acc = Acc::new(2 * n, 0, hn);
                            let lo = th * per;
                            let hi = if th + 1 == threads { fixed } else { lo + per };
                            for i in lo..hi {
                                let cs = call_seed(b"ntru-trapdoor/stats/samppre-fixed", k, i);
                                let mut prng = Shake256Prng::new(&cs);
                                let mut samp = SamplerZ::new(params, &mut prng);
                                let s = samp_pre_with(ek, t, &mut samp).expect("rounding guard");
                                acc.add(params, &s, &[], &[]);
                            }
                            acc
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mut acc = Acc::new(2 * n, 0, hn);
            for a in accs.iter() {
                acc.merge(a);
            }
            write_acc(
                &dir.join("fixed_key0.txt"),
                &params,
                &acc,
                "key 0, one fixed uniform target",
            );
            write_norms(&dir.join("fixed_norms_key0.bin"), &acc.norms);
            eprintln!("fixed target: {} calls", acc.calls);
        }
        if k == 0 && api > 0 {
            // The public API: hedged seeds, attempts, norm check.
            let per = api / threads;
            let parts: Vec<Vec<u64>> = std::thread::scope(|sc| {
                let handles: Vec<_> = (0..threads)
                    .map(|th| {
                        let (ek, params) = (&ek, &params);
                        sc.spawn(move || {
                            let lo = th * per;
                            let hi = if th + 1 == threads { api } else { lo + per };
                            let mut out = Vec::new();
                            for i in lo..hi {
                                let cs = call_seed(b"ntru-trapdoor/stats/samppre-api", k, i);
                                let mut tp = Shake256Prng::from_parts(&[b"target", &cs]);
                                let t = uniform_target(params, &mut tp);
                                let s = samp_pre(ek, &t, &cs).expect("samp_pre");
                                out.push(s.norm_sq());
                            }
                            out
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let norms: Vec<u64> = parts.concat();
            write_norms(&dir.join("api_norms_key0.bin"), &norms);
            eprintln!("public API: {} calls", norms.len());
        }
        drop(sk);
    }
}
