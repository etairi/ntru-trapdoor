//! Statistics tool: records every leaf call of ffSampling (centre,
//! leaf value 1/sigma', decision) for TrapGen keys at the PCS set, for the 200-bit replay of
//! `tools/stats/precision_hp.py`.
//!
//! Usage: `cargo run --release --example stats_precision -- OUT_DIR KEYS RANDOM_TARGETS`
//!
//! Keys: `trapgen(&Params::PCS, seed_k)` with seed_k = SHAKE256("ntru-trapdoor/stats/
//! precision-key" || k)[0..32]. Targets per key, in this order (the `kind` field):
//! 0 the zero target; 1 all coefficients q - 1; 2 all coefficients (q - 1)/2;
//! 3 a target aligned with the largest FFT coefficient of F (t_j = q - 1 where a cosine at that
//! evaluation point is positive, else 0), which maximises |FFT(t) FFT(F)| / q at that point,
//! the largest component of the first target vector t0 = -t F / q; 4 the same for f (the
//! second vector t1 = t f / q); 10 uniform targets (RANDOM_TARGETS of them).
//! Each call samples with `SamplerZ` over `Shake256Prng::new(SHAKE256("ntru-trapdoor/stats/
//! precision-call" || k || i))` through `samp_pre_with`, recording every leaf call.
//!
//! Output: `OUT_DIR/key<k>.txt` (lines `f`, `g`, `F`, `G`, `h` with decimal coefficients, and
//! `isigma` with the 2d leaf values as f64 bit patterns in sampling order) and
//! `OUT_DIR/key<k>.bin`, per target, little-endian: kind u32, t (d u32), s0 (d i32),
//! s1 (d i32), mu (2d f64), z (2d i32).

use std::io::Write;

use ntru_trapdoor::hazmat::{
    LeafSampler, Prng, SamplerZ, Shake256Prng, fft, ifft, samp_pre_with, shake256,
};
use ntru_trapdoor::{ExpandedKey, Params, RqPoly, trapgen};

/// Wraps the production leaf sampler and records its inputs and outputs.
struct Recorder<'a, P: Prng> {
    inner: SamplerZ<'a, P>,
    mu: Vec<f64>,
    isigma: Vec<f64>,
    z: Vec<i32>,
}

impl<P: Prng> LeafSampler for Recorder<'_, P> {
    fn sample(&mut self, mu: f64, isigma: f64) -> i32 {
        let z = self.inner.sample(mu, isigma);
        self.mu.push(mu);
        self.isigma.push(isigma);
        self.z.push(z);
        z
    }
}

/// The index (in the FFT representation: real parts in the first half, imaginary parts in the
/// second) of the largest evaluation of `a`.
fn argmax_fft(logn: u32, a: &[i16]) -> usize {
    let n = a.len();
    let hn = n / 2;
    let mut v: Vec<f64> = a.iter().map(|&x| x as f64).collect();
    fft(logn, &mut v);
    (0..hn)
        .max_by(|&i, &j| {
            let mi = v[i] * v[i] + v[i + hn] * v[i + hn];
            let mj = v[j] * v[j] + v[j + hn] * v[j + hn];
            mi.partial_cmp(&mj).unwrap()
        })
        .unwrap()
}

/// t_j = q - 1 where the real polynomial with the single evaluation e^{i 0} at FFT index `k`
/// (and its conjugate) is positive, 0 elsewhere: |FFT(t)_k| is then about (q - 1) d / pi.
fn aligned_target(params: &Params, k: usize) -> Vec<u32> {
    let n = params.n();
    let mut v = vec![0.0f64; n];
    v[k] = 1.0;
    ifft(params.logn, &mut v);
    v.iter()
        .map(|&c| if c > 0.0 { params.q - 1 } else { 0 })
        .collect()
}

/// max_k |FFT(t)_k| / q, the size of the target's largest evaluation.
fn fft_max(params: &Params, t: &[u32]) -> f64 {
    let hn = params.n() / 2;
    let mut v: Vec<f64> = t.iter().map(|&x| x as f64).collect();
    fft(params.logn, &mut v);
    (0..hn)
        .map(|k| (v[k] * v[k] + v[k + hn] * v[k + hn]).sqrt())
        .fold(0.0, f64::max)
        / params.q as f64
}

fn uniform_target(params: &Params, prng: &mut Shake256Prng) -> Vec<u32> {
    let bits = 32 - (params.q - 1).leading_zeros();
    let mask = (1u64 << bits) - 1;
    (0..params.n())
        .map(|_| {
            loop {
                let x = prng.next_u64() & mask;
                if x < params.q as u64 {
                    break x as u32;
                }
            }
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: stats_precision OUT_DIR KEYS RANDOM_TARGETS");
        std::process::exit(2);
    }
    let dir = std::path::PathBuf::from(&args[1]);
    std::fs::create_dir_all(&dir).unwrap();
    let keys: u32 = args[2].parse().expect("KEYS");
    let randoms: u32 = args[3].parse().expect("RANDOM_TARGETS");
    let params = Params::PCS;
    let n = params.n();
    for k in 0..keys {
        let mut seed = [0u8; 32];
        shake256(
            &[b"ntru-trapdoor/stats/precision-key", &k.to_le_bytes()],
            &mut seed,
        );
        let (pk, sk) = trapgen(&params, &seed).expect("trapgen");
        let ek = ExpandedKey::new(&sk).expect("expand");
        let mut targets: Vec<(u32, Vec<u32>)> = vec![
            (0, vec![0; n]),
            (1, vec![params.q - 1; n]),
            (2, vec![(params.q - 1) / 2; n]),
            (
                3,
                aligned_target(&params, argmax_fft(params.logn, sk.big_f())),
            ),
            (4, aligned_target(&params, argmax_fft(params.logn, sk.f()))),
        ];
        let mut tprng =
            Shake256Prng::from_parts(&[b"ntru-trapdoor/stats/precision-targets", &k.to_le_bytes()]);
        for _ in 0..randoms {
            targets.push((10, uniform_target(&params, &mut tprng)));
        }
        let sizes: Vec<String> = targets
            .iter()
            .take(6)
            .map(|(kind, tc)| format!("kind {kind}: {:.1}", fft_max(&params, tc)))
            .collect();
        eprintln!("key {k}: max |FFT(t)|/q by target: {}", sizes.join(", "));
        let mut bin = std::io::BufWriter::new(
            std::fs::File::create(dir.join(format!("key{k}.bin"))).unwrap(),
        );
        let mut leaves: Option<Vec<f64>> = None;
        let mut max_norm = 0u64;
        for (i, (kind, tc)) in targets.iter().enumerate() {
            let t = RqPoly::from_coeffs(&params, tc).unwrap();
            let mut cseed = [0u8; 32];
            shake256(
                &[
                    b"ntru-trapdoor/stats/precision-call",
                    &k.to_le_bytes(),
                    &(i as u32).to_le_bytes(),
                ],
                &mut cseed,
            );
            let mut prng = Shake256Prng::new(&cseed);
            let mut rec = Recorder {
                inner: SamplerZ::new(&params, &mut prng),
                mu: Vec::with_capacity(2 * n),
                isigma: Vec::with_capacity(2 * n),
                z: Vec::with_capacity(2 * n),
            };
            let s = samp_pre_with(&ek, &t, &mut rec).expect("rounding guard");
            assert_eq!(rec.z.len(), 2 * n);
            assert!(
                ntru_trapdoor::verify(&pk, &t, &s, params.beta_sq),
                "key {k} target {i}"
            );
            max_norm = max_norm.max(s.norm_sq());
            match &leaves {
                None => leaves = Some(rec.isigma.clone()),
                Some(l) => assert!(
                    l.iter()
                        .zip(rec.isigma.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                ),
            }
            bin.write_all(&kind.to_le_bytes()).unwrap();
            for &c in tc.iter() {
                bin.write_all(&c.to_le_bytes()).unwrap();
            }
            for &c in s.s0.iter().chain(s.s1.iter()) {
                bin.write_all(&c.to_le_bytes()).unwrap();
            }
            for &m in rec.mu.iter() {
                bin.write_all(&m.to_bits().to_le_bytes()).unwrap();
            }
            for &z in rec.z.iter() {
                bin.write_all(&z.to_le_bytes()).unwrap();
            }
        }
        bin.flush().unwrap();
        let mut txt = std::io::BufWriter::new(
            std::fs::File::create(dir.join(format!("key{k}.txt"))).unwrap(),
        );
        let ints = |v: &[i16]| {
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        };
        writeln!(txt, "# stats_precision key {k}: trapgen(Params::PCS, SHAKE256(\"ntru-trapdoor/stats/precision-key\" || LE32({k})))").unwrap();
        writeln!(
            txt,
            "seed {}",
            seed.iter().map(|b| format!("{b:02x}")).collect::<String>()
        )
        .unwrap();
        writeln!(txt, "targets {}", targets.len()).unwrap();
        writeln!(txt, "f {}", ints(sk.f())).unwrap();
        writeln!(txt, "g {}", ints(sk.g())).unwrap();
        writeln!(txt, "F {}", ints(sk.big_f())).unwrap();
        writeln!(txt, "G {}", ints(sk.big_g())).unwrap();
        writeln!(
            txt,
            "h {}",
            pk.h()
                .coeffs()
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        )
        .unwrap();
        let l = leaves.unwrap();
        writeln!(
            txt,
            "isigma {}",
            l.iter()
                .map(|v| format!("{:016x}", v.to_bits()))
                .collect::<Vec<_>>()
                .join(" ")
        )
        .unwrap();
        txt.flush().unwrap();
        let (lo, hi) = ek.leaf_sigma_range();
        eprintln!(
            "key {k}: leaves [{lo:.6}, {hi:.6}], max|F,G| {}, {} targets, max ||s||^2/(2d sigma^2) {:.4}",
            sk.big_f()
                .iter()
                .chain(sk.big_g().iter())
                .map(|x| x.unsigned_abs())
                .max()
                .unwrap(),
            targets.len(),
            max_norm as f64 / (2.0 * n as f64 * params.sigma * params.sigma)
        );
    }
}
