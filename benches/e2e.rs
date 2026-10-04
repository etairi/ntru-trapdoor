//! End-to-end benchmarks on keys made by this crate's own TrapGen (integrate stage), at both
//! presets: key setup (TrapGen plus the expanded key), SampPre per call, Verify, SampPre
//! followed by Verify, the key encodings, and the throughput of SampPre on several threads
//! sharing one `ExpandedKey` (it is `Send + Sync`).
//!
//! `benches/sign.rs` and `benches/keygen.rs` measure the components (R_q arithmetic, FFT, PRNG,
//! SamplerZ, NTRUSolve, ...); `benches/sign.rs` uses a Sage fixture key at the PCS set and an
//! fn-dsa key at Falcon-1024.
//!
//! Run: `CARGO_TARGET_DIR=target CARGO_BUILD_JOBS=4 cargo bench --bench e2e`. Criterion keeps
//! its results in `target/criterion/e2e*`.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ntru_trapdoor::hazmat::{Prng, Shake256Prng};
use ntru_trapdoor::{ExpandedKey, Params, PublicKey, RqPoly, SecretKey, samp_pre, trapgen, verify};

fn seed(tag: u8, k: u64) -> [u8; 32] {
    let mut s = [tag; 32];
    s[..8].copy_from_slice(&k.to_le_bytes());
    s
}

/// 64 uniformly random targets of `p`.
fn targets(p: &Params) -> Vec<RqPoly> {
    let mut prng = Shake256Prng::new(b"bench-e2e-targets");
    (0..64)
        .map(|_| {
            let c: Vec<u32> = (0..p.n())
                .map(|_| (prng.next_u64() % p.q as u64) as u32)
                .collect();
            RqPoly::from_coeffs(p, &c).unwrap()
        })
        .collect()
}

fn presets() -> [(&'static str, Params); 2] {
    [("pcs", Params::PCS), ("falcon", Params::FALCON_1024)]
}

fn bench_setup(c: &mut Criterion) {
    let mut g = c.benchmark_group("e2e_setup");
    g.sample_size(30);
    for (name, p) in presets() {
        // TrapGen and the expanded key, a fresh seed per iteration (the mean over the key
        // distribution, including the variable number of candidates).
        let mut k = 0u64;
        g.bench_function(format!("trapgen_expand/{name}"), |b| {
            b.iter(|| {
                k += 1;
                let (pk, sk) = trapgen(&p, &seed(0xe1, k)).unwrap();
                let ek = ExpandedKey::new(&sk).unwrap();
                black_box((pk, sk, ek))
            })
        });
    }
    g.finish();
}

fn bench_sign_verify(c: &mut Criterion) {
    let mut g = c.benchmark_group("e2e");
    for (name, p) in presets() {
        let (pk, sk) = trapgen(&p, &seed(0xe2, 0)).unwrap();
        let ek = ExpandedKey::new(&sk).unwrap();
        let ts = targets(&p);
        g.bench_function(format!("expand/{name}"), |b| {
            b.iter(|| ExpandedKey::new(black_box(&sk)).unwrap())
        });
        let mut ctr = 0u64;
        g.bench_function(format!("samp_pre/{name}"), |b| {
            b.iter(|| {
                ctr += 1;
                let t = &ts[(ctr % 64) as usize];
                samp_pre(&ek, black_box(t), &seed(0xe3, ctr)).unwrap()
            })
        });
        let pre: Vec<_> = ts
            .iter()
            .enumerate()
            .map(|(i, t)| samp_pre(&ek, t, &seed(0xe4, i as u64)).unwrap())
            .collect();
        let mut ctr = 0usize;
        g.bench_function(format!("verify/{name}"), |b| {
            b.iter(|| {
                ctr = (ctr + 1) % 64;
                assert!(verify(
                    &pk,
                    black_box(&ts[ctr]),
                    black_box(&pre[ctr]),
                    p.beta_sq
                ))
            })
        });
        let mut ctr = 0u64;
        g.bench_function(format!("samp_pre_verify/{name}"), |b| {
            b.iter(|| {
                ctr += 1;
                let t = &ts[(ctr % 64) as usize];
                let s = samp_pre(&ek, black_box(t), &seed(0xe5, ctr)).unwrap();
                assert!(verify(&pk, t, &s, p.beta_sq));
                s
            })
        });
        let pkb = pk.to_bytes();
        let skb = sk.to_bytes();
        g.bench_function(format!("pk_decode/{name}"), |b| {
            b.iter(|| PublicKey::from_bytes(&p, black_box(&pkb)).unwrap())
        });
        g.bench_function(format!("sk_decode_validate/{name}"), |b| {
            b.iter(|| SecretKey::from_bytes(&p, black_box(&skb)).unwrap())
        });
        let t0 = ts[0].clone();
        g.bench_function(format!("ring_inverse/{name}"), |b| {
            b.iter(|| black_box(&t0).inverse().unwrap())
        });
    }
    g.finish();
}

/// SampPre throughput with `threads` threads sharing one expanded key: each iteration runs
/// `threads * PER_THREAD` calls (criterion reports calls per second).
fn bench_throughput(c: &mut Criterion) {
    const PER_THREAD: u64 = 32;
    let p = Params::PCS;
    let (_, sk) = trapgen(&p, &seed(0xe6, 0)).unwrap();
    let ek = ExpandedKey::new(&sk).unwrap();
    let ts = targets(&p);
    let mut g = c.benchmark_group("e2e_throughput");
    g.sample_size(20);
    for threads in [1u64, 2, 4, 8] {
        g.throughput(Throughput::Elements(threads * PER_THREAD));
        let mut round = 0u64;
        g.bench_function(format!("samp_pre/pcs/{threads}_threads"), |b| {
            b.iter(|| {
                round += 1;
                std::thread::scope(|s| {
                    for th in 0..threads {
                        let (ek, ts) = (&ek, &ts);
                        s.spawn(move || {
                            for i in 0..PER_THREAD {
                                let k = (round * 64 + th) * PER_THREAD + i;
                                let t = &ts[(k % 64) as usize];
                                black_box(samp_pre(ek, t, &seed(0xe7, k)).unwrap());
                            }
                        });
                    }
                })
            })
        });
    }
    g.finish();
}

criterion_group!(group, bench_setup, bench_sign_verify, bench_throughput);
criterion_main!(group);
