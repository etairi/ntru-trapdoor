//! Benchmarks of key generation: TrapGen, NTRUSolve, the f, g sampler, the Gram–Schmidt norms
//! and the secret-key codec.
//!
//! Plan: design.md §8 (criterion; PCS set first, Falcon-1024 for comparison).
//!
//! Run: `CARGO_TARGET_DIR=target CARGO_BUILD_JOBS=4 cargo bench --bench keygen`. Criterion keeps
//! its results in `target/criterion/`. Before the benchmarks start, the candidate statistics
//! of 64 keys per set are printed to stderr: mean candidates and rejections per key, from
//! `KeygenStats`.
//!
//! `keygen/trapgen` draws a fresh seed for every iteration. So it measures the mean over the
//! key distribution, including the variable number of candidates (about 19 per PCS key). The
//! solver benchmarks use one fixed solvable pair per set (the first key of seed 0).

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use ntru_trapdoor::hazmat::{Shake256Prng, gram_schmidt_sq_norms, ntru_solve, sample_fg};
use ntru_trapdoor::{KeygenStats, Params, SecretKey, trapgen, trapgen_with_stats};

fn seed(k: u64) -> [u8; 32] {
    let mut s = [0u8; 32];
    s[..8].copy_from_slice(&k.to_le_bytes());
    s[31] = 0xbe;
    s
}

/// Mean candidates and rejections per key over `keys` seeds.
fn report_stats(p: &Params, keys: u64) {
    let mut sum = KeygenStats::default();
    let t = std::time::Instant::now();
    for k in 0..keys {
        let (_, _, s) = trapgen_with_stats(p, &seed(1_000_000 + k)).unwrap();
        sum.candidates += s.candidates;
        sum.rejected_norm_fg += s.rejected_norm_fg;
        sum.rejected_norm_ortho += s.rejected_norm_ortho;
        sum.rejected_not_invertible += s.rejected_not_invertible;
        sum.rejected_solve += s.rejected_solve;
        sum.rejected_size += s.rejected_size;
        sum.rejected_leaf += s.rejected_leaf;
    }
    let per = |x: u32| x as f64 / keys as f64;
    eprintln!(
        "keygen stats {} ({keys} keys, {:.1} ms/key): candidates {:.2}/key; rejected per key: \
         norm_fg {:.2}, norm_ortho {:.2}, not_invertible {:.3}, solve {:.3}, size {:.3}, leaf {:.3}",
        p.name,
        t.elapsed().as_secs_f64() * 1e3 / keys as f64,
        per(sum.candidates),
        per(sum.rejected_norm_fg),
        per(sum.rejected_norm_ortho),
        per(sum.rejected_not_invertible),
        per(sum.rejected_solve),
        per(sum.rejected_size),
        per(sum.rejected_leaf)
    );
}

fn benches(c: &mut Criterion) {
    let sets = [("pcs", Params::PCS), ("falcon1024", Params::FALCON_1024)];
    for (_, p) in &sets {
        report_stats(p, 64);
    }

    let mut g = c.benchmark_group("keygen");
    g.sample_size(30);
    for (name, p) in &sets {
        let mut k = 0u64;
        g.bench_function(format!("trapgen/{name}"), |b| {
            b.iter(|| {
                k += 1;
                black_box(trapgen(p, &seed(k)).unwrap())
            })
        });
        let (_, sk) = trapgen(p, &seed(0)).unwrap();
        g.bench_function(format!("ntru_solve/{name}"), |b| {
            b.iter(|| black_box(ntru_solve(p.q, black_box(sk.f()), black_box(sk.g())).unwrap()))
        });
    }
    g.finish();

    let mut g = c.benchmark_group("keygen");
    let p = Params::PCS;
    let (_, sk) = trapgen(&p, &seed(0)).unwrap();
    g.bench_function("sample_fg/pcs/2048", |b| {
        let mut prng = Shake256Prng::new(b"bench sample_fg");
        let mut out = vec![0i16; 2048];
        b.iter(|| {
            sample_fg(&p, &mut prng, &mut out);
            black_box(&out);
        })
    });
    g.bench_function("gs_norms/pcs", |b| {
        b.iter(|| {
            black_box(gram_schmidt_sq_norms(
                &p,
                black_box(sk.f()),
                black_box(sk.g()),
            ))
        })
    });
    let skb = sk.to_bytes();
    g.bench_function("codec/pcs/sk_encode", |b| {
        b.iter(|| black_box(sk.to_bytes()))
    });
    g.bench_function("codec/pcs/sk_decode", |b| {
        b.iter(|| black_box(SecretKey::from_bytes(&p, black_box(&skb)).unwrap()))
    });
    g.bench_function("validate/pcs", |b| b.iter(|| sk.validate().unwrap()));
    g.bench_function("public_key/pcs", |b| {
        b.iter(|| black_box(sk.public_key().unwrap()))
    });
    g.finish();
}

criterion_group!(group, benches);
criterion_main!(group);
