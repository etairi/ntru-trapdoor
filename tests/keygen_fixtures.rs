//! Keygen against independent oracles (design.md §7):
//! - K-GS-1: Falcon's two Gram–Schmidt norms against Sage (N₁ exact, N₂ at 200 bits);
//! - K-SOLVE-1: NTRUSolve on Sage pairs (20 at the PCS set, 50 toy pairs at logn 2-6): a
//!   solution exists iff the resultants are coprime (FLINT), fG − gF = q exactly, and the
//!   coefficients are within 2× of the exact Sage solver's;
//! - K-CDT-3: the f, g sampler on the key-generation stream against a Python re-derivation
//!   (hashlib SHAKE256 and the tables computed from their definition);
//! - K-ORACLE-1: the whole TrapGen candidate loop against a Sage re-derivation (accepted
//!   candidate, rejection counters, f, g, h).
//!
//! Fixtures (tests/fixtures/keygen/) and their generators:
//! - `pairs_pcs.txt`, `pairs_toy.txt`: `sage -python tools/keygen/fixtures.sage.py`;
//! - `cdt_kat.txt`: `python3 tools/keygen/cdt_kat.py` (needs mpmath);
//! - `trapgen_oracle.txt`: `sage -python tools/keygen/oracle.sage.py`.

use std::collections::HashMap;
use std::path::Path;

use ntru_trapdoor::hazmat::{Prng, Shake256Prng, gram_schmidt_sq_norms, ntru_solve, sample_fg};
use ntru_trapdoor::{Params, trapgen_with_stats};

/// One fixture record: the `key value` lines after a `pair` or `case` line.
type Record = HashMap<String, String>;

fn read_fixture(name: &str) -> Vec<Record> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/keygen")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut out: Vec<Record> = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let (k, v) = line.split_once(' ').unwrap_or((line, ""));
        if k == "pair" || k == "case" {
            out.push(Record::new());
        }
        if let Some(r) = out.last_mut() {
            r.insert(k.to_string(), v.to_string());
        }
    }
    out
}

fn ints<T: std::str::FromStr>(s: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',').map(|x| x.parse().unwrap()).collect()
}

/// fG − gF modulo x^n + 1 by schoolbook in i64.
fn ntru_lhs(f: &[i16], g: &[i16], big_f: &[i32], big_g: &[i32]) -> Vec<i64> {
    let n = f.len();
    let mut r = vec![0i64; n];
    for i in 0..n {
        let (fi, gi) = (f[i] as i64, g[i] as i64);
        if fi == 0 && gi == 0 {
            continue;
        }
        for j in 0..n {
            let t = fi * big_g[j] as i64 - gi * big_f[j] as i64;
            if i + j < n {
                r[i + j] += t;
            } else {
                r[i + j - n] -= t;
            }
        }
    }
    r
}

fn equation_holds(q: u32, f: &[i16], g: &[i16], big_f: &[i32], big_g: &[i32]) -> bool {
    let r = ntru_lhs(f, g, big_f, big_g);
    r[0] == q as i64 && r[1..].iter().all(|&x| x == 0)
}

/// K-GS-1.
#[test]
fn gram_schmidt_norms_against_sage() {
    let p = Params::PCS;
    let pairs = read_fixture("pairs_pcs.txt");
    assert_eq!(pairs.len(), 20);
    let mut worst = 0.0f64;
    for r in &pairs {
        let f: Vec<i16> = ints(&r["f"]);
        let g: Vec<i16> = ints(&r["g"]);
        let (n1, n2) = gram_schmidt_sq_norms(&p, &f, &g);
        assert_eq!(n1, r["n1"].parse::<u64>().unwrap(), "pair {}", r["pair"]);
        let exact: f64 = r["n2"].parse().unwrap();
        let rel = (n2 / exact - 1.0).abs();
        worst = worst.max(rel);
        assert!(
            rel <= 2f64.powi(-40),
            "pair {}: N2 {n2} vs {exact} (rel {rel:e})",
            r["pair"]
        );
    }
    eprintln!(
        "K-GS-1: worst relative N2 error {worst:e} = 2^{:.1}",
        worst.log2()
    );
}

/// K-SOLVE-1, PCS pairs.
#[test]
fn ntru_solve_against_sage_pcs() {
    let p = Params::PCS;
    let pairs = read_fixture("pairs_pcs.txt");
    let (mut solved, mut ratio_max) = (0, 0.0f64);
    for r in &pairs {
        let f: Vec<i16> = ints(&r["f"]);
        let g: Vec<i16> = ints(&r["g"]);
        let coprime = r["coprime"] == "1";
        match ntru_solve(p.q, &f, &g) {
            Some((big_f, big_g)) => {
                assert!(
                    coprime,
                    "pair {}: solved, but the resultants share a factor",
                    r["pair"]
                );
                assert!(
                    equation_holds(p.q, &f, &g, &big_f, &big_g),
                    "pair {}",
                    r["pair"]
                );
                let ours = big_f
                    .iter()
                    .chain(&big_g)
                    .map(|x| x.unsigned_abs())
                    .max()
                    .unwrap();
                let sage: u32 = r["sage_max"].parse().unwrap();
                let ratio = ours as f64 / sage as f64;
                ratio_max = ratio_max.max(ratio);
                assert!(
                    ratio <= 2.0,
                    "pair {}: max |F|,|G| {ours} vs Sage {sage}",
                    r["pair"]
                );
                solved += 1;
            }
            None => assert!(
                !coprime,
                "pair {}: coprime resultants but no solution",
                r["pair"]
            ),
        }
    }
    assert_eq!(solved, pairs.iter().filter(|r| r["coprime"] == "1").count());
    eprintln!("K-SOLVE-1 (PCS): {solved} solved; max |F|,|G| at most {ratio_max:.3} x Sage's");
}

/// K-SOLVE-1, toy pairs.
#[test]
fn ntru_solve_against_sage_toy() {
    let pairs = read_fixture("pairs_toy.txt");
    assert_eq!(pairs.len(), 50);
    let mut solved = 0;
    for r in &pairs {
        let q: u32 = r["q"].parse().unwrap();
        let f: Vec<i16> = ints(&r["f"]);
        let g: Vec<i16> = ints(&r["g"]);
        assert_eq!(f.len(), 1 << r["logn"].parse::<u32>().unwrap());
        let coprime = r["coprime"] == "1";
        match ntru_solve(q, &f, &g) {
            Some((big_f, big_g)) => {
                assert!(coprime, "pair {}", r["pair"]);
                assert!(
                    equation_holds(q, &f, &g, &big_f, &big_g),
                    "pair {}",
                    r["pair"]
                );
                solved += 1;
            }
            None => assert!(
                !coprime,
                "pair {}: coprime resultants but no solution",
                r["pair"]
            ),
        }
    }
    eprintln!("K-SOLVE-1 (toy): {solved} of 50 solvable pairs solved");
}

/// Counts the bytes drawn from the inner PRNG.
struct Counting<P: Prng> {
    inner: P,
    bytes: usize,
}

impl<P: Prng> Prng for Counting<P> {
    fn next_u8(&mut self) -> u8 {
        self.bytes += 1;
        self.inner.next_u8()
    }
    fn next_u16(&mut self) -> u16 {
        self.bytes += 2;
        self.inner.next_u16()
    }
    fn next_u64(&mut self) -> u64 {
        self.bytes += 8;
        self.inner.next_u64()
    }
    fn next_bytes(&mut self, dst: &mut [u8]) {
        self.bytes += dst.len();
        self.inner.next_bytes(dst)
    }
}

fn params_by_name(name: &str) -> Params {
    match name {
        "PCS" => Params::PCS,
        "FALCON_1024" => Params::FALCON_1024,
        other => panic!("unknown parameter set {other}"),
    }
}

fn seed32(hex_seed: &str) -> [u8; 32] {
    hex::decode(hex_seed).unwrap().try_into().unwrap()
}

/// K-CDT-3: the first candidate of the key-generation stream.
#[test]
fn sample_fg_kat_against_python() {
    let cases = read_fixture("cdt_kat.txt");
    assert_eq!(cases.len(), 6);
    for r in &cases {
        let p = params_by_name(&r["case"]);
        let seed = seed32(&r["seed"]);
        let inner = Shake256Prng::from_parts(&[
            b"ntru-trapdoor/v1/keygen",
            &p.q.to_le_bytes(),
            &[p.logn as u8],
            &seed,
        ]);
        let mut prng = Counting { inner, bytes: 0 };
        let mut f = vec![0i16; p.n()];
        let mut g = vec![0i16; p.n()];
        sample_fg(&p, &mut prng, &mut f);
        sample_fg(&p, &mut prng, &mut g);
        assert_eq!(f, ints::<i16>(&r["f"]), "{} {}", r["case"], r["seed"]);
        assert_eq!(g, ints::<i16>(&r["g"]), "{} {}", r["case"], r["seed"]);
        assert_eq!(prng.bytes, r["bytes"].parse::<usize>().unwrap());
    }
}

/// K-ORACLE-1: the TrapGen candidate loop against the Sage re-derivation.
#[test]
fn trapgen_against_sage_oracle() {
    let cases = read_fixture("trapgen_oracle.txt");
    assert_eq!(cases.len(), 6);
    for r in &cases {
        let p = params_by_name(&r["case"]);
        assert!(
            r["near"].is_empty(),
            "the oracle flagged near-boundary candidates"
        );
        let seed = seed32(&r["seed"]);
        let (pk, sk, stats) = trapgen_with_stats(&p, &seed).unwrap();
        let counts: HashMap<&str, u32> = r["counts"]
            .split(',')
            .map(|kv| {
                let (k, v) = kv.split_once('=').unwrap();
                (k, v.parse().unwrap())
            })
            .collect();
        let what = format!("{} seed {}: {stats:?}", r["case"], r["seed"]);
        assert_eq!(
            stats.candidates,
            r["candidates"].parse::<u32>().unwrap(),
            "{what}"
        );
        assert_eq!(stats.rejected_norm_fg, counts["norm_fg"], "{what}");
        assert_eq!(stats.rejected_norm_ortho, counts["norm_ortho"], "{what}");
        assert_eq!(
            stats.rejected_not_invertible, counts["not_invertible"],
            "{what}"
        );
        assert_eq!(stats.rejected_solve, counts["solve"], "{what}");
        assert_eq!((stats.rejected_size, stats.rejected_leaf), (0, 0), "{what}");
        assert_eq!(sk.f(), &ints::<i16>(&r["f"])[..], "{what}");
        assert_eq!(sk.g(), &ints::<i16>(&r["g"])[..], "{what}");
        assert_eq!(pk.h().coeffs(), &ints::<u32>(&r["h"])[..], "{what}");
        let big_f: Vec<i32> = sk.big_f().iter().map(|&x| x as i32).collect();
        let big_g: Vec<i32> = sk.big_g().iter().map(|&x| x as i32).collect();
        assert!(
            equation_holds(p.q, sk.f(), sk.g(), &big_f, &big_g),
            "{what}"
        );
    }
}
