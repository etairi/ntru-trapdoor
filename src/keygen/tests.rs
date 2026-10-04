//! Unit tests of the keygen module that need crate internals (the integration tests in
//! `tests/keygen_*.rs` use the public API).

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::One;

use super::bigpoly;
use super::flat;
use super::gs;
use super::solve::{ntru_solve, resultant};
use crate::Params;
use crate::prng::Prng;

/// SplitMix64 as a [`Prng`] (independent of the crate's SHAKE256 stream).
struct Mix(u64);

impl Mix {
    fn word(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

impl Prng for Mix {
    fn next_u8(&mut self) -> u8 {
        self.word() as u8
    }
    fn next_u16(&mut self) -> u16 {
        self.word() as u16
    }
    fn next_u64(&mut self) -> u64 {
        self.word()
    }
    fn next_bytes(&mut self, dst: &mut [u8]) {
        for b in dst {
            *b = self.word() as u8;
        }
    }
}

/// A scripted byte stream: returns the bytes of `data` in order.
struct Script {
    data: Vec<u8>,
    pos: usize,
}

impl Prng for Script {
    fn next_u8(&mut self) -> u8 {
        let b = self.data[self.pos];
        self.pos += 1;
        b
    }
    fn next_u16(&mut self) -> u16 {
        u16::from_le_bytes([self.next_u8(), self.next_u8()])
    }
    fn next_u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        self.next_bytes(&mut b);
        u64::from_le_bytes(b)
    }
    fn next_bytes(&mut self, dst: &mut [u8]) {
        for b in dst {
            *b = self.next_u8();
        }
    }
}

/// `true` iff the top 63 bits of `v` equal those of an entry of `table` (the lazy draw then
/// reads the low 65 bits too).
fn ties(table: &[u128], v: u128) -> bool {
    table.iter().any(|&t| t >> 65 == v >> 65)
}

/// Bytes of one lazy CDT draw of V with sign bit `b`: the word (H << 1) | b with H = V >> 65,
/// then, only if H ties with a table entry, the low 65 bits L as L >> 1 (8 bytes) and L & 1
/// (1 byte).
fn draw(table: &[u128], v: u128, b: u8) -> Vec<u8> {
    let h = (v >> 65) as u64;
    let mut out = ((h << 1) | (b as u64 & 1)).to_le_bytes().to_vec();
    if ties(table, v) {
        let l = v & ((1u128 << 65) - 1);
        out.extend_from_slice(&((l >> 1) as u64).to_le_bytes());
        out.push((l & 1) as u8);
    }
    out
}

/// K = #{j : V < T_j}, the full 128-bit count.
fn k_full(table: &[u128], v: u128) -> i16 {
    table.iter().filter(|&&t| v < t).count() as i16
}

#[test]
fn sample_fg_decisions_and_consumption_order() {
    let p = Params::PCS;
    let t = p.fg_table;
    let len = t.len() as i16;
    let mut data = Vec::new();
    // V = 0 < every entry: K = len. Its top bits tie with the last entries (1, 1, 2, 3).
    assert!(ties(t, 0));
    data.extend(draw(t, 0, 1)); // b = 1 -> +len
    data.extend(draw(t, 0, 0)); // b = 0 -> -len
    assert!(!ties(t, u128::MAX));
    data.extend(draw(t, u128::MAX, 1)); // K = 0, b = 1: redraw (8 bytes)
    data.extend(draw(t, u128::MAX, 0)); // K = 0, b = 0 -> 0
    data.extend(draw(t, t[0], 1)); // V = T_0 (a tie): not < T_0, so K = 0, b = 1: redraw
    data.extend(draw(t, t[0] - 1, 1)); // V < T_0 only: K = 1 -> +1
    data.extend(draw(t, t[5], 0)); // V = T_5: K = 5 -> -5
    data.extend(draw(t, t[5] - 1, 1)); // K = 6 -> +6
    let mut s = Script { data, pos: 0 };
    let mut out = [0i16; 6];
    super::sample_fg(&p, &mut s, &mut out);
    assert_eq!(out, [len, -len, 0, 1, -5, 6]);
    assert_eq!(s.pos, s.data.len());
}

/// The lazy draw equals the full 128-bit count, with the fallback forced: for every row j of
/// both tables, top bits H_j - 1, H_j, H_j + 1 and low bits around L_j and at the extremes.
/// The script holds exactly the bytes the draw may read (8, or 17 on a tie), so a missed or a
/// spurious tie fails the test.
#[test]
fn lazy_draw_matches_full_count() {
    let lmask = (1u128 << 65) - 1;
    for table in [Params::PCS.fg_table, Params::FALCON_1024.fg_table] {
        let mut cases = 0usize;
        let mut tie_cases = 0usize;
        for &tj in table {
            let (hj, lj) = (tj >> 65, tj & lmask);
            for h in [hj.wrapping_sub(1), hj, hj + 1] {
                if h >= 1 << 63 {
                    continue;
                }
                for l in [0, 1, lj.wrapping_sub(1), lj, lj + 1, lmask] {
                    if l > lmask {
                        continue;
                    }
                    let v = (h << 65) | l;
                    for b in [0u8, 1] {
                        let data = draw(table, v, b);
                        let tie = data.len() == 17;
                        let mut s = Script { data, pos: 0 };
                        let (k, bit) = super::gauss::draw_half(table, &mut s);
                        assert_eq!((k, bit), (k_full(table, v), b as u64), "v {v:#x}");
                        assert_eq!(s.pos, s.data.len(), "v {v:#x}");
                        cases += 1;
                        tie_cases += tie as usize;
                    }
                }
            }
        }
        assert!(tie_cases > 0 && tie_cases < cases);
    }
}

#[test]
fn sample_fg_moments() {
    // Mean 0 and variance sigma_fg^2 within 6 standard errors, both tables.
    for p in [Params::PCS, Params::FALCON_1024] {
        let mut r = Mix(11);
        let mut out = vec![0i16; 1 << 18];
        super::sample_fg(&p, &mut r, &mut out);
        let m = out.len() as f64;
        let mean = out.iter().map(|&x| x as f64).sum::<f64>() / m;
        let var = out.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / m;
        let s2 = p.sigma_fg * p.sigma_fg;
        assert!(
            mean.abs() < 6.0 * (s2 / m).sqrt(),
            "{}: mean {mean}",
            p.name
        );
        // Var(z^2) = 2 sigma^4 for a Gaussian.
        assert!(
            (var - s2).abs() < 6.0 * (2.0 * s2 * s2 / m).sqrt(),
            "{}: var {var} vs {s2}",
            p.name
        );
    }
}

/// Small random polynomials: coefficients uniform in [-b, b].
fn small_poly(r: &mut Mix, n: usize, b: i16) -> Vec<i16> {
    (0..n)
        .map(|_| (r.word() % (2 * b as u64 + 1)) as i16 - b)
        .collect()
}

#[test]
fn resultant_matches_definition_small() {
    // Res(f, x + 1) = f(-1) = f0 at n = 1; Res(f, x^2 + 1) = f(i) f(-i) = f0^2 + f1^2.
    assert_eq!(resultant(&[3]), BigInt::from(3));
    assert_eq!(resultant(&[3, -4]), BigInt::from(25));
    // x^4 + 1: N(f)(y) = fe(y)^2 - y fo(y)^2 modulo y^2 + 1, then |N(f)(i)|^2.
    let f = [1i16, 2, 3, 4];
    let (fe, fo) = ([1i64, 3], [2i64, 4]);
    let sq = |a: [i64; 2]| [a[0] * a[0] - a[1] * a[1], 2 * a[0] * a[1]];
    let (e2, o2) = (sq(fe), sq(fo));
    // y (o0 + o1 y) = -o1 + o0 y.
    let n1 = [e2[0] + o2[1], e2[1] - o2[0]];
    assert_eq!(resultant(&f), BigInt::from(n1[0] * n1[0] + n1[1] * n1[1]));
}

/// The NTT check of the NTRU equation (`ring::ntt::ntru_equation_holds`, used by key generation
/// and `SecretKey::validate`) against the big-integer oracle `bigpoly::ntru_equation_holds`:
/// solutions of toy pairs at every degree; the same solutions moved by (k f, k g) for random
/// polynomials k, which keeps fG - gF = q but makes F, G large (up to the i16 range); single
/// +-1 faults; and extreme operands (all coefficients at the i16 limits).
#[test]
fn ntt_equation_check_matches_bigint() {
    use crate::ring::ntt;
    let mut r = Mix(77);
    let (mut trues, mut falses) = (0, 0);
    let mut check = |logn: u32, q: u32, f: &[i16], g: &[i16], bf: &[i16], bg: &[i16]| -> bool {
        let w = |x: &[i16]| x.iter().map(|&v| v as i32).collect::<Vec<i32>>();
        let fast = ntt::ntru_equation_holds(logn, q, f, g, bf, bg);
        let slow = bigpoly::ntru_equation_holds(q, f, g, &w(bf), &w(bg));
        assert_eq!(fast, slow, "logn {logn} q {q}");
        if fast {
            trues += 1
        } else {
            falses += 1
        }
        fast
    };
    for logn in 1..=10u32 {
        let n = 1usize << logn;
        for &q in &[17u32, 12289, 1048573] {
            let mut found = 0;
            for _ in 0..40 {
                if found == 2 {
                    break;
                }
                let f = small_poly(&mut r, n, 4);
                let g = small_poly(&mut r, n, 4);
                let Some((bf, bg)) = ntru_solve(q, &f, &g) else {
                    continue;
                };
                let Ok(bf) = bf
                    .iter()
                    .map(|&x| i16::try_from(x))
                    .collect::<Result<Vec<_>, _>>()
                else {
                    continue;
                };
                let Ok(bg) = bg
                    .iter()
                    .map(|&x| i16::try_from(x))
                    .collect::<Result<Vec<_>, _>>()
                else {
                    continue;
                };
                found += 1;
                assert!(check(logn, q, &f, &g, &bf, &bg));
                // (F + k f, G + k g) for a random k: still a solution, with larger coefficients
                // (|k f|, |k g| <= 4 n |k| fill what the i16 range leaves).
                let big = bf
                    .iter()
                    .chain(&bg)
                    .map(|&x| (x as i64).abs())
                    .max()
                    .unwrap();
                let bound = (32767 - big) / (4 * n as i64);
                let k: Vec<i64> = (0..n)
                    .map(|_| (r.word() % (2 * bound as u64 + 1)) as i64 - bound)
                    .collect();
                let negamul = |a: &[i64], b: &[i16]| -> Vec<i64> {
                    let mut out = vec![0i64; n];
                    for i in 0..n {
                        for j in 0..n {
                            let t = a[i] * b[j] as i64;
                            if i + j < n {
                                out[i + j] += t;
                            } else {
                                out[i + j - n] -= t;
                            }
                        }
                    }
                    out
                };
                let (kf, kg) = (negamul(&k, &f), negamul(&k, &g));
                let bf2: Vec<i16> = bf
                    .iter()
                    .zip(&kf)
                    .map(|(&a, &b)| (a as i64 + b) as i16)
                    .collect();
                let bg2: Vec<i16> = bg
                    .iter()
                    .zip(&kg)
                    .map(|(&a, &b)| (a as i64 + b) as i16)
                    .collect();
                assert!(
                    bf.iter()
                        .zip(&kf)
                        .all(|(&a, &b)| (a as i64 + b).abs() <= 32767)
                );
                assert!(
                    bg.iter()
                        .zip(&kg)
                        .all(|(&a, &b)| (a as i64 + b).abs() <= 32767)
                );
                assert!(check(logn, q, &f, &g, &bf2, &bg2));
                // Faults.
                let i = (r.word() as usize) % n;
                let mut x = bf2.clone();
                x[i] = x[i].wrapping_add(1);
                assert!(!check(logn, q, &f, &g, &x, &bg2));
                let mut y = bg.clone();
                y[i] = y[i].wrapping_sub(1);
                assert!(!check(logn, q, &f, &g, &bf, &y));
                // The equation for another modulus fails.
                assert!(!check(logn, q + 2, &f, &g, &bf, &bg));
            }
            // At q = 1048573 and large degrees, F and G of these short pairs may exceed i16.
            assert!(
                found > 0 || q > 12289,
                "no toy solution at logn {logn}, q {q}"
            );
        }
        // Extreme operands: |fG - gF| reaches 2 n 2^30.
        let hi = vec![i16::MAX; n];
        let lo = vec![i16::MIN; n];
        check(logn, 12289, &hi, &lo, &lo, &hi);
        check(logn, 12289, &lo, &hi, &hi, &lo);
    }
    eprintln!("NTRU equation checks: {trues} true, {falses} false");
    assert!(trues > 40 && falses > 60, "{trues} true, {falses} false");
}

#[test]
fn ntru_solve_toy_pairs() {
    // For many random pairs at small degrees and several moduli: a solution exists iff the
    // resultants are coprime, and every solution satisfies fG - gF = q exactly.
    let mut r = Mix(21);
    let mut solved = 0;
    let mut unsolvable = 0;
    for logn in 0..=7u32 {
        let n = 1usize << logn;
        for &q in &[17u32, 257, 12289, 1048573] {
            for _ in 0..12 {
                let f = small_poly(&mut r, n, 4);
                let g = small_poly(&mut r, n, 4);
                let coprime = resultant(&f).gcd(&resultant(&g)).is_one();
                match ntru_solve(q, &f, &g) {
                    Some((big_f, big_g)) => {
                        assert!(coprime, "solution for non-coprime resultants");
                        assert!(
                            bigpoly::ntru_equation_holds(q, &f, &g, &big_f, &big_g),
                            "fG - gF != q at logn {logn}, q {q}"
                        );
                        solved += 1;
                    }
                    None => {
                        assert!(
                            !coprime,
                            "no solution for coprime resultants: logn {logn}, q {q}, f {f:?}, g {g:?}"
                        );
                        unsolvable += 1;
                    }
                }
            }
        }
    }
    assert!(
        solved > 100 && unsolvable > 20,
        "solved {solved}, unsolvable {unsolvable}"
    );
}

#[test]
fn ntru_solve_pcs_size_pairs() {
    // Gaussian pairs of the PCS width (and Falcon-1024's) at d = 1024, not norm-filtered: the
    // solution satisfies the equation and its coefficients have the oracle's sizes (Sage, 300
    // keys: max |F|, |G| = 1263 at the PCS set and 126 at Falcon-1024).
    for (p, limit) in [(Params::PCS, 1 << 12), (Params::FALCON_1024, 1 << 8)] {
        let mut r = Mix(31);
        let n = p.n();
        let mut f = vec![0i16; n];
        let mut g = vec![0i16; n];
        let mut done = 0;
        let mut tries = 0;
        while done < 2 {
            tries += 1;
            assert!(tries < 50);
            super::sample_fg(&p, &mut r, &mut f);
            super::sample_fg(&p, &mut r, &mut g);
            let t = std::time::Instant::now();
            let Some((big_f, big_g)) = ntru_solve(p.q, &f, &g) else {
                continue;
            };
            let dt = t.elapsed();
            assert!(bigpoly::ntru_equation_holds(p.q, &f, &g, &big_f, &big_g));
            let m = big_f
                .iter()
                .chain(&big_g)
                .map(|x| x.unsigned_abs())
                .max()
                .unwrap();
            eprintln!("{}: solved in {dt:?}, max |F|,|G| = {m}", p.name);
            assert!(m < limit, "{}: max |F|,|G| = {m}", p.name);
            done += 1;
        }
    }
}

#[test]
fn ortho_norm_degree_two_exact() {
    // n = 2: f f* + g g* = D = f0^2 + f1^2 + g0^2 + g1^2 (a rational integer), so
    // N2 = q^2 (f0^2 + f1^2 + g0^2 + g1^2) / D^2 = q^2 / D.
    let mut p = Params::PCS;
    p.logn = 1;
    for (f, g) in [([3i16, -4], [1i16, 2]), ([0, 1], [0, 0]), ([5, 5], [-7, 1])] {
        let d = [f[0], f[1], g[0], g[1]]
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>();
        let q = p.q as f64;
        let n2 = gs::ortho_sq_norm(&p, &f, &g);
        assert!(
            (n2 / (q * q / d) - 1.0).abs() < 1e-15,
            "{n2} vs {}",
            q * q / d
        );
        assert_eq!(gs::sq_norm_fg(&f, &g), d as u64);
    }
    // A common root of x^2 + 1 (f = g = 0): N2 is infinite and fails the test.
    let n2 = gs::ortho_sq_norm(&p, &[0, 0], &[0, 0]);
    assert!(!gs::passes_norm_ortho(&p, n2));
}

/// Phase timings of NTRUSolve at the PCS set (run with `--ignored --nocapture`).
#[test]
#[ignore]
fn profile_ntru_solve_phases() {
    use super::solve::babai_reduce;
    use super::xgcd::xgcd;
    use std::time::Instant;
    for p in [Params::PCS, Params::FALCON_1024] {
        let mut r = Mix(41);
        let n = p.n();
        let logn = p.logn as usize;
        let mut f = vec![0i16; n];
        let mut g = vec![0i16; n];
        let mut runs = 0;
        while runs < 5 {
            super::sample_fg(&p, &mut r, &mut f);
            super::sample_fg(&p, &mut r, &mut g);
            let t0 = Instant::now();
            let mut fs: Vec<Vec<BigInt>> = vec![f.iter().map(|&x| BigInt::from(x)).collect()];
            let mut gs: Vec<Vec<BigInt>> = vec![g.iter().map(|&x| BigInt::from(x)).collect()];
            for k in 0..logn {
                let nf = bigpoly::field_norm(&fs[k]);
                let ng = bigpoly::field_norm(&gs[k]);
                fs.push(nf);
                gs.push(ng);
            }
            let t_tower = t0.elapsed();
            let t1 = Instant::now();
            let (d, u, v) = xgcd(&fs[logn][0], &gs[logn][0]);
            let t_xgcd = t1.elapsed();
            if !d.is_one() {
                continue;
            }
            let qb = BigInt::from(p.q);
            let (fl, gl) = (-(v * &qb), u * &qb);
            let l = (fl.bits().max(gl.bits()) / 64 + 2) as usize;
            let mut big_f = flat::Flat::from_big(&[fl], l);
            let mut big_g = flat::Flat::from_big(&[gl], l);
            let mut t_lift = std::time::Duration::ZERO;
            let mut t_babai = vec![];
            for k in (0..logn).rev() {
                let t2 = Instant::now();
                let (mut nf, mut ng) = flat::lift(&big_f, &big_g, &fs[k], &gs[k], 160);
                t_lift += t2.elapsed();
                let t3 = Instant::now();
                babai_reduce(&fs[k], &gs[k], &mut nf, &mut ng);
                t_babai.push((k, t3.elapsed()));
                big_f = nf;
                big_g = ng;
            }
            let total = t0.elapsed();
            let tb: std::time::Duration = t_babai.iter().map(|x| x.1).sum();
            eprintln!(
                "{}: total {total:?}: tower {t_tower:?}, xgcd {t_xgcd:?}, lift {t_lift:?}, babai {tb:?}",
                p.name
            );
            eprintln!(
                "   babai per depth: {:?}",
                t_babai
                    .iter()
                    .map(|(k, t)| format!("{k}:{:.2}ms", t.as_secs_f64() * 1e3))
                    .collect::<Vec<_>>()
            );
            runs += 1;
        }
    }
}

/// A long NTRUSolve loop for sampling profilers (`--ignored`; `NTRU_PROFILE_SECS`, default 8).
#[test]
#[ignore]
fn profile_ntru_solve_loop() {
    let p = Params::PCS;
    let mut r = Mix(51);
    let n = p.n();
    let (mut f, mut g) = (vec![0i16; n], vec![0i16; n]);
    loop {
        super::sample_fg(&p, &mut r, &mut f);
        super::sample_fg(&p, &mut r, &mut g);
        if ntru_solve(p.q, &f, &g).is_some() {
            break;
        }
    }
    let secs: f64 = std::env::var("NTRU_PROFILE_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8.0);
    let t = std::time::Instant::now();
    let mut runs = 0u32;
    while t.elapsed().as_secs_f64() < secs {
        std::hint::black_box(ntru_solve(p.q, &f, &g));
        runs += 1;
    }
    eprintln!(
        "{runs} solves, {:.3} ms each",
        t.elapsed().as_secs_f64() * 1e3 / runs as f64
    );
}

/// The flat Babai reduction equals the `BigInt` reference bit for bit, on the real states of
/// NTRUSolve (every depth of 3 PCS and 2 Falcon-1024 solves) and on random toy states.
#[test]
fn babai_flat_matches_reference() {
    use super::solve::{babai_reduce, babai_reduce_reference};
    use super::xgcd::xgcd;
    let mut compared = 0;
    for (p, keys) in [(Params::PCS, 3), (Params::FALCON_1024, 2)] {
        let mut r = Mix(61);
        let n = p.n();
        let logn = p.logn as usize;
        let (mut f, mut g) = (vec![0i16; n], vec![0i16; n]);
        let mut done = 0;
        while done < keys {
            super::sample_fg(&p, &mut r, &mut f);
            super::sample_fg(&p, &mut r, &mut g);
            let mut fs: Vec<Vec<BigInt>> = vec![f.iter().map(|&x| BigInt::from(x)).collect()];
            let mut gs: Vec<Vec<BigInt>> = vec![g.iter().map(|&x| BigInt::from(x)).collect()];
            for k in 0..logn {
                let (nf, ng) = (bigpoly::field_norm(&fs[k]), bigpoly::field_norm(&gs[k]));
                fs.push(nf);
                gs.push(ng);
            }
            let (d, u, v) = xgcd(&fs[logn][0], &gs[logn][0]);
            if !d.is_one() {
                continue;
            }
            let qb = BigInt::from(p.q);
            let (mut big_f, mut big_g) = (vec![-(v * &qb)], vec![u * &qb]);
            for k in (0..logn).rev() {
                let (lf, lg) = bigpoly::lift(&big_f, &big_g, &fs[k], &gs[k]);
                // The flat lift equals the BigInt lift.
                let l =
                    (big_f.iter().chain(&big_g).map(|x| x.bits()).max().unwrap() / 64 + 2) as usize;
                let (ff, fg) = flat::lift(
                    &flat::Flat::from_big(&big_f, l),
                    &flat::Flat::from_big(&big_g, l),
                    &fs[k],
                    &gs[k],
                    160,
                );
                assert_eq!(
                    (ff.to_big(), fg.to_big()),
                    (lf.clone(), lg.clone()),
                    "lift depth {k}"
                );
                let (mut a, mut b) = (ff, fg);
                babai_reduce(&fs[k], &gs[k], &mut a, &mut b);
                let (a, b) = (a.to_big(), b.to_big());
                let (mut ra, mut rb) = (lf, lg);
                babai_reduce_reference(&fs[k], &gs[k], &mut ra, &mut rb);
                assert_eq!((&a, &b), (&ra, &rb), "{} depth {k}", p.name);
                compared += 1;
                big_f = a;
                big_g = b;
            }
            done += 1;
        }
    }
    // Random toy states: random f, g and (F, G) far larger than a solution would be.
    let mut r = Mix(62);
    for logn in 1..=6u32 {
        let n = 1usize << logn;
        for &(bf, bbig) in &[(5u64, 40u64), (60, 400), (300, 2000)] {
            let rnd = |r: &mut Mix, bits: u64| -> Vec<BigInt> {
                (0..n)
                    .map(|_| {
                        let words: Vec<u32> =
                            (0..bits.div_ceil(32)).map(|_| r.word() as u32).collect();
                        let m = num_bigint::BigUint::new(words) >> (bits.div_ceil(32) * 32 - bits);
                        let x = BigInt::from(m);
                        if r.word() & 1 == 1 { -x } else { x }
                    })
                    .collect()
            };
            let (f, g) = (rnd(&mut r, bf), rnd(&mut r, bf));
            let (big_f, big_g) = (rnd(&mut r, bbig), rnd(&mut r, bbig));
            let l = (bbig / 64 + 2) as usize;
            let (mut a, mut b) = (
                flat::Flat::from_big(&big_f, l),
                flat::Flat::from_big(&big_g, l),
            );
            babai_reduce(&f, &g, &mut a, &mut b);
            let (a, b) = (a.to_big(), b.to_big());
            let (mut ra, mut rb) = (big_f, big_g);
            babai_reduce_reference(&f, &g, &mut ra, &mut rb);
            assert_eq!((&a, &b), (&ra, &rb), "toy logn {logn} bits {bf} {bbig}");
            compared += 1;
        }
    }
    assert!(compared > 60);
}

/// Component timings of one accepted PCS candidate (run with `--ignored --nocapture`).
#[test]
#[ignore]
fn profile_trapgen_components() {
    use crate::{ExpandedKey, RqPoly, SecretKey};
    use std::time::Instant;
    let p = Params::PCS;
    let (_, sk) = super::trapgen(&p, &[7u8; 32]).unwrap();
    let (f, g) = (sk.f().to_vec(), sk.g().to_vec());
    let reps = 50u32;
    let time = |name: &str, mut op: Box<dyn FnMut() + '_>| {
        op();
        let t = Instant::now();
        for _ in 0..reps {
            op();
        }
        eprintln!(
            "{name:>28}: {:9.1} us",
            t.elapsed().as_secs_f64() * 1e6 / reps as f64
        );
    };
    let mut prng = Mix(5);
    let mut buf = vec![0i16; p.n()];
    time(
        "sample_fg (f and g)",
        Box::new(|| {
            super::sample_fg(&p, &mut prng, &mut buf);
            super::sample_fg(&p, &mut prng, &mut buf);
        }),
    );
    time(
        "N1",
        Box::new(|| {
            std::hint::black_box(gs::sq_norm_fg(&f, &g));
        }),
    );
    time(
        "N2",
        Box::new(|| {
            std::hint::black_box(gs::ortho_sq_norm(&p, &f, &g));
        }),
    );
    time(
        "inverse(f) and h = g/f",
        Box::new(|| {
            let fq = RqPoly::from_i16(&p, &f).unwrap();
            let gq = RqPoly::from_i16(&p, &g).unwrap();
            std::hint::black_box(gq.mul(&fq.inverse().unwrap()));
        }),
    );
    time(
        "ntru_solve",
        Box::new(|| {
            std::hint::black_box(ntru_solve(p.q, &f, &g).unwrap());
        }),
    );
    let bf: Vec<i32> = sk.big_f().iter().map(|&x| x as i32).collect();
    let bg: Vec<i32> = sk.big_g().iter().map(|&x| x as i32).collect();
    time(
        "exact fG - gF = q",
        Box::new(|| {
            std::hint::black_box(bigpoly::ntru_equation_holds(p.q, &f, &g, &bf, &bg));
        }),
    );
    let sk2 = SecretKey::from_parts(
        &p,
        f.clone(),
        g.clone(),
        sk.big_f().to_vec(),
        sk.big_g().to_vec(),
    )
    .unwrap();
    time(
        "ExpandedKey::new (leaves)",
        Box::new(|| {
            std::hint::black_box(ExpandedKey::new(&sk2).unwrap());
        }),
    );
    time(
        "is_invertible(f)",
        Box::new(|| {
            std::hint::black_box(RqPoly::from_i16(&p, &f).unwrap().is_invertible());
        }),
    );
}

/// Lehmer's xgcd against num-integer's generic `extended_gcd` on PCS resultants (run with
/// `--ignored --nocapture`).
#[test]
#[ignore]
fn profile_xgcd_against_num_integer() {
    use std::time::Instant;
    let p = Params::PCS;
    let mut r = Mix(71);
    let (mut f, mut g) = (vec![0i16; p.n()], vec![0i16; p.n()]);
    super::sample_fg(&p, &mut r, &mut f);
    super::sample_fg(&p, &mut r, &mut g);
    let (a, b) = (resultant(&f), resultant(&g));
    let t = Instant::now();
    let reps = 20;
    for _ in 0..reps {
        std::hint::black_box(super::xgcd::xgcd(&a, &b));
    }
    let lehmer = t.elapsed() / reps;
    let t = Instant::now();
    let e = a.extended_gcd(&b);
    let generic = t.elapsed();
    let (d, u, v) = super::xgcd::xgcd(&a, &b);
    assert_eq!(d, e.gcd);
    assert_eq!(&u * &a + &v * &b, d);
    eprintln!(
        "resultants of {} and {} bits: Lehmer {lehmer:?}, num-integer extended_gcd {generic:?}",
        a.bits(),
        b.bits()
    );
}
