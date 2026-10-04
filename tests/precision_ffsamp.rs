//! S-PREC-1 (design.md §7): the f64 ffSampling against a 200-bit replay.
//!
//! The fixtures (`tools/sign/precision_fixture.sage.py`) hold Sage keys, a target, the integer
//! decisions z of a 200-bit run of ffLDL + ffSampling (fn-dsa's leaf order), the 200-bit centre
//! and width at every leaf, and the exact preimage s = (hm, 0) - z B. A recording leaf sampler
//! feeds the same decisions to `samp_pre_with`, so both runs follow the same path; then:
//! - every f64 centre is within 2^-36 sigma' of the 200-bit centre (the survey measured
//!   2^-38.0 at the PCS set, so the bound has a 4x margin);
//! - every f64 width 1/isigma is within 2^-45 (relative) of the 200-bit width;
//! - the final s equals the exact s (the lattice point is rounded exactly).
//!
//! Run: `cargo test --test precision_ffsamp -- --nocapture` (prints the measured errors).

use ntru_trapdoor::hazmat::{LeafSampler, samp_pre_with};
use ntru_trapdoor::{ExpandedKey, Params, RqPoly, SecretKey};

/// Replays recorded decisions and records the sampler's inputs.
struct Replay {
    z: Vec<i32>,
    next: usize,
    seen: Vec<(f64, f64)>,
}

impl LeafSampler for Replay {
    fn sample(&mut self, mu: f64, isigma: f64) -> i32 {
        self.seen.push((mu, isigma));
        let z = self.z[self.next];
        self.next += 1;
        z
    }
}

struct Case {
    f: Vec<i16>,
    g: Vec<i16>,
    big_f: Vec<i16>,
    big_g: Vec<i16>,
    hm: Vec<u32>,
    s0: Vec<i32>,
    s1: Vec<i32>,
    /// (z, centre hi, centre lo, sigma') per leaf, in sampling order.
    leaves: Vec<(i32, f64, f64, f64)>,
}

fn f64_hex(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap())
}

fn load(name: &str) -> Vec<Case> {
    let path = format!("{}/tests/fixtures/sign/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).unwrap();
    let mut cases = Vec::new();
    let mut cur: Option<Case> = None;
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let head = it.next().unwrap();
        let ints = |it: std::str::SplitWhitespace| -> Vec<i64> {
            it.map(|x| x.parse::<i64>().unwrap()).collect()
        };
        match head {
            "case" => {
                if let Some(c) = cur.take() {
                    cases.push(c);
                }
                cur = Some(Case {
                    f: vec![],
                    g: vec![],
                    big_f: vec![],
                    big_g: vec![],
                    hm: vec![],
                    s0: vec![],
                    s1: vec![],
                    leaves: vec![],
                });
            }
            "f" | "g" | "F" | "G" => {
                let v: Vec<i16> = ints(it).iter().map(|&x| x as i16).collect();
                let c = cur.as_mut().unwrap();
                match head {
                    "f" => c.f = v,
                    "g" => c.g = v,
                    "F" => c.big_f = v,
                    _ => c.big_g = v,
                }
            }
            "h" => {}
            "hm" => cur.as_mut().unwrap().hm = ints(it).iter().map(|&x| x as u32).collect(),
            "s0" => cur.as_mut().unwrap().s0 = ints(it).iter().map(|&x| x as i32).collect(),
            "s1" => cur.as_mut().unwrap().s1 = ints(it).iter().map(|&x| x as i32).collect(),
            z => {
                let z: i32 = z.parse().unwrap();
                let hi = f64_hex(it.next().unwrap());
                let lo = f64_hex(it.next().unwrap());
                let s = f64_hex(it.next().unwrap());
                cur.as_mut().unwrap().leaves.push((z, hi, lo, s));
            }
        }
    }
    cases.extend(cur);
    cases
}

/// Returns (max centre error / sigma', max relative width error) over the leaves.
fn run(params: &Params, case: &Case) -> (f64, f64) {
    let sk = SecretKey::from_parts(
        params,
        case.f.clone(),
        case.g.clone(),
        case.big_f.clone(),
        case.big_g.clone(),
    )
    .unwrap();
    let ek = ExpandedKey::new(&sk).unwrap();
    let t = RqPoly::from_coeffs(params, &case.hm).unwrap();
    let mut rep = Replay {
        z: case.leaves.iter().map(|l| l.0).collect(),
        next: 0,
        seen: Vec::new(),
    };
    let s = samp_pre_with(&ek, &t, &mut rep).expect("rounding guard");
    assert_eq!(rep.seen.len(), 2 * params.n());
    assert_eq!(s.s0, case.s0, "s0 differs from the exact preimage");
    assert_eq!(s.s1, case.s1, "s1 differs from the exact preimage");
    let (mut dc_max, mut ds_max) = (0.0f64, 0.0f64);
    for (&(mu, isigma), &(_, hi, lo, sigma)) in rep.seen.iter().zip(case.leaves.iter()) {
        // mu - (hi + lo): mu - hi is exact when they are close (Sterbenz).
        let dc = ((mu - hi) - lo).abs() / sigma;
        let ds = (1.0 / isigma - sigma).abs() / sigma;
        dc_max = dc_max.max(dc);
        ds_max = ds_max.max(ds);
    }
    (dc_max, ds_max)
}

#[test]
fn precision_pcs() {
    for (k, case) in load("precision_pcs.txt").iter().enumerate() {
        let (dc, ds) = run(&Params::PCS, case);
        println!(
            "PCS case {k}: max |dc|/sigma' = {dc:.3e} (2^{:.1}), max |dsigma'|/sigma' = {ds:.3e} (2^{:.1})",
            dc.log2(),
            ds.log2()
        );
        assert!(dc <= 2f64.powi(-36), "centre error {dc:e}");
        assert!(ds <= 2f64.powi(-45), "width error {ds:e}");
    }
}

#[test]
fn precision_falcon1024() {
    for (k, case) in load("precision_falcon.txt").iter().enumerate() {
        let (dc, ds) = run(&Params::FALCON_1024, case);
        println!(
            "Falcon-1024 case {k}: max |dc|/sigma' = {dc:.3e} (2^{:.1}), max |dsigma'|/sigma' = {ds:.3e} (2^{:.1})",
            dc.log2(),
            ds.log2()
        );
        assert!(dc <= 2f64.powi(-36), "centre error {dc:e}");
        assert!(ds <= 2f64.powi(-45), "width error {ds:e}");
    }
}
