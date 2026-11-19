//! The kernels against the portable code, on random and extreme inputs (on processors without
//! the instructions the dispatch keeps the portable code, and these tests compare it with
//! itself).
use super::*;
#[cfg(target_arch = "x86_64")]
use crate::prng::keccak_f1600_portable;
use crate::prng::{Prng, Shake256Prng};
use crate::ring::ntt::{P1, P2, mp_add};

fn residues(rng: &mut Shake256Prng, p: u32, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| match i % 17 {
            0 => 0,
            1 => p - 1,
            2 => 1,
            _ => (rng.next_u64() % p as u64) as u32,
        })
        .collect()
}

/// The BMI1 and BMI2 Keccak equals the portable permutation.
#[cfg(target_arch = "x86_64")]
#[test]
fn keccak_bmi_equals_portable() {
    let mut rng = Shake256Prng::new(b"simd-keccak");
    for _ in 0..1000 {
        let mut a = [0u64; 25];
        for x in a.iter_mut() {
            *x = rng.next_u64();
        }
        let mut b = a;
        let ran = keccak_f1600(&mut a);
        keccak_f1600_portable(&mut b);
        if ran {
            assert_eq!(a, b);
        }
    }
}

/// The pointwise kernels equal the portable loops, lane by lane, at every size that they take.
#[test]
fn pointwise_kernels_equal_portable() {
    let mut rng = Shake256Prng::new(b"simd-pointwise");
    for pr in [&P1, &P2] {
        for n in [8usize, 16, 64, 1024] {
            let (a, b, c) = (
                residues(&mut rng, pr.p, n),
                residues(&mut rng, pr.p, n),
                residues(&mut rng, pr.p, n),
            );
            let k = residues(&mut rng, pr.p, 1)[0];
            let mut x = a.clone();
            if mmul_slices(pr, &mut x, &b) {
                let want: Vec<u32> = a.iter().zip(&b).map(|(&u, &v)| pr.mmul(u, v)).collect();
                assert_eq!(x, want, "mmul_slices p {} n {n}", pr.p);
            }
            let mut x = a.clone();
            if mmul_slices_const(pr, &mut x, Some(&b), k) {
                let want: Vec<u32> = a
                    .iter()
                    .zip(&b)
                    .map(|(&u, &v)| pr.mmul(pr.mmul(u, v), k))
                    .collect();
                assert_eq!(x, want, "mmul_slices_const p {} n {n}", pr.p);
            }
            let mut x = a.clone();
            if mmul_slices_const(pr, &mut x, None, k) {
                let want: Vec<u32> = a.iter().map(|&u| pr.mmul(u, k)).collect();
                assert_eq!(x, want, "mmul_slices_const (no b) p {} n {n}", pr.p);
            }
            let mut x = c.clone();
            if mmul_add_slices(pr, &mut x, &a, &b) {
                let want: Vec<u32> = c
                    .iter()
                    .zip(a.iter().zip(&b))
                    .map(|(&s, (&u, &v))| mp_add(s, pr.mmul(u, v), pr.p))
                    .collect();
                assert_eq!(x, want, "mmul_add_slices p {} n {n}", pr.p);
            }
        }
    }
}
