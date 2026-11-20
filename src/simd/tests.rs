//! The kernels against the portable code, on random and extreme inputs (on processors without
//! the instructions the dispatch keeps the portable code, and these tests compare it with
//! itself).
use super::*;
use crate::prng::{Prng, Shake256Prng, keccak_f1600_portable};
use crate::ring::ntt::{ModQ, P1, P2, mp_add, mp_sub};

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

/// Keccak states: all zeros, all ones, states with distinct lanes (counters, byte patterns, and
/// one set bit at a time in each lane), then `n` random states.
fn keccak_states(n: usize) -> Vec<[u64; 25]> {
    let mut v = vec![
        [0u64; 25],
        [u64::MAX; 25],
        core::array::from_fn(|i| i as u64 + 1),
        core::array::from_fn(|i| (0x0101_0101_0101_0101 * (i as u64 + 1)) ^ (1 << (63 - i))),
        core::array::from_fn(|i| !(((i as u64) << 32) | (i as u64 * 7))),
    ];
    for i in 0..25 {
        let mut s = [0u64; 25];
        s[i] = 1 << (i * 13 % 64);
        v.push(s);
    }
    let mut rng = Shake256Prng::new(b"simd-keccak");
    v.extend((0..n).map(|_| core::array::from_fn(|_| rng.next_u64())));
    v
}

/// Whether the processor has the features of `keccak_f1600_bmi`.
#[cfg(target_arch = "x86_64")]
fn bmi_features() -> bool {
    std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("bmi1")
        && std::is_x86_feature_detected!("bmi2")
        && std::is_x86_feature_detected!("lzcnt")
        && std::is_x86_feature_detected!("popcnt")
}

/// `f` permutes each state as the portable code does.
fn check_keccak_1(name: &str, states: &[[u64; 25]], f: impl Fn(&mut [u64; 25])) {
    for (i, s) in states.iter().enumerate() {
        let (mut got, mut want) = (*s, *s);
        f(&mut got);
        keccak_f1600_portable(&mut want);
        assert_eq!(got, want, "{name}, state {i}");
    }
}

/// `f` permutes `N` states at once as the portable code permutes each, with every state of
/// `states` in every position.
fn check_keccak_n<const N: usize>(
    name: &str,
    states: &[[u64; 25]],
    f: impl Fn(&mut [[u64; 25]; N]),
) {
    for shift in 0..N {
        for (i, group) in states[shift..].as_chunks::<N>().0.iter().enumerate() {
            let mut got = *group;
            f(&mut got);
            for (k, (g, s)) in got.iter().zip(group).enumerate() {
                let mut want = *s;
                keccak_f1600_portable(&mut want);
                assert_eq!(*g, want, "{name}, shift {shift}, group {i}, position {k}");
            }
        }
    }
}

/// Every Keccak kernel of this processor equals the portable permutation, whatever the dispatch
/// chooses.
#[test]
fn keccak_kernels_equal_portable() {
    let states = keccak_states(4000);
    #[cfg(target_arch = "x86_64")]
    {
        if bmi_features() {
            // SAFETY: the processor has AVX2, BMI1, BMI2, LZCNT and POPCNT.
            check_keccak_1("bmi", &states, |s| unsafe {
                crate::prng::keccak_f1600_bmi(s)
            });
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the processor has AVX2.
            let f = |s: &mut _| unsafe { keccak_x86::keccak_f1600_x4_avx2(s) };
            check_keccak_n::<4>("x4 avx2", &states, f);
        }
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
            // SAFETY: the processor has AVX-512F and AVX-512VL.
            let f = |s: &mut _| unsafe { keccak_x86::keccak_f1600_avx512(s) };
            check_keccak_1("avx512", &states, f);
            // SAFETY: as above.
            let f = |s: &mut _| unsafe { keccak_x86::keccak_f1600_x4_avx512(s) };
            check_keccak_n::<4>("x4 avx512", &states, f);
            // SAFETY: as above.
            let f = |s: &mut _| unsafe { keccak_x86::keccak_f1600_x8_avx512(s) };
            check_keccak_n::<8>("x8 avx512", &states, f);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("sha3") {
        // SAFETY: the processor has the SHA3 extension.
        check_keccak_1("sha3", &states, |s| unsafe { keccak_neon::keccak_f1600(s) });
        // SAFETY: as above.
        let f = |s: &mut _| unsafe { keccak_neon::keccak_f1600_x2(s) };
        check_keccak_n::<2>("x2 sha3", &states, f);
        // SAFETY: as above.
        let f = |s: &mut _| unsafe { keccak_neon::keccak_f1600_x2_mixed(s) };
        check_keccak_n::<2>("x2 mixed", &states, f);
        // SAFETY: as above.
        let f = |s: &mut _| unsafe { keccak_neon::keccak_f1600_x3(s) };
        check_keccak_n::<3>("x3 hybrid", &states, f);
    }
}

/// The Keccak dispatch: each function permutes as the portable code does, or returns `false`
/// and leaves its states untouched.
#[test]
fn keccak_dispatch_equals_portable() {
    fn or_portable<const N: usize>(ran: bool, before: &[[u64; 25]; N], s: &mut [[u64; 25]; N]) {
        if !ran {
            assert_eq!(s, before);
            s.iter_mut().for_each(keccak_f1600_portable);
        }
    }
    let states = keccak_states(1000);
    check_keccak_1("keccak_f1600", &states, |s| {
        let before = *s;
        if !keccak_f1600(s) {
            assert_eq!(*s, before);
            keccak_f1600_portable(s);
        }
    });
    check_keccak_n::<2>("keccak_f1600_x2", &states, |s| {
        let before = *s;
        or_portable(keccak_f1600_x2(s), &before, s);
    });
    check_keccak_n::<3>("keccak_f1600_x3", &states, |s| {
        let before = *s;
        or_portable(keccak_f1600_x3(s), &before, s);
    });
    check_keccak_n::<4>("keccak_f1600_x4", &states, |s| {
        let before = *s;
        or_portable(keccak_f1600_x4(s), &before, s);
    });
    check_keccak_n::<8>("keccak_f1600_x8", &states, |s| {
        let before = *s;
        or_portable(keccak_f1600_x8(s), &before, s);
    });
}

/// Nanoseconds per call and per state of the Keccak kernels of this processor, each call
/// permuting the output of the previous one, as in a SHAKE256 squeeze. The kernels alternate
/// batch by batch, and the median of 21 batches is printed. Run with `cargo test --release
/// --features simd --lib keccak_timings -- --ignored --nocapture`.
#[test]
#[ignore]
fn keccak_timings() {
    use std::hint::black_box;
    use std::time::Instant;

    /// A kernel: its name, its number of states, and a closure that runs it `n` times.
    type Kernel = (&'static str, usize, Box<dyn FnMut(u32)>);
    fn one(name: &'static str, mut f: impl FnMut(&mut [u64; 25]) + 'static) -> Kernel {
        let mut s = [7u64; 25];
        let run = move |n: u32| {
            for _ in 0..n {
                f(black_box(&mut s));
            }
        };
        (name, 1, Box::new(run))
    }
    fn many<const N: usize>(
        name: &'static str,
        mut f: impl FnMut(&mut [[u64; 25]; N]) + 'static,
    ) -> Kernel {
        let mut s: [[u64; 25]; N] = core::array::from_fn(|k| [k as u64 + 1; 25]);
        let run = move |n: u32| {
            for _ in 0..n {
                f(black_box(&mut s));
            }
        };
        (name, N, Box::new(run))
    }

    let mut kernels: Vec<Kernel> = vec![one("portable", keccak_f1600_portable)];
    #[cfg(target_arch = "x86_64")]
    {
        if bmi_features() {
            // SAFETY: the processor has AVX2, BMI1, BMI2, LZCNT and POPCNT.
            kernels.push(one("bmi", |s| unsafe { crate::prng::keccak_f1600_bmi(s) }));
        }

        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the processor has AVX2.
            let f = |s: &mut _| unsafe { keccak_x86::keccak_f1600_x4_avx2(s) };
            kernels.push(many::<4>("x4 avx2", f));
        }
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
            // SAFETY: the processor has AVX-512F and AVX-512VL.
            let f = |s: &mut _| unsafe { keccak_x86::keccak_f1600_avx512(s) };
            kernels.push(one("avx512", f));
            // SAFETY: as above.
            let f = |s: &mut _| unsafe { keccak_x86::keccak_f1600_x4_avx512(s) };
            kernels.push(many::<4>("x4 avx512", f));
            // SAFETY: as above.
            let f = |s: &mut _| unsafe { keccak_x86::keccak_f1600_x8_avx512(s) };
            kernels.push(many::<8>("x8 avx512", f));
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("sha3") {
        // SAFETY: the processor has the SHA3 extension.
        kernels.push(one("sha3", |s| unsafe { keccak_neon::keccak_f1600(s) }));
        // SAFETY: as above.
        let f = |s: &mut _| unsafe { keccak_neon::keccak_f1600_x2(s) };
        kernels.push(many::<2>("x2 sha3", f));
        // SAFETY: as above.
        let f = |s: &mut _| unsafe { keccak_neon::keccak_f1600_x2_mixed(s) };
        kernels.push(many::<2>("x2 mixed", f));
        // SAFETY: as above.
        let f = |s: &mut _| unsafe { keccak_neon::keccak_f1600_x3(s) };
        kernels.push(many::<3>("x3 hybrid", f));
    }
    kernels.push(one("dispatch", |s| {
        if !keccak_f1600(s) {
            keccak_f1600_portable(s)
        }
    }));

    const CALLS: u32 = 20_000;
    let mut ns: Vec<Vec<f64>> = vec![Vec::new(); kernels.len()];
    for (_, _, run) in kernels.iter_mut() {
        run(CALLS / 10);
    }
    for _ in 0..21 {
        for ((_, _, run), t) in kernels.iter_mut().zip(ns.iter_mut()) {
            let start = Instant::now();
            run(CALLS);
            t.push(start.elapsed().as_nanos() as f64 / CALLS as f64);
        }
    }
    for ((name, ways, _), t) in kernels.iter().zip(ns.iter_mut()) {
        t.sort_by(f64::total_cmp);
        let (med, lo, hi) = (t[t.len() / 2], t[0], t[t.len() - 1]);
        println!(
            "keccak {name:>10}: {med:7.1} ns per call, {:6.1} ns per state (min {lo:.1}, max {hi:.1})",
            med / *ways as f64
        );
    }
}

/// `samp_pre_batch` returns what `samp_pre` returns, call by call, for 1 to 5, 7 and 256 calls
/// (groups of every width up to 8, with and without calls left over), and so it does for a key
/// that allows no attempt, which `samp_pre_group` runs call by call (`Params::validate` refuses
/// such a key, so that guard is reached only from inside the crate).
#[test]
fn samp_pre_batch_sizes_equal_samp_pre() {
    let params = crate::Params::PCS;
    let (_, sk) = crate::trapgen(&params, &[0x64; 32]).unwrap();
    let ek = crate::ExpandedKey::new(&sk).unwrap();
    let mut rng = Shake256Prng::new(b"batch-sizes");
    let targets: Vec<crate::RqPoly> = (0..256)
        .map(|_| {
            let c: Vec<u32> = (0..params.n())
                .map(|_| (rng.next_u64() % params.q as u64) as u32)
                .collect();
            crate::RqPoly::from_coeffs(&params, &c).unwrap()
        })
        .collect();
    let seeds: Vec<[u8; 32]> = (0..256u32)
        .map(|i| core::array::from_fn(|j| (i as u8).wrapping_mul(29) ^ j as u8))
        .collect();
    let items: Vec<(&crate::RqPoly, &[u8; 32])> = targets.iter().zip(seeds.iter()).collect();
    let want: Vec<_> = items
        .iter()
        .map(|&(t, seed)| crate::samp_pre(&ek, t, seed))
        .collect();
    for n in [1, 2, 3, 4, 5, 7, 256] {
        assert_eq!(
            crate::samp_pre_batch(&ek, &items[..n]),
            want[..n],
            "{n} calls"
        );
    }
    let mut none = crate::ExpandedKey::new(&sk).unwrap();
    none.params.samppre_max_attempts = 0;
    for n in [1, 2, 3, 4, 5, 7] {
        let got = crate::samp_pre_batch(&none, &items[..n]);
        assert_eq!(got.len(), n);
        for (i, (g, &(t, seed))) in got.iter().zip(items.iter()).enumerate() {
            assert_eq!(g, &crate::samp_pre(&none, t, seed), "{n} calls, item {i}");
            assert_eq!(
                g,
                &Err(crate::Error::SampPreExhausted),
                "{n} calls, item {i}"
            );
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

/// The x86-64-v3 copy of a SampPre attempt returns, bit for bit, what the portable code returns,
/// and reads the same stream: at both presets, on several keys, targets and streams (on
/// processors without the instructions the copy does not run and the test is empty).
#[cfg(target_arch = "x86_64")]
#[test]
fn samp_pre_attempt_v3_equals_portable() {
    use crate::hazmat::SamplerZ;
    use crate::{Params, RqPoly};
    if !v3() {
        return;
    }
    for params in [Params::PCS, Params::FALCON_1024] {
        for k in 0..3u8 {
            let (_, sk) = crate::trapgen(&params, &[k ^ 0x3c; 32]).expect("trapgen");
            let ek = crate::ExpandedKey::new(&sk).expect("expand");
            let mut rng = Shake256Prng::from_parts(&[b"v3-targets", &[k]]);
            for case in 0..6u8 {
                let c: Vec<u32> = (0..params.n())
                    .map(|_| (rng.next_u64() % params.q as u64) as u32)
                    .collect();
                let t = RqPoly::from_coeffs(&params, &c).unwrap();
                let seed = [b"v3-stream".as_slice(), &[k, case]].concat();
                let mut a = Shake256Prng::new(&seed);
                let mut b = Shake256Prng::new(&seed);
                // SAFETY: the processor has the instructions of the copy (`v3`).
                let got = unsafe { crate::sign::samp_pre_attempt_v3(&ek, &t, &mut a) };
                let mut samp = SamplerZ::new(&params, &mut b);
                let want = crate::sign::samp_pre_with(&ek, &t, &mut samp)
                    .filter(|s| s.norm_sq() <= params.beta_sq);
                assert_eq!(got, want, "{} key {k} case {case}", params.name);
                assert_eq!(a.next_u64(), b.next_u64(), "stream positions");
            }
        }
    }
}

/// The x86-64-v3 FFT passes (AVX2 kernels from `logn = 4` or the layers with half-blocks of at
/// least 4 values on) equal the portable passes bit for bit, at every degree, on FFT-domain
/// inputs (non-integer values) and on values with signed zeros.
#[cfg(target_arch = "x86_64")]
#[test]
fn fft_passes_v3_equal_portable() {
    use crate::fft::{
        fft, fft_v3, ifft, ifft_v3, merge_sub_mul_add_split, merge_sub_mul_add_split_v3,
        poly_merge_fft, poly_merge_fft_v3, poly_split_fft, poly_split_fft_v3,
    };
    if !v3() {
        return;
    }
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
    let mut rng = Shake256Prng::new(b"simd-fft-v3");
    for logn in 1..=10u32 {
        let n = 1usize << logn;
        let hn = n / 2;
        for round in 0..20 {
            let mut v: Vec<Vec<f64>> = (0..4)
                .map(|_| {
                    (0..n)
                        .map(|i| match (round, i % 13) {
                            (0, 0) => 0.0,
                            (0, 1) => -0.0,
                            _ => ((rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2e4,
                        })
                        .collect()
                })
                .collect();
            for x in v.iter_mut() {
                fft(logn, x);
            }
            // fft and ifft.
            let (mut a, mut b) = (v[0].clone(), v[0].clone());
            fft(logn, &mut a);
            // SAFETY: the processor has the instructions of the copies (`v3`).
            unsafe { fft_v3(logn, &mut b) };
            assert_eq!(bits(&a), bits(&b), "fft {logn}");
            let (mut a, mut b) = (v[1].clone(), v[1].clone());
            ifft(logn, &mut a);
            // SAFETY: as above.
            unsafe { ifft_v3(logn, &mut b) };
            assert_eq!(bits(&a), bits(&b), "ifft {logn}");
            if logn < 2 {
                continue;
            }
            // Split and merge.
            let (mut a0, mut a1, mut b0, mut b1) =
                (vec![0.0; hn], vec![0.0; hn], vec![0.0; hn], vec![0.0; hn]);
            poly_split_fft(logn, &mut a0, &mut a1, &v[2]);
            // SAFETY: as above.
            unsafe { poly_split_fft_v3(logn, &mut b0, &mut b1, &v[2]) };
            assert_eq!(
                (bits(&a0), bits(&a1)),
                (bits(&b0), bits(&b1)),
                "split {logn}"
            );
            let (mut ma, mut mb) = (vec![0.0; n], vec![0.0; n]);
            poly_merge_fft(logn, &mut ma, &a0, &a1);
            // SAFETY: as above.
            unsafe { poly_merge_fft_v3(logn, &mut mb, &a0, &a1) };
            assert_eq!(bits(&ma), bits(&mb), "merge {logn}");
            // The fused node pass.
            let l10: Vec<f64> = v[3].iter().map(|x| x / 1024.5).collect();
            let (mut wa, mut wb) = (v[2].clone(), v[2].clone());
            let (mut ta, mut tb) = (v[1].clone(), v[1].clone());
            merge_sub_mul_add_split(logn, &mut wa, &v[0], &mut ta, &l10);
            // SAFETY: as above.
            unsafe { merge_sub_mul_add_split_v3(logn, &mut wb, &v[0], &mut tb, &l10) };
            assert_eq!(
                (bits(&wa), bits(&ta)),
                (bits(&wb), bits(&tb)),
                "fused {logn}"
            );
        }
    }
}

/// Montgomery product, as `NttPrime::mmul`.
fn mm(pr: &NttPrime, a: u32, b: u32) -> u32 {
    pr.mmul(a, b)
}

/// The portable forward layers from blocks of `t` values (`m` blocks) down to blocks of `end`
/// values: the loops of `NttPrime::ntt`, one layer at a time.
fn ntt_layers_ref(pr: &NttPrime, a: &mut [u32], mut t: usize, mut m: usize, end: usize) {
    let p = pr.p;
    while t > end {
        let ht = t >> 1;
        for (blk, &s) in a.chunks_exact_mut(t).zip(pr.gm[m..2 * m].iter()) {
            let (x, y) = blk.split_at_mut(ht);
            for (x1, x2) in x.iter_mut().zip(y.iter_mut()) {
                let v = mm(pr, *x2, s);
                (*x1, *x2) = (mp_add(*x1, v, p), mp_sub(*x1, v, p));
            }
        }
        t = ht;
        m <<= 1;
    }
}

/// The portable inverse layers from half-blocks of `t` values (`hm` blocks) up to half-blocks
/// of `stop / 2` values (`NttPrime::intt_n`).
fn intt_layers_ref(pr: &NttPrime, a: &mut [u32], mut t: usize, mut hm: usize, stop: usize) {
    let p = pr.p;
    while t < stop {
        let dt = t << 1;
        for (blk, &s) in a.chunks_exact_mut(dt).zip(pr.igm[hm..2 * hm].iter()) {
            let (x, y) = blk.split_at_mut(t);
            for (x1, x2) in x.iter_mut().zip(y.iter_mut()) {
                let (u, v) = (*x1, *x2);
                (*x1, *x2) = (mp_add(u, v, p), mm(pr, mp_sub(u, v, p), s));
            }
        }
        t = dt;
        hm >>= 1;
    }
}

/// Inputs of a kernel test: random residues with 0, 1 and p - 1 mixed in, then all 0, all 1 and
/// all p - 1.
fn test_vectors(rng: &mut Shake256Prng, p: u32, n: usize) -> Vec<Vec<u32>> {
    let mut v = vec![residues(rng, p, n), residues(rng, p, n)];
    v.extend([0, 1, p - 1].map(|x| vec![x; n]));
    v
}

/// The narrow layers of both transforms equal the portable layers, value by value, at every
/// size that they take (n = 16 to 1024).
#[test]
fn narrow_layers_equal_portable() {
    let mut rng = Shake256Prng::new(b"simd-narrow");
    for pr in [&P1, &P2] {
        for logn in 4..=10u32 {
            let n = 1usize << logn;
            for a in test_vectors(&mut rng, pr.p, n) {
                // Forward: the wide layers, then the kernel against the portable loops.
                let mut x = a.clone();
                let (t, m) = ntt_wide_layers(pr, logn, &mut x);
                let mut want = x.clone();
                if ntt_narrow_layers(pr, logn, &mut x, t) {
                    ntt_layers_ref(pr, &mut want, t, m, 1);
                    assert_eq!(x, want, "ntt p {} logn {logn}", pr.p);
                }
                // Inverse: the kernel against the portable loops up to where it stops.
                let mut x = a.clone();
                if let Some((t, hm)) = intt_narrow_layers(pr, logn, &mut x) {
                    let mut want = a.clone();
                    intt_layers_ref(pr, &mut want, 1, n >> 1, t);
                    assert_eq!(x, want, "intt p {} logn {logn}", pr.p);
                    assert_eq!(hm, n / (2 * t));
                }
            }
        }
    }
}

/// The wide layers of both transforms (AVX-512 and AVX2, or NEON) equal the portable layers,
/// value by value: the forward ones down to where they stop, the inverse ones from every
/// half-block size that they take.
#[test]
fn wide_layers_equal_portable() {
    let mut rng = Shake256Prng::new(b"simd-wide");
    for pr in [&P1, &P2] {
        for logn in 3..=10u32 {
            let n = 1usize << logn;
            for a in test_vectors(&mut rng, pr.p, n) {
                let mut x = a.clone();
                let (t, m) = ntt_wide_layers(pr, logn, &mut x);
                let mut want = a.clone();
                ntt_layers_ref(pr, &mut want, n, 1, t);
                assert_eq!(x, want, "ntt p {} logn {logn}", pr.p);
                assert_eq!(m, n / t);
                for t0 in [4usize, 8, 16, 32] {
                    let mut x = a.clone();
                    if t0 < n && intt_wide_layers(pr, logn, &mut x, t0, n / (2 * t0)) {
                        let mut want = a.clone();
                        intt_layers_ref(pr, &mut want, t0, n / (2 * t0), n);
                        assert_eq!(x, want, "intt p {} logn {logn} t {t0}", pr.p);
                    }
                }
            }
        }
    }
}

/// Odd and even moduli of the tests, prime and composite, from 3 to the largest prime below
/// 2^25.
const QS: [u32; 9] = [
    3,
    4,
    15,
    17,
    12289,
    1048573,
    1 << 20,
    3 * 3 * 1048573,
    33554393,
];

/// The CRT kernels equal `ModQ::crt` (and its sum with `mp_add`), on random residue pairs, on
/// 0, 1 and p - 1, and around the centring boundary (t = (p2 - 1)/2 with u1 near (p1 - 1)/2).
/// Even moduli stay on the portable code.
#[test]
fn crt_kernel_equals_portable() {
    let mut rng = Shake256Prng::new(b"simd-crt");
    let (h1, h2) = ((P1.p - 1) / 2, (P2.p - 1) / 2);
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for x in [0, 1, P1.p - 1, h1 - 1, h1, h1 + 1] {
        for y in [0, 1, P2.p - 1, h2 - 1, h2, h2 + 1] {
            pairs.push((x, y));
        }
        // u2 with t = h2 exactly: u2 = u1 + p1 h2 modulo p2.
        let u2 = ((x as u64 + P1.p as u64 * h2 as u64) % P2.p as u64) as u32;
        for d in [0u32, 1, P2.p - 1] {
            pairs.push((x, ((u2 as u64 + d as u64) % P2.p as u64) as u32));
        }
    }
    while !pairs.len().is_multiple_of(8) || pairs.len() < 4096 {
        pairs.push((
            (rng.next_u64() % P1.p as u64) as u32,
            (rng.next_u64() % P2.p as u64) as u32,
        ));
    }
    let (u1, u2): (Vec<u32>, Vec<u32>) = pairs.into_iter().unzip();
    for q in QS {
        let mq = ModQ::new(q);
        let want: Vec<u32> = u1.iter().zip(&u2).map(|(&x, &y)| mq.crt(x, y)).collect();
        let mut out = vec![0u32; u1.len()];
        let ran = crt_slices(q, &u1, &u2, &mut out, false);
        assert!(!ran || !q.is_multiple_of(2), "q {q}");
        if ran {
            assert_eq!(out, want, "crt q {q}");
        }
        let base: Vec<u32> = (0..u1.len()).map(|i| (i as u32 * 7919) % q).collect();
        let mut acc = base.clone();
        if crt_slices(q, &u1, &u2, &mut acc, true) {
            let want: Vec<u32> = base
                .iter()
                .zip(&want)
                .map(|(&b, &w)| mp_add(b, w, q))
                .collect();
            assert_eq!(acc, want, "crt accumulate q {q}");
        }
    }
}

/// Signed values for the reduction tests: the extremes of i32, small values, multiples of q
/// and their neighbours, and random values of every size.
fn signed_values(rng: &mut Shake256Prng, q: u32) -> Vec<i32> {
    let q = q as i32;
    let mut v = vec![
        0,
        1,
        -1,
        i32::MAX,
        i32::MIN,
        i32::MIN + 1,
        q,
        -q,
        q - 1,
        1 - q,
        q + 1,
        -q - 1,
        2 * q,
        -2 * q,
    ];
    while !v.len().is_multiple_of(8) || v.len() < 4096 {
        let x = rng.next_u64();
        v.push((x as i32) >> (x >> 59));
    }
    v
}

/// The reduction of signed values, the comparison of Verify and the centred lift modulo p1
/// equal the portable code.
#[test]
fn reduction_kernels_equal_portable() {
    let mut rng = Shake256Prng::new(b"simd-reduce");
    for q in QS {
        let mq = ModQ::new(q);
        let x = signed_values(&mut rng, q);
        let want: Vec<u32> = x.iter().map(|&v| mq.reduce_i32(v)).collect();
        let mut out = vec![0u32; x.len()];
        if reduce_i32_slices(q, &x, &mut out) {
            assert_eq!(out, want, "reduce_i32 q {q}");
        }
        // add_i32_diff: random y, z = x + y mod q except where perturbed.
        let y: Vec<u32> = (0..x.len())
            .map(|_| (rng.next_u64() % q as u64) as u32)
            .collect();
        let mut z: Vec<u32> = want
            .iter()
            .zip(&y)
            .map(|(&a, &b)| mp_add(a, b, q))
            .collect();
        for k in [None, Some(0usize), Some(x.len() - 1), Some(1000)] {
            if let Some(k) = k {
                z[k] = (z[k] + 1) % q;
            }
            let diff = x.iter().zip(&y).zip(&z).fold(0u32, |d, ((&a, &b), &c)| {
                d | (mp_add(mq.reduce_i32(a), b, q) ^ c)
            });
            if let Some(got) = add_i32_diff(q, &x, &y, &z) {
                assert_eq!(got, diff, "add_i32_diff q {q} {k:?}");
            }
        }
        // The centred lift of residues modulo p1.
        let h = P1.p >> 1;
        let mut u = vec![0, 1, h - 1, h, h + 1, P1.p - 2, P1.p - 1, 2];
        u.extend(residues(&mut rng, P1.p, 4096));
        let want: Vec<u32> = u
            .iter()
            .map(|&v| {
                let c = if v > h {
                    v as i64 - P1.p as i64
                } else {
                    v as i64
                };
                mq.reduce_i32(c as i32)
            })
            .collect();
        let mut out = vec![0u32; u.len()];
        if centred_p1_mod_q(q, &u, &mut out) {
            assert_eq!(out, want, "centred_p1_mod_q q {q}");
        }
    }
}

/// Verify's norm kernel equals the u128 sum, for every tail length and for values whose sum
/// exceeds 2^64.
#[test]
fn norm_kernel_equals_portable() {
    let mut rng = Shake256Prng::new(b"simd-norm");
    let exact = |a: &[i32], b: &[i32]| -> u128 {
        a.iter()
            .chain(b)
            .map(|&x| (x.unsigned_abs() as u128).pow(2))
            .sum()
    };
    for len in (0..20).chain([1024, 1031]) {
        for shift in [0u32, 8, 20, 31] {
            let a: Vec<i32> = (0..len).map(|_| (rng.next_u64() as i32) >> shift).collect();
            let b: Vec<i32> = (0..len + 3)
                .map(|_| (rng.next_u64() as i32) >> shift)
                .collect();
            if let Some(got) = norm_sq(&a, &b) {
                assert_eq!(got, exact(&a, &b), "len {len} shift {shift}");
            }
        }
    }
    let big = vec![i32::MIN; 4096];
    let max = vec![i32::MAX; 4093];
    if let Some(got) = norm_sq(&big, &max) {
        assert_eq!(got, exact(&big, &max));
        assert!(got > u64::MAX as u128);
    }
}

/// The norm of the inversion tower in the NTT domain equals the portable formula of
/// `ring::inv::norm_ntt`, at every size that the kernels take.
#[test]
fn norm_ntt_kernel_equals_portable() {
    let mut rng = Shake256Prng::new(b"simd-norm-ntt");
    for pr in [&P1, &P2] {
        for logh in 1..=9u32 {
            let h = 1usize << logh;
            for (e, o) in test_vectors(&mut rng, pr.p, h)
                .into_iter()
                .zip(test_vectors(&mut rng, pr.p, h).into_iter().rev())
            {
                let c = pr.ninv_r2[logh as usize];
                let want: Vec<u32> = (0..h)
                    .map(|k| {
                        let z = pr.gm[h / 2 + k / 2];
                        let e2 = pr.mmul(pr.mmul(e[k], e[k]), c);
                        let zo2 = pr.mmul(pr.mmul(pr.mmul(o[k], o[k]), c), z);
                        if k % 2 == 1 {
                            mp_add(e2, zo2, pr.p)
                        } else {
                            mp_sub(e2, zo2, pr.p)
                        }
                    })
                    .collect();
                let mut out = vec![0u32; h];
                if norm_ntt(pr, logh, &e, &o, &mut out) {
                    assert_eq!(out, want, "p {} logh {logh}", pr.p);
                }
            }
        }
    }
}

/// The check of the NTRU equation in the NTT domain equals the portable loop, on random
/// residues and on slots where the equation holds.
#[test]
fn ntru_diff_kernel_equals_portable() {
    let mut rng = Shake256Prng::new(b"simd-ntru");
    for pr in [&P1, &P2] {
        for q in [12289u32, 1048573] {
            for case in 0..4 {
                let n = 1024;
                let mut v: Vec<Vec<u32>> = (0..4).map(|_| residues(&mut rng, pr.p, n)).collect();
                if case >= 2 {
                    // f = 1, g = 0, G = q: f G - g F = q in every slot, except one if case 3.
                    v[0] = vec![1; n];
                    v[1] = vec![0; n];
                    v[3] = vec![q; n];
                    if case == 3 {
                        v[3][517] = q + 1;
                    }
                }
                let want = (0..n).fold(0u32, |d, i| {
                    d | (mp_sub(pr.mul(v[0][i], v[3][i]), pr.mul(v[1][i], v[2][i]), pr.p) ^ q)
                });
                if let Some(got) = ntru_diff(pr, q, &v[0], &v[1], &v[2], &v[3]) {
                    assert_eq!(got, want, "p {} q {q} case {case}", pr.p);
                    assert_eq!(got == 0, case == 2);
                }
            }
        }
    }
}
