//! The end-to-end pipeline at both presets (integrate stage): `trapgen` → `ExpandedKey::new`
//! → `samp_pre` → `verify`, through the public API only. Every property is re-checked here by
//! an independent computation that does not use the crate's ring, FFT or solver:
//! - $`fG-gF=q`$ exactly (schoolbook product in $`\mathbb Z[x]/(x^d+1)`$, `i128`) and
//!   $`hf=g\bmod q`$ (schoolbook modulo $`q`$);
//! - Falcon's two Gram–Schmidt norms are at most $`(1.17\sqrt q)^2`$: $`N_1`$ exactly, and
//!   $`N_2`$ by a direct $`O(d^2)`$ evaluation of $`f,g`$ at the primitive $`2d`$-th roots of
//!   unity, which must also agree with the crate's FFT value;
//! - every leaf width lies in $`[\sigma_{\min},\sigma_0)`$ and in the envelope
//!   $`[\sigma_{\min},1.17^2\sigma_{\min}]`$;
//! - SampPre on uniformly random targets (and $`t=0`$, $`t=-1`$): $`s_0+hs_1=t`$ (schoolbook),
//!   $`\|s\|^2\le\lfloor\beta^2\rfloor`$, `verify`, determinism in the seed;
//! - the encodings: round trips and sizes; a decoded secret key gives the same preimages;
//! - error cases: non-invertible elements and keys, wrong lengths, mismatched rings,
//!   malformed encodings, a degenerate key, invalid parameters.
//!
//! The ignored `many_keys` test runs the pipeline on 1024 PCS and 512 Falcon-1024 keys and
//! prints the key-generation statistics:
//! `cargo test --release --test pipeline -- --ignored --nocapture` (about 15 s).

use ntru_trapdoor::hazmat::{Prng, Shake256Prng, gram_schmidt_sq_norms};
use ntru_trapdoor::{
    Error, ExpandedKey, KeygenReject, Params, Preimage, PublicKey, RqPoly, SecretKey, samp_pre,
    trapgen, trapgen_from_fg, trapgen_with_stats, verify,
};

// ---------------------------------------------------------------------------------------------
// Independent arithmetic.

/// $`a\cdot b`$ in $`\mathbb Z[x]/(x^n+1)`$, schoolbook in `i128`.
fn negacyclic(a: &[i64], b: &[i64]) -> Vec<i128> {
    let n = a.len();
    assert_eq!(b.len(), n);
    let mut r = vec![0i128; n];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            let p = ai as i128 * bj as i128;
            if i + j < n {
                r[i + j] += p;
            } else {
                r[i + j - n] -= p;
            }
        }
    }
    r
}

fn wide<T: Copy + Into<i64>>(x: &[T]) -> Vec<i64> {
    x.iter().map(|&v| v.into()).collect()
}

fn mod_q(x: &[i128], q: u32) -> Vec<u32> {
    x.iter().map(|&v| v.rem_euclid(q as i128) as u32).collect()
}

/// Falcon's second Gram–Schmidt norm,
/// $`N_2=\frac{2q^2}{n}\sum_{k<n/2}(|f(\zeta_k)|^2+|g(\zeta_k)|^2)^{-1}`$ with
/// $`\zeta_k=e^{i\pi(2k+1)/n}`$ (one root per conjugate pair), by direct evaluation.
fn n2_direct(q: u32, f: &[i16], g: &[i16]) -> f64 {
    let n = f.len();
    let two_n = 2 * n;
    let (cos, sin): (Vec<f64>, Vec<f64>) = (0..two_n)
        .map(|m| {
            let a = std::f64::consts::PI * m as f64 / n as f64;
            (a.cos(), a.sin())
        })
        .unzip();
    let eval = |p: &[i16], k: usize| -> (f64, f64) {
        let (mut re, mut im) = (0.0, 0.0);
        for (j, &c) in p.iter().enumerate() {
            let m = ((2 * k + 1) * j) % two_n;
            re += c as f64 * cos[m];
            im += c as f64 * sin[m];
        }
        (re, im)
    };
    let mut sum = 0.0;
    for k in 0..n / 2 {
        let (fr, fi) = eval(f, k);
        let (gr, gi) = eval(g, k);
        sum += 1.0 / (fr * fr + fi * fi + gr * gr + gi * gi);
    }
    2.0 * (q as f64) * (q as f64) / n as f64 * sum
}

fn random_target(p: &Params, prng: &mut Shake256Prng) -> RqPoly {
    let c: Vec<u32> = (0..p.n())
        .map(|_| (prng.next_u64() % p.q as u64) as u32)
        .collect();
    RqPoly::from_coeffs(p, &c).unwrap()
}

fn seed_of(tag: u8, k: u64) -> [u8; 32] {
    let mut s = [tag; 32];
    s[..8].copy_from_slice(&k.to_le_bytes());
    s
}

/// The envelope $`1.17^2\sigma_{\min}`$ of the largest leaf width.
fn sigma_max(p: &Params) -> f64 {
    1.17 * 1.17 * p.sigma_min
}

// ---------------------------------------------------------------------------------------------
// The checks of one key.

struct KeyReport {
    leaf: (f64, f64),
    max_fg: u16,
    max_big_fg: u16,
    n2_rel_err: f64,
}

/// Every key property, checked independently.
fn check_key(p: &Params, pk: &PublicKey, sk: &SecretKey, ek: &ExpandedKey) -> KeyReport {
    let n = p.n();
    let q = p.q;
    let (f, g, big_f, big_g) = (sk.f(), sk.g(), sk.big_f(), sk.big_g());
    assert!([f, g, big_f, big_g].iter().all(|x| x.len() == n));

    // fG - gF = q exactly.
    let fg = negacyclic(&wide(f), &wide(big_g));
    let gf = negacyclic(&wide(g), &wide(big_f));
    let lhs: Vec<i128> = fg.iter().zip(&gf).map(|(a, b)| a - b).collect();
    assert_eq!(lhs[0], q as i128, "{}: constant term of fG - gF", p.name);
    assert!(lhs[1..].iter().all(|&c| c == 0), "{}: fG - gF != q", p.name);

    // h f = g mod q (A = [1 | h] annihilates (g, -f)), and G = h F mod q likewise.
    let h = wide(pk.h().coeffs());
    let as_i128 = |x: &[i16]| x.iter().map(|&v| v as i128).collect::<Vec<i128>>();
    assert_eq!(mod_q(&negacyclic(&h, &wide(f)), q), mod_q(&as_i128(g), q));
    assert_eq!(
        mod_q(&negacyclic(&h, &wide(big_f)), q),
        mod_q(&as_i128(big_g), q)
    );

    // Falcon's two Gram-Schmidt norms.
    let n1: u64 = f
        .iter()
        .chain(g)
        .map(|&x| (x as i64 * x as i64) as u64)
        .sum();
    assert!(n1 * p.gs_bound_den <= p.gs_bound_num, "{}: N1 {n1}", p.name);
    let n2 = n2_direct(q, f, g);
    let (n1_crate, n2_crate) = gram_schmidt_sq_norms(p, f, g);
    assert_eq!(n1_crate, n1);
    let n2_rel_err = (n2_crate - n2).abs() / n2;
    assert!(n2_rel_err < 1e-9, "{}: N2 {n2_crate} vs {n2}", p.name);
    assert!(n2 <= p.gs_bound(), "{}: N2 {n2}", p.name);

    // The leaf widths.
    let (lo, hi) = ek.leaf_sigma_range();
    assert!(lo >= p.sigma_min * (1.0 - 1e-12), "{}: leaf {lo}", p.name);
    assert!(hi < p.base.sigma0(), "{}: leaf {hi}", p.name);
    assert!(
        hi <= sigma_max(p) * (1.0 + 1e-9),
        "{}: leaf {hi} above the envelope",
        p.name
    );
    assert!(lo <= hi);

    // The key validates, and the public key is a function of the secret key.
    sk.validate().unwrap();
    assert_eq!(&sk.public_key().unwrap(), pk);

    let max_abs = |x: &[i16]| x.iter().map(|v| v.unsigned_abs()).max().unwrap();
    KeyReport {
        leaf: (lo, hi),
        max_fg: max_abs(f).max(max_abs(g)),
        max_big_fg: max_abs(big_f).max(max_abs(big_g)),
        n2_rel_err,
    }
}

/// SampPre's output for `t`, checked independently.
fn check_preimage(p: &Params, pk: &PublicKey, t: &RqPoly, s: &Preimage) {
    let n = p.n();
    assert_eq!((s.s0.len(), s.s1.len(), s.n()), (n, n, n));
    // ||s||^2 <= floor(beta^2).
    let norm: u128 =
        s.s0.iter()
            .chain(&s.s1)
            .map(|&x| (x as i128 * x as i128) as u128)
            .sum();
    assert!(norm <= p.beta_sq as u128, "{}: ||s||^2 = {norm}", p.name);
    assert_eq!(s.norm_sq() as u128, norm);
    // s0 + h s1 = t mod q.
    let hs1 = negacyclic(&wide(pk.h().coeffs()), &wide(&s.s1));
    let lhs: Vec<i128> = hs1
        .iter()
        .zip(&s.s0)
        .map(|(&a, &b)| a + b as i128)
        .collect();
    assert_eq!(mod_q(&lhs, p.q), t.coeffs(), "{}: s0 + h s1 != t", p.name);
    // The crate's verifier agrees, and is tight at the bound.
    assert!(verify(pk, t, s, p.beta_sq));
    assert!(verify(pk, t, s, norm as u64));
    assert!(!verify(pk, t, s, norm as u64 - 1));
}

/// The pipeline on `keys` keys of `p`, with `targets` random targets per key (plus 0 and -1).
fn pipeline(p: &Params, keys: u64, targets: u64) {
    let mut prng = Shake256Prng::new(&[b'P', p.q as u8, (p.q >> 8) as u8]);
    for k in 0..keys {
        let (pk, sk, stats) = trapgen_with_stats(p, &seed_of(0x11, k)).unwrap();
        assert!(stats.candidates >= 1 && stats.rejected_leaf == 0 && stats.rejected_size == 0);
        let ek = ExpandedKey::new(&sk).unwrap();
        assert_eq!(ek.params(), p);
        let rep = check_key(p, &pk, &sk, &ek);
        eprintln!(
            "{} key {k}: {} candidates, leaves [{:.5}, {:.5}], max |f,g| {}, max |F,G| {}, N2 \
             relative error {:.1e}",
            p.name,
            stats.candidates,
            rep.leaf.0,
            rep.leaf.1,
            rep.max_fg,
            rep.max_big_fg,
            rep.n2_rel_err
        );

        // The encodings, and the expanded key of the decoded secret key.
        let pkb = pk.to_bytes();
        let skb = sk.to_bytes();
        assert_eq!(pkb.len(), p.rq_bytes());
        assert_eq!(PublicKey::from_bytes(p, &pkb).unwrap(), pk);
        let sk2 = SecretKey::from_bytes(p, &skb).unwrap();
        assert_eq!(
            (sk2.f(), sk2.g(), sk2.big_f(), sk2.big_g()),
            (sk.f(), sk.g(), sk.big_f(), sk.big_g())
        );
        assert_eq!(&*sk2.to_bytes(), &*skb);
        let ek2 = ExpandedKey::new(&sk2).unwrap();

        let mut ts = vec![
            RqPoly::zero(p).unwrap(),
            RqPoly::from_coeffs(p, &vec![p.q - 1; p.n()]).unwrap(),
        ];
        for _ in 0..targets {
            ts.push(random_target(p, &mut prng));
        }
        for (i, t) in ts.iter().enumerate() {
            assert_eq!(RqPoly::from_bytes(p, &t.to_bytes()).unwrap(), *t);
            let seed = seed_of(0x22, k * 1000 + i as u64);
            let s = samp_pre(&ek, t, &seed).unwrap();
            check_preimage(p, &pk, t, &s);
            // Deterministic in (key, target, seed); the decoded key gives the same preimage.
            assert_eq!(samp_pre(&ek, t, &seed).unwrap(), s);
            assert_eq!(samp_pre(&ek2, t, &seed).unwrap(), s);
            let other = samp_pre(&ek, t, &seed_of(0x33, k * 1000 + i as u64)).unwrap();
            check_preimage(p, &pk, t, &other);
            assert_ne!(other, s);
        }
    }
}

#[test]
fn pipeline_pcs() {
    pipeline(&Params::PCS, 3, 8);
}

#[test]
fn pipeline_falcon1024() {
    pipeline(&Params::FALCON_1024, 3, 8);
}

// ---------------------------------------------------------------------------------------------
// Error cases.

/// A primitive 2n-th root of unity modulo the prime q (a root of x^n + 1), by search.
fn root_of_xn_plus_1(q: u32, n: u64) -> u64 {
    let pow = |mut b: u64, mut e: u64| {
        let (m, mut r) = (q as u64, 1u64);
        b %= m;
        while e > 0 {
            if e & 1 == 1 {
                r = r * b % m;
            }
            b = b * b % m;
            e >>= 1;
        }
        r
    };
    assert_eq!((q as u64 - 1) % (2 * n), 0);
    (2..q as u64)
        .map(|c| pow(c, (q as u64 - 1) / (2 * n)))
        .find(|&z| pow(z, n) == q as u64 - 1)
        .unwrap()
}

#[test]
fn non_invertible_elements_and_keys() {
    // PCS: x^1024 + 1 = (x^512 - r)(x^512 + r) with r^2 = -1 mod q, r = 365259.
    let p = Params::PCS;
    let (q, n) = (p.q, p.n());
    let r = 365_259u32;
    assert_eq!((r as u64 * r as u64) % q as u64, q as u64 - 1);
    let mut c = vec![0u32; n];
    c[512] = 1;
    for (r0, what) in [(q - r, "x^512 - r"), (r, "x^512 + r")] {
        c[0] = r0;
        let a = RqPoly::from_coeffs(&p, &c).unwrap();
        assert!(a.inverse().is_none(), "{what}");
        assert!(!a.is_invertible(), "{what}");
    }
    let zero = RqPoly::zero(&p).unwrap();
    assert!(zero.inverse().is_none() && !zero.is_invertible());
    // A short non-unit: a + b x^512 with a + r b = 0 mod q and a^2 + b^2 = q (the ideal
    // generated by x^512 - r is principal of norm q in Z[i], r playing the role of i).
    let (a, b) = (1..=1024i64)
        .find_map(|b| {
            let a = (-(r as i64) * b).rem_euclid(q as i64);
            let a = if a > q as i64 / 2 { a - q as i64 } else { a };
            (a * a + b * b == q as i64).then_some((a, b))
        })
        .unwrap();
    let mut short = vec![0i64; n];
    short[0] = a;
    short[512] = b;
    let short_q = RqPoly::from_i64(&p, &short).unwrap();
    assert!(short_q.inverse().is_none() && !short_q.is_invertible());
    eprintln!("PCS short non-unit: {a} + {b} x^512 (squared norm q = {q})");
    // A unit and its inverse.
    let mut prng = Shake256Prng::new(b"units");
    let u = random_target(&p, &mut prng);
    let ui = u.inverse().unwrap();
    let mut one = vec![0u32; n];
    one[0] = 1;
    assert_eq!(u.mul(&ui).coeffs(), &one[..]);
    assert!(u.is_invertible());

    // A secret key whose f is that non-unit: no public key, and validate fails.
    let (_, sk) = trapgen(&p, &seed_of(0x44, 0)).unwrap();
    let f_bad: Vec<i16> = short.iter().map(|&x| x as i16).collect();
    let bad = SecretKey::from_parts(
        &p,
        f_bad,
        sk.g().to_vec(),
        sk.big_f().to_vec(),
        sk.big_g().to_vec(),
    )
    .unwrap();
    assert_eq!(bad.public_key().unwrap_err(), Error::InvalidKey);
    assert_eq!(bad.validate().unwrap_err(), Error::InvalidKey);

    // Falcon-1024: x - zeta for a root zeta of x^1024 + 1 modulo 12289.
    let p = Params::FALCON_1024;
    let zeta = root_of_xn_plus_1(p.q, p.n() as u64);
    let mut c = vec![0u32; p.n()];
    c[0] = p.q - zeta as u32;
    c[1] = 1;
    let a = RqPoly::from_coeffs(&p, &c).unwrap();
    assert!(a.inverse().is_none() && !a.is_invertible());
}

#[test]
fn wrong_lengths_and_encodings() {
    let p = Params::PCS;
    let n = p.n();
    let (pk, sk) = trapgen(&p, &seed_of(0x55, 0)).unwrap();
    // R_q constructors.
    assert_eq!(
        RqPoly::from_coeffs(&p, &vec![0; n - 1]).unwrap_err(),
        Error::InvalidLength
    );
    assert_eq!(
        RqPoly::from_i64(&p, &vec![0; n + 1]).unwrap_err(),
        Error::InvalidLength
    );
    assert_eq!(RqPoly::from_i32(&p, &[]).unwrap_err(), Error::InvalidLength);
    assert_eq!(
        RqPoly::from_i16(&p, &vec![0; 512]).unwrap_err(),
        Error::InvalidLength
    );
    let mut c = vec![0u32; n];
    c[3] = p.q;
    assert_eq!(
        RqPoly::from_coeffs(&p, &c).unwrap_err(),
        Error::InvalidEncoding
    );
    // R_q encoding.
    let hb = pk.h().to_bytes();
    assert_eq!(hb.len(), 2560);
    assert_eq!(
        RqPoly::from_bytes(&p, &hb[..2559]).unwrap_err(),
        Error::InvalidLength
    );
    let mut longer = hb.clone();
    longer.push(0);
    assert_eq!(
        RqPoly::from_bytes(&p, &longer).unwrap_err(),
        Error::InvalidLength
    );
    let mut big = hb.clone();
    big[0] = 0xff;
    big[1] = 0xff;
    big[2] |= 0x0f; // coefficient 0 = 2^20 - 1 >= q
    assert_eq!(
        RqPoly::from_bytes(&p, &big).unwrap_err(),
        Error::InvalidEncoding
    );
    assert_eq!(
        PublicKey::from_bytes(&p, &big).unwrap_err(),
        Error::InvalidEncoding
    );
    // Key encodings.
    assert_eq!(
        PublicKey::from_bytes(&p, &hb[1..]).unwrap_err(),
        Error::InvalidLength
    );
    let skb = sk.to_bytes();
    assert_eq!(skb.len(), 5889);
    assert_eq!(
        SecretKey::from_bytes(&p, &skb[..5888]).unwrap_err(),
        Error::InvalidLength
    );
    assert_eq!(
        SecretKey::from_bytes(&p, &[]).unwrap_err(),
        Error::InvalidLength
    );
    let mut bad = skb.to_vec();
    bad[0] ^= 1;
    assert_eq!(
        SecretKey::from_bytes(&p, &bad).unwrap_err(),
        Error::InvalidEncoding
    );
    let mut bad = skb.to_vec();
    bad[5000] ^= 0x10; // a bit of G: fG - gF != q
    assert_eq!(
        SecretKey::from_bytes(&p, &bad).unwrap_err(),
        Error::InvalidKey
    );
    // A Falcon-1024 encoding is not a PCS one, and conversely.
    let (fpk, fsk) = trapgen(&Params::FALCON_1024, &seed_of(0x55, 1)).unwrap();
    assert_eq!(
        PublicKey::from_bytes(&p, &fpk.to_bytes()).unwrap_err(),
        Error::InvalidLength
    );
    assert_eq!(
        SecretKey::from_bytes(&p, &fsk.to_bytes()).unwrap_err(),
        Error::InvalidLength
    );
    // Secret-key parts and key generation inputs.
    let short = vec![0i16; n - 1];
    assert_eq!(
        SecretKey::from_parts(
            &p,
            short.clone(),
            sk.g().to_vec(),
            sk.big_f().to_vec(),
            sk.big_g().to_vec()
        )
        .unwrap_err(),
        Error::InvalidLength
    );
    assert_eq!(
        trapgen_from_fg(&p, &short, sk.g()).unwrap_err(),
        KeygenReject::Input
    );
    assert_eq!(
        trapgen_from_fg(&p, sk.f(), &short).unwrap_err(),
        KeygenReject::Input
    );
    // A preimage of the wrong length never verifies.
    let ek = ExpandedKey::new(&sk).unwrap();
    let t = RqPoly::zero(&p).unwrap();
    let s = samp_pre(&ek, &t, &[1; 32]).unwrap();
    assert!(verify(&pk, &t, &s, p.beta_sq));
    let mut cut = s.clone();
    cut.s1.pop();
    assert!(!verify(&pk, &t, &cut, p.beta_sq));
    let mut cut = s.clone();
    cut.s0.push(0);
    assert!(!verify(&pk, &t, &cut, p.beta_sq));
}

#[test]
fn mismatched_rings() {
    let pcs = Params::PCS;
    let fal = Params::FALCON_1024;
    let (pk, sk) = trapgen(&pcs, &seed_of(0x66, 0)).unwrap();
    let ek = ExpandedKey::new(&sk).unwrap();
    let t_fal = RqPoly::zero(&fal).unwrap();
    assert_eq!(
        samp_pre(&ek, &t_fal, &[0; 32]).unwrap_err(),
        Error::Mismatch
    );
    let s = samp_pre(&ek, &RqPoly::zero(&pcs).unwrap(), &[0; 32]).unwrap();
    assert!(!verify(&pk, &t_fal, &s, pcs.beta_sq));
    assert_eq!(PublicKey::from_h(&pcs, t_fal).unwrap_err(), Error::Mismatch);
    // A preimage under another key does not verify.
    let (pk2, _) = trapgen(&pcs, &seed_of(0x66, 1)).unwrap();
    assert!(!verify(&pk2, &RqPoly::zero(&pcs).unwrap(), &s, pcs.beta_sq));
}

#[test]
#[should_panic(expected = "modulus or degree mismatch")]
fn ring_operations_panic_on_mismatch() {
    let _ = RqPoly::zero(&Params::PCS)
        .unwrap()
        .mul(&RqPoly::zero(&Params::FALCON_1024).unwrap());
}

#[test]
fn degenerate_key_and_invalid_parameters() {
    // The zero "key": fG - gF = 0, so `ExpandedKey::new`, which validates keys from
    // `from_parts`, refuses it before building the (singular) tree.
    let p = Params::PCS;
    let z = vec![0i16; p.n()];
    let sk = SecretKey::from_parts(&p, z.clone(), z.clone(), z.clone(), z).unwrap();
    assert_eq!(ExpandedKey::new(&sk).err(), Some(Error::InvalidKey));
    assert_eq!(sk.validate().unwrap_err(), Error::InvalidKey);
    // A valid key under a set whose leaf window reaches the base width (sigma 10% larger: the
    // leaves rise above 3.1): the leaf check refuses it, and key generation refuses the set.
    let (_, good) = trapgen(&p, &seed_of(0x66, 0)).unwrap();
    let mut wide = p;
    wide.sigma *= 1.1;
    wide.inv_sigma = 1.0 / wide.sigma;
    wide.validate().unwrap();
    assert!(matches!(
        wide.check_leaf_window(),
        Err(Error::InvalidParams(_))
    ));
    assert!(matches!(
        trapgen(&wide, &[0; 32]),
        Err(Error::InvalidParams(_))
    ));
    let moved = SecretKey::from_parts(
        &wide,
        good.f().to_vec(),
        good.g().to_vec(),
        good.big_f().to_vec(),
        good.big_g().to_vec(),
    )
    .unwrap();
    assert_eq!(ExpandedKey::new(&moved).err(), Some(Error::LeafOutOfRange));
    // Invalid parameter sets are refused everywhere.
    let mut bad = Params::PCS;
    bad.q = 1_048_575; // 3 * 5^2 * 11 * 31 * 41: not prime
    assert!(matches!(bad.validate(), Err(Error::InvalidParams(_))));
    assert!(matches!(
        trapgen(&bad, &[0; 32]),
        Err(Error::InvalidParams(_))
    ));
    assert!(matches!(
        SecretKey::from_parts(
            &bad,
            vec![0; 1024],
            vec![0; 1024],
            vec![0; 1024],
            vec![0; 1024]
        ),
        Err(Error::InvalidParams(_))
    ));
    assert!(matches!(
        PublicKey::from_bytes(&bad, &[0; 2560]),
        Err(Error::InvalidParams(_))
    ));
    let mut bad = Params::PCS;
    bad.logn = 11;
    assert!(matches!(bad.validate(), Err(Error::InvalidParams(_))));
    // Every error displays a message.
    for e in [
        Error::InvalidParams("x"),
        Error::Mismatch,
        Error::InvalidLength,
        Error::InvalidEncoding,
        Error::NotInvertible,
        Error::InvalidKey,
        Error::LeafOutOfRange,
        Error::KeygenExhausted,
        Error::SampPreExhausted,
    ] {
        assert!(!e.to_string().is_empty());
    }
}

// ---------------------------------------------------------------------------------------------
// Statistics over many keys (ignored: about 15 s in release).

#[test]
#[ignore = "1024 PCS keys and 512 Falcon-1024 keys; run with --release"]
fn many_keys() {
    for (p, keys) in [(Params::PCS, 1024u64), (Params::FALCON_1024, 512)] {
        let t0 = std::time::Instant::now();
        let mut prng = Shake256Prng::new(b"many keys targets");
        let (mut cand, mut fg, mut ortho, mut noinv, mut solve) = (0u64, 0u64, 0u64, 0u64, 0u64);
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        let (mut max_fg, mut max_big) = (0u16, 0u16);
        let mut ratio = 0.0;
        let mut keygen_s = 0.0;
        for k in 0..keys {
            let tk = std::time::Instant::now();
            let (pk, sk, st) = trapgen_with_stats(&p, &seed_of(0x77, k)).unwrap();
            keygen_s += tk.elapsed().as_secs_f64();
            assert_eq!((st.rejected_size, st.rejected_leaf), (0, 0));
            cand += st.candidates as u64;
            fg += st.rejected_norm_fg as u64;
            ortho += st.rejected_norm_ortho as u64;
            noinv += st.rejected_not_invertible as u64;
            solve += st.rejected_solve as u64;
            let ek = ExpandedKey::new(&sk).unwrap();
            let (l, h) = ek.leaf_sigma_range();
            assert!(l >= p.sigma_min * (1.0 - 1e-12) && h < p.base.sigma0());
            lo = lo.min(l);
            hi = hi.max(h);
            let m = |x: &[i16]| x.iter().map(|v| v.unsigned_abs()).max().unwrap();
            max_fg = max_fg.max(m(sk.f())).max(m(sk.g()));
            max_big = max_big.max(m(sk.big_f())).max(m(sk.big_g()));
            sk.validate().unwrap();
            let t = random_target(&p, &mut prng);
            let s = samp_pre(&ek, &t, &seed_of(0x78, k)).unwrap();
            assert!(verify(&pk, &t, &s, p.beta_sq));
            ratio += s.norm_sq() as f64 / (2.0 * p.n() as f64 * p.sigma * p.sigma);
        }
        let gs_pass = cand - fg - ortho;
        let solve_runs = gs_pass - noinv;
        eprintln!(
            "{}: {keys} keys, total {:.1} s; keygen {:.2} ms/key; {:.2} candidates/key; GS pass \
             {gs_pass}/{cand} = {:.4}; not invertible {noinv}; solver failures \
             {solve}/{solve_runs} = {:.4}; max |f,g| {max_fg}, max |F,G| {max_big}; leaves \
             [{lo:.6}, {hi:.6}] (envelope [{:.6}, {:.6}], base {}); mean ||s||^2/(2 d sigma^2) = \
             {:.4}",
            p.name,
            t0.elapsed().as_secs_f64(),
            keygen_s * 1e3 / keys as f64,
            cand as f64 / keys as f64,
            gs_pass as f64 / cand as f64,
            solve as f64 / solve_runs as f64,
            p.sigma_min,
            sigma_max(&p),
            p.base.sigma0(),
            ratio / keys as f64
        );
        assert!(hi <= sigma_max(&p) * (1.0 + 1e-9));
        assert!((ratio / keys as f64 - 1.0).abs() < 0.01);
    }
}
