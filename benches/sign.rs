//! Benchmarks of the signing side: R_q multiplication and inversion, FFT, SampPre, Verify, the
//! expanded key, SamplerZ and the PRNG, plus fn-dsa's Falcon-1024 signing as a baseline.
//!
//! Plan: design.md §8 (criterion; PCS set first, Falcon-1024 for comparison).
//! Keys: the first Sage fixture key of `tests/fixtures/sign/pcs_keys.txt` (PCS), and an fn-dsa
//! Falcon-1024 key from a fixed seed. Run: `cargo bench --bench sign`.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use fn_dsa::{
    CryptoRng, DOMAIN_NONE, HASH_ID_RAW, KeyPairGenerator, KeyPairGenerator1024, RngCore, RngError,
    SigningKey, SigningKey1024, sign_key_size, signature_size, vrfy_key_size,
};
use fn_dsa_comm::codec::trim_i8_decode;
use ntru_trapdoor::hazmat::{LeafSampler, Prng, SamplerZ, Shake256Prng, fft, ifft};
use ntru_trapdoor::{ExpandedKey, Params, PublicKey, RqPoly, SecretKey, samp_pre, verify};

/// The first PCS fixture key: (secret key, public key).
fn pcs_key(params: &Params) -> (SecretKey, PublicKey) {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sign/pcs_keys.txt"
    ))
    .unwrap();
    let mut polys: Vec<Vec<i64>> = Vec::new();
    for line in text.lines().skip_while(|l| !l.starts_with("key 0")).skip(1) {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("f" | "g" | "F" | "G" | "h") => {
                polys.push(it.map(|x| x.parse().unwrap()).collect())
            }
            _ => break,
        }
    }
    let i16v = |v: &Vec<i64>| v.iter().map(|&x| x as i16).collect::<Vec<i16>>();
    let sk = SecretKey::from_parts(
        params,
        i16v(&polys[0]),
        i16v(&polys[1]),
        i16v(&polys[2]),
        i16v(&polys[3]),
    )
    .unwrap();
    let pk = PublicKey::from_h(params, RqPoly::from_i64(params, &polys[4]).unwrap()).unwrap();
    (sk, pk)
}

/// A deterministic rand_core 0.6 RNG for fn-dsa.
struct TestRng(Shake256Prng);
impl CryptoRng for TestRng {}
impl RngCore for TestRng {
    fn next_u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        self.0.next_bytes(&mut b);
        u32::from_le_bytes(b)
    }
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.next_bytes(dest);
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RngError> {
        self.fill_bytes(dest);
        Ok(())
    }
}

/// An fn-dsa Falcon-1024 key: the encoded signing key and our decoded SecretKey.
fn falcon_key() -> (Vec<u8>, SecretKey) {
    let params = Params::FALCON_1024;
    let mut kg = KeyPairGenerator1024::default();
    let mut rng = TestRng(Shake256Prng::new(b"bench-falcon-key"));
    let mut sk = vec![0u8; sign_key_size(10)];
    let mut vk = vec![0u8; vrfy_key_size(10)];
    kg.keygen(10, &mut rng, &mut sk, &mut vk);
    let n = 1024;
    let (mut f, mut g, mut big_f) = (vec![0i8; n], vec![0i8; n], vec![0i8; n]);
    let j = 1 + trim_i8_decode(&sk[1..], &mut f, 5).unwrap();
    let j = j + trim_i8_decode(&sk[j..], &mut g, 5).unwrap();
    trim_i8_decode(&sk[j..], &mut big_f, 8).unwrap();
    let to16 = |v: &[i8]| v.iter().map(|&x| x as i16).collect::<Vec<i16>>();
    let (f, g, big_f) = (to16(&f), to16(&g), to16(&big_f));
    let fr = RqPoly::from_i16(&params, &f).unwrap();
    let gr = RqPoly::from_i16(&params, &g).unwrap();
    let br = RqPoly::from_i16(&params, &big_f).unwrap();
    let big_g: Vec<i16> = gr
        .mul(&br)
        .mul(&fr.inverse().unwrap())
        .to_centered()
        .iter()
        .map(|&x| x as i16)
        .collect();
    (
        sk,
        SecretKey::from_parts(&params, f, g, big_f, big_g).unwrap(),
    )
}

fn rand_poly(params: &Params, rng: &mut Shake256Prng) -> RqPoly {
    let c: Vec<u32> = (0..params.n())
        .map(|_| (rng.next_u64() % params.q as u64) as u32)
        .collect();
    RqPoly::from_coeffs(params, &c).unwrap()
}

fn bench_ring(c: &mut Criterion) {
    let mut g = c.benchmark_group("ring");
    let mut rng = Shake256Prng::new(b"bench-ring");
    for (name, params) in [("pcs", Params::PCS), ("falcon", Params::FALCON_1024)] {
        let a = rand_poly(&params, &mut rng);
        let b = rand_poly(&params, &mut rng);
        g.bench_function(format!("mul/{name}"), |bch| {
            bch.iter(|| black_box(&a).mul(black_box(&b)))
        });
    }
    let params = Params::PCS;
    let a = rand_poly(&params, &mut rng);
    g.bench_function("inverse/pcs", |bch| bch.iter(|| black_box(&a).inverse()));
    g.bench_function("is_invertible/pcs", |bch| {
        bch.iter(|| black_box(&a).is_invertible())
    });
    let b = rand_poly(&params, &mut rng);
    g.bench_function("prepare/pcs", |bch| bch.iter(|| black_box(&a).prepare()));
    let ap = a.prepare();
    g.bench_function("mul_prepared/pcs", |bch| {
        bch.iter(|| black_box(&ap).mul(black_box(&b)))
    });
    g.bench_function("add/pcs", |bch| {
        bch.iter(|| black_box(&a).add(black_box(&b)))
    });
    let enc = a.to_bytes();
    g.bench_function("encode/pcs", |bch| bch.iter(|| black_box(&a).to_bytes()));
    g.bench_function("decode/pcs", |bch| {
        bch.iter(|| RqPoly::from_bytes(&params, black_box(&enc)))
    });
    g.finish();
}

fn bench_fft(c: &mut Criterion) {
    let mut rng = Shake256Prng::new(b"bench-fft");
    let f: Vec<f64> = (0..1024)
        .map(|_| (rng.next_u16() as f64) - 32768.0)
        .collect();
    let mut buf = f.clone();
    c.bench_function("fft/fft_ifft/1024", |bch| {
        bch.iter(|| {
            fft(10, black_box(&mut buf));
            ifft(10, black_box(&mut buf));
        })
    });
}

fn bench_prng(c: &mut Criterion) {
    let mut out = vec![0u8; 65536];
    c.bench_function("prng/shake256_64k", |bch| {
        bch.iter(|| {
            let mut p = Shake256Prng::new(b"bench-prng");
            p.next_bytes(black_box(&mut out));
        })
    });
}

fn bench_sampler(c: &mut Criterion) {
    let mut g = c.benchmark_group("sampler");
    for (name, params, sigma) in [
        ("pcs_2.6", Params::PCS, 2.6),
        ("falcon_1.5", Params::FALCON_1024, 1.5),
    ] {
        let mut prng = Shake256Prng::new(b"bench-sampler");
        let mut s = SamplerZ::new(&params, &mut prng);
        let isigma = 1.0 / sigma;
        let mut mu = 0.37;
        g.bench_function(format!("leaf/{name}"), |bch| {
            bch.iter(|| {
                mu += 1.618;
                s.sample(black_box(mu), black_box(isigma))
            })
        });
    }
    g.finish();
}

fn bench_expand_samppre_verify(c: &mut Criterion) {
    let pcs = Params::PCS;
    let (pcs_sk, pcs_pk) = pcs_key(&pcs);
    let (falcon_enc, falcon_sk) = falcon_key();
    // `expand/*`: keys that are known valid (decoded, as a stored key would be), so that only the
    // basis, the tree and the leaf check are timed. `expand_validate/*`: keys from `from_parts`,
    // which `ExpandedKey::new` first validates (fG - gF = q, widths, f invertible).
    let pcs_valid = SecretKey::from_bytes(&pcs, &pcs_sk.to_bytes()).unwrap();
    let falcon_valid = SecretKey::from_bytes(&Params::FALCON_1024, &falcon_sk.to_bytes()).unwrap();
    let mut g = c.benchmark_group("expand");
    g.bench_function("pcs", |bch| {
        bch.iter(|| ExpandedKey::new(black_box(&pcs_valid)).unwrap())
    });
    g.bench_function("falcon", |bch| {
        bch.iter(|| ExpandedKey::new(black_box(&falcon_valid)).unwrap())
    });
    g.finish();
    let mut g = c.benchmark_group("expand_validate");
    g.bench_function("pcs", |bch| {
        bch.iter(|| ExpandedKey::new(black_box(&pcs_sk)).unwrap())
    });
    g.bench_function("falcon", |bch| {
        bch.iter(|| ExpandedKey::new(black_box(&falcon_sk)).unwrap())
    });
    g.finish();

    let mut g = c.benchmark_group("samp_pre");
    let mut rng = Shake256Prng::new(b"bench-samppre");
    for (name, params, sk) in [
        ("pcs", pcs, &pcs_sk),
        ("falcon", Params::FALCON_1024, &falcon_sk),
    ] {
        let ek = ExpandedKey::new(sk).unwrap();
        let t = rand_poly(&params, &mut rng);
        let mut ctr = 0u64;
        g.bench_function(name, |bch| {
            bch.iter(|| {
                ctr += 1;
                let mut seed = [0u8; 32];
                seed[..8].copy_from_slice(&ctr.to_le_bytes());
                samp_pre(&ek, black_box(&t), &seed).unwrap()
            })
        });
    }
    g.finish();

    let ek = ExpandedKey::new(&pcs_sk).unwrap();
    let t = rand_poly(&pcs, &mut rng);
    let s = samp_pre(&ek, &t, &[9u8; 32]).unwrap();
    c.bench_function("verify/pcs", |bch| {
        bch.iter(|| assert!(verify(&pcs_pk, black_box(&t), black_box(&s), pcs.beta_sq)))
    });

    // fn-dsa's signing at Falcon-1024 (same key repeated: its "+sign"), the on-the-fly
    // ffLDL baseline for samp_pre/falcon. It includes hashing the message to a point.
    let mut signer = SigningKey1024::decode(&falcon_enc).unwrap();
    let mut rng = TestRng(Shake256Prng::new(b"bench-fndsa"));
    let mut sig = vec![0u8; signature_size(10)];
    c.bench_function("fndsa/sign_1024", |bch| {
        bch.iter(|| {
            signer
                .sign(
                    &mut rng,
                    &DOMAIN_NONE,
                    &HASH_ID_RAW,
                    black_box(b"message"),
                    &mut sig,
                )
                .unwrap()
        })
    });
}

criterion_group!(
    group,
    bench_ring,
    bench_fft,
    bench_prng,
    bench_sampler,
    bench_expand_samppre_verify
);
criterion_main!(group);
