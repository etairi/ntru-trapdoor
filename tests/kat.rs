//! Known answers of this version: TrapGen and SampPre at both presets, for fixed seeds and
//! targets.
//!
//! SampPre derives its coins from (key, target, seed) under the label
//! `ntru-trapdoor/v1/samp-pre` (src/sign.rs). LTYZ25 (cited by the paper) shows that the same
//! coins on the same target under slightly different floating-point arithmetic can give two
//! lattice points whose difference leaks the key. So the map (key, target, seed) -> s must be
//! fixed for a given label: **if `samp_pre_known_answers` fails after a change, the change
//! altered SampPre's outputs (its arithmetic, tables or use of the stream), and the label in
//! src/sign.rs must move to the next version (`v2`, ...) before the new answers are pinned.**
//! A change of `trapgen_known_answers` alone (a new key for the same seed) is harmless, but the
//! keys of the SampPre test change with it, so that test must be re-pinned too.
//!
//! Every digest is SHAKE256 (32 bytes) of the data named in the test.

use ntru_trapdoor::hazmat::shake256;
use ntru_trapdoor::{ExpandedKey, Params, RqPoly, samp_pre, trapgen, verify};

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn digest(parts: &[&[u8]]) -> String {
    let mut out = [0u8; 32];
    shake256(parts, &mut out);
    hex(&out)
}

/// The key-generation seed of preset `name`: SHAKE256("ntru-trapdoor/kat/keygen" || name).
fn keygen_seed(name: &str) -> [u8; 32] {
    let mut s = [0u8; 32];
    shake256(&[b"ntru-trapdoor/kat/keygen", name.as_bytes()], &mut s);
    s
}

/// Target `j`: coefficient `i` is the little-endian u64 at offset 8i of
/// SHAKE256("ntru-trapdoor/kat/target" || name || j), reduced modulo q.
fn target(p: &Params, j: u8) -> RqPoly {
    let mut bytes = vec![0u8; 8 * p.n()];
    shake256(
        &[b"ntru-trapdoor/kat/target", p.name.as_bytes(), &[j]],
        &mut bytes,
    );
    let c: Vec<u32> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|w| (u64::from_le_bytes(*w) % p.q as u64) as u32)
        .collect();
    RqPoly::from_coeffs(p, &c).unwrap()
}

/// TrapGen: SHAKE256 of the public-key encoding and of the secret-key encoding.
#[test]
fn trapgen_known_answers() {
    let want = [
        (
            Params::PCS,
            "d55ff292ec45fd59825ddb451753c7c2db2433e8689d9e5b38129261445b2385",
            "f393c9330fd6ed533ccac59d57f687c760bf0a6482d4cd6f3a614d69f0ac22bf",
        ),
        (
            Params::FALCON_1024,
            "0cc089ae273001b4cca77d3cf08787ee37b3940010869d1dedcbc76984c06d6f",
            "1e4db1c2268e5ec135ad892f9747ff0a72470c1897f5e798f50b687475666d4d",
        ),
    ];
    for (p, want_pk, want_sk) in want {
        let (pk, sk) = trapgen(&p, &keygen_seed(p.name)).unwrap();
        assert_eq!(digest(&[&pk.to_bytes()]), want_pk, "{} pk", p.name);
        assert_eq!(digest(&[&sk.to_bytes()]), want_sk, "{} sk", p.name);
    }
}

/// SampPre: for targets j = 0, 1, 2 and seeds [j; 32], SHAKE256 of every s0_i, then every s1_i,
/// as little-endian i32, over the three calls; and the squared norms.
#[test]
fn samp_pre_known_answers() {
    let want = [
        (
            Params::PCS,
            "a08df50fa53f0b02875b9c53b10ecb651c3394b0307795c7df95667cad8e2c69",
            [14617776345, 14423212561, 14514700041],
        ),
        (
            Params::FALCON_1024,
            "4fbe684b3d71e5185680fcf2dab166e8d847bc5965f7d3a4664f3b60e3f85771",
            [58095581, 57560538, 59100624],
        ),
    ];
    for (p, want_s, want_norms) in want {
        let (pk, sk) = trapgen(&p, &keygen_seed(p.name)).unwrap();
        let ek = ExpandedKey::new(&sk).unwrap();
        let mut data = Vec::new();
        let mut norms = [0u64; 3];
        for j in 0..3u8 {
            let t = target(&p, j);
            let s = samp_pre(&ek, &t, &[j; 32]).unwrap();
            assert!(verify(&pk, &t, &s, p.beta_sq));
            for x in s.s0.iter().chain(s.s1.iter()) {
                data.extend_from_slice(&x.to_le_bytes());
            }
            norms[j as usize] = s.norm_sq();
        }
        assert_eq!(digest(&[&data]), want_s, "{} s", p.name);
        assert_eq!(norms, want_norms, "{} norms", p.name);
    }
}
