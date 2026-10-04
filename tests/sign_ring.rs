//! R_q encoding and arithmetic against fn-dsa-comm at q = 12289 (design.md §7, S-RING-3), and
//! the public operator API.
//!
//! fn-dsa-comm's `modq_encode` (fn-dsa-comm/src/codec.rs:73-100) packs four 14-bit values into
//! 7 little-endian bytes; `RqPoly::to_bytes` must produce the same bytes at q = 12289, and
//! `from_bytes` must invert `modq_encode`. fn-dsa-comm's NTT arithmetic modulo 12289
//! (`mqpoly_*`) is an independent oracle for `mul`.
//!
//! Run: `cargo test --test sign_ring`.

use fn_dsa_comm::codec::{modq_decode, modq_encode};
use fn_dsa_comm::mq::{
    mqpoly_NTT_to_int, mqpoly_ext_to_int, mqpoly_int_to_NTT, mqpoly_int_to_ext, mqpoly_mul_ntt,
};
use ntru_trapdoor::hazmat::{Prng, Shake256Prng};
use ntru_trapdoor::{Params, RqPoly};

fn rand_poly(params: &Params, rng: &mut Shake256Prng) -> RqPoly {
    let c: Vec<u32> = (0..params.n())
        .map(|_| (rng.next_u64() % params.q as u64) as u32)
        .collect();
    RqPoly::from_coeffs(params, &c).unwrap()
}

#[test]
fn encoding_equals_modq_encode() {
    let params = Params::FALCON_1024;
    let mut rng = Shake256Prng::new(b"modq");
    for _ in 0..16 {
        let a = rand_poly(&params, &mut rng);
        let h: Vec<u16> = a.coeffs().iter().map(|&x| x as u16).collect();
        let mut enc = vec![0u8; 1792];
        assert_eq!(modq_encode(&h, &mut enc), 1792);
        assert_eq!(a.to_bytes(), enc);
        assert_eq!(RqPoly::from_bytes(&params, &enc).unwrap(), a);
        let mut back = vec![0u16; 1024];
        modq_decode(&a.to_bytes(), &mut back).unwrap();
        assert_eq!(back, h);
    }
}

/// `mul` against fn-dsa-comm's NTT product modulo 12289 (logn 2..10).
#[test]
fn mul_matches_fndsa_mq() {
    let mut rng = Shake256Prng::new(b"mq-mul");
    for logn in 2..=10u32 {
        let params = Params {
            logn,
            ..Params::FALCON_1024
        };
        for _ in 0..4 {
            let a = rand_poly(&params, &mut rng);
            let b = rand_poly(&params, &mut rng);
            let mut x: Vec<u16> = a.coeffs().iter().map(|&v| v as u16).collect();
            let mut y: Vec<u16> = b.coeffs().iter().map(|&v| v as u16).collect();
            mqpoly_ext_to_int(logn, &mut x);
            mqpoly_ext_to_int(logn, &mut y);
            mqpoly_int_to_NTT(logn, &mut x);
            mqpoly_int_to_NTT(logn, &mut y);
            mqpoly_mul_ntt(logn, &mut x, &y);
            mqpoly_NTT_to_int(logn, &mut x);
            mqpoly_int_to_ext(logn, &mut x);
            let want: Vec<u32> = x.iter().map(|&v| v as u32).collect();
            assert_eq!(a.mul(&b).coeffs(), &want[..], "logn {logn}");
            assert_eq!((&a * &b).coeffs(), &want[..]);
        }
    }
}

/// The PCS ring: operators, distributivity and inverses through the public API.
#[test]
fn pcs_ring_identities() {
    let params = Params::PCS;
    let mut rng = Shake256Prng::new(b"pcs-ring");
    let a = rand_poly(&params, &mut rng);
    let b = rand_poly(&params, &mut rng);
    let c = rand_poly(&params, &mut rng);
    assert_eq!(&a * &(&b + &c), &(&a * &b) + &(&a * &c));
    assert_eq!(&(&a - &b) + &b, a);
    assert_eq!(&a + &(-&a), RqPoly::zero(&params).unwrap());
    let ai = a.inverse().unwrap();
    assert!(a.is_invertible());
    let mut one = vec![0u32; 1024];
    one[0] = 1;
    assert_eq!(&a * &ai, RqPoly::from_coeffs(&params, &one).unwrap());
    // (ab)^-1 = a^-1 b^-1
    let bi = b.inverse().unwrap();
    assert_eq!((&a * &b).inverse().unwrap(), &ai * &bi);
    // Encoding: 2560 bytes, round trip.
    let enc = c.to_bytes();
    assert_eq!(enc.len(), 2560);
    assert_eq!(RqPoly::from_bytes(&params, &enc).unwrap(), c);
}
