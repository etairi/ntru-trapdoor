//! Key encodings (design.md §7, K-ENC-1):
//! - round trips at both presets;
//! - sizes: public key 2560 B (PCS) and 1792 B (Falcon-1024); secret key 5889 B and 4097 B;
//! - rejections: a wrong header, a wrong length, a coefficient at −2^(w−1), nonzero padding, a
//!   public-key coefficient ≥ q, and a corrupted F (the NTRU equation fails).

use ntru_trapdoor::{Error, Params, PublicKey, SecretKey, hazmat, tables, trapgen};

/// Sets bits `[off, off + w)` of `bytes` (little-endian bit order) to `v`.
fn set_bits(bytes: &mut [u8], off: usize, w: usize, v: u32) {
    for i in 0..w {
        let bit = off + i;
        let (byte, sh) = (bit / 8, bit % 8);
        if (v >> i) & 1 == 1 {
            bytes[byte] |= 1 << sh;
        } else {
            bytes[byte] &= !(1 << sh);
        }
    }
}

#[test]
fn round_trips_and_sizes() {
    for (p, pk_len, sk_len) in [(Params::PCS, 2560, 5889), (Params::FALCON_1024, 1792, 4097)] {
        let (pk, sk) = trapgen(&p, &[0x42; 32]).unwrap();
        let pkb = pk.to_bytes();
        assert_eq!(pkb.len(), pk_len);
        assert_eq!(pkb.len(), p.rq_bytes());
        assert_eq!(PublicKey::from_bytes(&p, &pkb).unwrap(), pk);
        let skb = sk.to_bytes();
        assert_eq!(skb.len(), sk_len, "{}", p.name);
        assert_eq!(skb[0], 0xA0 | p.logn as u8);
        let sk2 = SecretKey::from_bytes(&p, &skb).unwrap();
        assert_eq!(
            (sk2.f(), sk2.g(), sk2.big_f(), sk2.big_g()),
            (sk.f(), sk.g(), sk.big_f(), sk.big_g())
        );
        assert_eq!(sk2.public_key().unwrap(), pk);
        // The encoding is canonical: decoding then encoding gives the same bytes.
        assert_eq!(&*sk2.to_bytes(), &*skb);
    }
}

#[test]
fn secret_key_rejections() {
    let p = Params::PCS;
    let (_, sk) = trapgen(&p, &[0x17; 32]).unwrap();
    let skb = sk.to_bytes().to_vec();
    let n = p.n();
    // Lengths.
    assert_eq!(
        SecretKey::from_bytes(&p, &skb[..skb.len() - 1]).unwrap_err(),
        Error::InvalidLength
    );
    let mut longer = skb.clone();
    longer.push(0);
    assert_eq!(
        SecretKey::from_bytes(&p, &longer).unwrap_err(),
        Error::InvalidLength
    );
    // The Falcon-1024 encoding has the same header (logn = 10) but another length.
    assert_eq!(
        SecretKey::from_bytes(&Params::FALCON_1024, &skb).unwrap_err(),
        Error::InvalidLength
    );
    // Header.
    let mut bad = skb.clone();
    bad[0] = 0xA9;
    assert_eq!(
        SecretKey::from_bytes(&p, &bad).unwrap_err(),
        Error::InvalidEncoding
    );
    // -2^(w-1) in f (w = 10), g (w = 10), F (w = 13), G (w = 13).
    let (wfg, wbig) = (p.sk_fg_bits as usize, p.sk_big_fg_bits as usize);
    let starts = [0, n * wfg, 2 * n * wfg, 2 * n * wfg + n * wbig];
    for (k, &start) in starts.iter().enumerate() {
        let w = if k < 2 { wfg } else { wbig };
        let mut bad = skb.clone();
        // Coefficient 5 of polynomial k.
        set_bits(&mut bad[1..], start + 5 * w, w, 1 << (w - 1));
        assert_eq!(
            SecretKey::from_bytes(&p, &bad).unwrap_err(),
            Error::InvalidEncoding,
            "polynomial {k}"
        );
    }
    // A corrupted F: F_3 + 1 still decodes, but fG - gF != q.
    let mut bad = skb.clone();
    let off = starts[2] + 3 * wbig;
    let mut v = 0u32;
    for i in 0..wbig {
        v |= (((bad[1 + (off + i) / 8] >> ((off + i) % 8)) & 1) as u32) << i;
    }
    let f3 = ((v << (32 - wbig)) as i32) >> (32 - wbig);
    assert_eq!(f3, sk.big_f()[3] as i32);
    set_bits(
        &mut bad[1..],
        off,
        wbig,
        ((f3 + 1) as u32) & ((1 << wbig) - 1),
    );
    assert_eq!(
        SecretKey::from_bytes(&p, &bad).unwrap_err(),
        Error::InvalidKey
    );
}

#[test]
fn public_key_rejections() {
    for p in [Params::PCS, Params::FALCON_1024] {
        let (pk, _) = trapgen(&p, &[0x18; 32]).unwrap();
        let pkb = pk.to_bytes();
        assert_eq!(
            PublicKey::from_bytes(&p, &pkb[1..]).unwrap_err(),
            Error::InvalidLength
        );
        let mut longer = pkb.clone();
        longer.push(0);
        assert_eq!(
            PublicKey::from_bytes(&p, &longer).unwrap_err(),
            Error::InvalidLength
        );
        // Coefficient 0 set to all ones (2^q_bits - 1 >= q).
        let mut bad = pkb.clone();
        let qb = p.q_bits() as usize;
        set_bits(&mut bad, 0, qb, (1 << qb) - 1);
        assert_eq!(
            PublicKey::from_bytes(&p, &bad).unwrap_err(),
            Error::InvalidEncoding
        );
        // Coefficient 1 set to exactly q.
        let mut bad = pkb.clone();
        set_bits(&mut bad, qb, qb, p.q);
        assert_eq!(
            PublicKey::from_bytes(&p, &bad).unwrap_err(),
            Error::InvalidEncoding
        );
        // And q - 1 is accepted (every canonical h is a public key).
        let mut ok = pkb.clone();
        set_bits(&mut ok, qb, qb, p.q - 1);
        assert_eq!(
            PublicKey::from_bytes(&p, &ok).unwrap().h().coeffs()[1],
            p.q - 1
        );
    }
}

/// A toy set with n = 2 (logn = 1), q = 17, where every polynomial leaves padding bits:
/// f, g at 7 bits (14 of 16) and F, G at 9 bits (18 of 24), with Falcon-1024's table.
fn toy_params() -> Params {
    Params {
        name: "toy (n = 2, q = 17)",
        logn: 1,
        q: 17,
        sigma: 2.0,
        inv_sigma: 0.5,
        sigma_min: 0.5,
        base: Params::FALCON_1024.base,
        inv_q: 1.0 / 17.0,
        beta_sq: 1000,
        sigma_fg: 1.0,
        fg_table: &tables::FG_FALCON1024_128,
        gs_bound_num: 13689 * 17,
        gs_bound_den: 10000,
        sk_fg_bits: 7,
        sk_big_fg_bits: 9,
        samppre_max_attempts: 27,
        keygen_max_attempts: 64,
    }
}

#[test]
fn padding_bits_are_checked() {
    let p = toy_params();
    // A valid toy key: f = 1 + x (f(4) = 5, f(-4) = -3: a unit mod 17), g = 2 - x, (F, G) from
    // NTRUSolve; Res(f) = 2 and Res(g) = 5 are coprime.
    let (f, g) = (vec![1i16, 1], vec![2i16, -1]);
    let (big_f, big_g) = hazmat::ntru_solve(p.q, &f, &g).unwrap();
    let to16 = |x: Vec<i32>| x.into_iter().map(|v| v as i16).collect::<Vec<i16>>();
    let sk = SecretKey::from_parts(&p, f, g, to16(big_f), to16(big_g)).unwrap();
    sk.validate().unwrap();
    let skb = sk.to_bytes().to_vec();
    assert_eq!(skb.len(), 1 + 2 + 2 + 3 + 3);
    let sk2 = SecretKey::from_bytes(&p, &skb).unwrap();
    assert_eq!(sk2.big_g(), sk.big_g());
    // The top two bits of the second byte of f, of g, and the top six bits of the third
    // byte of F and of G are padding.
    for (byte, mask) in [(2usize, 0xC0u8), (4, 0xC0), (7, 0xFC), (10, 0xFC)] {
        for bit in 0..8 {
            if mask & (1 << bit) == 0 {
                continue;
            }
            let mut bad = skb.clone();
            bad[byte] |= 1 << bit;
            assert_eq!(
                SecretKey::from_bytes(&p, &bad).unwrap_err(),
                Error::InvalidEncoding,
                "byte {byte} bit {bit}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "does not fit the encoding widths")]
fn to_bytes_panics_on_an_unencodable_key() {
    let p = Params::FALCON_1024;
    let (_, sk) = trapgen(&p, &[5u8; 32]).unwrap();
    let mut big_f = sk.big_f().to_vec();
    big_f[0] = 300; // F, G are 9 bits wide at Falcon-1024: |F_i| < 256
    let bad = SecretKey::from_parts(
        &p,
        sk.f().to_vec(),
        sk.g().to_vec(),
        big_f,
        sk.big_g().to_vec(),
    )
    .unwrap();
    let _ = bad.to_bytes();
}
