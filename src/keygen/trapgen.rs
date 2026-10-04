//! The TrapGen candidate loop and its deterministic core (design.md §5.9).
//!
//! Per candidate, the checks run cheapest first; the accepted key does not depend on the
//! order, only the rejection counters do:
//! 1. the encoding width of $`f,g`$ (never fails for CDT samples);
//! 2. $`N_1=\|(g,-f)\|^2`$, exact (rejects about half of the candidates at the PCS set, where
//!    $`\mathbb E[N_1]=2d\sigma_{fg}^2`$ equals the bound);
//! 3. $`N_2`$, two FFTs;
//! 4. $`f`$ invertible modulo $`q`$, and $`h=g\,f^{-1}`$;
//! 5. NTRUSolve (big integers, by far the most expensive step);
//! 6. the encoding width of $`F,G`$, then the exact check $`fG-gF=q`$ (eight NTTs,
//!    `ring::ntt::ntru_equation_holds`);
//! 7. the expanded key and its leaf check (`ExpandedKey::from_valid`). Under
//!    [`Params::check_leaf_window`], which [`crate::trapgen`] requires, this check can only fire
//!    through `f64` rounding at the window's boundary.

use zeroize::{Zeroize, Zeroizing};

use super::gs::{ortho_sq_norm, passes_norm_fg, passes_norm_ortho, sq_norm_fg};
use super::{KeygenReject, KeygenStats, gauss, solve};
use crate::prng::Shake256Prng;
use crate::ring::ntt;
use crate::{Error, ExpandedKey, Params, PublicKey, RqPoly, SecretKey};

/// Domain separator of the key-generation stream (design.md §5.1).
pub(crate) const KEYGEN_DOMAIN: &[u8] = b"ntru-trapdoor/v1/keygen";

/// `true` iff every $`|x_i|<2^{w-1}`$ (so $`-2^{w-1}`$, which the encoding rejects, fails).
pub(crate) fn fits_width<T: Copy + Into<i64>>(x: &[T], w: u32) -> bool {
    let lim = 1i64 << (w - 1);
    x.iter().all(|&v| {
        let v: i64 = v.into();
        -lim < v && v < lim
    })
}

/// The keygen PRNG: SHAKE256("ntru-trapdoor/v1/keygen" || LE32(q) || logn || seed).
pub(crate) fn keygen_prng(params: &Params, seed: &[u8; 32]) -> Shake256Prng {
    Shake256Prng::from_parts(&[
        KEYGEN_DOMAIN,
        &params.q.to_le_bytes(),
        &[params.logn as u8],
        seed,
    ])
}

/// [`crate::trapgen_with_stats`]. The loop has the structure of fn-dsa's `keygen_from_seed`
/// (fn-dsa-kgen/src/lib.rs:266-315: one seeded PRNG, then sample, test and solve until a
/// candidate passes), with this crate's sampler, tests and solver.
pub(crate) fn trapgen_with_stats(
    params: &Params,
    seed: &[u8; 32],
) -> Result<(PublicKey, SecretKey, KeygenStats), Error> {
    params.validate()?;
    params.check_leaf_window()?;
    let n = params.n();
    let mut prng = keygen_prng(params, seed);
    let mut f = Zeroizing::new(vec![0i16; n]);
    let mut g = Zeroizing::new(vec![0i16; n]);
    let mut stats = KeygenStats::default();
    for _ in 0..params.keygen_max_attempts {
        stats.candidates += 1;
        gauss::sample_fg(params, &mut prng, &mut f);
        gauss::sample_fg(params, &mut prng, &mut g);
        match from_fg_checked(params, &f, &g) {
            Ok((pk, sk)) => return Ok((pk, sk, stats)),
            Err(KeygenReject::NormFg) => stats.rejected_norm_fg += 1,
            Err(KeygenReject::NormOrtho) => stats.rejected_norm_ortho += 1,
            Err(KeygenReject::NotInvertible) => stats.rejected_not_invertible += 1,
            Err(KeygenReject::Solve) => stats.rejected_solve += 1,
            Err(KeygenReject::Size) => stats.rejected_size += 1,
            Err(KeygenReject::Leaf) => stats.rejected_leaf += 1,
            // Validated parameters and lengths: unreachable unless the crate is inconsistent.
            Err(KeygenReject::Input) => {
                return Err(Error::InvalidParams(
                    "key generation: inconsistent candidate",
                ));
            }
        }
    }
    Err(Error::KeygenExhausted)
}

/// [`crate::trapgen_from_fg`].
pub(crate) fn trapgen_from_fg(
    params: &Params,
    f: &[i16],
    g: &[i16],
) -> Result<(PublicKey, SecretKey), KeygenReject> {
    if params.validate().is_err() {
        return Err(KeygenReject::Input);
    }
    from_fg_checked(params, f, g)
}

/// [`trapgen_from_fg`] for parameters that passed [`Params::validate`].
fn from_fg_checked(
    params: &Params,
    f: &[i16],
    g: &[i16],
) -> Result<(PublicKey, SecretKey), KeygenReject> {
    let n = params.n();
    if f.len() != n || g.len() != n {
        return Err(KeygenReject::Input);
    }
    // 1. Widths of f and g.
    if !fits_width(f, params.sk_fg_bits) || !fits_width(g, params.sk_fg_bits) {
        return Err(KeygenReject::Size);
    }
    // 2-3. Falcon's two Gram-Schmidt norms.
    if !passes_norm_fg(params, sq_norm_fg(f, g)) {
        return Err(KeygenReject::NormFg);
    }
    if !passes_norm_ortho(params, ortho_sq_norm(params, f, g)) {
        return Err(KeygenReject::NormOrtho);
    }
    // 4. f invertible modulo q; h = g / f.
    let h = {
        let fq = Zeroizing::new(RqPoly::from_i16(params, f).map_err(|_| KeygenReject::Input)?);
        let Some(finv) = fq.inverse() else {
            return Err(KeygenReject::NotInvertible);
        };
        let finv = Zeroizing::new(finv);
        let gq = Zeroizing::new(RqPoly::from_i16(params, g).map_err(|_| KeygenReject::Input)?);
        gq.mul(&finv)
    };
    // 5. NTRUSolve.
    let Some((big_f, big_g)) = solve::ntru_solve(params.q, f, g) else {
        return Err(KeygenReject::Solve);
    };
    let (mut big_f, mut big_g) = (Zeroizing::new(big_f), Zeroizing::new(big_g));
    // 6. Widths of F and G (at most 16 bits, so they fit i16), then the exact NTRU equation.
    if !fits_width(&big_f, params.sk_big_fg_bits) || !fits_width(&big_g, params.sk_big_fg_bits) {
        return Err(KeygenReject::Size);
    }
    let to16 = |x: &[i32]| x.iter().map(|&v| v as i16).collect::<Vec<i16>>();
    // The parameters passed `Params::validate` and every length is n, so the checks of
    // `SecretKey::from_parts` and `PublicKey::from_h` would only repeat work. The key is
    // complete below: f is invertible (step 4), the widths fit, and fG - gF = q.
    let mut sk = SecretKey {
        params: *params,
        f: f.to_vec(),
        g: g.to_vec(),
        big_f: to16(&big_f),
        big_g: to16(&big_g),
        validated: false,
    };
    big_f.zeroize();
    big_g.zeroize();
    if !ntt::ntru_equation_holds(params.logn, params.q, &sk.f, &sk.g, &sk.big_f, &sk.big_g) {
        return Err(KeygenReject::Solve);
    }
    sk.validated = true;
    // 7. The ffLDL tree and its leaf check (the expanded key is wiped when dropped).
    match ExpandedKey::from_valid(&sk) {
        Ok(ek) => drop(ek),
        Err(Error::LeafOutOfRange) => return Err(KeygenReject::Leaf),
        Err(_) => return Err(KeygenReject::Input),
    }
    Ok((PublicKey::from_valid(params, h), sk))
}

/// [`SecretKey::public_key`].
pub(crate) fn public_key(sk: &SecretKey) -> Result<PublicKey, Error> {
    let params = &sk.params;
    let fq = Zeroizing::new(RqPoly::from_i16(params, &sk.f)?);
    let finv = Zeroizing::new(fq.inverse().ok_or(Error::InvalidKey)?);
    let gq = Zeroizing::new(RqPoly::from_i16(params, &sk.g)?);
    Ok(PublicKey::from_valid(params, gq.mul(&finv)))
}

/// [`SecretKey::validate`].
pub(crate) fn validate(sk: &SecretKey) -> Result<(), Error> {
    let p = &sk.params;
    let n = p.n();
    if sk.f.len() != n || sk.g.len() != n || sk.big_f.len() != n || sk.big_g.len() != n {
        return Err(Error::InvalidKey);
    }
    if !fits_width(&sk.f, p.sk_fg_bits)
        || !fits_width(&sk.g, p.sk_fg_bits)
        || !fits_width(&sk.big_f, p.sk_big_fg_bits)
        || !fits_width(&sk.big_g, p.sk_big_fg_bits)
    {
        return Err(Error::InvalidKey);
    }
    // Exact, with eight NTTs whose residues are wiped (the big-integer check that this replaces
    // left copies of the trapdoor in freed memory and took about four times as long).
    if !ntt::ntru_equation_holds(p.logn, p.q, &sk.f, &sk.g, &sk.big_f, &sk.big_g) {
        return Err(Error::InvalidKey);
    }
    let fq = Zeroizing::new(RqPoly::from_i16(p, &sk.f)?);
    if !fq.is_invertible() {
        return Err(Error::InvalidKey);
    }
    Ok(())
}
