//! Inversion and the invertibility test in $`R_q=\mathbb Z_q[x]/(x^n+1)`$ by the norm tower.
//!
//! New code (design.md §5.2). Write $`a(x)=a_e(x^2)+x\,a_o(x^2)`$. The relative norm to
//! $`\mathbb Z_q[y]/(y^{n/2}+1)`$, $`y=x^2`$, is
//! $`N(a)(y)=a(x)\,a(-x)=a_e(y)^2-y\,a_o(y)^2`$: the determinant of multiplication by $`a`$ on the
//! basis $`\{1,x\}`$. Iterating down to $`\mathbb Z_q`$ gives the determinant of multiplication by
//! $`a`$ over $`\mathbb Z_q`$, which is nonzero iff $`a`$ is a unit (for prime $`q`$; this holds for
//! every factorisation pattern of $`x^n+1`$, including the PCS's two degree-512 factors). On the
//! way back, $`a^{-1}(x)=a(-x)\,N(a)^{-1}(x^2)=(a_eN^{-1})(x^2)-x\,(a_oN^{-1})(x^2)`$.
//!
//! Each level computes its products exactly over $`\mathbb Z`$ with the two-prime NTT of
//! [`super::ntt`] (the NTTs of $`a_e`$ and $`a_o`$ are computed once and reused on the way back up;
//! multiplication by $`y`$ is pointwise, by the evaluation points) and reduces modulo $`q`$. The
//! scalar at the bottom is inverted by Fermat's little theorem, $`a_0^{q-2}`$, a fixed-exponent
//! power, which is the inverse whenever $`q`$ is prime and $`a_0\ne0`$. Only when the result fails
//! the check $`a_0\cdot a_0^{q-2}\equiv1`$ (for a prime $`q`$: exactly when $`a_0=0`$, the `None`
//! case) does the extended Euclidean algorithm decide; it also makes the inverse exact for a
//! composite $`q`$ (the tower is a ring identity over any $`\mathbb Z_q`$, and an element is a unit
//! iff its norm is a unit of $`\mathbb Z_q`$). For a prime $`q`$ the sequence of operations does not
//! depend on the value of $`a`$; only the final `None`/`Some` reveals whether $`a`$ is a unit.
//! Intermediate buffers are wiped.

use zeroize::Zeroize;

use super::ntt::{ModQ, NttPrime, P1, P2, mp_add, mp_sub};

/// The NTTs of $`a_e`$ and $`a_o`$ at one level, modulo both primes. Wiped on drop (the element
/// being inverted may be secret, e.g. $`f`$ in key generation).
struct Level {
    /// log2 of the half size $`h`$ (the NTT size); $`h\ge2`$.
    logh: u32,
    e1: Vec<u32>,
    o1: Vec<u32>,
    e2: Vec<u32>,
    o2: Vec<u32>,
}

impl Drop for Level {
    fn drop(&mut self) {
        // Slice wipes: `Vec::zeroize` would also wipe the whole capacity byte by byte, about
        // ten times the cost [measured, sign/README.md].
        for v in [&mut self.e1, &mut self.o1, &mut self.e2, &mut self.o2] {
            v.as_mut_slice().zeroize();
        }
    }
}

/// In the NTT domain of size $`h=2^{\mathrm{logh}}\ge2`$: `e <- e^2 - y o^2`. Output $`2i`$ is
/// evaluated at $`\zeta_i`$ and output $`2i+1`$ at $`-\zeta_i`$, $`\zeta_i`$ = `gm[h/2 + i]` (see
/// the `ntt` module documentation).
fn norm_ntt(pr: &NttPrime, logh: u32, e: &[u32], o: &[u32]) -> Vec<u32> {
    let h = 1usize << logh;
    let mut out = vec![0u32; h];
    for i in 0..h / 2 {
        let z = pr.gm[h / 2 + i]; // Montgomery form of zeta_i
        for (k, sign_plus) in [(2 * i, false), (2 * i + 1, true)] {
            let e2 = pr.mul(e[k], e[k]);
            let o2 = pr.mul(o[k], o[k]);
            let zo2 = pr.mmul(o2, z); // zeta_i * o^2, plain
            // At -zeta_i: e^2 - (-zeta_i) o^2 = e^2 + zeta_i o^2.
            out[k] = if sign_plus {
                mp_add(e2, zo2, pr.p)
            } else {
                mp_sub(e2, zo2, pr.p)
            };
        }
    }
    out
}

/// One descent step: from $`a`$ of size $`2h`$ ($`h=2^{\mathrm{logh}}\ge2`$, coefficients in
/// $`[0,q)`$) to $`N(a)`$ of size $`h`$; returns the level data for the ascent.
fn descend(mq: &ModQ, logh: u32, a: &[u32]) -> (Vec<u32>, Level) {
    let h = 1usize << logh;
    let ae: Vec<u32> = a[..2 * h].iter().step_by(2).copied().collect();
    let ao: Vec<u32> = a[..2 * h].iter().skip(1).step_by(2).copied().collect();
    let (mut e1, mut o1) = (ae.clone(), ao.clone());
    let (mut e2, mut o2) = (ae, ao);
    P1.ntt(logh, &mut e1);
    P1.ntt(logh, &mut o1);
    P2.ntt(logh, &mut e2);
    P2.ntt(logh, &mut o2);
    let mut n1 = norm_ntt(&P1, logh, &e1, &o1);
    let mut n2 = norm_ntt(&P2, logh, &e2, &o2);
    P1.intt(logh, &mut n1);
    P2.intt(logh, &mut n2);
    let norm: Vec<u32> = n1
        .iter()
        .zip(n2.iter())
        .map(|(&u1, &u2)| mq.crt(u1, u2))
        .collect();
    n1.as_mut_slice().zeroize();
    n2.as_mut_slice().zeroize();
    (
        norm,
        Level {
            logh,
            e1,
            o1,
            e2,
            o2,
        },
    )
}

/// $`-x \bmod q`$ for $`x\in[0,q)`$.
#[inline(always)]
fn neg_mod(x: u32, q: u32) -> u32 {
    mp_sub(0, x, q)
}

/// $`b^e \bmod q`$ by square-and-multiply over the public exponent `e`.
fn pow_mod(mq: &ModQ, b: u32, e: u32) -> u32 {
    let mut r = 1u32;
    let mut b = b;
    let mut e = e;
    while e > 0 {
        let rb = mq.reduce(r as u64 * b as u64);
        // Select without branching on secret data (the exponent bits are public).
        let m = (e & 1).wrapping_neg();
        r = (rb & m) | (r & !m);
        b = mq.reduce(b as u64 * b as u64);
        e >>= 1;
    }
    r
}

/// The inverse of the scalar $`x\in[0,q)`$ modulo $`q\ge2`$, or `None` if $`\gcd(x,q)\ne1`$:
/// $`x^{q-2}`$ (a fixed sequence of operations) when it is the inverse, which is always the case
/// for a prime $`q`$ and $`x\ne0`$; otherwise the extended Euclidean algorithm (variable time,
/// reached for a prime $`q`$ only when $`x=0`$).
fn scalar_inverse(mq: &ModQ, x: u32, q: u32) -> Option<u32> {
    let f = pow_mod(mq, x, q - 2);
    if mq.reduce(x as u64 * f as u64) == 1 {
        return Some(f);
    }
    // Extended Euclid on (q, x): r_i = s_i q + t_i x.
    let (mut r0, mut r1) = (q as i64, x as i64);
    let (mut t0, mut t1) = (0i64, 1i64);
    while r1 != 0 {
        let k = r0 / r1;
        (r0, r1) = (r1, r0 - k * r1);
        (t0, t1) = (t1, t0 - k * t1);
    }
    (r0 == 1).then(|| t0.rem_euclid(q as i64) as u32)
}

/// The scalar norm $`N_{R/\mathbb Z_q}(a)`$ of $`a`$ (size $`2^{\mathrm{logn}}`$, coefficients in
/// $`[0,q)`$), and the per-level data when `keep` is set.
fn descend_all(logn: u32, q: u32, a: &[u32], keep: bool) -> (u32, [u32; 2], Vec<Level>) {
    let mq = ModQ::new(q);
    let mut levels = Vec::new();
    let mut cur = a[..1usize << logn].to_vec();
    let mut lg = logn;
    while lg > 1 {
        let (norm, level) = descend(&mq, lg - 1, &cur);
        cur.as_mut_slice().zeroize();
        cur = norm;
        if keep {
            levels.push(level);
        }
        lg -= 1;
    }
    // Size 2: a0 + a1 x modulo x^2 + 1; the norm is a0^2 + a1^2.
    let (a0, a1) = (cur[0], cur[1]);
    cur.as_mut_slice().zeroize();
    let nrm = mq.reduce(a0 as u64 * a0 as u64 + a1 as u64 * a1 as u64);
    (nrm, [a0, a1], levels)
}

/// `true` iff $`a`$ (size $`2^{\mathrm{logn}}`$, $`\mathrm{logn}\ge1`$, coefficients in $`[0,q)`$,
/// $`3\le q<2^{25}`$) is a unit of $`R_q`$: iff its norm is a unit of $`\mathbb Z_q`$.
pub(crate) fn is_invertible(logn: u32, q: u32, a: &[u32]) -> bool {
    let nrm = descend_all(logn, q, a, false).0;
    scalar_inverse(&ModQ::new(q), nrm, q).is_some()
}

/// The inverse of $`a`$ in $`R_q`$ (size $`2^{\mathrm{logn}}`$, $`\mathrm{logn}\ge1`$, coefficients in
/// $`[0,q)`$, $`3\le q<2^{25}`$), or `None` if $`a`$ is not a unit.
pub(crate) fn invert(logn: u32, q: u32, a: &[u32]) -> Option<Vec<u32>> {
    let mq = ModQ::new(q);
    let (nrm, [a0, a1], levels) = descend_all(logn, q, a, true);
    let ninv = scalar_inverse(&mq, nrm, q)?;
    // Size 2: (a0 + a1 x)^-1 = (a0 - a1 x) / (a0^2 + a1^2).
    let mut inv = vec![
        mq.reduce(a0 as u64 * ninv as u64),
        neg_mod(mq.reduce(a1 as u64 * ninv as u64), q),
    ];
    for lvl in levels.iter().rev() {
        let logh = lvl.logh;
        let h = 1usize << logh;
        debug_assert_eq!(inv.len(), h);
        let mut i1 = inv.clone();
        let mut i2 = inv;
        P1.ntt(logh, &mut i1);
        P2.ntt(logh, &mut i2);
        let mut pe1 = lvl.e1.clone();
        let mut po1 = lvl.o1.clone();
        let mut pe2 = lvl.e2.clone();
        let mut po2 = lvl.o2.clone();
        P1.pointwise_mul(&mut pe1, &i1);
        P1.pointwise_mul(&mut po1, &i1);
        P2.pointwise_mul(&mut pe2, &i2);
        P2.pointwise_mul(&mut po2, &i2);
        P1.intt(logh, &mut pe1);
        P1.intt(logh, &mut po1);
        P2.intt(logh, &mut pe2);
        P2.intt(logh, &mut po2);
        let mut out = vec![0u32; 2 * h];
        for i in 0..h {
            out[2 * i] = mq.crt(pe1[i], pe2[i]);
            out[2 * i + 1] = neg_mod(mq.crt(po1[i], po2[i]), q);
        }
        for v in [&mut i1, &mut i2, &mut pe1, &mut po1, &mut pe2, &mut po2] {
            v.as_mut_slice().zeroize();
        }
        inv = out;
    }
    Some(inv)
}
