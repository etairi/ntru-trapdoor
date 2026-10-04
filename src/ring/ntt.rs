//! Exact negacyclic convolution over $`\mathbb Z`$ through two 31-bit NTT primes and the CRT.
//!
//! Port of fn-dsa-kgen (Thomas Pornin, The Unlicense), commit 0629bb1:
//! - the arithmetic modulo a 31-bit prime: `tbmask`, `mp_set_u`, `mp_R`, `mp_hR`, `mp_add`,
//!   `mp_sub`, `mp_half`, `mp_mmul` (fn-dsa-kgen/src/mp31.rs:45-132);
//! - the root tables `mp_mkgmigm` (fn-dsa-kgen/src/poly.rs:22-46), evaluated at compile time;
//! - `mp_NTT` and `mp_iNTT` (fn-dsa-kgen/src/poly.rs:83-138);
//! - the first two primes of `PRIMES` with their constants (fn-dsa-kgen/src/mp31.rs:343-347).
//!
//! New here: the CRT reconstruction and the reduction modulo $`q`$; the prepared products
//! ([`prepare`], [`mul_prepared_mod_q`]); and the exact check of the NTRU equation
//! ([`ntru_equation_holds`]).
//!
//! **Tables.** One forward table `gm` and one inverse table `igm` of 1024 entries per prime
//! serve every size $`2^k\le1024`$: fn-dsa's `mp_mkgmigm(logn, ...)` squares the 2048-th root
//! $`10-\mathrm{logn}`$ times and writes the entries at indices `rev10(i << (10 - logn))`, which
//! are exactly the first $`2^{\mathrm{logn}}`$ entries of the logn = 10 table [derived; checked by
//! `tests::tables_are_prefix_closed`].
//!
//! **Evaluation points.** The NTT of size $`N`$ leaves output $`2i`$ at $`a(\zeta_i)`$ and output
//! $`2i+1`$ at $`a(-\zeta_i)`$, with $`\zeta_i`$ = `gm[N/2 + i]` (Montgomery form): the last
//! butterfly layer splits $`y^2-\zeta_i^2`$ into $`(y-\zeta_i)(y+\zeta_i)`$ [derived; checked by
//! `tests::evaluation_points`]. The norm tower uses this to multiply by $`y`$ pointwise.
//!
//! **Bounds.** Inputs in $`[0,q)`$ with $`q<2^{25}`$ and size $`N\le1024`$ give integer results
//! with $`|c|\le2N(q-1)^2<P/2`$, $`P=p_1p_2>2^{61.99}`$ (the largest admissible $`q`$ at
//! $`N=1024`$ is 47,451,978 [computed]), so the centred CRT value is the exact integer.
//!
//! All arithmetic is branch-free (masks), as in fn-dsa. Temporary residues are wiped.

use zeroize::Zeroize;

/// One NTT prime with its Montgomery constants (fn-dsa-kgen `SmallPrime`, mp31.rs:276-287).
pub(crate) struct NttPrime {
    /// The modulus, $`1.34\cdot2^{30}<p<2^{31}`$, $`p\equiv1\pmod{2048}`$.
    pub(crate) p: u32,
    /// $`-1/p \bmod 2^{32}`$.
    pub(crate) p0i: u32,
    /// $`2^{64}\bmod p`$.
    pub(crate) r2: u32,
    /// Forward roots: `gm[rev10(i)]` = $`2^{32}g^i \bmod p`$ (Montgomery form).
    pub(crate) gm: [u32; 1024],
    /// Inverse roots: `igm[rev10(i)]` = $`2^{31}g^{-i}\bmod p`$ (Montgomery form of
    /// $`g^{-i}/2`$; the inverse NTT halves at every layer).
    pub(crate) igm: [u32; 1024],
}

/// fn-dsa-kgen `PRIMES[0]` (mp31.rs:344-345): p, p0i, R2, g, ig.
pub(crate) static P1: NttPrime =
    NttPrime::new(2147473409, 2042615807, 419348484, 1790111537, 786166065);
/// fn-dsa-kgen `PRIMES[1]` (mp31.rs:346-347): p, p0i, R2, g, ig.
pub(crate) static P2: NttPrime =
    NttPrime::new(2147389441, 1862176767, 1141604340, 677655126, 2024968256);
/// $`p_1^{-1}\bmod p_2`$ in Montgomery form modulo $`p_2`$: fn-dsa-kgen `PRIMES[1].s`
/// (mp31.rs:347; $`1885487106\cdot2^{32}\bmod p_2`$ [computed]).
const P1_INV_MOD_P2_MONT: u32 = 942807490;

impl NttPrime {
    const fn new(p: u32, p0i: u32, r2: u32, g: u32, ig: u32) -> NttPrime {
        // fn-dsa-kgen poly.rs:22-46 (mp_mkgmigm) at logn = 10.
        let mut gm = [0u32; 1024];
        let mut igm = [0u32; 1024];
        let mut x1 = mp_r(p);
        let mut x2 = mp_hr(p);
        let mut i = 0;
        while i < 1024 {
            let v = rev10(i);
            gm[v] = x1;
            igm[v] = x2;
            x1 = mp_mmul(x1, g, p, p0i);
            x2 = mp_mmul(x2, ig, p, p0i);
            i += 1;
        }
        NttPrime {
            p,
            p0i,
            r2,
            gm,
            igm,
        }
    }

    /// Montgomery multiplication modulo this prime.
    #[inline(always)]
    pub(crate) fn mmul(&self, a: u32, b: u32) -> u32 {
        mp_mmul(a, b, self.p, self.p0i)
    }

    /// The plain product $`ab \bmod p`$ of two values in unsigned representation.
    #[inline(always)]
    pub(crate) fn mul(&self, a: u32, b: u32) -> u32 {
        self.mmul(self.mmul(a, b), self.r2)
    }

    /// In-place NTT of size $`2^{\mathrm{logn}}`$ (fn-dsa-kgen `mp_NTT`, poly.rs:83-105).
    pub(crate) fn ntt(&self, logn: u32, a: &mut [u32]) {
        if logn == 0 {
            return;
        }
        let (p, p0i) = (self.p, self.p0i);
        let n = 1usize << logn;
        let a = &mut a[..n];
        let mut t = n;
        for lm in 0..logn {
            let m = 1usize << lm;
            let ht = t >> 1;
            for i in 0..m {
                let s = self.gm[i + m];
                let j0 = i * t;
                let (x, y) = a[j0..j0 + t].split_at_mut(ht);
                for (x1, x2) in x.iter_mut().zip(y.iter_mut()) {
                    let u = *x1;
                    let v = mp_mmul(*x2, s, p, p0i);
                    *x1 = mp_add(u, v, p);
                    *x2 = mp_sub(u, v, p);
                }
            }
            t = ht;
        }
    }

    /// In-place inverse NTT of size $`2^{\mathrm{logn}}`$, including the division by $`2^{\mathrm{logn}}`$
    /// (fn-dsa-kgen `mp_iNTT`, poly.rs:113-138).
    pub(crate) fn intt(&self, logn: u32, a: &mut [u32]) {
        if logn == 0 {
            return;
        }
        let (p, p0i) = (self.p, self.p0i);
        let n = 1usize << logn;
        let a = &mut a[..n];
        let mut t = 1usize;
        for lm in 0..logn {
            let hm = 1usize << (logn - 1 - lm);
            let dt = t << 1;
            for i in 0..hm {
                let s = self.igm[i + hm];
                let j0 = i * dt;
                let (x, y) = a[j0..j0 + dt].split_at_mut(t);
                for (x1, x2) in x.iter_mut().zip(y.iter_mut()) {
                    let u = *x1;
                    let v = *x2;
                    *x1 = mp_half(mp_add(u, v, p), p);
                    *x2 = mp_mmul(mp_sub(u, v, p), s, p, p0i);
                }
            }
            t = dt;
        }
    }

    /// Pointwise product `a <- a * b` of two NTT vectors (plain values, not Montgomery).
    pub(crate) fn pointwise_mul(&self, a: &mut [u32], b: &[u32]) {
        for (x, &y) in a.iter_mut().zip(b.iter()) {
            *x = self.mul(*x, y);
        }
    }
}

/// Bit reversal over 10 bits (fn-dsa-kgen's `REV10` table, mp31.rs:252-341).
const fn rev10(i: usize) -> usize {
    ((i as u32).reverse_bits() >> 22) as usize
}

/// `0xFFFFFFFF` if the top bit of `x` is set, else 0 (mp31.rs:45-49).
#[inline(always)]
const fn tbmask(x: u32) -> u32 {
    ((x as i32) >> 31) as u32
}

/// $`v \bmod p`$ for $`v\in[0,2p)`$ (mp31.rs:57-62).
#[inline(always)]
pub(crate) const fn mp_set_u(v: u32, p: u32) -> u32 {
    let w = v.wrapping_sub(p);
    w.wrapping_add(p & tbmask(w))
}

/// $`2^{32}\bmod p`$, the Montgomery form of 1 (mp31.rs:72-80).
const fn mp_r(p: u32) -> u32 {
    p.wrapping_neg() << 1
}

/// $`2^{31}\bmod p`$, the Montgomery form of 1/2 (mp31.rs:82-90).
const fn mp_hr(p: u32) -> u32 {
    0x8000_0000 - p
}

/// $`a+b \bmod p`$ (mp31.rs:92-100).
#[inline(always)]
pub(crate) const fn mp_add(a: u32, b: u32, p: u32) -> u32 {
    let d = a.wrapping_add(b).wrapping_sub(p);
    d.wrapping_add(p & tbmask(d))
}

/// $`a-b \bmod p`$ (mp31.rs:102-110).
#[inline(always)]
pub(crate) const fn mp_sub(a: u32, b: u32, p: u32) -> u32 {
    let d = a.wrapping_sub(b);
    d.wrapping_add(p & tbmask(d))
}

/// $`a/2 \bmod p`$ (mp31.rs:112-119).
#[inline(always)]
const fn mp_half(a: u32, p: u32) -> u32 {
    a.wrapping_add(p & (a & 1).wrapping_neg()) >> 1
}

/// Montgomery multiplication $`ab/2^{32}\bmod p`$ (mp31.rs:121-132).
#[inline(always)]
const fn mp_mmul(a: u32, b: u32, p: u32, p0i: u32) -> u32 {
    let z = (a as u64) * (b as u64);
    let w = (z as u32).wrapping_mul(p0i);
    let d = (((z + (w as u64) * (p as u64)) >> 32) as u32).wrapping_sub(p);
    d.wrapping_add(p & tbmask(d))
}

/// Reduction modulo a runtime $`q<2^{25}`$ of values below $`2^{63}`$, by a Barrett quotient
/// estimate and one masked correction.
#[derive(Clone, Copy)]
pub(crate) struct ModQ {
    q: u64,
    /// $`\lfloor(2^{64}-1)/q\rfloor`$.
    m: u64,
    /// $`P \bmod q`$, for the centred CRT lift.
    p_mod_q: u64,
}

/// $`P=p_1p_2`$.
const P12: u64 = 2147473409u64 * 2147389441u64;

impl ModQ {
    pub(crate) fn new(q: u32) -> ModQ {
        debug_assert!(q >= 2);
        let q = q as u64;
        ModQ {
            q,
            m: u64::MAX / q,
            p_mod_q: P12 % q,
        }
    }

    /// $`x \bmod q`$ for a signed `x`, in $`[0,q)`$, without a division.
    #[inline(always)]
    pub(crate) fn reduce_i32(&self, x: i32) -> u32 {
        let r = self.reduce(x.unsigned_abs() as u64);
        // For x < 0: (q - r) mod q, selected by a mask.
        let neg = (x >> 31) as u32;
        let nr = mp_sub(0, r, self.q as u32);
        (r & !neg) | (nr & neg)
    }

    /// $`x \bmod q`$ for $`x<2^{63}`$.
    #[inline(always)]
    pub(crate) fn reduce(&self, x: u64) -> u32 {
        // qhat in {floor(x/q) - 1, floor(x/q)}, so r in [0, 2q).
        let qhat = ((x as u128 * self.m as u128) >> 64) as u64;
        let r = x - qhat * self.q;
        let r2 = r.wrapping_sub(self.q);
        // r2 < 0 (top bit set) iff r < q.
        let keep = ((r2 as i64) >> 63) as u64;
        ((r & keep) | (r2 & !keep)) as u32
    }

    /// The centred CRT lift of $`(u_1 \bmod p_1, u_2\bmod p_2)`$, reduced modulo $`q`$: the integer
    /// $`c\in(-P/2,P/2)`$ with $`c\equiv u_i`$, then $`c\bmod q`$.
    #[inline(always)]
    pub(crate) fn crt(&self, u1: u32, u2: u32) -> u32 {
        // t = (u2 - u1) / p1 mod p2; c' = u1 + p1 t in [0, P).
        let (p1, p2) = (P1.p, P2.p);
        let d = mp_sub(u2, mp_set_u(u1, p2), p2);
        let t = mp_mmul(d, P1_INV_MOD_P2_MONT, p2, P2.p0i);
        let c = u1 as u64 + (p1 as u64) * (t as u64);
        let r = self.reduce(c) as u64;
        // If c' > P/2 the integer is c' - P: subtract P mod q.
        let neg = ((P12 >> 1).wrapping_sub(c) as i64 >> 63) as u64;
        let r = r + self.q - (self.p_mod_q & neg);
        let r2 = r.wrapping_sub(self.q);
        let keep = ((r2 as i64) >> 63) as u64;
        ((r & keep) | (r2 & !keep)) as u32
    }
}

/// The residues of `a` (coefficients in $`[0,q)`$, $`q<p_1,p_2`$) modulo $`p_1`$ and $`p_2`$, in NTT
/// form: the prepared operand of [`mul_prepared_mod_q`].
pub(crate) fn prepare(logn: u32, a: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let n = 1usize << logn;
    let mut a1 = a[..n].to_vec();
    let mut a2 = a[..n].to_vec();
    P1.ntt(logn, &mut a1);
    P2.ntt(logn, &mut a2);
    (a1, a2)
}

/// `out <- a * b mod (x^n + 1, q)` for `a` given by its prepared residues `(a1, a2)` and `b` with
/// coefficients in $`[0,q)`$, $`n=2^{\mathrm{logn}}`$. Exact: the negacyclic product is computed
/// over $`\mathbb Z`$ modulo $`p_1`$ and $`p_2`$, lifted by the CRT and reduced modulo $`q`$. Four
/// NTTs: two forward (of `b`), two inverse.
pub(crate) fn mul_prepared_mod_q(
    logn: u32,
    q: u32,
    a1: &[u32],
    a2: &[u32],
    b: &[u32],
    out: &mut [u32],
) {
    let n = 1usize << logn;
    let mq = ModQ::new(q);
    let mut b1 = b[..n].to_vec();
    let mut b2 = b[..n].to_vec();
    P1.ntt(logn, &mut b1);
    P2.ntt(logn, &mut b2);
    P1.pointwise_mul(&mut b1, &a1[..n]);
    P2.pointwise_mul(&mut b2, &a2[..n]);
    P1.intt(logn, &mut b1);
    P2.intt(logn, &mut b2);
    for ((o, &u1), &u2) in out[..n].iter_mut().zip(b1.iter()).zip(b2.iter()) {
        *o = mq.crt(u1, u2);
    }
    // `b` may be secret (f in key generation, the PCS's witnesses): wipe its residues (slice
    // wipes: `Vec::zeroize` would also wipe the capacity byte by byte, about ten times the cost
    // [measured, sign/README.md]).
    for v in [&mut b1, &mut b2] {
        v.as_mut_slice().zeroize();
    }
}

/// `out <- a * b mod (x^n + 1, q)` for `a`, `b` with coefficients in $`[0,q)`$, $`n=2^{\mathrm{logn}}`$:
/// [`prepare`] then [`mul_prepared_mod_q`] (six NTTs). Every residue is wiped.
pub(crate) fn mul_mod_q(logn: u32, q: u32, a: &[u32], b: &[u32], out: &mut [u32]) {
    let (mut a1, mut a2) = prepare(logn, a);
    mul_prepared_mod_q(logn, q, &a1, &a2, b, out);
    for v in [&mut a1, &mut a2] {
        v.as_mut_slice().zeroize();
    }
}

/// `true` iff $`fG-gF=q`$ in $`\mathbb Z[x]/(x^n+1)`$, exactly, for $`n=2^{\mathrm{logn}}`$ and
/// coefficients of absolute value at most $`2^{15}`$ (any `i16`), $`q<2^{25}`$.
///
/// Each coefficient of $`fG-gF`$ is a sum of $`2n`$ products, so its absolute value is at most
/// $`2n\cdot2^{30}\le2^{41}<P/2`$ ($`P=p_1p_2>2^{61.99}`$): it is determined by its residues
/// modulo $`p_1`$ and $`p_2`$. Modulo each prime the NTT is a ring isomorphism onto
/// $`\mathbb Z_p^n`$ ($`2n\mid p-1`$), and the NTT of the constant $`q`$ is $`q`$ in every slot. So
/// the equation holds iff $`\hat f\hat G-\hat g\hat F=q`$ in every slot modulo both primes: eight
/// forward NTTs, no inverse NTT and no CRT. The residues (the trapdoor) are wiped; the comparison
/// accumulates a mask, so the time does not depend on where a mismatch occurs.
pub(crate) fn ntru_equation_holds(
    logn: u32,
    q: u32,
    f: &[i16],
    g: &[i16],
    big_f: &[i16],
    big_g: &[i16],
) -> bool {
    let n = 1usize << logn;
    assert!(
        f.len() == n && g.len() == n && big_f.len() == n && big_g.len() == n,
        "ntru_equation_holds: lengths"
    );
    let mut buf = vec![0u32; 4 * n];
    let mut diff = 0u32;
    for pr in [&P1, &P2] {
        let p = pr.p;
        for (k, src) in [f, g, big_f, big_g].into_iter().enumerate() {
            let dst = &mut buf[k * n..(k + 1) * n];
            for (d, &x) in dst.iter_mut().zip(src.iter()) {
                // x mod p for |x| <= 2^15: x, or p + x for x < 0.
                let u = x as i32 as u32;
                *d = u.wrapping_add(p & tbmask(u));
            }
            pr.ntt(logn, dst);
        }
        let (rf, rest) = buf.split_at(n);
        let (rg, rest) = rest.split_at(n);
        let (rbf, rbg) = rest.split_at(n);
        for i in 0..n {
            let lhs = mp_sub(pr.mul(rf[i], rbg[i]), pr.mul(rg[i], rbf[i]), p);
            diff |= lhs ^ q;
        }
    }
    buf.as_mut_slice().zeroize();
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pow_mod(mut b: u64, mut e: u64, m: u64) -> u64 {
        let mut r = 1u64;
        b %= m;
        while e > 0 {
            if e & 1 == 1 {
                r = ((r as u128 * b as u128) % m as u128) as u64;
            }
            b = ((b as u128 * b as u128) % m as u128) as u64;
            e >>= 1;
        }
        r
    }

    /// The constants copied from fn-dsa-kgen: p prime, p = 1 mod 2048, p0i, R2, g of order 2048,
    /// and the CRT constant.
    #[test]
    fn prime_constants() {
        for pr in [&P1, &P2] {
            let p = pr.p as u64;
            assert_eq!(p % 2048, 1);
            // Trial division up to sqrt(p) < 46341.
            assert!((2..46341u64).all(|d| !p.is_multiple_of(d)));
            assert_eq!((pr.p.wrapping_mul(pr.p0i)), u32::MAX); // p * p0i = -1 mod 2^32
            assert_eq!(pr.r2 as u64, ((1u128 << 64) % p as u128) as u64);
            // gm[rev10(1)] = R g: g^1024 = -1, g^2048 = 1.
            let g = pr.mmul(pr.gm[rev10(1)], 1) as u64;
            assert_eq!(pow_mod(g, 1024, p), p - 1);
            // igm[rev10(1)] = R g^-1 / 2.
            let ig2 = pr.mmul(pr.igm[rev10(1)], 1) as u64;
            assert_eq!((g as u128 * ig2 as u128 * 2 % p as u128) as u64, 1);
        }
        let inv = P2.mmul(P1_INV_MOD_P2_MONT, 1) as u64;
        assert_eq!((inv as u128 * P1.p as u128 % P2.p as u128) as u64, 1);
        assert_eq!(inv, 1885487106);
    }

    /// Tables for smaller sizes are prefixes of the logn = 10 table (fn-dsa's mp_mkgmigm).
    #[test]
    fn tables_are_prefix_closed() {
        for pr in [&P1, &P2] {
            for logn in 0..=10u32 {
                // mp_mkgmigm(logn): square g (10 - logn) times, entries at rev10(i << k).
                let k = 10 - logn;
                let mut g = pr.gm[rev10(1)];
                let mut ig = pr.igm[rev10(1)];
                // Recover the Montgomery roots g, ig from the table (gm[rev10(1)] = R g in
                // Montgomery form means mmul(gm, R2)... use plain values instead).
                g = pr.mmul(g, 1); // plain g
                ig = pr.mmul(ig, 1); // plain g^-1/2
                let ig = pr.mul(ig, 2); // plain g^-1
                let (mut gg, mut igg) = (g, ig);
                for _ in 0..k {
                    gg = pr.mul(gg, gg);
                    igg = pr.mul(igg, igg);
                }
                let mut x1 = mp_r(pr.p);
                let mut x2 = mp_hr(pr.p);
                for i in 0..(1usize << logn) {
                    let v = rev10(i << k);
                    assert!(v < (1 << logn));
                    assert_eq!(pr.gm[v], x1);
                    assert_eq!(pr.igm[v], x2);
                    x1 = pr.mul(x1, gg);
                    x2 = pr.mul(x2, igg);
                }
            }
        }
    }

    /// NTT output 2i is a(zeta_i), output 2i+1 is a(-zeta_i), zeta_i = gm[N/2 + i].
    #[test]
    fn evaluation_points() {
        for pr in [&P1, &P2] {
            for logn in 1..=10u32 {
                let n = 1usize << logn;
                let mut a = vec![0u32; n];
                a[1] = 1; // the polynomial y
                pr.ntt(logn, &mut a);
                for i in 0..n / 2 {
                    let z = pr.mmul(pr.gm[n / 2 + i], 1);
                    assert_eq!(a[2 * i], z);
                    assert_eq!(a[2 * i + 1], mp_sub(0, z, pr.p));
                    // zeta^n = -1.
                    assert_eq!(pow_mod(z as u64, n as u64, pr.p as u64), pr.p as u64 - 1);
                }
            }
        }
    }

    /// `reduce_i32` against `rem_euclid`, at the extremes too.
    #[test]
    fn reduce_signed() {
        for q in [3u32, 17, 12289, 1048573, (1 << 25) - 39] {
            let mq = ModQ::new(q);
            let mut x: u64 = 0x0123_4567_89ab_cdef;
            let mut vals = vec![
                0i32,
                1,
                -1,
                i32::MAX,
                i32::MIN,
                i32::MIN + 1,
                q as i32,
                -(q as i32),
            ];
            for _ in 0..10000 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                vals.push(x as i32);
                vals.push((x as i32) >> 12);
            }
            for &v in &vals {
                assert_eq!(
                    mq.reduce_i32(v) as i64,
                    (v as i64).rem_euclid(q as i64),
                    "q {q} v {v}"
                );
            }
        }
    }

    #[test]
    fn crt_and_reduction() {
        for q in [3u32, 12289, 1048573, (1 << 25) - 39] {
            let mq = ModQ::new(q);
            let mut x: u64 = 0x1234_5678_9abc_def0;
            for _ in 0..10000 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let v = x >> 1;
                assert_eq!(mq.reduce(v) as u64, v % q as u64);
                // A signed integer |c| < 2^61, through the CRT.
                let c = (x as i64) >> 3;
                let u1 = c.rem_euclid(P1.p as i64) as u32;
                let u2 = c.rem_euclid(P2.p as i64) as u32;
                assert_eq!(mq.crt(u1, u2) as i64, c.rem_euclid(q as i64));
            }
        }
    }
}
