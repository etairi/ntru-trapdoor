//! Exact negacyclic convolution over $`\mathbb Z`$ through two 31-bit NTT primes and the CRT.
//!
//! Port of fn-dsa-kgen (Thomas Pornin, The Unlicense), commit 0629bb1:
//! - the arithmetic modulo a 31-bit prime: `tbmask`, `mp_set_u`, `mp_R`, `mp_hR`, `mp_add`,
//!   `mp_sub`, `mp_half`, `mp_mmul` (fn-dsa-kgen/src/mp31.rs:45-132);
//! - the root tables `mp_mkgmigm` (fn-dsa-kgen/src/poly.rs:22-46), evaluated at compile time;
//! - `mp_NTT` and `mp_iNTT` (fn-dsa-kgen/src/poly.rs:83-138);
//! - the first two primes of `PRIMES` with their constants (fn-dsa-kgen/src/mp31.rs:343-347).
//!
//! Two changes to the transforms, both with the same residues as fn-dsa's code (checked by
//! `tests::transforms_equal_fndsa` against verbatim copies of `mp_NTT` and `mp_iNTT`):
//! - The loops run over fixed-size blocks, and two layers (the last two of the NTT, the first
//!   two of the inverse) run together on blocks of four values.
//! - The inverse NTT [`NttPrime::intt_n`] omits fn-dsa's halving at every layer, so it
//!   multiplies by $`n`$. Every caller folds the factor $`1/n`$ into a pointwise product that it
//!   computes anyway ([`NttPrime::ninv_r2`]), and prepared operands are stored times
//!   $`2^{32}/n`$, so that a prepared product needs one Montgomery multiplication per point
//!   instead of two.
//!
//! New here: the CRT reconstruction and the reduction modulo $`q`$; the prepared products
//! ([`prepare`], [`mul_prepared_mod_q`]), their sums ([`sum_of_prepared_products_mod_q`]) and the
//! single-prime product with a ternary operand ([`mul_ternary_p1_mod_q`]); and the exact check
//! of the NTRU equation ([`ntru_equation_holds`]).
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
    /// Inverse roots: `igm[rev10(i)]` = $`2^{32}g^{-i}\bmod p`$ (Montgomery form of $`g^{-i}`$).
    /// fn-dsa's table holds $`g^{-i}/2`$ and its inverse NTT halves at every layer. Here the
    /// inverse NTT ([`NttPrime::intt_n`]) does not halve, and the factor $`1/n`$ is folded into
    /// a pointwise product that is computed anyway ([`NttPrime::ninv_r2`]).
    pub(crate) igm: [u32; 1024],
    /// `ninv_r2[logn]` = $`2^{64}/2^{\mathrm{logn}} \bmod p`$: `mmul(mmul(a, b), ninv_r2[logn])` is
    /// $`ab/n`$ and `mmul(a, ninv_r2[logn])` is $`2^{32}a/n`$ (the prepared form).
    pub(crate) ninv_r2: [u32; 11],
}

/// fn-dsa-kgen `PRIMES[0]` (mp31.rs:344-345): p, p0i, R2, g, ig.
pub(crate) static P1: NttPrime =
    NttPrime::new(2147473409, 2042615807, 419348484, 1790111537, 786166065);
/// fn-dsa-kgen `PRIMES[1]` (mp31.rs:346-347): p, p0i, R2, g, ig.
pub(crate) static P2: NttPrime =
    NttPrime::new(2147389441, 1862176767, 1141604340, 677655126, 2024968256);
/// $`p_1^{-1}\bmod p_2`$ in Montgomery form modulo $`p_2`$: fn-dsa-kgen `PRIMES[1].s`
/// (mp31.rs:347; $`1885487106\cdot2^{32}\bmod p_2`$ [computed]).
pub(crate) const P1_INV_MOD_P2_MONT: u32 = 942807490;

impl NttPrime {
    const fn new(p: u32, p0i: u32, r2: u32, g: u32, ig: u32) -> NttPrime {
        // fn-dsa-kgen poly.rs:22-46 (mp_mkgmigm) at logn = 10.
        let mut gm = [0u32; 1024];
        let mut igm = [0u32; 1024];
        let mut x1 = mp_r(p);
        // fn-dsa starts the inverse table at mp_hR (1/2); this crate at mp_R (1), see `igm`.
        let mut x2 = mp_r(p);
        let mut i = 0;
        while i < 1024 {
            let v = rev10(i);
            gm[v] = x1;
            igm[v] = x2;
            x1 = mp_mmul(x1, g, p, p0i);
            x2 = mp_mmul(x2, ig, p, p0i);
            i += 1;
        }
        // 1/n = p - (p - 1)/n modulo p, since n divides p - 1.
        let mut ninv_r2 = [0u32; 11];
        let mut logn = 0;
        while logn <= 10 {
            let ninv = (p - ((p - 1) >> logn)) as u64;
            ninv_r2[logn] = ((r2 as u64 * ninv) % p as u64) as u32;
            logn += 1;
        }
        NttPrime {
            p,
            p0i,
            r2,
            gm,
            igm,
            ninv_r2,
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

    /// In-place NTT of size $`2^{\mathrm{logn}}`$ (fn-dsa-kgen `mp_NTT`, poly.rs:83-105): the same
    /// butterflies with the same roots, so the same output. The layers run over `chunks_exact`
    /// blocks, and the last two layers (half-blocks of 2, then 1) run together on blocks of 4
    /// values, which removes the per-block index arithmetic of the inner loops (15 % faster at
    /// $`n=1024`$ on an Apple M4, 2026-11-18 study).
    pub(crate) fn ntt(&self, logn: u32, a: &mut [u32]) {
        if logn == 0 {
            return;
        }
        let (p, p0i) = (self.p, self.p0i);
        let n = 1usize << logn;
        let a = &mut a[..n];
        if logn == 1 {
            let v = mp_mmul(a[1], self.gm[1], p, p0i);
            let u = a[0];
            a[0] = mp_add(u, v, p);
            a[1] = mp_sub(u, v, p);
            return;
        }
        // Layer lm has m = 2^lm blocks of t = n/m values, block i with the root gm[m + i]. With
        // the `simd` feature, AVX-512 and AVX2 run the layers whose half-blocks have 8 values
        // or more, NEON those with 4 or more.
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
        let (mut t, mut m) = crate::simd::ntt_wide_layers(self, logn, a);
        #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
        let (mut t, mut m) = (n, 1usize);
        // The layers below them, with the lanes regrouped in registers (from t = 8 with AVX2,
        // t = 4 with NEON, for n >= 16).
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
        if crate::simd::ntt_narrow_layers(self, logn, a, t) {
            return;
        }
        while t > 4 {
            let ht = t >> 1;
            for (blk, &s) in a.chunks_exact_mut(t).zip(self.gm[m..2 * m].iter()) {
                let (x, y) = blk.split_at_mut(ht);
                for (x1, x2) in x.iter_mut().zip(y.iter_mut()) {
                    let u = *x1;
                    let v = mp_mmul(*x2, s, p, p0i);
                    *x1 = mp_add(u, v, p);
                    *x2 = mp_sub(u, v, p);
                }
            }
            t = ht;
            m <<= 1;
        }
        // The last two layers: block i of 4 values (m = n/4) uses gm[m + i] for the pairs
        // (0, 2) and (1, 3), then gm[2m + 2i] for (0, 1) and gm[2m + 2i + 1] for (2, 3).
        let (blocks, _) = a.as_chunks_mut::<4>();
        let (g2, _) = self.gm[2 * m..4 * m].as_chunks::<2>();
        for ((b, &s), &[s0, s1]) in blocks.iter_mut().zip(self.gm[m..2 * m].iter()).zip(g2) {
            let v = mp_mmul(b[2], s, p, p0i);
            let w = mp_mmul(b[3], s, p, p0i);
            let (x0, x2) = (mp_add(b[0], v, p), mp_sub(b[0], v, p));
            let (x1, x3) = (mp_add(b[1], w, p), mp_sub(b[1], w, p));
            let v = mp_mmul(x1, s0, p, p0i);
            let w = mp_mmul(x3, s1, p, p0i);
            *b = [
                mp_add(x0, v, p),
                mp_sub(x0, v, p),
                mp_add(x2, w, p),
                mp_sub(x2, w, p),
            ];
        }
    }

    /// In-place inverse NTT of size $`n=2^{\mathrm{logn}}`$, times $`n`$: fn-dsa-kgen's `mp_iNTT`
    /// (poly.rs:113-138) without its halving at every layer. The callers fold the factor $`1/n`$
    /// into the pointwise product before it ([`NttPrime::pointwise_mul_div_n`], [`prepare`],
    /// the norm of `ring::inv`), so the result is the same residue. The first two layers run
    /// together on blocks of 4 values, the others over `chunks_exact` blocks.
    pub(crate) fn intt_n(&self, logn: u32, a: &mut [u32]) {
        if logn == 0 {
            return;
        }
        let (p, p0i) = (self.p, self.p0i);
        let n = 1usize << logn;
        let a = &mut a[..n];
        if logn == 1 {
            let (u, v) = (a[0], a[1]);
            a[0] = mp_add(u, v, p);
            a[1] = mp_mmul(mp_sub(u, v, p), self.igm[1], p, p0i);
            return;
        }
        // With the `simd` feature, the first layers with the lanes regrouped in registers (up
        // to t = 4 with AVX2, t = 2 with NEON, for n >= 16).
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
        let narrow = crate::simd::intt_narrow_layers(self, logn, a);
        #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
        let narrow: Option<(usize, usize)> = None;
        let (mut t, mut hm) = match narrow {
            Some(next) => next,
            None => {
                self.intt_first_two_layers(logn, a);
                (4usize, n >> 3)
            }
        };
        // Layer lm >= 2: hm = n / 2^(lm + 1) blocks of 2t values, block i with igm[hm + i]. With
        // the `simd` feature, AVX2 and AVX-512 run the layers from t = 8 on (NEON from t = 4).
        while t < n {
            #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
            if crate::simd::intt_wide_layers(self, logn, a, t, hm) {
                break;
            }
            let dt = t << 1;
            for (blk, &s) in a.chunks_exact_mut(dt).zip(self.igm[hm..2 * hm].iter()) {
                let (x, y) = blk.split_at_mut(t);
                for (x1, x2) in x.iter_mut().zip(y.iter_mut()) {
                    let (u, v) = (*x1, *x2);
                    *x1 = mp_add(u, v, p);
                    *x2 = mp_mmul(mp_sub(u, v, p), s, p, p0i);
                }
            }
            t = dt;
            hm >>= 1;
        }
    }

    /// The first two layers of [`NttPrime::intt_n`] ($`n\ge4`$), on blocks of 4 values: layer 0
    /// pairs (2j, 2j + 1) with igm[n/2 + j], layer 1 pairs (4i, 4i + 2) and (4i + 1, 4i + 3)
    /// with igm[n/4 + i].
    fn intt_first_two_layers(&self, logn: u32, a: &mut [u32]) {
        let (p, p0i) = (self.p, self.p0i);
        let n = 1usize << logn;
        let hn = n >> 1;
        let (blocks, _) = a[..n].as_chunks_mut::<4>();
        let (g0, _) = self.igm[hn..n].as_chunks::<2>();
        for ((b, &[s0, s1]), &s) in blocks.iter_mut().zip(g0).zip(self.igm[hn >> 1..hn].iter()) {
            let (x0, x1) = (
                mp_add(b[0], b[1], p),
                mp_mmul(mp_sub(b[0], b[1], p), s0, p, p0i),
            );
            let (x2, x3) = (
                mp_add(b[2], b[3], p),
                mp_mmul(mp_sub(b[2], b[3], p), s1, p, p0i),
            );
            *b = [
                mp_add(x0, x2, p),
                mp_add(x1, x3, p),
                mp_mmul(mp_sub(x0, x2, p), s, p, p0i),
                mp_mmul(mp_sub(x1, x3, p), s, p, p0i),
            ];
        }
    }

    /// Pointwise `a <- a * b / n` of two NTT vectors of size $`n=2^{\mathrm{logn}}`$ (plain
    /// values): the input of [`NttPrime::intt_n`] for the product. Two Montgomery products, as
    /// the plain product [`NttPrime::mul`].
    pub(crate) fn pointwise_mul_div_n(&self, logn: u32, a: &mut [u32], b: &[u32]) {
        let c = self.ninv_r2[logn as usize];
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
        if crate::simd::mmul_slices_const(self, a, Some(b), c) {
            return;
        }
        for (x, &y) in a.iter_mut().zip(b.iter()) {
            *x = self.mmul(self.mmul(*x, y), c);
        }
    }

    /// `a <- 2^32 a / n` (the prepared form): a later `mmul(b, a)` is then $`ab/n`$, one
    /// Montgomery product per point instead of two.
    pub(crate) fn scale_prepared(&self, logn: u32, a: &mut [u32]) {
        let c = self.ninv_r2[logn as usize];
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
        if crate::simd::mmul_slices_const(self, a, None, c) {
            return;
        }
        for x in a.iter_mut() {
            *x = self.mmul(*x, c);
        }
    }

    /// Pointwise `a <- a * b / n` for `b` in the prepared form ([`NttPrime::scale_prepared`]).
    pub(crate) fn pointwise_mul_prepared(&self, a: &mut [u32], b: &[u32]) {
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
        if crate::simd::mmul_slices(self, a, b) {
            return;
        }
        for (x, &y) in a.iter_mut().zip(b.iter()) {
            *x = self.mmul(*x, y);
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

/// $`2^{31}\bmod p`$, the Montgomery form of 1/2 (mp31.rs:82-90): fn-dsa's first inverse root,
/// for the tests' copy of fn-dsa's inverse NTT.
#[cfg(test)]
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

/// $`a/2 \bmod p`$ (mp31.rs:112-119), for the tests' copy of fn-dsa's inverse NTT.
#[cfg(test)]
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

    /// $`x \bmod q`$ for a signed `x`, in $`[0,q)`$, without a division. The magnitude is at
    /// most $`2^{63}`$, within the range of [`ModQ::reduce`].
    #[inline(always)]
    pub(crate) fn reduce_i64(&self, x: i64) -> u32 {
        let r = self.reduce(x.unsigned_abs());
        // For x < 0: (q - r) mod q, selected by a mask.
        let neg = (x >> 63) as u32;
        let nr = mp_sub(0, r, self.q as u32);
        (r & !neg) | (nr & neg)
    }

    /// $`x \bmod q`$ for any $`x<2^{64}`$.
    #[inline(always)]
    pub(crate) fn reduce(&self, x: u64) -> u32 {
        // qhat in {floor(x/q) - 1, floor(x/q)}, so r in [0, 2q): with m = floor((2^64 - 1)/q)
        // >= (2^64 - q)/q, x m / 2^64 >= x/q - x/2^64 > x/q - 1, and x m / 2^64 < x/q.
        let qhat = ((x as u128 * self.m as u128) >> 64) as u64;
        let r = x - qhat * self.q;
        let r2 = r.wrapping_sub(self.q);
        // r2 < 0 (top bit set) iff r < q.
        let keep = ((r2 as i64) >> 63) as u64;
        ((r & keep) | (r2 & !keep)) as u32
    }

    /// The modulus $`q`$.
    #[inline(always)]
    pub(crate) fn q(&self) -> u32 {
        self.q as u32
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

/// `out[i] <- mq.crt(u1[i], u2[i])` for slices of equal length.
pub(crate) fn crt_slice(mq: &ModQ, u1: &[u32], u2: &[u32], out: &mut [u32]) {
    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    if crate::simd::crt_slices(mq.q(), u1, u2, out, false) {
        return;
    }
    for ((o, &x), &y) in out.iter_mut().zip(u1.iter()).zip(u2.iter()) {
        *o = mq.crt(x, y);
    }
}

/// `out[i] <- mq.reduce_i32(x[i])` for slices of equal length.
pub(crate) fn reduce_i32_slice(mq: &ModQ, x: &[i32], out: &mut [u32]) {
    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    if crate::simd::reduce_i32_slices(mq.q(), x, out) {
        return;
    }
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = mq.reduce_i32(v);
    }
}

/// `true` iff $`x_i+y_i\equiv z_i\pmod q`$ for every `i`, for signed `x` and canonical `y`, `z`
/// of equal length. The comparison accumulates a mask over all values.
pub(crate) fn add_i32_equals(mq: &ModQ, x: &[i32], y: &[u32], z: &[u32]) -> bool {
    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    if let Some(diff) = crate::simd::add_i32_diff(mq.q(), x, y, z) {
        return diff == 0;
    }
    let q = mq.q();
    let mut diff = 0u32;
    for ((&a, &b), &c) in x.iter().zip(y.iter()).zip(z.iter()) {
        diff |= mp_add(mq.reduce_i32(a), b, q) ^ c;
    }
    diff == 0
}

/// The residues of `a` (coefficients in $`[0,q)`$, $`q<p_1,p_2`$) modulo $`p_1`$ and $`p_2`$, in NTT
/// form and scaled by $`2^{32}/n`$ ([`NttPrime::scale_prepared`]): the prepared operand of
/// [`mul_prepared_mod_q`].
pub(crate) fn prepare(logn: u32, a: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let n = 1usize << logn;
    let mut a1 = a[..n].to_vec();
    let mut a2 = a[..n].to_vec();
    P1.ntt(logn, &mut a1);
    P2.ntt(logn, &mut a2);
    P1.scale_prepared(logn, &mut a1);
    P2.scale_prepared(logn, &mut a2);
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
    P1.pointwise_mul_prepared(&mut b1, &a1[..n]);
    P2.pointwise_mul_prepared(&mut b2, &a2[..n]);
    P1.intt_n(logn, &mut b1);
    P2.intt_n(logn, &mut b2);
    crt_slice(&mq, &b1, &b2, &mut out[..n]);
    // `b` may be secret (f in key generation, the PCS's witnesses): wipe its residues (slice
    // wipes: `Vec::zeroize` would also wipe the capacity byte by byte, about ten times the cost
    // [measured, sign/README.md]).
    for v in [&mut b1, &mut b2] {
        crate::wipe::slice(v);
    }
}

/// `out <- a * b mod (x^n + 1, q)` for `a`, `b` with coefficients in $`[0,q)`$, $`n=2^{\mathrm{logn}}`$:
/// [`prepare`] then [`mul_prepared_mod_q`] (six NTTs). Every residue is wiped.
pub(crate) fn mul_mod_q(logn: u32, q: u32, a: &[u32], b: &[u32], out: &mut [u32]) {
    let (mut a1, mut a2) = prepare(logn, a);
    mul_prepared_mod_q(logn, q, &a1, &a2, b, out);
    for v in [&mut a1, &mut a2] {
        crate::wipe::slice(v);
    }
}

/// The largest number of products $`a_ib_i`$ (coefficients in $`[0,q)`$, degree $`n`$) whose sum
/// the CRT still determines: each product has integer coefficients in
/// $`[-(n-1)(q-1)^2,\,n(q-1)^2]`$, and the centred lift recovers integers of absolute value at
/// most $`(P-1)/2`$. At least 1 for every admissible ring (module documentation, "Bounds"),
/// 2,047 at $`d=1024`$, $`q=1048573`$ and $`1.49\cdot10^{7}`$ at $`q=12289`$.
pub(crate) fn max_sum_terms(logn: u32, q: u32) -> usize {
    let per = (1u128 << logn) * ((q as u128 - 1) * (q as u128 - 1)).max(1);
    let k = ((P12 as u128 - 1) / 2) / per;
    k.clamp(1, usize::MAX as u128) as usize
}

/// `out <- sum_i a_i * b_i mod (x^n + 1, q)` for prepared `a_i` (residues `(a1_i, a2_i)` from
/// [`prepare`]) and `b_i` with coefficients in $`[0,q)`$, exactly: the pointwise products are
/// added modulo $`p_1`$ and $`p_2`$ in the NTT domain, and each block of at most
/// [`max_sum_terms`] terms gets one inverse NTT per prime and one CRT. Equal to the sum modulo
/// $`q`$ of the [`mul_prepared_mod_q`] products. Every residue is wiped.
pub(crate) fn sum_of_prepared_products_mod_q(
    logn: u32,
    q: u32,
    terms: &[(&[u32], &[u32], &[u32])],
    out: &mut [u32],
) {
    let n = 1usize << logn;
    let mq = ModQ::new(q);
    let out = &mut out[..n];
    out.fill(0);
    let mut acc1 = vec![0u32; n];
    let mut acc2 = vec![0u32; n];
    let mut b1 = vec![0u32; n];
    let mut b2 = vec![0u32; n];
    for block in terms.chunks(max_sum_terms(logn, q)) {
        acc1.fill(0);
        acc2.fill(0);
        for &(a1, a2, b) in block {
            b1.copy_from_slice(&b[..n]);
            b2.copy_from_slice(&b[..n]);
            P1.ntt(logn, &mut b1);
            P2.ntt(logn, &mut b2);
            for (pr, acc, bb, aa) in [
                (&P1, &mut acc1, &b1, &a1[..n]),
                (&P2, &mut acc2, &b2, &a2[..n]),
            ] {
                #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
                if crate::simd::mmul_add_slices(pr, acc, bb, aa) {
                    continue;
                }
                for ((s, &x), &y) in acc.iter_mut().zip(bb.iter()).zip(aa.iter()) {
                    *s = mp_add(*s, pr.mmul(x, y), pr.p);
                }
            }
        }
        P1.intt_n(logn, &mut acc1);
        P2.intt_n(logn, &mut acc2);
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
        if crate::simd::crt_slices(q, &acc1, &acc2, out, true) {
            continue;
        }
        for ((o, &u1), &u2) in out.iter_mut().zip(acc1.iter()).zip(acc2.iter()) {
            *o = mp_add(*o, mq.crt(u1, u2), q);
        }
    }
    for v in [&mut acc1, &mut acc2, &mut b1, &mut b2] {
        crate::wipe::slice(v);
    }
}

/// Whether a product with a ternary operand fits the single prime $`p_1`$: with the other
/// operand lifted to centred values (absolute value at most $`\lfloor q/2\rfloor`$), every
/// integer coefficient has absolute value at most $`n\lfloor q/2\rfloor`$, which the centred lift
/// modulo $`p_1`$ recovers iff $`2n\lfloor q/2\rfloor<p_1`$. True at $`n=1024`$ for $`q=1048573`$
/// ($`2\cdot1024\cdot524286=1{,}073{,}737{,}728`$) and $`q=12289`$, false for $`q`$ above about
/// $`2^{21}`$ at $`n=1024`$. With operands in $`[0,q)`$ instead, the bound would be
/// $`n(q-1)`$, and $`2n(q-1)`$ exceeds $`p_1`$ by 2,047 at $`q=1048573`$ [computed].
pub(crate) fn ternary_fits_p1(logn: u32, q: u32) -> bool {
    2 * (1u64 << logn) * (q as u64 / 2) < P1.p as u64
}

/// The residues modulo $`p_1`$ of the centred lift of `a` (coefficients in $`[0,q)`$; $`a_i`$ above
/// $`q/2`$ stands for $`a_i-q`$), in NTT form and prepared ([`NttPrime::scale_prepared`]): the
/// operand of [`mul_ternary_p1_mod_q`].
pub(crate) fn prepare_centred_p1(logn: u32, q: u32, a: &[u32]) -> Vec<u32> {
    let n = 1usize << logn;
    let p = P1.p;
    let mut r: Vec<u32> = a[..n]
        .iter()
        .map(|&x| {
            // x - q + p1 when 2x > q, by a mask.
            let up = ((q.wrapping_sub(x << 1) as i32) >> 31) as u32;
            x.wrapping_add((p.wrapping_sub(q)) & up)
        })
        .collect();
    P1.ntt(logn, &mut r);
    P1.scale_prepared(logn, &mut r);
    r
}

/// [`prepare_centred_p1`] from the prepared residues modulo $`p_1`$ ([`prepare`]) instead of the
/// coefficients: `mmul(r1, 1)` is $`\hat a/n`$, so the inverse NTT returns $`a \bmod p_1`$, which
/// is $`a`$ since $`q<p_1`$. One inverse and one forward NTT.
pub(crate) fn centred_p1_from_prepared(logn: u32, q: u32, r1: &[u32]) -> Vec<u32> {
    let n = 1usize << logn;
    let mut a = r1[..n].to_vec();
    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    let done = crate::simd::mmul_slices_const(&P1, &mut a, None, 1);
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    let done = false;
    if !done {
        for x in a.iter_mut() {
            *x = P1.mmul(*x, 1);
        }
    }
    P1.intt_n(logn, &mut a);
    let r = prepare_centred_p1(logn, q, &a);
    crate::wipe::slice(&mut a);
    r
}

/// `true` iff every coefficient of `t` (canonical, in $`[0,q)`$) is $`0`$, $`1`$ or $`q-1`$. The
/// test accumulates a mask over all coefficients, so its time does not depend on `t`.
pub(crate) fn is_ternary(q: u32, t: &[u32]) -> bool {
    let mut bad = 0u32;
    for &x in t {
        bad |= ((x != 0) & (x != 1) & (x != q - 1)) as u32;
    }
    bad == 0
}

/// `out <- a * t mod (x^n + 1, q)` for `a` given by [`prepare_centred_p1`] and a ternary `t`
/// ([`is_ternary`]), when [`ternary_fits_p1`] holds: one forward and one inverse NTT modulo
/// $`p_1`$, then the centred lift of each residue (the exact integer coefficient) reduced modulo
/// $`q`$. Equal to [`mul_prepared_mod_q`]. The residues are wiped.
pub(crate) fn mul_ternary_p1_mod_q(logn: u32, q: u32, ac1: &[u32], t: &[u32], out: &mut [u32]) {
    debug_assert!(ternary_fits_p1(logn, q) && is_ternary(q, t));
    let n = 1usize << logn;
    let p = P1.p;
    let mq = ModQ::new(q);
    // 0, 1, q - 1 to 0, 1, p1 - 1, by masks.
    let mut b: Vec<u32> = t[..n]
        .iter()
        .map(|&x| {
            let neg = ((x == q - 1) as u32).wrapping_neg();
            (x & !neg) | ((p - 1) & neg)
        })
        .collect();
    P1.ntt(logn, &mut b);
    P1.pointwise_mul_prepared(&mut b, &ac1[..n]);
    P1.intt_n(logn, &mut b);
    #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
    let done = crate::simd::centred_p1_mod_q(q, &b, &mut out[..n]);
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    let done = false;
    if !done {
        let half = p >> 1;
        for (o, &u) in out[..n].iter_mut().zip(b.iter()) {
            // The integer is u if u <= (p1 - 1)/2, else u - p1, and |c| < 2^31.
            let neg = ((half.wrapping_sub(u) as i32) >> 31) as u32;
            let c = (u as i64 - (p & neg) as i64) as i32;
            *o = mq.reduce_i32(c);
        }
    }
    crate::wipe::slice(&mut b);
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
        #[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
        if let Some(d) = crate::simd::ntru_diff(pr, q, rf, rg, rbf, rbg) {
            diff |= d;
            continue;
        }
        for i in 0..n {
            let lhs = mp_sub(pr.mul(rf[i], rbg[i]), pr.mul(rg[i], rbf[i]), p);
            diff |= lhs ^ q;
        }
    }
    crate::wipe::slice(&mut buf);
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
            // igm[rev10(1)] = R g^-1.
            let ig = pr.mmul(pr.igm[rev10(1)], 1) as u64;
            assert_eq!((g as u128 * ig as u128 % p as u128) as u64, 1);
            // ninv_r2[logn] = R^2 / n.
            for logn in 0..=10u32 {
                let c = pr.ninv_r2[logn as usize] as u128;
                let want = (pr.r2 as u128) % p as u128;
                assert_eq!(c * (1u128 << logn) % p as u128, want, "logn {logn}");
            }
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
                ig = pr.mmul(ig, 1); // plain g^-1
                let (mut gg, mut igg) = (g, ig);
                for _ in 0..k {
                    gg = pr.mul(gg, gg);
                    igg = pr.mul(igg, igg);
                }
                let mut x1 = mp_r(pr.p);
                let mut x2 = mp_r(pr.p);
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

    /// `reduce_i64` is `rem_euclid` on the whole range of `i64`, its ends included (the
    /// magnitude of `i64::MIN` is 2^63), and `reduce` on the whole range of `u64`.
    #[test]
    fn reduce_signed_i64() {
        for q in [3u32, 17, 12289, 1048573, (1 << 25) - 39] {
            let mq = ModQ::new(q);
            let qi = q as i64;
            let mut vals = vec![
                0i64,
                1,
                -1,
                i64::MAX,
                i64::MIN,
                i64::MIN + 1,
                i64::MAX - 1,
                qi,
                -qi,
                qi - 1,
                1 - qi,
                i64::MIN / qi * qi,
                i64::MAX / qi * qi,
            ];
            let mut x: u64 = 0x0fed_cba9_8765_4321;
            for _ in 0..10000 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                vals.push(x as i64);
                vals.push((x as i64) >> 20);
                vals.push((x as i64) >> 40);
            }
            for &v in &vals {
                assert_eq!(mq.reduce_i64(v) as i64, v.rem_euclid(qi), "q {q} v {v}");
                let u = v as u64;
                assert_eq!(mq.reduce(u) as u64, u % q as u64, "q {q} u {u}");
            }
            assert_eq!(mq.reduce(u64::MAX) as u64, u64::MAX % q as u64);
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

    /// fn-dsa-kgen `mp_NTT` (poly.rs:83-105), verbatim: the reference of [`NttPrime::ntt`].
    fn ntt_fndsa(pr: &NttPrime, logn: u32, a: &mut [u32]) {
        if logn == 0 {
            return;
        }
        let (p, p0i) = (pr.p, pr.p0i);
        let n = 1usize << logn;
        let mut t = n;
        for lm in 0..logn {
            let m = 1usize << lm;
            let ht = t >> 1;
            for i in 0..m {
                let s = pr.gm[i + m];
                let j0 = i * t;
                for j in j0..j0 + ht {
                    let u = a[j];
                    let v = mp_mmul(a[j + ht], s, p, p0i);
                    a[j] = mp_add(u, v, p);
                    a[j + ht] = mp_sub(u, v, p);
                }
            }
            t = ht;
        }
    }

    /// fn-dsa-kgen `mp_iNTT` (poly.rs:113-138), verbatim, with fn-dsa's inverse table (roots
    /// times 1/2, from mp_hR): the reference of [`NttPrime::intt_n`] and the 1/n folding.
    fn intt_fndsa(pr: &NttPrime, logn: u32, a: &mut [u32]) {
        if logn == 0 {
            return;
        }
        let (p, p0i) = (pr.p, pr.p0i);
        // fn-dsa's igm: rev10(i) -> 2^31 g^-i, that is the half of this crate's entry.
        let igm_half: Vec<u32> = pr.igm.iter().map(|&x| mp_half(x, p)).collect();
        assert_eq!(igm_half[0], mp_hr(p));
        let mut t = 1;
        for lm in 0..logn {
            let hm = 1usize << (logn - 1 - lm);
            let dt = t << 1;
            for i in 0..hm {
                let s = igm_half[i + hm];
                let j0 = i * dt;
                for j in j0..j0 + t {
                    let u = a[j];
                    let v = a[j + t];
                    a[j] = mp_half(mp_add(u, v, p), p);
                    a[j + t] = mp_mmul(mp_sub(u, v, p), s, p, p0i);
                }
            }
            t = dt;
        }
    }

    /// The restructured transforms equal fn-dsa's: the forward NTT exactly, and the inverse
    /// with the factor 1/n folded into a pointwise product (both prepared forms), at every
    /// size, on random and extreme residues.
    #[test]
    fn transforms_equal_fndsa() {
        let mut x: u64 = 0x5eed_0123_4567_89ab;
        let mut rnd = |m: u32| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % m as u64) as u32
        };
        for pr in [&P1, &P2] {
            let p = pr.p;
            for logn in 0..=10u32 {
                let n = 1usize << logn;
                for case in 0..6 {
                    let a: Vec<u32> = (0..n)
                        .map(|_| match case {
                            0 => p - 1,
                            1 => 0,
                            _ => rnd(p),
                        })
                        .collect();
                    let b: Vec<u32> = (0..n).map(|_| rnd(p)).collect();
                    // Forward.
                    let (mut fa, mut ra) = (a.clone(), a.clone());
                    pr.ntt(logn, &mut fa);
                    ntt_fndsa(pr, logn, &mut ra);
                    assert_eq!(fa, ra, "ntt p {p} logn {logn}");
                    // Inverse of a pointwise product, three ways.
                    let mut fb = b.clone();
                    pr.ntt(logn, &mut fb);
                    let mut want = fa.clone();
                    for (w, &y) in want.iter_mut().zip(fb.iter()) {
                        *w = pr.mul(*w, y);
                    }
                    intt_fndsa(pr, logn, &mut want);
                    let mut got = fa.clone();
                    pr.pointwise_mul_div_n(logn, &mut got, &fb);
                    pr.intt_n(logn, &mut got);
                    assert_eq!(got, want, "pointwise_mul_div_n p {p} logn {logn}");
                    let mut prep = fb.clone();
                    pr.scale_prepared(logn, &mut prep);
                    let mut got = fa.clone();
                    pr.pointwise_mul_prepared(&mut got, &prep);
                    pr.intt_n(logn, &mut got);
                    assert_eq!(got, want, "prepared p {p} logn {logn}");
                    // The round trip of a alone.
                    let mut back = fa.clone();
                    let ones = vec![1u32; n];
                    pr.pointwise_mul_div_n(logn, &mut back, &ones);
                    pr.intt_n(logn, &mut back);
                    assert_eq!(back, a, "round trip p {p} logn {logn}");
                }
            }
        }
    }
}
