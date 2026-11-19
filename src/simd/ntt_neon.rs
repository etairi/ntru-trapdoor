//! NEON kernels of the two-prime NTT (4 lanes of 32 bits), on aarch64 where NEON is part of
//! the baseline. The same arithmetic as the AVX2 kernels (`ntt_avx2`): Montgomery
//! multiplication with `umull` and `umlal` ($`t=(z+wp)/2^{32}<1.5p`$), and the final reductions
//! as unsigned minima, lane by lane equal to the portable code.
//!
//! The narrow layers (half-blocks of 2 and 1 values) run on two blocks of 4 values at a time:
//! `zip` and `trn` regroup the lanes between the layers, so that each layer is one butterfly on
//! two vectors (on an Apple M4 equal or faster than `vld4q` and `vst4q` on four blocks). The
//! CRT, the reductions modulo $`q`$ and Verify's norm compute what the AVX2 kernels (`ntt_avx2`)
//! compute.
use core::arch::aarch64::*;

use super::QConsts;
use crate::ring::ntt::{NttPrime, P1, P1_INV_MOD_P2_MONT, P2};

/// Montgomery multiplication of 4 lanes (canonical inputs, canonical output).
#[target_feature(enable = "neon")]
#[inline]
fn mmul4(a: uint32x4_t, b: uint32x4_t, p: uint32x4_t, p0i: uint32x2_t) -> uint32x4_t {
    let pl = vget_low_u32(p);
    let zl = vmull_u32(vget_low_u32(a), vget_low_u32(b));
    let zh = vmull_high_u32(a, b);
    let sl = vmlal_u32(zl, vmul_u32(vmovn_u64(zl), p0i), pl);
    let sh = vmlal_u32(zh, vmul_u32(vmovn_u64(zh), p0i), pl);
    let t = vcombine_u32(vshrn_n_u64::<32>(sl), vshrn_n_u64::<32>(sh));
    vminq_u32(t, vsubq_u32(t, p))
}

/// $`a+b \bmod p`$ on 4 lanes.
#[target_feature(enable = "neon")]
#[inline]
fn add4(a: uint32x4_t, b: uint32x4_t, p: uint32x4_t) -> uint32x4_t {
    let s = vaddq_u32(a, b);
    vminq_u32(s, vsubq_u32(s, p))
}

/// $`a-b \bmod p`$ on 4 lanes.
#[target_feature(enable = "neon")]
#[inline]
fn sub4(a: uint32x4_t, b: uint32x4_t, p: uint32x4_t) -> uint32x4_t {
    let d = vsubq_u32(a, b);
    vminq_u32(d, vaddq_u32(d, p))
}

#[target_feature(enable = "neon")]
#[inline]
fn load(s: &[u32; 4]) -> uint32x4_t {
    // SAFETY: `s` is 16 bytes.
    unsafe { vld1q_u32(s.as_ptr()) }
}

#[target_feature(enable = "neon")]
#[inline]
fn store(s: &mut [u32; 4], v: uint32x4_t) {
    // SAFETY: `s` is 16 bytes.
    unsafe { vst1q_u32(s.as_mut_ptr(), v) }
}

#[target_feature(enable = "neon")]
#[inline]
fn consts(pr: &NttPrime) -> (uint32x4_t, uint32x2_t) {
    (vdupq_n_u32(pr.p), vdup_n_u32(pr.p0i))
}

/// `NttPrime::ntt`'s layers with half-blocks of at least 4 values: returns the block size and
/// the block count of the next layer (block size 4: the portable code's last two layers).
#[target_feature(enable = "neon")]
pub(super) fn ntt_wide_layers(pr: &NttPrime, logn: u32, a: &mut [u32]) -> (usize, usize) {
    let n = 1usize << logn;
    let (p, p0i) = consts(pr);
    let mut t = n;
    let mut m = 1usize;
    while t >= 8 {
        let ht = t >> 1;
        for (blk, &s) in a[..n].chunks_exact_mut(t).zip(pr.gm[m..2 * m].iter()) {
            let sv = vdupq_n_u32(s);
            let (x, y) = blk.split_at_mut(ht);
            for (xc, yc) in x
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(y.as_chunks_mut::<4>().0)
            {
                let (xv, yv) = (load(xc), load(yc));
                let v = mmul4(yv, sv, p, p0i);
                store(xc, add4(xv, v, p));
                store(yc, sub4(xv, v, p));
            }
        }
        t = ht;
        m <<= 1;
    }
    (t, m)
}

/// `NttPrime::intt_n`'s layers from half-block size `t >= 4` (with `hm` blocks) to the end.
#[target_feature(enable = "neon")]
pub(super) fn intt_wide_layers(pr: &NttPrime, logn: u32, a: &mut [u32], t: usize, hm: usize) {
    let n = 1usize << logn;
    let (p, p0i) = consts(pr);
    let (mut t, mut hm) = (t, hm);
    while t < n {
        let dt = t << 1;
        for (blk, &s) in a[..n].chunks_exact_mut(dt).zip(pr.igm[hm..2 * hm].iter()) {
            let sv = vdupq_n_u32(s);
            let (x, y) = blk.split_at_mut(t);
            for (xc, yc) in x
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(y.as_chunks_mut::<4>().0)
            {
                let (u, v) = (load(xc), load(yc));
                store(xc, add4(u, v, p));
                store(yc, mmul4(sub4(u, v, p), sv, p, p0i));
            }
        }
        t = dt;
        hm >>= 1;
    }
}

/// `a[i] <- mmul(a[i], b[i])`, lengths equal multiples of 4.
#[target_feature(enable = "neon")]
pub(super) fn mmul_slices(pr: &NttPrime, a: &mut [u32], b: &[u32]) {
    let (p, p0i) = consts(pr);
    for (ac, bc) in a
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(b.as_chunks::<4>().0)
    {
        store(ac, mmul4(load(ac), load(bc), p, p0i));
    }
}

/// `a[i] <- mmul(mmul(a[i], b[i]), c)`, or `mmul(a[i], c)` without `b`; length a multiple of 4.
#[target_feature(enable = "neon")]
pub(super) fn mmul_slices_const(pr: &NttPrime, a: &mut [u32], b: Option<&[u32]>, c: u32) {
    let (p, p0i) = consts(pr);
    let cv = vdupq_n_u32(c);
    match b {
        Some(b) => {
            for (ac, bc) in a
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(b.as_chunks::<4>().0)
            {
                store(ac, mmul4(mmul4(load(ac), load(bc), p, p0i), cv, p, p0i));
            }
        }
        None => {
            for ac in a.as_chunks_mut::<4>().0.iter_mut() {
                store(ac, mmul4(load(ac), cv, p, p0i));
            }
        }
    }
}

/// `acc[i] <- mp_add(acc[i], mmul(x[i], y[i]))`, lengths equal multiples of 4.
#[target_feature(enable = "neon")]
pub(super) fn mmul_add_slices(pr: &NttPrime, acc: &mut [u32], x: &[u32], y: &[u32]) {
    let (p, p0i) = consts(pr);
    for ((sc, xc), yc) in acc
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
        .zip(y.as_chunks::<4>().0)
    {
        store(sc, add4(load(sc), mmul4(load(xc), load(yc), p, p0i), p));
    }
}

/// `[r0 r0 r1 r1]`: the roots of a layer with half-blocks of 2 values, for 2 blocks of 4.
#[target_feature(enable = "neon")]
#[inline]
fn roots_dup2(r: &[u32; 2]) -> uint32x4_t {
    // SAFETY: `r` is 8 bytes.
    let r = unsafe { vld1_u32(r.as_ptr()) };
    let rr = vcombine_u32(r, r);
    vzip1q_u32(rr, rr)
}

/// The forward butterfly `(x + y s, x - y s)` on 4 lanes.
#[target_feature(enable = "neon")]
#[inline]
fn fwd4(
    x: uint32x4_t,
    y: uint32x4_t,
    s: uint32x4_t,
    p: uint32x4_t,
    p0i: uint32x2_t,
) -> (uint32x4_t, uint32x4_t) {
    let v = mmul4(y, s, p, p0i);
    (add4(x, v, p), sub4(x, v, p))
}

/// The inverse butterfly `(x + y, (x - y) s)` on 4 lanes.
#[target_feature(enable = "neon")]
#[inline]
fn inv4(
    x: uint32x4_t,
    y: uint32x4_t,
    s: uint32x4_t,
    p: uint32x4_t,
    p0i: uint32x2_t,
) -> (uint32x4_t, uint32x4_t) {
    (add4(x, y, p), mmul4(sub4(x, y, p), s, p, p0i))
}

/// `NttPrime::ntt`'s last two layers, from blocks of 4 values (`n / 4` blocks, block `i` with
/// the root `gm[n/4 + i]` for the pairs (0, 2), (1, 3), then `gm[n/2 + 2i]` for (0, 1) and
/// `gm[n/2 + 2i + 1]` for (2, 3)), two blocks per iteration: `zip` and `trn` regroup the lanes
/// so that each layer is one butterfly on two vectors.
#[target_feature(enable = "neon")]
pub(super) fn ntt_narrow_layers(pr: &NttPrime, logn: u32, a: &mut [u32]) {
    let n = 1usize << logn;
    let m = n >> 2;
    let (p, p0i) = consts(pr);
    let rb = pr.gm[m..2 * m].as_chunks::<2>().0;
    let rc = pr.gm[2 * m..4 * m].as_chunks::<4>().0;
    let blocks = a[..n].as_chunks_mut::<4>().0.as_chunks_mut::<2>().0;
    for ((pair, sb), sc) in blocks.iter_mut().zip(rb).zip(rc) {
        let [u, v] = pair;
        let (u2, v2) = (
            vreinterpretq_u64_u32(load(u)),
            vreinterpretq_u64_u32(load(v)),
        );
        // Half-blocks of 2: x = [u0 u1 v0 v1], y = [u2 u3 v2 v3], roots [sb0 sb0 sb1 sb1].
        let x = vreinterpretq_u32_u64(vzip1q_u64(u2, v2));
        let y = vreinterpretq_u32_u64(vzip2q_u64(u2, v2));
        let (x, y) = fwd4(x, y, roots_dup2(sb), p, p0i);
        // Pairs: x = [u0 u2 v0 v2], y = [u1 u3 v1 v3], roots sc.
        let (x, y) = (vtrn1q_u32(x, y), vtrn2q_u32(x, y));
        let (x, y) = fwd4(x, y, load(sc), p, p0i);
        // Back in order.
        let (x, y) = (
            vreinterpretq_u64_u32(vtrn1q_u32(x, y)),
            vreinterpretq_u64_u32(vtrn2q_u32(x, y)),
        );
        store(u, vreinterpretq_u32_u64(vzip1q_u64(x, y)));
        store(v, vreinterpretq_u32_u64(vzip2q_u64(x, y)));
    }
}

/// `NttPrime::intt_n`'s first two layers (pairs (0, 1) and (2, 3) of block `i` with
/// `igm[n/2 + 2i]` and `igm[n/2 + 2i + 1]`, then (0, 2), (1, 3) with `igm[n/4 + i]`), two
/// blocks of 4 per iteration: the steps of [`ntt_narrow_layers`] in reverse order.
#[target_feature(enable = "neon")]
pub(super) fn intt_narrow_layers(pr: &NttPrime, logn: u32, a: &mut [u32]) {
    let n = 1usize << logn;
    let m = n >> 2;
    let (p, p0i) = consts(pr);
    let rc = pr.igm[2 * m..4 * m].as_chunks::<4>().0;
    let rb = pr.igm[m..2 * m].as_chunks::<2>().0;
    let blocks = a[..n].as_chunks_mut::<4>().0.as_chunks_mut::<2>().0;
    for ((pair, sc), sb) in blocks.iter_mut().zip(rc).zip(rb) {
        let [u, v] = pair;
        let (u2, v2) = (
            vreinterpretq_u64_u32(load(u)),
            vreinterpretq_u64_u32(load(v)),
        );
        // Pairs: x = [u0 u2 v0 v2], y = [u1 u3 v1 v3], roots sc.
        let lo = vreinterpretq_u32_u64(vzip1q_u64(u2, v2));
        let hi = vreinterpretq_u32_u64(vzip2q_u64(u2, v2));
        let (x, y) = (vtrn1q_u32(lo, hi), vtrn2q_u32(lo, hi));
        let (x, y) = inv4(x, y, load(sc), p, p0i);
        // Half-blocks of 2: x = [u0 u1 v0 v1], y = [u2 u3 v2 v3], roots [sb0 sb0 sb1 sb1].
        let (x, y) = (vtrn1q_u32(x, y), vtrn2q_u32(x, y));
        let (x, y) = inv4(x, y, roots_dup2(sb), p, p0i);
        // Back in order.
        let (x, y) = (vreinterpretq_u64_u32(x), vreinterpretq_u64_u32(y));
        store(u, vreinterpretq_u32_u64(vzip1q_u64(x, y)));
        store(v, vreinterpretq_u32_u64(vzip2q_u64(x, y)));
    }
}

/// The constants of [`crt4`] as lanes.
struct Crt4 {
    p2: uint32x4_t,
    p2i: uint32x2_t,
    inv: uint32x4_t,
    h1: uint32x4_t,
    h2: uint32x4_t,
    q: uint32x4_t,
    q0i: uint32x2_t,
    r: uint32x2_t,
    p1r: uint32x2_t,
    pq: uint32x4_t,
}

#[target_feature(enable = "neon")]
#[inline]
fn crt_consts(c: &QConsts) -> Crt4 {
    Crt4 {
        p2: vdupq_n_u32(P2.p),
        p2i: vdup_n_u32(P2.p0i),
        inv: vdupq_n_u32(P1_INV_MOD_P2_MONT),
        h1: vdupq_n_u32((P1.p - 1) / 2),
        h2: vdupq_n_u32((P2.p - 1) / 2),
        q: vdupq_n_u32(c.q),
        q0i: vdup_n_u32(c.q0i),
        r: vdup_n_u32(c.r),
        p1r: vdup_n_u32(c.p1r),
        pq: vdupq_n_u32(c.pq),
    }
}

/// `ModQ::crt` on 4 lanes, for an odd $`q`$: the arithmetic of the AVX2 kernel (`ntt_avx2::crt8`),
/// with $`z=u_1(2^{32}\bmod q)+t(p_12^{32}\bmod q)`$ from `umull` and `umlal`.
#[target_feature(enable = "neon")]
#[inline]
fn crt4(u1: uint32x4_t, u2: uint32x4_t, k: &Crt4) -> uint32x4_t {
    let d = sub4(u2, vminq_u32(u1, vsubq_u32(u1, k.p2)), k.p2);
    let t = mmul4(d, k.inv, k.p2, k.p2i);
    let zl = vmlal_u32(vmull_u32(vget_low_u32(u1), k.r), vget_low_u32(t), k.p1r);
    let zh = vmlal_u32(
        vmull_high_u32(u1, vcombine_u32(k.r, k.r)),
        vget_high_u32(t),
        k.p1r,
    );
    let ql = vget_low_u32(k.q);
    let sl = vmlal_u32(zl, vmul_u32(vmovn_u64(zl), k.q0i), ql);
    let sh = vmlal_u32(zh, vmul_u32(vmovn_u64(zh), k.q0i), ql);
    let r = vcombine_u32(vshrn_n_u64::<32>(sl), vshrn_n_u64::<32>(sh));
    let r = vminq_u32(r, vsubq_u32(r, k.q));
    let neg = vorrq_u32(
        vcgtq_u32(t, k.h2),
        vandq_u32(vceqq_u32(t, k.h2), vcgtq_u32(u1, k.h1)),
    );
    sub4(r, vandq_u32(k.pq, neg), k.q)
}

/// `out[i] <- crt(u1[i], u2[i])`, or `mp_add(out[i], crt(u1[i], u2[i]), q)` with `ACC`, for an
/// odd $`q`$. Equal lengths, multiples of 4.
#[target_feature(enable = "neon")]
pub(super) fn crt_slices<const ACC: bool>(c: &QConsts, u1: &[u32], u2: &[u32], out: &mut [u32]) {
    let k = crt_consts(c);
    for ((o, x), y) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(u1.as_chunks::<4>().0)
        .zip(u2.as_chunks::<4>().0)
    {
        let r = crt4(load(x), load(y), &k);
        store(o, if ACC { add4(load(o), r, k.q) } else { r });
    }
}

/// `ModQ::reduce_i32` on 4 lanes: the Barrett step of the AVX2 kernel (`ntt_avx2::reduce_i32x8`),
/// `vabsq_s32` leaving $`-2^{31}`$ as $`2^{31}`$ unsigned.
#[target_feature(enable = "neon")]
#[inline]
fn reduce_i32x4(x: int32x4_t, q: uint32x4_t, m: uint32x2_t) -> uint32x4_t {
    let ax = vreinterpretq_u32_s32(vabsq_s32(x));
    let k = vcombine_u32(
        vshrn_n_u64::<32>(vmull_u32(vget_low_u32(ax), m)),
        vshrn_n_u64::<32>(vmull_high_u32(ax, vcombine_u32(m, m))),
    );
    let r = vmlsq_u32(ax, k, q);
    let r = vminq_u32(r, vsubq_u32(r, q));
    let nr = sub4(vdupq_n_u32(0), r, q);
    vbslq_u32(vreinterpretq_u32_s32(vshrq_n_s32::<31>(x)), nr, r)
}

#[target_feature(enable = "neon")]
#[inline]
fn load_i32(s: &[i32; 4]) -> int32x4_t {
    // SAFETY: `s` is 16 bytes.
    unsafe { vld1q_s32(s.as_ptr()) }
}

/// `out[i] <- x[i] mod q` (`ModQ::reduce_i32`). Equal lengths, multiples of 4.
#[target_feature(enable = "neon")]
pub(super) fn reduce_i32_slices(c: &QConsts, x: &[i32], out: &mut [u32]) {
    let (q, m) = (vdupq_n_u32(c.q), vdup_n_u32(c.m));
    for (o, v) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<4>().0)
    {
        store(o, reduce_i32x4(load_i32(v), q, m));
    }
}

/// The OR over `i` of `(x[i] mod q + y[i] mod q) XOR z[i]`, for signed `x` and canonical `y`.
/// Equal lengths, multiples of 4.
#[target_feature(enable = "neon")]
pub(super) fn add_i32_diff(c: &QConsts, x: &[i32], y: &[u32], z: &[u32]) -> u32 {
    let (q, m) = (vdupq_n_u32(c.q), vdup_n_u32(c.m));
    let mut acc = vdupq_n_u32(0);
    for ((xc, yc), zc) in x
        .as_chunks::<4>()
        .0
        .iter()
        .zip(y.as_chunks::<4>().0)
        .zip(z.as_chunks::<4>().0)
    {
        let s = add4(reduce_i32x4(load_i32(xc), q, m), load(yc), q);
        acc = vorrq_u32(acc, veorq_u32(s, load(zc)));
    }
    or_lanes(acc)
}

/// `out[i] <- c mod q` for the centred lift `c` of `u[i] mod p1`. Equal lengths, multiples of 4.
#[target_feature(enable = "neon")]
pub(super) fn centred_p1_mod_q(c: &QConsts, u: &[u32], out: &mut [u32]) {
    let (q, m) = (vdupq_n_u32(c.q), vdup_n_u32(c.m));
    let (p1, h1) = (vdupq_n_u32(P1.p), vdupq_n_u32(P1.p >> 1));
    for (o, uc) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(u.as_chunks::<4>().0)
    {
        let v = load(uc);
        let lifted = vsubq_u32(v, vandq_u32(p1, vcgtq_u32(v, h1)));
        store(o, reduce_i32x4(vreinterpretq_s32_u32(lifted), q, m));
    }
}

/// $`\sum_ix_i^2`$ over both slices, exactly: the low and high 32 bits of each square
/// ($`\le2^{62}`$) are added separately in 64-bit lanes (`uaddw`, `usra`).
#[target_feature(enable = "neon")]
pub(super) fn norm_sq(a: &[i32], b: &[i32]) -> u128 {
    let (mut lo, mut hi) = (vdupq_n_u64(0), vdupq_n_u64(0));
    let mut tail = 0u128;
    for s in [a, b] {
        let (chunks, rest) = s.as_chunks::<4>();
        for c in chunks {
            let x = vreinterpretq_u32_s32(vabsq_s32(load_i32(c)));
            for sq in [
                vmull_u32(vget_low_u32(x), vget_low_u32(x)),
                vmull_high_u32(x, x),
            ] {
                lo = vaddw_u32(lo, vmovn_u64(sq));
                hi = vsraq_n_u64::<32>(hi, sq);
            }
        }
        tail += rest
            .iter()
            .map(|&x| (x.unsigned_abs() as u128).pow(2))
            .sum::<u128>();
    }
    let l = vgetq_lane_u64::<0>(lo) as u128 + vgetq_lane_u64::<1>(lo) as u128;
    let h = vgetq_lane_u64::<0>(hi) as u128 + vgetq_lane_u64::<1>(hi) as u128;
    (h << 32) + l + tail
}

/// The norm of `ring::inv` in the NTT domain (`norm_ntt`) on 4 lanes (see the AVX2 kernel).
/// Length `h` a multiple of 4.
#[target_feature(enable = "neon")]
pub(super) fn norm_ntt(pr: &NttPrime, logh: u32, e: &[u32], o: &[u32], out: &mut [u32]) {
    let h = 1usize << logh;
    let (p, p0i) = consts(pr);
    let c = vdupq_n_u32(pr.ninv_r2[logh as usize]);
    let odd = vcombine_u32(
        vcreate_u32(0xFFFF_FFFF_0000_0000),
        vcreate_u32(0xFFFF_FFFF_0000_0000),
    );
    let z = pr.gm[h / 2..h].as_chunks::<2>().0;
    for (((oc, ec), occ), zc) in out[..h]
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(e[..h].as_chunks::<4>().0)
        .zip(o[..h].as_chunks::<4>().0)
        .zip(z)
    {
        let (ev, ov) = (load(ec), load(occ));
        let e2 = mmul4(mmul4(ev, ev, p, p0i), c, p, p0i);
        let o2 = mmul4(mmul4(ov, ov, p, p0i), c, p, p0i);
        let zz = vcombine_u32(vdup_n_u32(zc[0]), vdup_n_u32(zc[1]));
        let zo2 = mmul4(o2, zz, p, p0i);
        store(oc, vbslq_u32(odd, add4(e2, zo2, p), sub4(e2, zo2, p)));
    }
}

/// The OR over `i` of `(mul(f[i], G[i]) - mul(g[i], F[i])) XOR q`. Equal lengths, multiples of
/// 4.
#[target_feature(enable = "neon")]
pub(super) fn ntru_diff(
    pr: &NttPrime,
    q: u32,
    f: &[u32],
    g: &[u32],
    bf: &[u32],
    bg: &[u32],
) -> u32 {
    let (p, p0i) = consts(pr);
    let r2 = vdupq_n_u32(pr.r2);
    let qv = vdupq_n_u32(q);
    let mut acc = vdupq_n_u32(0);
    for (((fc, gc), bfc), bgc) in f
        .as_chunks::<4>()
        .0
        .iter()
        .zip(g.as_chunks::<4>().0)
        .zip(bf.as_chunks::<4>().0)
        .zip(bg.as_chunks::<4>().0)
    {
        let fg = mmul4(mmul4(load(fc), load(bgc), p, p0i), r2, p, p0i);
        let gf = mmul4(mmul4(load(gc), load(bfc), p, p0i), r2, p, p0i);
        acc = vorrq_u32(acc, veorq_u32(sub4(fg, gf, p), qv));
    }
    or_lanes(acc)
}

/// The OR of the 4 lanes.
#[target_feature(enable = "neon")]
#[inline]
fn or_lanes(v: uint32x4_t) -> u32 {
    let h = vorr_u32(vget_low_u32(v), vget_high_u32(v));
    vget_lane_u32::<0>(h) | vget_lane_u32::<1>(h)
}
