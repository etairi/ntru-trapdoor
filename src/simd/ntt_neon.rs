//! NEON kernels of the two-prime NTT (4 lanes of 32 bits), on aarch64 where NEON is part of
//! the baseline. The same arithmetic as the AVX2 kernels (`ntt_avx2`): Montgomery
//! multiplication with `umull` and `umlal` ($`t=(z+wp)/2^{32}<1.5p`$), and the final reductions
//! as unsigned minima, lane by lane equal to the portable code.
use core::arch::aarch64::*;

use crate::ring::ntt::NttPrime;

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
