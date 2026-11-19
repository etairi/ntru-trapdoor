//! Keccak-f\[1600\] with the SHA3 extension of aarch64, on two states at once (one per 64-bit
//! element of 25 registers, `keccak_lanes`) or on one. θ takes `EOR3` and `RAX1`, θ's sums with
//! ρ one `XAR` per lane, and χ one `BCAX` per lane. For each state the kernels return what
//! `prng::keccak_f1600_portable` returns.
use core::arch::aarch64::*;

type V = uint64x2_t;

#[target_feature(enable = "neon,sha3")]
#[inline]
fn xor5(a: V, b: V, c: V, d: V, e: V) -> V {
    veor3q_u64(veor3q_u64(a, b, c), d, e)
}

/// `RAX1`: $`c\oplus\mathrm{rol}(e,1)`$.
#[target_feature(enable = "neon,sha3")]
#[inline]
fn theta_d(c: V, e: V) -> V {
    vrax1q_u64(c, e)
}

#[target_feature(enable = "neon,sha3")]
#[inline]
fn theta_xor(a: V, d: V) -> V {
    veorq_u64(a, d)
}

/// `XAR`: $`\mathrm{ror}(a\oplus d,L)=\mathrm{rol}(a\oplus d,R)`$.
#[target_feature(enable = "neon,sha3")]
#[inline]
fn rho<const R: i32, const L: i32>(a: V, d: V) -> V {
    vxarq_u64::<L>(a, d)
}

/// `BCAX`: $`a\oplus(c\wedge\lnot b)`$.
#[target_feature(enable = "neon,sha3")]
#[inline]
fn chi(a: V, b: V, c: V) -> V {
    vbcaxq_u64(a, c, b)
}

#[target_feature(enable = "neon,sha3")]
#[inline]
fn iota(a: V, rc: u64) -> V {
    veorq_u64(a, vdupq_n_u64(rc))
}

/// Lanes `j` and `j + 1` of a state.
#[target_feature(enable = "neon")]
#[inline]
fn load2(s: &[u64; 25], j: usize) -> V {
    let c = &s[j..j + 2];
    // SAFETY: `c` is two lanes, 16 bytes.
    unsafe { vld1q_u64(c.as_ptr()) }
}

/// Writes lanes `j` and `j + 1` of a state.
#[target_feature(enable = "neon")]
#[inline]
fn store2(s: &mut [u64; 25], j: usize, x: V) {
    let c = &mut s[j..j + 2];
    // SAFETY: `c` is two lanes, 16 bytes.
    unsafe { vst1q_u64(c.as_mut_ptr(), x) }
}

/// Keccak-f\[1600\] on one state: the code of [`keccak_f1600_x2`], with the state in element 0
/// of each register and zeros in element 1 (a don't-care, which ι changes). Dispatched on Apple's
/// processors only (see `simd::keccak_f1600`), and tested on every processor with the extension.
#[cfg(any(test, target_vendor = "apple"))]
#[target_feature(enable = "neon,sha3")]
pub(super) fn keccak_f1600(state: &mut [u64; 25]) {
    let mut a = [vdupq_n_u64(0); 25];
    for (ai, &si) in a.iter_mut().zip(state.iter()) {
        *ai = vcombine_u64(vdup_n_u64(si), vdup_n_u64(0));
    }
    keccak_f1600_lanes!(a);
    for (si, ai) in state.iter_mut().zip(a) {
        *si = vgetq_lane_u64::<0>(ai);
    }
}

/// Keccak-f\[1600\] on two states.
#[target_feature(enable = "neon,sha3")]
pub(super) fn keccak_f1600_x2(states: &mut [[u64; 25]; 2]) {
    let mut a = [vdupq_n_u64(0); 25];
    for i in 0..12 {
        let j = 2 * i;
        let (x, y) = (load2(&states[0], j), load2(&states[1], j));
        a[j] = vzip1q_u64(x, y);
        a[j + 1] = vzip2q_u64(x, y);
    }
    a[24] = vcombine_u64(vdup_n_u64(states[0][24]), vdup_n_u64(states[1][24]));
    keccak_f1600_lanes!(a);
    for i in 0..12 {
        let j = 2 * i;
        store2(&mut states[0], j, vzip1q_u64(a[j], a[j + 1]));
        store2(&mut states[1], j, vzip2q_u64(a[j], a[j + 1]));
    }
    states[0][24] = vgetq_lane_u64::<0>(a[24]);
    states[1][24] = vgetq_lane_u64::<1>(a[24]);
}
