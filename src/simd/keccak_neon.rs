//! Keccak-f\[1600\] with the SHA3 extension of aarch64, on two states at once (one per 64-bit
//! element of 25 registers, `keccak_lanes`), on one, or on three. For each state the kernels
//! return what `prng::keccak_f1600_portable` returns.
//!
//! - [`keccak_f1600_x2`] and [`keccak_f1600`]: θ takes `EOR3` and `RAX1`, θ's sums with ρ one
//!   `XAR` per lane, and χ one `BCAX` per lane, 64 SHA3 instructions per round. For processors
//!   that run them as fast as the other vector instructions, such as Apple's.
//! - [`keccak_f1600_x2_mixed`] and [`keccak_f1600_x3`]: for processors that run the SHA3
//!   instructions on a single pipe. A Neoverse V2 issues them once per cycle, against four times
//!   per cycle for `EOR`, `AND`, `ORR` and `ORN` and twice for `USHR` and `SLI` (measured), so the
//!   64 SHA3 instructions cost 64 cycles per round there. The mixed round ([`mixed::round`])
//!   keeps them for θ and for 15 of the 24 rotations, rotates the other lanes with `USHR` and
//!   `SLI`, and computes χ with the plain instructions. `keccak_f1600_x3` permutes a third state
//!   in general-purpose registers during the same rounds, on the scalar pipes.
use core::arch::aarch64::*;

type V = uint64x2_t;

#[cfg(any(test, target_vendor = "apple"))]
#[target_feature(enable = "neon,sha3")]
#[inline]
fn xor5(a: V, b: V, c: V, d: V, e: V) -> V {
    veor3q_u64(veor3q_u64(a, b, c), d, e)
}

/// `RAX1`: $`c\oplus\mathrm{rol}(e,1)`$.
#[cfg(any(test, target_vendor = "apple"))]
#[target_feature(enable = "neon,sha3")]
#[inline]
fn theta_d(c: V, e: V) -> V {
    vrax1q_u64(c, e)
}

#[cfg(any(test, target_vendor = "apple"))]
#[target_feature(enable = "neon,sha3")]
#[inline]
fn theta_xor(a: V, d: V) -> V {
    veorq_u64(a, d)
}

/// `XAR`: $`\mathrm{ror}(a\oplus d,L)=\mathrm{rol}(a\oplus d,R)`$.
#[cfg(any(test, target_vendor = "apple"))]
#[target_feature(enable = "neon,sha3")]
#[inline]
fn rho<const R: i32, const L: i32>(a: V, d: V) -> V {
    vxarq_u64::<L>(a, d)
}

/// `BCAX`: $`a\oplus(c\wedge\lnot b)`$.
#[cfg(any(test, target_vendor = "apple"))]
#[target_feature(enable = "neon,sha3")]
#[inline]
fn chi(a: V, b: V, c: V) -> V {
    vbcaxq_u64(a, c, b)
}

#[cfg(any(test, target_vendor = "apple"))]
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

/// Keccak-f\[1600\] on two states. Dispatched on Apple's processors only (see
/// `simd::keccak_f1600_x2`), and tested on every processor with the extension.
#[cfg(any(test, target_vendor = "apple"))]
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

/// The lanes that the mixed kernels hold complemented, those of the portable code
/// (`prng::keccak_f1600_portable`): with the χ of [`mixed::round`], a round maps lanes held with
/// these complemented to lanes held the same way.
const COMPLEMENTED: [usize; 6] = [1, 2, 8, 12, 17, 20];

/// The ρ offsets of the lanes that [`mixed::round`] rotates with `XAR`: those of lanes 1 to 15 of
/// `keccak_theta_rho_pi`. The other nine take `EOR`, `USHR` and `SLI`. Of 0, 6, 9, 12, 15, 18 and
/// 24 such lanes, 15 gave the fastest three-state kernel on a Neoverse V2.
const XAR_OFFSETS: u64 = {
    let offsets = [44, 43, 21, 14, 28, 20, 3, 45, 61, 1, 6, 25, 8, 18, 27];
    let mut m = 0u64;
    let mut i = 0;
    while i < offsets.len() {
        m |= 1 << offsets[i];
        i += 1;
    }
    m
};

/// The round of the mixed kernels and its primitives.
mod mixed {
    use super::*;

    /// θ's parities with two `EOR3`.
    #[target_feature(enable = "neon,sha3")]
    #[inline]
    fn xor5(a: V, b: V, c: V, d: V, e: V) -> V {
        veor3q_u64(veor3q_u64(a, b, c), d, e)
    }

    /// `RAX1`: $`c\oplus\mathrm{rol}(e,1)`$. Since five complemented lanes cancel in pairs, `D`
    /// is the same whichever lanes are complemented.
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

    /// $`\mathrm{rol}(a\oplus d,R)`$: `XAR` for the offsets of [`XAR_OFFSETS`], else `EOR`, then
    /// `USHR` and `SLI` on the shift pipes (LLVM merges `SHL` and `SRI` into `XAR`, not these).
    #[target_feature(enable = "neon,sha3")]
    #[inline]
    fn rho<const R: i32, const L: i32>(a: V, d: V) -> V {
        if (XAR_OFFSETS >> R) & 1 == 1 {
            vxarq_u64::<L>(a, d)
        } else {
            let t = veorq_u64(a, d);
            vsliq_n_u64::<R>(vshrq_n_u64::<L>(t), t)
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn not(a: V) -> V {
        vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(a)))
    }

    /// The lanes of [`COMPLEMENTED`] complemented, in place.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn complement(a: &mut [V; 25]) {
        for &i in COMPLEMENTED.iter() {
            a[i] = not(a[i]);
        }
    }

    /// One round on lanes held with those of [`COMPLEMENTED`] complemented, which it leaves held
    /// the same way. χ takes the forms of the portable code: $`b\oplus(c\vee d)`$,
    /// $`b\oplus(c\wedge d)`$ and $`b\oplus(c\vee\lnot d)`$ (`ORN`) in 20 lanes, one `BCAX` in
    /// lanes 12 and 20, and a complemented $`b`$ in lanes 13, 18 and 21. LLVM merges `BIC` and
    /// `EOR` into `BCAX`, but not `ORR`, `AND` or `ORN` and `EOR`, so these run on the other
    /// pipes.
    #[target_feature(enable = "neon,sha3")]
    #[inline]
    pub(super) fn round(a: &mut [V; 25], rc: u64) {
        let b = keccak_theta_rho_pi!(a);
        let x = |p: V, q: V| veorq_u64(p, q);
        a[0] = x(x(b[0], vorrq_u64(b[1], b[2])), vdupq_n_u64(rc));
        a[1] = x(b[1], vornq_u64(b[3], b[2]));
        a[2] = x(b[2], vandq_u64(b[3], b[4]));
        a[3] = x(b[3], vorrq_u64(b[4], b[0]));
        a[4] = x(b[4], vandq_u64(b[0], b[1]));
        a[5] = x(b[5], vorrq_u64(b[6], b[7]));
        a[6] = x(b[6], vandq_u64(b[7], b[8]));
        a[7] = x(b[7], vornq_u64(b[8], b[9]));
        a[8] = x(b[8], vorrq_u64(b[9], b[5]));
        a[9] = x(b[9], vandq_u64(b[5], b[6]));
        a[10] = x(b[10], vorrq_u64(b[11], b[12]));
        a[11] = x(b[11], vandq_u64(b[12], b[13]));
        a[12] = vbcaxq_u64(b[12], b[14], b[13]);
        a[13] = x(not(b[13]), vorrq_u64(b[14], b[10]));
        a[14] = x(b[14], vandq_u64(b[10], b[11]));
        a[15] = x(b[15], vandq_u64(b[16], b[17]));
        a[16] = x(b[16], vorrq_u64(b[17], b[18]));
        a[17] = x(b[17], vornq_u64(b[19], b[18]));
        a[18] = x(not(b[18]), vandq_u64(b[19], b[15]));
        a[19] = x(b[19], vorrq_u64(b[15], b[16]));
        a[20] = vbcaxq_u64(b[20], b[22], b[21]);
        a[21] = x(not(b[21]), vorrq_u64(b[22], b[23]));
        a[22] = x(b[22], vandq_u64(b[23], b[24]));
        a[23] = x(b[23], vorrq_u64(b[24], b[20]));
        a[24] = x(b[24], vandq_u64(b[20], b[21]));
    }
}

/// Two states in 25 registers, lane $`j`$ of `s0` in element 0 and of `s1` in element 1, with the
/// lanes of [`COMPLEMENTED`] complemented.
#[target_feature(enable = "neon")]
#[inline]
fn load_complemented(s0: &[u64; 25], s1: &[u64; 25]) -> [V; 25] {
    let mut a = [vdupq_n_u64(0); 25];
    for i in 0..12 {
        let j = 2 * i;
        let (x, y) = (load2(s0, j), load2(s1, j));
        a[j] = vzip1q_u64(x, y);
        a[j + 1] = vzip2q_u64(x, y);
    }
    a[24] = vcombine_u64(vdup_n_u64(s0[24]), vdup_n_u64(s1[24]));
    mixed::complement(&mut a);
    a
}

/// The inverse of [`load_complemented`].
#[target_feature(enable = "neon")]
#[inline]
fn store_complemented(s0: &mut [u64; 25], s1: &mut [u64; 25], mut a: [V; 25]) {
    mixed::complement(&mut a);
    for i in 0..12 {
        let j = 2 * i;
        store2(s0, j, vzip1q_u64(a[j], a[j + 1]));
        store2(s1, j, vzip2q_u64(a[j], a[j + 1]));
    }
    s0[24] = vgetq_lane_u64::<0>(a[24]);
    s1[24] = vgetq_lane_u64::<1>(a[24]);
}

/// Keccak-f\[1600\] on two states with [`mixed::round`]. Dispatched on aarch64 processors other
/// than Apple's (see `simd::keccak_f1600_x2`), and tested on every processor with the extension.
#[cfg(any(test, not(target_vendor = "apple")))]
#[target_feature(enable = "neon,sha3")]
pub(super) fn keccak_f1600_x2_mixed(states: &mut [[u64; 25]; 2]) {
    let [s0, s1] = states;
    let mut a = load_complemented(s0, s1);
    for &rc in crate::prng::RC.iter() {
        mixed::round(&mut a, rc);
    }
    store_complemented(s0, s1, a);
}

/// Keccak-f\[1600\] on three states: states 0 and 1 with [`mixed::round`] in vector registers,
/// state 2 with `prng::keccak_round_textbook` in general-purpose registers, round by round in one
/// loop, so that the scalar pipes work while the vector pipes do.
#[target_feature(enable = "neon,sha3")]
pub(super) fn keccak_f1600_x3(states: &mut [[u64; 25]; 3]) {
    let [s0, s1, s2] = states;
    let mut a = load_complemented(s0, s1);
    let mut s = *s2;
    for &rc in crate::prng::RC.iter() {
        mixed::round(&mut a, rc);
        crate::prng::keccak_round_textbook(&mut s, rc);
    }
    store_complemented(s0, s1, a);
    *s2 = s;
}
