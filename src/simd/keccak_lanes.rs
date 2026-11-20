//! The rounds of Keccak-f\[1600\] on vectors that hold the same lane of several states, one state
//! per 64-bit element: the multi-state kernels (`keccak_x86`, `keccak_neon`) expand
//! [`keccak_theta_rho_pi`] and [`keccak_f1600_lanes`] with their own primitives.

/// θ, ρ and π of one round on `$a: [V; 25]`, where `$a[x + 5y]` holds lane $`(x,y)`$ of every
/// state: the lanes $`B[y+5((2x+3y)\bmod5)]=\mathrm{rol}(A[x+5y]\oplus D[x],r[x,y])`$ as a
/// `[V; 25]` (the textbook form, with the offsets of the portable code), which χ combines by
/// rows of five. The calling module provides, on its vector type `V`:
/// - `xor5(a, b, c, d, e)`: $`a\oplus b\oplus c\oplus d\oplus e`$.
/// - `theta_d(c, e)`: θ's $`D=c\oplus\mathrm{rol}(e,1)`$, for $`c=C[x-1]`$ and $`e=C[x+1]`$.
/// - `theta_xor(a, d)`: $`a\oplus d`$ (lane $`(0,0)`$, whose ρ offset is 0).
/// - `rho::<R, L>(a, d)`: $`\mathrm{rol}(a\oplus d,R)`$, where $`L=64-R`$. The 24 offsets $`R`$
///   are distinct, so `rho` may choose its instructions lane by lane.
macro_rules! keccak_theta_rho_pi {
    ($a:ident) => {{
        let c0 = xor5($a[0], $a[5], $a[10], $a[15], $a[20]);
        let c1 = xor5($a[1], $a[6], $a[11], $a[16], $a[21]);
        let c2 = xor5($a[2], $a[7], $a[12], $a[17], $a[22]);
        let c3 = xor5($a[3], $a[8], $a[13], $a[18], $a[23]);
        let c4 = xor5($a[4], $a[9], $a[14], $a[19], $a[24]);
        let d0 = theta_d(c4, c1);
        let d1 = theta_d(c0, c2);
        let d2 = theta_d(c1, c3);
        let d3 = theta_d(c2, c4);
        let d4 = theta_d(c3, c0);
        [
            theta_xor($a[0], d0),
            rho::<44, 20>($a[6], d1),
            rho::<43, 21>($a[12], d2),
            rho::<21, 43>($a[18], d3),
            rho::<14, 50>($a[24], d4),
            rho::<28, 36>($a[3], d3),
            rho::<20, 44>($a[9], d4),
            rho::<3, 61>($a[10], d0),
            rho::<45, 19>($a[16], d1),
            rho::<61, 3>($a[22], d2),
            rho::<1, 63>($a[1], d1),
            rho::<6, 58>($a[7], d2),
            rho::<25, 39>($a[13], d3),
            rho::<8, 56>($a[19], d4),
            rho::<18, 46>($a[20], d0),
            rho::<27, 37>($a[4], d4),
            rho::<36, 28>($a[5], d0),
            rho::<10, 54>($a[11], d1),
            rho::<15, 49>($a[17], d2),
            rho::<56, 8>($a[23], d3),
            rho::<62, 2>($a[2], d2),
            rho::<55, 9>($a[8], d3),
            rho::<39, 25>($a[14], d4),
            rho::<41, 23>($a[15], d0),
            rho::<2, 62>($a[21], d1),
        ]
    }};
}

/// The 24 rounds on `$a: [V; 25]`: θ, ρ and π by [`keccak_theta_rho_pi`], then χ and ι write the
/// next state back into `$a`. Besides the primitives of `keccak_theta_rho_pi`, the calling module
/// provides:
/// - `chi(a, b, c)`: $`a\oplus(\lnot b\wedge c)`$.
/// - `iota(a, rc)`: $`a\oplus rc`$ in every element.
///
/// The tests compare every kernel with the portable code.
#[cfg_attr(
    all(target_arch = "aarch64", not(target_vendor = "apple"), not(test)),
    allow(
        unused_macros,
        reason = "its aarch64 kernels run on Apple's processors only"
    )
)]
macro_rules! keccak_f1600_lanes {
    ($a:ident) => {
        for &rc in crate::prng::RC.iter() {
            let b = keccak_theta_rho_pi!($a);
            $a[0] = iota(chi(b[0], b[1], b[2]), rc);
            $a[1] = chi(b[1], b[2], b[3]);
            $a[2] = chi(b[2], b[3], b[4]);
            $a[3] = chi(b[3], b[4], b[0]);
            $a[4] = chi(b[4], b[0], b[1]);
            $a[5] = chi(b[5], b[6], b[7]);
            $a[6] = chi(b[6], b[7], b[8]);
            $a[7] = chi(b[7], b[8], b[9]);
            $a[8] = chi(b[8], b[9], b[5]);
            $a[9] = chi(b[9], b[5], b[6]);
            $a[10] = chi(b[10], b[11], b[12]);
            $a[11] = chi(b[11], b[12], b[13]);
            $a[12] = chi(b[12], b[13], b[14]);
            $a[13] = chi(b[13], b[14], b[10]);
            $a[14] = chi(b[14], b[10], b[11]);
            $a[15] = chi(b[15], b[16], b[17]);
            $a[16] = chi(b[16], b[17], b[18]);
            $a[17] = chi(b[17], b[18], b[19]);
            $a[18] = chi(b[18], b[19], b[15]);
            $a[19] = chi(b[19], b[15], b[16]);
            $a[20] = chi(b[20], b[21], b[22]);
            $a[21] = chi(b[21], b[22], b[23]);
            $a[22] = chi(b[22], b[23], b[24]);
            $a[23] = chi(b[23], b[24], b[20]);
            $a[24] = chi(b[24], b[20], b[21]);
        }
    };
}
