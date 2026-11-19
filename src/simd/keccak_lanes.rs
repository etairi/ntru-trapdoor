//! The 24 rounds of Keccak-f\[1600\] on vectors that hold the same lane of several states, one
//! state per 64-bit element: the multi-state kernels (`keccak_x86`, `keccak_neon`) expand
//! [`keccak_f1600_lanes`] with their own primitives.

/// The 24 rounds on `$a: [V; 25]`, where `$a[x + 5y]` holds lane $`(x,y)`$ of every state. The
/// calling module provides, on its vector type `V`:
/// - `xor5(a, b, c, d, e)`: $`a\oplus b\oplus c\oplus d\oplus e`$.
/// - `theta_d(c, e)`: θ's $`D=c\oplus\mathrm{rol}(e,1)`$, for $`c=C[x-1]`$ and $`e=C[x+1]`$.
/// - `theta_xor(a, d)`: $`a\oplus d`$ (lane $`(0,0)`$, whose ρ offset is 0).
/// - `rho::<R, L>(a, d)`: $`\mathrm{rol}(a\oplus d,R)`$, where $`L=64-R`$.
/// - `chi(a, b, c)`: $`a\oplus(\lnot b\wedge c)`$.
/// - `iota(a, rc)`: $`a\oplus rc`$ in every element.
///
/// θ, ρ and π go into fresh values $`B[y+5((2x+3y)\bmod5)]=\mathrm{rol}(A[x+5y]\oplus
/// D[x],r[x,y])`$ (the textbook form, with the offsets of the portable code), then χ and ι write
/// the next state back into `$a`. The tests compare every kernel with the portable code.
macro_rules! keccak_f1600_lanes {
    ($a:ident) => {
        for &rc in crate::prng::RC.iter() {
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

            let b0 = theta_xor($a[0], d0);
            let b1 = rho::<44, 20>($a[6], d1);
            let b2 = rho::<43, 21>($a[12], d2);
            let b3 = rho::<21, 43>($a[18], d3);
            let b4 = rho::<14, 50>($a[24], d4);
            let b5 = rho::<28, 36>($a[3], d3);
            let b6 = rho::<20, 44>($a[9], d4);
            let b7 = rho::<3, 61>($a[10], d0);
            let b8 = rho::<45, 19>($a[16], d1);
            let b9 = rho::<61, 3>($a[22], d2);
            let b10 = rho::<1, 63>($a[1], d1);
            let b11 = rho::<6, 58>($a[7], d2);
            let b12 = rho::<25, 39>($a[13], d3);
            let b13 = rho::<8, 56>($a[19], d4);
            let b14 = rho::<18, 46>($a[20], d0);
            let b15 = rho::<27, 37>($a[4], d4);
            let b16 = rho::<36, 28>($a[5], d0);
            let b17 = rho::<10, 54>($a[11], d1);
            let b18 = rho::<15, 49>($a[17], d2);
            let b19 = rho::<56, 8>($a[23], d3);
            let b20 = rho::<62, 2>($a[2], d2);
            let b21 = rho::<55, 9>($a[8], d3);
            let b22 = rho::<39, 25>($a[14], d4);
            let b23 = rho::<41, 23>($a[15], d0);
            let b24 = rho::<2, 62>($a[21], d1);

            $a[0] = iota(chi(b0, b1, b2), rc);
            $a[1] = chi(b1, b2, b3);
            $a[2] = chi(b2, b3, b4);
            $a[3] = chi(b3, b4, b0);
            $a[4] = chi(b4, b0, b1);
            $a[5] = chi(b5, b6, b7);
            $a[6] = chi(b6, b7, b8);
            $a[7] = chi(b7, b8, b9);
            $a[8] = chi(b8, b9, b5);
            $a[9] = chi(b9, b5, b6);
            $a[10] = chi(b10, b11, b12);
            $a[11] = chi(b11, b12, b13);
            $a[12] = chi(b12, b13, b14);
            $a[13] = chi(b13, b14, b10);
            $a[14] = chi(b14, b10, b11);
            $a[15] = chi(b15, b16, b17);
            $a[16] = chi(b16, b17, b18);
            $a[17] = chi(b17, b18, b19);
            $a[18] = chi(b18, b19, b15);
            $a[19] = chi(b19, b15, b16);
            $a[20] = chi(b20, b21, b22);
            $a[21] = chi(b21, b22, b23);
            $a[22] = chi(b22, b23, b24);
            $a[23] = chi(b23, b24, b20);
            $a[24] = chi(b24, b20, b21);
        }
    };
}
