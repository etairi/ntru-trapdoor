//! The sampler of $`f`$ and $`g`$: exact $`D_{\mathbb Z,\sigma_{fg}}`$ from a 128-bit reverse
//! CDT of the half-Gaussian (design.md §5.9, step 1).
//!
//! Per coefficient: $`K=\#\{j:V<T_j\}`$ for $`V`$ uniform on $`[0,2^{128})`$ and a sign bit
//! $`b`$; return $`-K`$ if $`b=0`$, $`K`$ if $`b=1`$ and $`K\ne0`$, and draw again if $`b=1`$,
//! $`K=0`$.
//!
//! **Lazy draw** (the construction of the width-3.1 base sampler in `crate::sampler`). One
//! `next_u64` $`w`$ gives $`b=w\bmod2`$ and the top 63 bits $`H=\lfloor w/2\rfloor`$ of
//! $`V=H\cdot2^{65}+L`$. Write $`T_j=H_j\cdot2^{65}+L_j`$. Since
//! $`V<T_j\iff H<H_j\lor(H=H_j\land L<L_j)`$, if $`H`$ equals no $`H_j`$ then
//! $`K=\#\{j:H<H_j\}`$, a binary search over the top bits. Otherwise (probability at most
//! (number of rows)$`\cdot2^{-63}`$, $`2^{-54.6}`$ at the PCS set) the low 65 bits $`L`$ are
//! drawn (`next_u64`, then the low bit of `next_u8`) and $`K`$ is counted on the full $`V`$.
//! So $`(K,b)`$ has exactly the distribution of the full 128-bit draw [derived], and a draw
//! takes 8 PRNG bytes instead of 17 (keygen stage R4; adopted by the integrator).
//!
//! **Why this is exact** [derived]: $`\Pr[K=i]=(T_{i-1}-T_i)/2^{128}`$ with $`T_{-1}=2^{128}`$,
//! the half-Gaussian $`\propto\rho(i)`$ up to the 128-bit rounding of the table. With the sign
//! bit, $`\Pr[z=0]\propto\frac12\rho(0)`$ (only $`b=0,K=0`$) and
//! $`\Pr[z=\pm i]\propto\frac12\rho(i)`$ for $`i\ge1`$; conditioned on not redrawing,
//! $`\Pr[z]\propto\rho(z)`$. The redraw probability is $`\frac12\Pr[K=0]`$, 1.5% at the PCS set.
//!
//! The same construction as fn-dsa's `sample_f` (fn-dsa-kgen/src/gauss.rs:30-81: a sign bit and
//! a half-Gaussian CDT) with four changes: 128-bit entries instead of 15-bit ones, a binary
//! search instead of a full scan, the lazy draw, and no parity conditioning (unsolvable pairs
//! are rejected by NTRUSolve instead). Variable time: the binary search, the rare fallback and
//! the redraw (design.md §10).

use crate::Params;
use crate::prng::Prng;

/// [`crate::hazmat::sample_fg`].
pub(crate) fn sample_fg<P: Prng>(params: &Params, prng: &mut P, out: &mut [i16]) {
    let table = params.fg_table;
    assert!(
        table.len() <= i16::MAX as usize,
        "sample_fg: table too long for i16"
    );
    for x in out.iter_mut() {
        *x = loop {
            let (k, b) = draw_half(table, prng);
            if b == 0 {
                break -k;
            }
            if k != 0 {
                break k;
            }
        };
    }
}

/// One lazy draw of $`(K,b)`$ (module documentation). `table` is non-increasing, as
/// [`Params::validate`] requires, so both predicates below are true on a prefix.
#[inline]
pub(super) fn draw_half<P: Prng>(table: &[u128], prng: &mut P) -> (i16, u64) {
    let w = prng.next_u64();
    let b = w & 1;
    let h = w >> 1;
    // K_H = #{j : H < H_j}; the entries with H_j = H, if any, start at index K_H.
    let mut k = table.partition_point(|&e| ((e >> 65) as u64) > h);
    if k < table.len() && ((table[k] >> 65) as u64) == h {
        // A tie: the low 65 bits of V decide.
        let lo = prng.next_u64();
        let last = (prng.next_u8() & 1) as u128;
        let v = ((h as u128) << 65) | ((lo as u128) << 1) | last;
        k = table.partition_point(|&e| e > v);
    }
    (k as i16, b)
}
