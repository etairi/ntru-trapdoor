# Security notes

## Randomness

Every function that needs randomness takes a 32-byte seed. Seeds must be secret, uniformly random
and fresh: draw them from the operating system's CSPRNG. A fixed seed makes a key, or a sample,
public.

`samp_pre` derives its coins from the key, the target and the seed, under the label
`ntru-trapdoor/v1/samp-pre`. A repeated seed with another target therefore still gives
independent randomness, and a repeated (key, target, seed) gives the same preimage.

**Never run the sampler twice on the same key, target and seed under different floating-point
arithmetic**, for example two versions of this crate, or two platforms. The same coins under
slightly different rounding can return two different lattice points whose difference reveals the
secret key (LTYZ25, EUROCRYPT 2025). [`tests/kat.rs`](../tests/kat.rs) pins SampPre's outputs;
any change that alters them must also move the label to `v2`.

## Key distribution: no unusually dense sublattice

Key generation follows DLP14 and Falcon: `f, g` have standard deviation `σ_fg = 1.17 √(q/2d)`,
and a key is accepted only if both of Falcon's Gram–Schmidt norms are at most `1.17 √q`. The
second test also rules out the dense sublattice that overstretched NTRU attacks exploit, whatever
the modulus.

**The attack.** The NTRU lattice `Λ = {(s₀, s₁) : s₀ + h s₁ = 0 mod q}` has rank `2d` and volume
`q^d`; with `h = g/f` it contains the sublattice `R·(g, −f)`. The overstretched NTRU attacks find
that sublattice when it is much denser than `Λ`, that is when `f` and `g` are small compared with
`√q` (Kirchner and Fouque, EUROCRYPT 2017; Ducas and van Woerden, "NTRU Fatigue", ASIACRYPT
2021). For ternary `f, g`, Ducas and van Woerden estimate the threshold at about
`q = 0.004 d^2.484` (a fit over prime `d` from 199 to 499, matrix NTRU): 2^16.9 at `d = 1024`,
below the PCS modulus 2^20 − 3. The threshold grows with the variance of the secret: Hough,
Sandsbråten and Silde (IACR Communications in Cryptology 1(4), 2024, Eq. (3)) fit
`q = 0.0058 σ² d^2.484` over the same range of `d`.

**Proved for every accepted key.**
- The trapdoor basis has Gram–Schmidt norms at most `M = 1.17 √q`. All `2d` norms multiply to
  `q^d`, so the lemma of Pataki and Tural (arXiv:0804.4014, Lemma 1) gives every rank-`d`
  sublattice of `Λ`, in particular `R·(g, −f)`, a volume of at least `(q/M)^d = (√q/1.17)^d`.
- Directly: with `s_j = |f(z_j)|² + |g(z_j)|²` at the `d` roots of `x^d + 1`,
  `vol(R·(g, −f))² = ∏ s_j`, `‖(g, f)‖² = (1/d) Σ s_j`, and Falcon's second squared Gram–Schmidt
  norm is `N₂ = (q²/d) Σ 1/s_j`. By AM–GM, `N₂ ≤ 1.17² q` gives
  `vol(R·(g, −f))^(1/d) ≥ √q/1.17`, and `vol(R·(g, −f))^(1/d) ≤ ‖(g, f)‖ ≤ 1.17 √q`.

So `vol(R·(g, −f))^(1/d)/√q` lies in `[1/1.17, 1.17] ≈ [0.8547, 1.17]`: no rank-`d` sublattice is
denser, per dimension, than `Λ` itself by more than a factor 1.17. For ternary `f, g` at the PCS
`(d, q)` this ratio would be about 0.032.

**What enforces it** is the second norm test, not `σ_fg`: with a looser Gram–Schmidt bound, small
`f, g` could pass at a large `q`. `Params::validate` therefore refuses a bound above
`(1.17 √q)² = 13689 q / 10000`; tighter bounds are accepted. Key generation evaluates `N₂` in `f64`,
with a relative error of about 2^-43, so the per-dimension bound reads
`vol(R·(g, −f))^(1/d) ≥ (1 − 2^-44) √q/1.17`.

**Measured.** On 2000 PCS keys, `vol(R·(g, −f))^(1/d)/√q` lies in `[0.9964, 1.0457]` (mean 1.0245)
and `‖(g, f)‖/√q` in `[1.1223, 1.1700]` (mean 1.1603)
(`cargo run --release --example stats_density`).

**Estimated** (heuristic: Gaussian heuristic, geometric series assumption). Ducas and van
Woerden's own estimator, run by `tools/stats/ntru_fatigue.py` from a checkout of their code
(https://github.com/WvanWoerden/NTRUFatigue), for progressive BKZ with 8 tours per block size:

| instance | expected successful block size | P[dense sublattice found first] |
|---|---|---|
| `pcs`: `q = 2^20 − 3`, `σ² = σ_fg²` (keys before the norm tests) | 948.08 | 0 |
| `pcs-worst`: the worst key the norm tests admit | 873.29 | 0 |
| `falcon1024`: `q = 12289`, its `σ_fg²` | 948.08 | 0 |

"0" means that at no block size did the estimator find a dense-sublattice probability above its
10^-7 threshold. For keys of this distribution `√q` cancels out of the estimates, so recovering a
PCS key is estimated to cost what recovering a Falcon-1024 key costs; Falcon's own estimate is BKZ
block size 936 (specification v1.2, Sect. 2.5.1 and Table 3.3).

**Limits.** The estimates use the Gaussian heuristic for the intersections of `R·(g, −f)` with
the sublattices spanned by the reduced basis, which Ducas and van Woerden leave unvalidated for
intersections of large dimension (their Sect. 4.3), and they do not model the rare "lucky lift"
events. Only the volume bound above is proved.

## Side channels

The library is **not constant time**. SampPre keeps fn-dsa's isochronous structure where it is
cheap: fixed-length table scans, and a Bernoulli test scaled by `σ_min / σ'`, whose acceptance
rate does not depend on `σ'` or on the centre (0.634 at the PCS set's base width, 0.585 at
fn-dsa's). The following run in variable time:
- native `f64` arithmetic, on some CPUs;
- the rejection loops, and the base draw's rare fallback (probability 2^-58.1 per draw);
- key generation: the binary-search CDT, the candidate loop and the big-integer NTRUSolve.

The ring arithmetic uses masks; `inverse` reveals only whether an element is a unit.

## Wiping

Secret keys, expanded keys, PRNG states, preimages and the scratch buffers that hold the trapdoor
are wiped on drop (`zeroize`). The exception is NTRUSolve's big integers: `num-bigint` cannot
wipe its buffers.

## Input validation

Every `RqPoly` constructor checks the ring (`1 ≤ logn ≤ 10`, `3 ≤ q < 2^25`) and returns
`InvalidParams` outside it. `ExpandedKey::new` refuses trapdoors with `fG − gF ≠ q` and keys
whose leaf widths fall outside the base sampler's range. Decoders reject non-canonical
encodings.
