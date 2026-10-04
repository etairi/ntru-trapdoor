# Distribution and precision

This note records what is exact in the sampler, what was measured, and what is not established.
Measurements were taken on an Apple M4. The analysis scripts are in
[`tools/stats/`](../tools/stats), and the replay tools in [`examples/`](../examples). The SamplerZ
and SampPre statistics below were collected before fixes F1 and F2 (end of this note). Those fixes
change outputs only on events of probability below 2^-57 per draw, and the statistical tests of
the test suite give identical results on the final code. The precision replay was repeated on the
final code.

## Exact parts

- **Base sampler.** The width-3.1 half-Gaussian of the PCS set uses the 41 rows of its reverse
  CDT, each rounded at 192 bits, so the base draw `K` is at most 41. Every `Pr[K = k]` with
  `k ≤ 40` has a relative error below 2^-69. `Pr[K = 41]` also carries the omitted tail
  `Pr[K > 41] ≈ 2^-134.5`, which is 2^-6.2 of its own value. The table is interval-certified by
  its generator ([`tools/gauss_tables.py`](../tools/gauss_tables.py)) and equals an independent
  mpmath computation.
- **The Bernoulli test (`BerExp`).** fn-dsa saturates the shift of its exponential at 63, which
  over-weights far-tail outputs by up to 2^69. This crate rejects there instead (finding F1
  below), so no acceptance threshold exceeds the exact one by more than the relative error of the
  63-bit exponential, about 2^-51.
- **The `f, g` sampler.** 128-bit CDTs, drawn exactly; statistical distance at most 2^-106.7 per
  key.
- **Leaf widths.** For every key that passes Falcon's two norm tests, every leaf width of the
  ffLDL tree lies in `Params::leaf_window()`: the leaf values lie in `[q²/M, M]` with
  `M = max(N₁, N₂)`, because each split of the tree produces arithmetic and harmonic means of its
  parent's values, and `d₀₀ d₁₁ = q²` at the root (proved, and checked on 600 keys). Key
  generation requires the window to lie inside the base sampler's range, and
  `ExpandedKey::new` still checks every leaf.

## Measured

- **SamplerZ** against the exact `D_{Z,σ',c}`: 70 pairs `(σ', c)` across the leaf range of the
  PCS set, 10^8 draws each. The χ² p-values are uniform (Kolmogorov–Smirnov p = 0.50).
- **SampPre** at the PCS set: 3.3·10^6 preimages on 16 keys. Their law agrees with the coset
  Gaussian of standard deviation σ in norm (`E‖s‖² / (2dσ²)` = 1.00004 and 0.99997 in two
  batches), in coordinates, in random directions and in key-adapted directions, to the resolution
  of the sample. At toy sizes the crate matches the exactly enumerated coset Gaussian (χ²
  p = 0.574) and Sage's `DiscreteGaussianDistributionLatticeSampler` on the same coset
  (p = 0.699).
- **Floating-point precision.** 8 keys × 133 targets = 1064 calls (2,179,072 leaves), replayed at
  200 bits. The output `s` is exact on every call.

  | | max \|Δ\|/σ' | rms | 99.99 % quantile |
  |---|---|---|---|
  | all leaves (structured targets included) | 2^-35.44 | 2^-40.14 | 2^-36.80 |
  | uniform targets only | 2^-36.81 | | |
  | leaf widths, relative error | 2^-45.93 | | |

## Rényi-divergence budget

For applications that argue about the sampler through the Rényi divergence, as Falcon does: order
`a = 256`, per SampPre call, in nats (`ln R`). The per-leaf terms add.

| Term | per call | × 2^48 calls | × 2^57 calls |
|---|---|---|---|
| leaf centres, `Σᵢ a Δᵢ² / (2σ'ᵢ²)`, worst of 1064 calls | 2^-57.86 | 2^-9.86 | 2^-0.86 |
| leaf centres, uniform targets: worst / mean | 2^-59.93 / 2^-62.37 | | |
| leaf widths | 2^-76.24 | | |
| the sampler itself (192-bit table, BerExp fixed), worst of 8 keys | 2^-85.26 | | |
| **total** | **2^-57.86** | **0.0011 nats** | **0.55 nats** |

Probability preservation gives `Pr_real(E) ≤ (R · Pr_ideal(E))^((a−1)/a)`. So an event of ideal
probability 2^-128 loses 0.50 bits at 2^48 calls per key, and 1.29 bits at 2^57 calls.

## Not established

- **A worst case over all targets and sampling paths.** The figures above are the worst of 1064
  measured calls; structured targets were 1.4 bits worse than uniform ones. A proof needs an a
  priori bound on the floating-point error of the ffSampling recursion. If every leaf had the
  largest error seen, the centre term would be 2^-52.9 per call; that is not a bound either.
- **A per-call statistical distance of 2^-128.** Native `f64` does not allow one.
- The composition over calls and the probability-preservation step are standard results (Bai,
  Langlois, Lepoint, Stehlé and Steinfeld, ASIACRYPT 2015; Prest, ASIACRYPT 2017).

## Findings fixed before release

- **F1: far-tail acceptance in `BerExp`.** fn-dsa caps the shift of its exponential at 63, so
  trials far in the tail were accepted with probability 2^-64 where the exact value is far
  smaller. That made the order-256 Rényi divergence of SampPre about 2^14 nats per call, with no
  useful bound. The crate now rejects those trials. The difference shows only on events of
  probability below 2^-57 per draw, so fn-dsa parity is unaffected in practice.
- **F2: the last row of the base table.** At 128 bits, the last row (`K = 41`) of the width-3.1
  table had a +24 % relative error, which dominated the sampler's Rényi term once F1 was fixed.
  The table is now rounded at 192 bits.
