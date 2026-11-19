# Changelog

All notable changes to this crate. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Faster, with the same outputs as 0.1.2: the known-answer tests and the fn-dsa parity tests pass
unchanged, with and without the new feature. Figures below are against 0.1.2 on an Intel Xeon
8581C and an Apple M4.

### Added

- The `simd` feature, off by default: SIMD kernels chosen at run time, in one module that may
  use `unsafe` code (the rest of the crate denies it). The environment variable
  `NTRU_TRAPDOOR_SCALAR` keeps the portable code.
  - x86-64: SampPre runs a copy of its code compiled for x86-64-v3 (AVX2, BMI1, BMI2, LZCNT and
    POPCNT, without FMA), with AVX2 kernels for its FFT passes. Keccak runs with AVX-512, else
    compiled for x86-64-v3. The NTT runs with AVX-512F and AVX2, and so do the CRT, the
    reductions modulo `q`, the norm of the inversion tower, the check of the NTRU equation and
    Verify's norm and final comparison.
  - aarch64: NEON for the FFT passes, the NTT and the same code around it, and the SHA3
    extension for Keccak on Apple's processors.
  - `samp_pre_batch` derives the seeds of several calls together and squeezes the first blocks
    of their PRNG streams together, 8 streams at a time with AVX-512, 4 with AVX2 and 2 on
    Apple's processors. Each call still reads its own stream.
  - With it, on the Xeon, SampPre takes 30 % less time at `Params::PCS` and 28 % less at
    Falcon-1024, Verify 76 % less, a product in `R_q` 73 % less and a batch of 256 calls 43 %
    less. On the M4: 14 %, 14 %, 53 %, 56 % and 19 % less.

### Changed

- SampPre is faster without the feature too: 12 % at `Params::PCS` and 13 % at Falcon-1024 on
  the Xeon, 9 % and 10 % on the M4.
  - The Bernoulli step decides its first byte from an `f64` estimate of the threshold and
    computes FACCT's polynomial only in 2 of 256 trials, with the same result and the same PRNG
    bytes (a lemma, recorded in the sampler's documentation). That byte is compared with the
    estimate in floating point, without a conversion to an integer.
  - ffSampling's nodes of degree 4 are unrolled.
  - fn-dsa's base sampler reads its table by columns, which the compiler vectorizes: the same
    comparison on every row.
- `RqPoly::to_bytes` and `PublicKey::to_bytes` write 32 bits at a time: 2.4 times faster on the
  M4 and 2.7 times on the Xeon, with the same bytes.

## [0.1.2] - 2026-11-19

Faster, with the same outputs: for the same inputs and seeds, every key, preimage, product and
encoding is the same bytes as in 0.1.1. The known-answer tests and the fn-dsa parity tests pass
unchanged, so the seed label stays `ntru-trapdoor/v1/samp-pre`.

### Added

- `PreparedRqPoly::mul_ternary`: the product with a ternary operand (coefficients 0, 1 and
  q − 1), computed modulo a single NTT prime when 2d⌊q/2⌋ is below that prime (q up to about
  2^21 at d = 1024, both presets included). Equal to `PreparedRqPoly::mul`. Its running time
  reveals whether the operand is ternary.
- `PreparedRqPoly::sum_of_products`: a sum of products with prepared operands, added in the NTT
  domain, so that each block of terms needs one inverse NTT per prime and one CRT. Equal to
  adding the products.
- `trapgen_expanded`: `trapgen` that also returns the `ExpandedKey` that its leaf check builds,
  the one `ExpandedKey::new` builds from the new key.
- `samp_pre_batch`: `samp_pre` for many targets and seeds, with the same results entry by entry.
- The `parallel` feature, off by default, with the optional dependency `rayon`.
  `samp_pre_batch` then runs on rayon's thread pool.

### Changed

- SampPre is faster: by 6 to 8 % on an Intel Xeon 8581C and by 8 to 12 % on an Apple M4, at both
  presets.
  - The SHAKE256 reads inline into the sampler. A read across a block boundary copies the bytes
    on each side instead of reading them one at a time.
  - ffSampling runs the work between its two recursive calls as one pass, and SampPre computes
    its target and its lattice point in one pass each, with the same IEEE operations on each
    value as before.
- Products in `R_q` are faster, by about 12 % on the Xeon (Verify and the inversion by 10 %) and by
  3 to 9 % on the M4. The inverse NTT no longer halves at every layer, and the factor 1/n moves
  into the pointwise product. Prepared operands are stored scaled, so that a prepared product
  needs one Montgomery multiplication per point instead of two. The NTT layers run over
  fixed-size blocks, with the two smallest layers merged.

## [0.1.1] - 2026-11-02

Documentation only: the code does not change.

### Documentation

- The README writes its formulas in Unicode and HTML instead of GitHub's `` $`…`$ `` math, which
  crates.io shows as raw LaTeX between dollar signs.

## [0.1.0] - 2026-11-02

First version. NTRU trapdoors in the DLP14 / Falcon convention for the lattice instantiation of the
predicate credential system (PCS): `R_q` arithmetic for `q = 5 mod 8`, TrapGen, SampPre at
arbitrary targets, Verify, and key encodings, at the PCS set and at Falcon-1024's parameters.

Documentation: a README in the usual crate shape, and the detailed notes in `docs/` (distribution
and precision, security, testing and benchmarks). The analysis scripts behind those notes are in
`tools/` (`tools/stats/`, `tools/ntru_sizes.py`, `tools/constants.py`), and
`examples/quickstart.rs` runs the README's example.

The entries below record what changed after the adversarial review and the statistics study of
2026-10-03, and after the overstretched-NTRU check of 2026-10-04 (marked). Earlier states of the
crate were never released; the label `ntru-trapdoor/v1/samp-pre` denotes this version.

### Fixed

- **SamplerZ's Bernoulli step no longer over-weights the far tail** (stats finding F1, high).
  fn-dsa's `ber_exp` saturates its shift at 63, so for `x >= 64 ln 2` it accepted with
  probability 2^-64 where the exact value is smaller, down to 2^-133 at the PCS leaves. That made
  the order-256 Rényi divergence of one SampPre call about 2^14 nats. The threshold is now 0
  whenever the shift reaches 64 (a mask, no branch). The per-trial acceptance changes by less than
  2^-124.9 and the byte consumption is unchanged; at Falcon-1024 the outcome differs from fn-dsa's
  with probability about 2^-96 per trial.
- **`ExpandedKey::new` validates the key** (review finding 1, medium). A key assembled with
  `SecretKey::from_parts` is checked with `SecretKey::validate` (fG - gF = q, the encoding widths,
  f invertible) before the tree is built; keys from `trapgen` and `SecretKey::from_bytes` carry a
  flag that skips the repeated check. Before, an inconsistent trapdoor gave "preimages" with
  `A s != t` that expose the sampler's integer coordinates.
- **`RqPoly` refuses rings it cannot compute in** (review finding 2, medium). Every constructor
  checks `1 <= logn <= 10` and `3 <= q < 2^25` and returns `Error::InvalidParams`; before, larger
  moduli gave wrong products and wrong `Some` inverses, and some parameters panicked.
- **`RqPoly::inverse` and `is_invertible` are exact for a composite modulus too**: the scalar at the
  bottom of the norm tower falls back to the extended Euclidean algorithm when Fermat's inverse
  fails (for a prime `q` only when the element is not a unit).
- `SecretKey::validate` (and so `SecretKey::from_bytes`) no longer leaves copies of the trapdoor
  in freed memory (review finding 3): see the NTT check below.

### Changed

- **`Params::validate` refuses a Gram–Schmidt bound above `(1.17 √q)² = 13689 q / 10000`**
  (2026-10-04). The second norm test, not `σ_fg`, is what rules out the dense sublattice that
  overstretched NTRU attacks exploit: with the bound, `vol(R·(g, −f))^(1/d)` lies within a factor
  1.17 of `√q` (security notes, "Key distribution"). A custom set may tighten the bound but no
  longer loosen it. Both presets sit exactly on the bound; no output changes.
- **The width-3.1 base table is rounded at 192 bits** (stats finding F2, medium): `BASE_3_1_192`,
  the same 41 rows as before, each rounded at 192 instead of 128 bits. The last row's relative
  error falls from +24% to 1.4% (the cut tail), and the sampler's own Rényi term per call from
  2^-57.5 to 2^-85.3, the floor set by FACCT's 2^-51 relative error. `BASE_3_1_128` is gone.
- **The base draw scans 28 entries instead of 41** (review finding 7): the 13 entries whose top 63
  bits are 0 can only tie, which one `H == 0` test covers. Same outputs, same fixed length per
  draw; SampPre at the PCS set takes 161 µs instead of 174 µs. The fallback now reads the low 129
  bits (25 bytes per draw instead of 17, with probability at most 2^-58.1).
- **Exact check of fG - gF = q with the two-prime NTT** (eight forward NTTs, residues wiped)
  instead of big-integer Kronecker products, in key generation and `SecretKey::validate`: the
  check takes about 41 µs instead of about 170 µs at d = 1024, so `validate` takes 72 µs instead
  of 200 µs and decoding a secret key 77 µs instead of 206 µs. The big-integer version remains as
  the tests' oracle.
- **Verify multiplies with a prepared `h`**: `PublicKey` keeps `h` in NTT form after the first use,
  so a call computes four NTTs instead of six, and `s mod q` uses a Barrett reduction instead of a
  division per coefficient: 30.4 µs -> 20.8 µs.
- **Rounding guard in SampPre** (review finding 4): an attempt whose `f64` lattice point lies
  farther than 2^-10 from an integer vector in some coordinate is discarded. The measured margin
  is about 2^26 (no valid key trips it). `hazmat::samp_pre_with` now returns `Option<Preimage>`.
- `RqPoly::zero` returns `Result<RqPoly, Error>`, like the other constructors.
- `Params::validate` rejects `samppre_max_attempts > 256` (the attempt counter is one byte).
- `trapgen` and `trapgen_with_stats` require `Params::check_leaf_window` (review finding 6): the
  window `[sigma / sqrt(B), sigma sqrt(B) / q]` of possible leaf widths, `B` the Gram-Schmidt bound,
  must lie inside the base sampler's range. The review proved that every leaf value lies in
  `[q^2/M, M]` with `M = max(N1, N2)`, so under the window check the leaf rejection of key
  generation fires only through rounding at the boundary. `trapgen_from_fg` does not require it,
  so that the leaf rejection stays testable.
- `KeygenReject` is `#[non_exhaustive]`; `verify` is `#[must_use]`.
- `Debug` of `RqPoly` (and of the new `PreparedRqPoly` and of `PublicKey`) prints the modulus and
  the degree only, not coefficients (an element may be secret, such as `f^-1`).
- The zeroize dependency no longer enables its `derive` feature (unused), which removes syn and
  quote from the library's build.

### Added

- `PreparedRqPoly`, `RqPoly::prepare` and `PublicKey::h_prepared`: products with a fixed operand
  (the public key, or the PCS's public parameters) in four NTTs instead of six.
- `Params::leaf_window` and `Params::check_leaf_window`.
- `tests/kat.rs`: known answers of TrapGen and SampPre at both presets. A change of SampPre's
  answers must come with a new seed-derivation label (LTYZ25: the same coins under different
  arithmetic leak the key).
- `tests/hardening.rs`: regression tests of the review and statistics findings through the public
  API (inconsistent trapdoors, invalid rings, leaf windows, the far-tail trial, API details).
- Unit tests: the precomputed ffLDL tree against a copy of fn-dsa's per-attempt recursion, bit for
  bit (design claim D1, which output parity alone cannot establish); the NTT check of the NTRU
  equation against the big-integer oracle; the far tail of `ber_exp`; prepared products; inverses
  for composite moduli; the rounding guard; Verify against the plain formula.
- `CHANGELOG.md`.
- (2026-10-04) `examples/stats_density.rs` (the density of `R·(g, −f)` over generated keys, by a
  direct DFT) and `tools/stats/ntru_fatigue.py` (Ducas and van Woerden's estimator, read from a
  checkout of their code), behind the key-distribution section of the security notes; the
  regression test H-6 in `tests/hardening.rs`.

### Documentation

- (2026-10-04) The API documentation renders its formulas: `docs/katex-header.html` (KaTeX 0.17.0
  from jsDelivr with SRI hashes, the header of the PCS and Jali crates) is passed to rustdoc by the
  `cargo docs` alias and by `[package.metadata.docs.rs]`. All 903 inline formulas of `src/` parse
  with KaTeX 0.17.0.
- (2026-10-04) Security notes: a key-distribution section. Every accepted key has
  `vol(R·(g, −f))^(1/d)/√q` in `[1/1.17, 1.17]` (proved: the lower end by Pataki–Tural, or by
  AM–GM on the second norm test; the upper end by AM–GM and the first test), so the key's lattice
  lacks the unusually dense sublattice that overstretched NTRU attacks exploit, although the PCS
  modulus is above the ternary fatigue point. With measurements on 2000 keys and the authors'
  estimator (heuristic: expected block size 948, the same as for Falcon-1024 keys, and no
  dense-sublattice event above its 10^-7 threshold). README: a note on the bound.
- README: performance table, precision and distribution evidence with the Rényi budget per number
  of calls, the leaf-range proposition, and the limitations.
- Corrected statements: output parity with fn-dsa certifies decisions and constants (to about
  10^-8 relative), not IEEE bit patterns; the rounding margin of SampPre is measured (2^-26.9 at
  the PCS set), not guessed (2^-15); `solve.rs` says that another FFT may return another, equally
  valid (F, G); NOTICE cites falcon-rust `src/math.rs:68-216`; the table generator says
  "non-increasing".
