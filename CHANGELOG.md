# Changelog

All notable changes to this crate. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate follows
[Semantic Versioning](https://semver.org/) once published.

## [0.1.0] - unreleased

First version. NTRU trapdoors in the DLP14 / Falcon convention for the lattice instantiation of the
predicate credential system (PCS): `R_q` arithmetic for `q = 5 mod 8`, TrapGen, SampPre at
arbitrary targets, Verify, and key encodings, at the PCS set and at Falcon-1024's parameters.

Documentation: a README in the usual crate shape, and the detailed notes in `docs/` (distribution
and precision, security, testing and benchmarks). The analysis scripts behind those notes are in
`tools/` (`tools/stats/`, `tools/ntru_sizes.py`, `tools/constants.py`), and
`examples/quickstart.rs` runs the README's example.

The entries below record what changed after the adversarial review and the statistics study of
2026-10-03. Earlier states of the crate were never released; the label `ntru-trapdoor/v1/samp-pre`
denotes this version.

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

### Documentation

- README: performance table, precision and distribution evidence with the Rényi budget per number
  of calls, the leaf-range proposition, and the limitations.
- Corrected statements: output parity with fn-dsa certifies decisions and constants (to about
  10^-8 relative), not IEEE bit patterns; the rounding margin of SampPre is measured (2^-26.9 at
  the PCS set), not guessed (2^-15); `solve.rs` says that another FFT may return another, equally
  valid (F, G); NOTICE cites falcon-rust `src/math.rs:68-216`; the table generator says
  "non-increasing".
