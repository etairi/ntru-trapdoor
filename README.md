# ntru-trapdoor

ntru-trapdoor is a Rust library for NTRU lattice trapdoors. It generates NTRU key pairs together
with a short trapdoor basis, and samples short Gaussian preimages with Falcon's fast Fourier
sampler. Unlike a Falcon signature library, it is parametric:

- the sampler takes **any target** in
  *R*<sub>*q*</sub> = ℤ<sub>*q*</sub>\[*x*\]/(*x*<sup>*d*</sup> + 1), not only the hash of a message;
- the modulus can be **any prime** *q* < 2<sup>25</sup>, including *q* ≡ 5 (mod 8), where
  *x*<sup>*d*</sup> + 1 has no full NTT;
- the Gaussian width is the one your application needs.

It also provides exact arithmetic in *R*<sub>*q*</sub>: multiplication, inversion and canonical
encodings. It was written for the lattice instantiation of a predicate credential system, whose
credentials are signatures from vanishing short integer solutions (Dubois, Klooß, Lai and Woo,
PKC 2025), and ships that parameter set next to Falcon-1024's.

## Status

ntru-trapdoor is research code, version 0.1.

- **Not audited.** No independent review of the code has taken place.
- **Checked against fn-dsa.** At Falcon-1024's parameters, the preimage sampler reproduces Thomas
  Pornin's [fn-dsa](https://github.com/pornin/rust-fn-dsa) signatures bit for bit.
- **No constant-time guarantee.** The sampler keeps Falcon's isochronous structure, but native
  floating point, the rejection loops and key generation run in variable time. See the
  [security notes](docs/security.md).
- **Distribution.** The sampling tables and rejection steps are exact. The floating-point error
  of the fast Fourier sampler is measured, not bounded a priori. See
  [distribution and precision](docs/precision.md).
- **Randomness.** Every function that needs randomness takes a 32-byte seed, which must be
  secret, uniform and fresh. Never sample twice with the same key, target and seed under two
  versions of this crate: different floating-point rounding of the same coins can leak the key.
- **Unstable formats.** The API and the encodings may change between versions.

## Installation

ntru-trapdoor is on [crates.io](https://crates.io/crates/ntru-trapdoor). It needs Rust 1.95 or
later (edition 2024).

```toml
[dependencies]
ntru-trapdoor = "0.1"
```

The optional `parallel` feature runs `samp_pre_batch` on all cores, through rayon:
`ntru-trapdoor = { version = "0.1", features = ["parallel"] }`. The optional `simd` feature
chooses SIMD kernels at run time, with the same outputs, in the crate's only module with
`unsafe` code:
- On x86-64, SampPre runs a copy of its code compiled for x86-64-v3 (AVX2, BMI1, BMI2, LZCNT
  and POPCNT, no FMA) with AVX2 FFT passes. Keccak runs with AVX-512 or the x86-64-v3
  instructions, and the NTT, the CRT and Verify with AVX-512F and AVX2.
- On aarch64, the FFT passes, the NTT, the CRT and Verify run with NEON, and Keccak with the
  SHA3 extension on Apple's processors.
- `samp_pre_batch` derives the seeds of several calls and squeezes their PRNG streams
  together, 8 at a time with AVX-512, 4 with AVX2 and 2 on Apple's processors.

The environment variable `NTRU_TRAPDOOR_SCALAR` keeps the portable code.

Without the `simd` feature the crate forbids `unsafe` code. Its only dependencies are `zeroize`
and, for key generation, `num-bigint`, `num-integer` and `num-traits`.

## Quick start

Generate a key pair, then sample a short preimage of a target and verify it. The fixed seeds keep
the example reproducible; real keys and samples need fresh random seeds.

```rust
use ntru_trapdoor::{Error, ExpandedKey, Params, RqPoly, samp_pre, trapgen, verify};

fn main() -> Result<(), Error> {
    let params = Params::PCS;
    let (pk, sk) = trapgen(&params, &[7u8; 32])?;
    // The sampling tree, built once per key and shared by every call.
    let ek = ExpandedKey::new(&sk)?;
    // Any target in R_q.
    let t = RqPoly::from_coeffs(&params, &vec![12345; params.n()])?;
    // A short s = (s0, s1) with s0 + h·s1 = t mod q and ‖s‖² ≤ β².
    let s = samp_pre(&ek, &t, &[9u8; 32])?;
    assert!(verify(&pk, &t, &s, params.beta_sq));
    Ok(())
}
```

[`examples/quickstart.rs`](examples/quickstart.rs) runs this example.

## API overview

| Item | Purpose |
|---|---|
| `Params` | A parameter set: the presets `Params::PCS` and `Params::FALCON_1024`, or your own `(d, q, widths)` |
| `trapgen(params, seed)` | A key pair `(PublicKey, SecretKey)` |
| `trapgen_expanded(params, seed)` | The key pair and its `ExpandedKey`, built once |
| `PublicKey` | `h = g/f mod q`, so that `A = [1 \| h]` |
| `SecretKey` | The trapdoor `(f, g, F, G)` with `fG − gF = q`; wiped on drop |
| `ExpandedKey::new(&sk)` | The FFT basis and the sampling tree, built once per key; `Send + Sync` |
| `samp_pre(&ek, &t, &seed)` | A short preimage `s` of `t`, for a fresh seed per call |
| `samp_pre_batch(&ek, &items)` | `samp_pre` for many `(t, seed)` pairs, on all cores with the `parallel` feature |
| `verify(&pk, &t, &s, beta_sq)` | The norm bound and the linear relation |
| `RqPoly`, `PreparedRqPoly` | Elements of `R_q`: exact `mul`, `inverse`, `is_invertible`, canonical encodings, and for a prepared operand `mul_ternary` and `sum_of_products` |

Errors are values of `Error`. The crate documentation gives the details.

## Parameter sets

| | `Params::PCS` | `Params::FALCON_1024` |
|---|---|---|
| ring degree `d` | 1024 | 1024 |
| modulus `q` | 1,048,573 = 2^20 − 3 (prime, ≡ 5 mod 8) | 12,289 |
| standard deviation `σ` of each coefficient of `s` | 2656.42 | 168.39 |
| GPV width `ς = σ√(2π)` | 6658.66 | 422.09 |
| smoothing `ε` | 2^-128 | 2^-36 |
| norm bound `⌊β²⌋` | 90,803,788,942 | 70,265,242 |
| leaf widths of the sampling tree | 2.217 to 3.035 | 1.298 to 1.777 |
| base sampler | width 3.1, 192-bit table | width 1.8205, fn-dsa's table |
| public key / secret key | 2560 B / 5889 B | 1792 B / 4097 B |

Widths follow Falcon's convention, the standard deviation `σ`. Papers in the GPV tradition use
the width parameter `ς`, with `σ = ς/√(2π)`. `Params::FALCON_1024` has fn-dsa's constants bit for
bit and exists for the parity tests.

A custom set may tighten Falcon's Gram–Schmidt bound `(1.17 √q)²`, but `Params::validate` refuses
to loosen it. With the bound, every accepted key has `vol(R·(g, −f))^(1/d)` within a factor 1.17 of
`√q`, whatever the modulus, so the key's lattice has none of the unusually dense sublattices that
overstretched NTRU attacks exploit. The [security notes](docs/security.md) give the proof, the
measurements and the estimates.

## Performance

Indicative medians on an Apple M4, one thread, rustc 1.100.0, release profile, with other
applications running. [Testing and benchmarks](docs/testing.md) has the full tables, an Intel
Xeon (x86-64) table and the setup.

| Operation | PCS | Falcon-1024 | PCS, `simd` | Falcon-1024, `simd` |
|---|---|---|---|---|
| key generation (`trapgen`) | 5.9 ms | 4.1 ms | 5.8 ms | 4.0 ms |
| sampling tree (`ExpandedKey::new`) | 52 µs | 52 µs | 49 µs | 49 µs |
| preimage (`samp_pre`) | 133 µs | 163 µs | 130 µs | 157 µs |
| `verify` | 20 µs | 20 µs | 9.3 µs | 9.4 µs |
| `R_q` multiplication / inversion | 27 µs / 55 µs | 26 µs / 56 µs | 11 µs / 28 µs | 11 µs / 28 µs |

On the Xeon, with `simd`, a preimage takes 204 µs at `Params::PCS` against 255 µs without it,
and Verify 10.5 µs against 43 µs. For comparison, fn-dsa 0.4.0 signs a Falcon-1024 message in
213 µs on the M4 and 434 µs on the Xeon, where this crate samples a Falcon-1024 preimage, tree
included, in 203 µs and 326 µs with `simd`. Sampling scales with threads that share one key:
about 36,000 preimages per second on 8 threads of the M4 at `Params::PCS` with `simd`, and
`samp_pre_batch` runs 256 calls in 29 ms on one M4 core with `simd`.

## Documentation

- [Distribution and precision](docs/precision.md): what is exact, what was measured, and a
  Rényi-divergence budget.
- [Security notes](docs/security.md): randomness, the key distribution (overstretched NTRU),
  side channels and wiping.
- [Testing and benchmarks](docs/testing.md).
- [Changelog](CHANGELOG.md).

The API documentation writes its formulas as KaTeX, inside code (`` $`…`$ ``). `cargo doc` leaves
them readable as LaTeX; `cargo docs` builds the documentation with the KaTeX header in
`docs/katex-header.html`, as docs.rs does, so that they render (KaTeX is loaded from a CDN).

## Credits

This crate ports code from the following works. [`NOTICE`](NOTICE) gives the details and the
licence texts.

- **[fn-dsa](https://github.com/pornin/rust-fn-dsa)** by Thomas Pornin (The Unlicense), commit
  `0629bb1`, from its portable code paths: the `f64` FFT, LDL and ffSampling, the sampler
  (`SamplerZ` with `BerExp`, whose far-tail rejection is changed here), SHAKE256 and its PRNG,
  the 31-bit NTT, and the structure of key generation.
- **[falcon-rust](https://github.com/aszepieniec/falcon-rust)** by Alan Szepieniec (MIT), v0.3.1:
  the structure of the big-integer NTRUSolve and of its Babai reduction.

## References

- P.-A. Fouque, J. Hoffstein, P. Kirchner, V. Lyubashevsky, T. Pornin, T. Prest, T. Ricosset,
  G. Seiler, W. Whyte and Z. Zhang. Falcon: Fast-Fourier Lattice-based Compact Signatures over
  NTRU. Specification v1.2, 2020.
- L. Ducas, V. Lyubashevsky and T. Prest. Efficient Identity-Based Encryption over NTRU Lattices.
  ASIACRYPT 2014. (The NTRU trapdoor and its key generation.)
- A. Dubois, M. Klooß, R. W. F. Lai and I. K. Y. Woo. Lattice-Based Proof-Friendly Signatures
  from Vanishing Short Integer Solutions. PKC 2025. Cryptology ePrint Archive, Paper 2025/356.
  (The PCS parameter set.)

## License

MIT; see [LICENSE](LICENSE).
