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
