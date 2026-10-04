#!/usr/bin/env sage -python
"""Fixtures of the sign tests: PCS keys (Sage), and a 200-bit replay of ffSampling.

Adapted from an earlier precision experiment (same author), with fn-dsa's leaf order (design.md §0 finding 3): at
each fn-dsa `logn = 1` node the sampler draws t1[0] (the even part) before t1[1] (the odd
part), i.e. at each leaf of the Falcon-spec recursion below the t0 side before the t1 side.
At those leaves l10 is 0 (the split of a self-adjoint polynomial has a zero odd part), so the
order does not change the mathematics, only the order of the recorded decisions.

Keys: f, g <- D_{Z, 1.17 sqrt(q/2n)} (Sage's DiscreteGaussianDistributionIntegerSampler, the
standard-deviation convention), Falcon's Gram-Schmidt acceptance at 1.17 sqrt(q), NTRUSolve by
the independent exact Sage solver of tools/ntru_sizes.py,
exact check f G - g F = q, and h = g / f mod q computed in Sage.

Replay: target hm uniform in [0, q); the same ffLDL + ffSampling (Falcon's Alg. 9/11, all n
complex evaluation points) runs at 200-bit precision (mpmath). The integer decisions z come from
a seeded perturbation of the 200-bit centres (z = round(c + sigma' N(0,1))); the Rust test feeds
them back through a recording LeafSampler and compares its f64 centres and widths with the
200-bit ones, and its final s with the exact s = (hm, 0) - z B computed here over Z.

Output (plain text, decimal integers; each 200-bit centre as a double-double hi + lo and each
200-bit width rounded to binary64, as IEEE-754 bit patterns in hex):
  tests/fixtures/sign/pcs_keys.txt         3 PCS keys: f g F G h, and the 200-bit leaf range
  tests/fixtures/sign/precision_pcs.txt    2 replay cases at the PCS set
  tests/fixtures/sign/precision_falcon.txt 1 replay case at Falcon-1024 (its own Sage key)

Reproduce (from the crate root; about 20 s):
  sage -python tools/sign/precision_fixture.sage.py tests/fixtures/sign
Seeds: PCS keys 22000 + k, Falcon key 11000; targets random.Random(seed*1000003 + k*101 + t).
"""
import math
import os
import random
import struct
import sys

from mpmath import mp, mpf, mpc, nstr

NS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")  # tools/, which holds ntru_sizes.py
sys.path.insert(0, NS)
import ntru_sizes as ns  # noqa: E402  (exact Sage NTRUSolve and GS norm)
from sage.all import GF, PolynomialRing, set_random_seed  # noqa: E402
from sage.stats.distributions.discrete_gaussian_integer import \
    DiscreteGaussianDistributionIntegerSampler  # noqa: E402

PREC = 200
mp.prec = PREC
N = 1024


class HP:
    def __init__(self):
        self.cache = {}

    def num(self, x):
        return mpc(int(x), 0)

    def zeta(self, n):
        if n not in self.cache:
            self.cache[n] = [mp.expjpi(mpf(2 * j + 1) / n) for j in range(n // 2)]
        return self.cache[n]


def fft(B, c):
    n = len(c)
    if n == 1:
        return [c[0]]
    F0 = fft(B, c[0::2])
    F1 = fft(B, c[1::2])
    z = B.zeta(n)
    h = n // 2
    out = [None] * n
    for j in range(h):
        t = z[j] * F1[j]
        out[j] = F0[j] + t
        out[j + h] = F0[j] - t
    return out


def split(B, F):
    n = len(F)
    h = n // 2
    z = B.zeta(n)
    return ([(F[j] + F[j + h]) / 2 for j in range(h)],
            [(F[j] - F[j + h]) / (2 * z[j]) for j in range(h)])


def merge(B, f0, f1):
    h = len(f0)
    n = 2 * h
    z = B.zeta(n)
    out = [None] * n
    for j in range(h):
        t = z[j] * f1[j]
        out[j] = f0[j] + t
        out[j + h] = f0[j] - t
    return out


def ifft(B, F):
    n = len(F)
    if n == 1:
        return [F[0]]
    f0, f1 = split(B, F)
    c0, c1 = ifft(B, f0), ifft(B, f1)
    out = [None] * n
    out[0::2] = c0
    out[1::2] = c1
    return out


def ffldl(B, G00, G01, G11):
    n = len(G00)
    L10 = [G01[j].conjugate() / G00[j] for j in range(n)]
    D11 = [G11[j] - L10[j] * L10[j].conjugate() * G00[j] for j in range(n)]
    if n == 1:
        return ("leaf", L10, G00[0].real, D11[0].real)
    d0, d1 = split(B, G00)
    T0 = ffldl(B, d0, d1, d0)
    e0, e1 = split(B, D11)
    T1 = ffldl(B, e0, e1, e0)
    return ("node", L10, T0, T1)


def ffsamp(B, t0, t1, T, sigma, sampler):
    if T[0] == "leaf":
        # fn-dsa's order: the t0 side (even part) first; l10 = 0 here (see the docstring).
        _, L10, D00, D11 = T
        z0 = sampler(t0[0].real, sigma / mp.sqrt(D00))
        z1 = sampler(t1[0].real, sigma / mp.sqrt(D11))
        return [mpc(z0, 0)], [mpc(z1, 0)]
    _, L10, T0, T1 = T
    a, b = split(B, t1)
    za, zb = ffsamp(B, a, b, T1, sigma, sampler)
    z1 = merge(B, za, zb)
    t0p = [t0[j] + (t1[j] - z1[j]) * L10[j] for j in range(len(t0))]
    a, b = split(B, t0p)
    za, zb = ffsamp(B, a, b, T0, sigma, sampler)
    z0 = merge(B, za, zb)
    return z0, z1


def leaf_widths(T, sigma, out):
    if T[0] == "leaf":
        out.append(sigma / mp.sqrt(T[2]))
        out.append(sigma / mp.sqrt(T[3]))
        return
    leaf_widths(T[3], sigma, out)
    leaf_widths(T[2], sigma, out)


def keygen(q, seed):
    set_random_seed(seed)
    D = DiscreteGaussianDistributionIntegerSampler(sigma=1.17 * math.sqrt(q / (2 * N)))
    for draw in range(1, 5001):
        fc = [int(D()) for _ in range(N)]
        gc = [int(D()) for _ in range(N)]
        if ns.gs_norm_sq(fc, gc, q) > (1.17 ** 2) * q:
            continue
        f, g = ns.R(fc), ns.R(gc)
        r = ns.solve(f, g, q, N, 0, {})
        if r is None:
            continue
        F, G = r
        assert ns.modneg(f * G - g * F, N) == ns.R(q), "NTRU equation"
        Rq = PolynomialRing(GF(q), "y")
        M = Rq.gen() ** N + 1
        hq = (Rq(gc) * Rq(fc).inverse_mod(M)) % M
        h = [int(c) for c in hq.list()] + [0] * (N - len(hq.list()))
        return fc, gc, ns.coeffs(F, N), ns.coeffs(G, N), h
    raise RuntimeError("no key")


def tree(B, key, sigma):
    fc, gc, Fc, Gc = key[:4]
    f, g, F, G = (fft(B, [B.num(v) for v in c]) for c in (fc, gc, Fc, Gc))
    G00 = [g[j] * g[j].conjugate() + f[j] * f[j].conjugate() for j in range(N)]
    G01 = [g[j] * G[j].conjugate() + f[j] * F[j].conjugate() for j in range(N)]
    G11 = [G[j] * G[j].conjugate() + F[j] * F[j].conjugate() for j in range(N)]
    return (f, g, F, G), ffldl(B, G00, G01, G11)


def replay_case(B, key, kf, T, sigma, q, rng):
    fc, gc, Fc, Gc = key[:4]
    f, g, F, G = kf
    hm = [rng.randrange(q) for _ in range(N)]
    hf = fft(B, [B.num(v) for v in hm])
    qq = B.num(q)
    t0 = [-(hf[j] * F[j]) / qq for j in range(N)]
    t1 = [(hf[j] * f[j]) / qq for j in range(N)]
    rec = []

    def samp(c, s):
        z = int(round(float(c) + float(s) * rng.gauss(0.0, 1.0)))
        rec.append((z, c, s))
        return z

    z0h, z1h = ffsamp(B, t0, t1, T, mpf(sigma), samp)
    z0 = [int(mp.nint(v.real)) for v in ifft(B, z0h)]
    z1 = [int(mp.nint(v.real)) for v in ifft(B, z1h)]
    R = ns.R
    s0 = ns.coeffs(R(hm) - ns.modneg(R(z0) * R(gc) + R(z1) * R(Gc), N), N)
    s1 = ns.coeffs(ns.modneg(R(z0) * R(fc) + R(z1) * R(Fc), N), N)
    # Exact check: s0 + h s1 = hm mod q.
    h = key[4]
    Rq = PolynomialRing(GF(q), "y")
    M = Rq.gen() ** N + 1
    assert (Rq(s0) + Rq(h) * Rq(s1)) % M == Rq(hm), "A s = t"
    return hm, s0, s1, rec


def f64hex(x):
    return "%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]


def ints(xs):
    return " ".join(str(int(v)) for v in xs)


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    B = HP()
    eta = lambda eps2: mp.sqrt(mp.log(4 * N * (1 + mpf(2) ** eps2)) / mp.pi)
    sets = [
        ("pcs", 1048573, eta(128) / mp.sqrt(2 * mp.pi), 22, 3, [(0, 0), (1, 0)]),
        ("falcon", 12289, eta(36) / mp.sqrt(2 * mp.pi), 11, 1, [(0, 0)]),
    ]
    for name, q, sigma_min, seed, nkeys, cases in sets:
        sigma = mpf("1.17") * mp.sqrt(q) * sigma_min
        keys = []
        lines_keys = [f"# {name} keys: q {q}, sigma {nstr(sigma, 20)}, seeds {seed}000+k; "
                      f"generated by tools/sign/precision_fixture.sage.py ({PREC}-bit)"]
        for k in range(nkeys):
            key = keygen(q, seed * 1000 + k)
            kf, T = tree(B, key, sigma)
            widths = []
            leaf_widths(T, sigma, widths)
            keys.append((key, kf, T))
            lines_keys.append(f"key {k}")
            for lab, poly in zip(("f", "g", "F", "G", "h"), key):
                lines_keys.append(f"{lab} {ints(poly)}")
            lines_keys.append(f"leaf_min {nstr(min(widths), 25)}")
            lines_keys.append(f"leaf_max {nstr(max(widths), 25)}")
            print(f"{name} key {k}: leaves [{nstr(min(widths), 8)}, {nstr(max(widths), 8)}], "
                  f"max|F,G| {max(abs(v) for v in key[2] + key[3])}", flush=True)
        if name == "pcs":
            with open(os.path.join(outdir, "pcs_keys.txt"), "w") as fh:
                fh.write("\n".join(lines_keys) + "\n")
        lines = [f"# {name} ffSampling replay at {PREC} bits (fn-dsa leaf order); "
                 f"generated by tools/sign/precision_fixture.sage.py",
                 "# per case: key polynomials, target hm, exact s0 s1, then 2n lines "
                 "'z centre_hi centre_lo sigma' in sampling order (f64 bit patterns in hex; "
                 "the 200-bit centre is hi + lo, sigma is rounded to binary64)"]
        for (k, t) in cases:
            key, kf, T = keys[k]
            rng = random.Random(seed * 1000003 + k * 101 + t)
            hm, s0, s1, rec = replay_case(B, key, kf, T, sigma, q, rng)
            lines.append(f"case {k} {t}")
            for lab, poly in zip(("f", "g", "F", "G", "h"), key):
                lines.append(f"{lab} {ints(poly)}")
            lines.append(f"hm {ints(hm)}")
            lines.append(f"s0 {ints(s0)}")
            lines.append(f"s1 {ints(s1)}")
            for z, c, s in rec:
                # The centre as a double-double hi + lo (|lo| <= ulp(hi)/2), and the width
                # rounded to binary64, as IEEE bit patterns.
                hi = float(c)
                lo = float(c - mpf(hi))
                lines.append(f"{z} {f64hex(hi)} {f64hex(lo)} {f64hex(float(s))}")
            nrm = sum(v * v for v in s0) + sum(v * v for v in s1)
            print(f"{name} case {k} {t}: leaves {len(rec)}, |s|^2/(2n sigma^2) "
                  f"{float(nrm / (2 * N * sigma ** 2)):.4f}", flush=True)
        with open(os.path.join(outdir, f"precision_{name}.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
