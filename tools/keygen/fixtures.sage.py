#!/usr/bin/env sage -python
"""Sage fixtures for the keygen tests K-GS-1 and K-SOLVE-1 (design.md section 7).

Writes, under tests/fixtures/keygen/:

  pairs_pcs.txt   20 pairs (f, g) at the PCS set (n = 1024, q = 1048573), coefficients from
                  Sage's DiscreteGaussianDistributionIntegerSampler(sigma = sigma_fg) (rho(x) =
                  exp(-x^2 / (2 sigma^2)), the standard-deviation convention), not norm-filtered:
                    n1      ||f||^2 + ||g||^2 (exact);
                    n2      (2 q^2 / n) sum_{j < n/2} 1 / (|f(z_j)|^2 + |g(z_j)|^2), z_j =
                            exp(i pi (2j + 1) / n), evaluated with 200-bit complex arithmetic
                            (one root per conjugate pair), printed with 40 significant digits;
                    coprime 1 iff gcd(Res(f, x^n + 1), Res(g, x^n + 1)) = 1 (exact, FLINT);
                    sage_max the largest |F_i|, |G_i| of the exact Sage NTRUSolve below
                            (0 when not coprime).
  pairs_toy.txt   50 pairs at logn in 2..6 and q in {17, 257, 12289, 1048573}, coefficients
                  uniform in [-4, 4]: logn, q, f, g, coprime.

The NTRUSolve is the exact Sage oracle of
tools/ntru_sizes.py (same author; integer arithmetic in
ZZ[x], float only for the Babai rounding, KBITS = 30 bits per round), copied below with the size
bookkeeping removed.

Cross-check of the n2 method [computed, 2026-10-03]: for one PCS pair the exact rational
q^2 * (D^-1)_0, D = f f* + g g* (the trace identity sum_{all n roots} 1/D(z) = n (D^-1)_0),
took 104 s in Sage and agreed with the 200-bit evaluation to 57 significant digits.

Usage (about 1 minute):  sage -python tools/keygen/fixtures.sage.py
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
from sage.all import ZZ, ComplexField, PolynomialRing, RealField, set_random_seed, xgcd
from sage.stats.distributions.discrete_gaussian_integer import \
    DiscreteGaussianDistributionIntegerSampler

OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "keygen"
R = PolynomialRing(ZZ, 'x')
X = R.gen()


# ---- exact Sage NTRUSolve (ntru_sizes.py, size bookkeeping removed) ----------------------

def coeffs(p, m):
    c = p.list()
    return [int(v) for v in c] + [0] * (m - len(c))


def modneg(p, m):
    c = p.list()
    out = [0] * m
    for i, v in enumerate(c):
        if (i // m) % 2 == 0:
            out[i % m] += v
        else:
            out[i % m] -= v
    return R(out)


def field_norm(f, m):
    c = coeffs(f, m)
    fe = R(c[0::2])
    fo = R(c[1::2])
    return modneg(fe * fe - X * fo * fo, m // 2)


def galois(f, m):
    c = coeffs(f, m)
    return R([v if i % 2 == 0 else -v for i, v in enumerate(c)])


def lift(f, m_small):
    c = coeffs(f, m_small)
    out = [0] * (2 * m_small)
    out[0::2] = c
    return R(out)


def nfft(c):
    m = len(c)
    tw = np.exp(1j * np.pi * np.arange(m) / m)
    return np.fft.fft(np.array(c, dtype=np.float64) * tw)


def infft(v):
    m = len(v)
    tw = np.exp(-1j * np.pi * np.arange(m) / m)
    return (np.fft.ifft(v) * tw).real


def bits(cs):
    return max((abs(int(v)).bit_length() for v in cs), default=0)


KBITS = 30


def reduce(f, g, F, G, m):
    fc, gc = coeffs(f, m), coeffs(g, m)
    size = max(53, bits(fc), bits(gc))
    fa = nfft([v >> (size - 53) for v in fc])
    ga = nfft([v >> (size - 53) for v in gc])
    den = fa * np.conj(fa) + ga * np.conj(ga)
    rounds = 0
    while True:
        Fc, Gc = coeffs(F, m), coeffs(G, m)
        Size = max(53, bits(Fc), bits(Gc))
        if Size < size:
            break
        s = max(0, Size - size - KBITS)
        Fa = nfft([v >> (Size - 53) for v in Fc])
        Ga = nfft([v >> (Size - 53) for v in Gc])
        num = Fa * np.conj(fa) + Ga * np.conj(ga)
        ratio = infft(num / den) * float(2 ** (Size - size - s))
        k = [int(round(v)) for v in ratio]
        if all(v == 0 for v in k):
            if s == 0:
                break
            continue
        kp = R(k)
        F = F - modneg(kp * f, m) * (1 << s)
        G = G - modneg(kp * g, m) * (1 << s)
        rounds += 1
        if rounds > 100000:
            raise RuntimeError("reduce did not converge")
    return F, G


def solve(f, g, q, m):
    if m == 1:
        fL, gL = coeffs(f, 1)[0], coeffs(g, 1)[0]
        d, u, v = xgcd(fL, gL)
        if d != 1:
            return None
        return R(-v * q), R(u * q)
    r = solve(field_norm(f, m), field_norm(g, m), q, m // 2)
    if r is None:
        return None
    Fd, Gd = r
    F = modneg(lift(Fd, m // 2) * galois(g, m), m)
    G = modneg(lift(Gd, m // 2) * galois(f, m), m)
    return reduce(f, g, F, G, m)


# ---- fixture quantities ----------------------------------------------------------------------

def resultant(c, n):
    return R(c).resultant(X ** n + 1)


def n2_highprec(fc, gc, q, prec=200):
    """(2 q^2 / n) sum_{j < n/2} 1 / (|f(z_j)|^2 + |g(z_j)|^2), 200-bit complex arithmetic."""
    n = len(fc)
    CF = ComplexField(prec)
    RF = RealField(prec)
    P = PolynomialRing(CF, 'y')
    fC, gC = P(fc), P(gc)
    s = RF(0)
    for j in range(n // 2):
        z = (CF(0, 1) * CF.pi() * (2 * j + 1) / n).exp()
        s += 1 / (fC(z).norm() + gC(z).norm())
    return RF(2) * RF(q) ** 2 * s / n


def fmt_list(c):
    return ",".join(str(int(v)) for v in c)


def pcs_pairs(count=20, seed=20261003):
    n, q = 1024, 1048573
    sigma = 1.17 * math.sqrt(q / (2 * n))
    set_random_seed(seed)
    D = DiscreteGaussianDistributionIntegerSampler(sigma=sigma)
    lines = [
        "# K-GS-1 / K-SOLVE-1 fixture: tools/keygen/fixtures.sage.py (Sage DiscreteGaussian, "
        f"sigma = {sigma!r}, set_random_seed({seed}))",
        "# per pair: f, g, n1 (exact), n2 (200-bit evaluation, 40 digits), coprime, sage_max",
        f"params n={n} q={q}",
    ]
    t0 = time.time()
    for k in range(count):
        fc = [int(D()) for _ in range(n)]
        gc = [int(D()) for _ in range(n)]
        n1 = sum(v * v for v in fc) + sum(v * v for v in gc)
        n2 = n2_highprec(fc, gc, q)
        coprime = resultant(fc, n).gcd(resultant(gc, n)) == 1
        sage_max = 0
        if coprime:
            F, G = solve(R(fc), R(gc), q, n)
            lhs = modneg(R(fc) * G - R(gc) * F, n)
            assert lhs == R(q)
            sage_max = max(abs(int(v)) for v in coeffs(F, n) + coeffs(G, n))
        lines += [f"pair {k}", f"f {fmt_list(fc)}", f"g {fmt_list(gc)}", f"n1 {n1}",
                  f"n2 {n2.n(digits=40)}", f"coprime {int(coprime)}", f"sage_max {sage_max}"]
        print(f"pcs pair {k}: n1 {n1} n2 {float(n2):.6f} coprime {coprime} sage_max {sage_max} "
              f"({time.time() - t0:.1f} s)", file=sys.stderr)
    return "\n".join(lines) + "\n"


def toy_pairs(count=50, seed=7):
    set_random_seed(seed)
    import random
    rnd = random.Random(seed)
    qs = [17, 257, 12289, 1048573]
    lines = ["# K-SOLVE-1 toy fixture: tools/keygen/fixtures.sage.py (Python random.Random"
             f"({seed}), coefficients uniform in [-4, 4])",
             "# per pair: logn, q, f, g, coprime (gcd of the resultants is 1)"]
    for k in range(count):
        logn = 2 + k % 5
        n = 1 << logn
        q = qs[(k // 5) % 4]
        fc = [rnd.randint(-4, 4) for _ in range(n)]
        gc = [rnd.randint(-4, 4) for _ in range(n)]
        coprime = resultant(fc, n).gcd(resultant(gc, n)) == 1
        if coprime:
            F, G = solve(R(fc), R(gc), q, n)
            assert modneg(R(fc) * G - R(gc) * F, n) == R(q)
        lines += [f"pair {k}", f"logn {logn}", f"q {q}", f"f {fmt_list(fc)}",
                  f"g {fmt_list(gc)}", f"coprime {int(coprime)}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pairs_toy.txt").write_text(toy_pairs())
    (OUT / "pairs_pcs.txt").write_text(pcs_pairs())
    print("wrote", OUT / "pairs_toy.txt", OUT / "pairs_pcs.txt", file=sys.stderr)
