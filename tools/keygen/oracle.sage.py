#!/usr/bin/env sage -python
"""An independent re-derivation of TrapGen's candidate loop: test K-ORACLE-1 of design.md
section 7, as a fixture that the Rust test `tests/keygen_fixtures.rs` compares against.

For each (parameter set, seed), the oracle replays the key-generation stream (hashlib.shake_256,
tools/keygen/cdt_kat.py) and applies, per candidate (f, g), the checks of design.md 5.9:
  1. N1 = ||f||^2 + ||g||^2:    10000 N1 <= 13689 q (exact);
  2. N2 = (2 q^2 / n) sum_{j < n/2} 1 / (|f(z_j)|^2 + |g(z_j)|^2) <= 13689 q / 10000, with
     200-bit complex arithmetic (tools/keygen/fixtures.sage.py); a candidate within 2^-30
     (relative) of the bound is reported as `near`, where the f64 decision of the crate may
     legitimately differ;
  3. f invertible in GF(q)[x]/(x^n + 1): gcd(f, x^n + 1) = 1 over GF(q);
  4. NTRUSolve succeeds iff gcd(Res(f, x^n + 1), Res(g, x^n + 1)) = 1 (exact, FLINT).
The width and leaf checks (steps 6-7) are not re-derived: the oracle assumes they pass, and the
Rust test checks that the crate rejected nothing for those reasons.

For the first candidate that passes, it records the 1-based candidate count, the rejection
counters, f, g and h = g / f in GF(q)[x]/(x^n + 1).

Writes tests/fixtures/keygen/trapgen_oracle.txt. Usage (about 1-2 minutes):
    sage -python tools/keygen/oracle.sage.py
"""
import sys
import time
from pathlib import Path

from sage.all import GF, ZZ, PolynomialRing

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "keygen"))
import gauss_tables  # noqa: E402
from cdt_kat import DOMAIN, Stream, sample  # noqa: E402

# n2_highprec from the fixtures script (same author; imported by path).
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "fixtures_sage", ROOT / "tools" / "keygen" / "fixtures.sage.py")
_fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fx)

SETS = {
    "PCS": (1048573, 10, "FG_PCS_128"),
    "FALCON_1024": (12289, 10, "FG_FALCON1024_128"),
}
SEEDS = [bytes(32), bytes(range(1, 33)), bytes([0xA5] * 32)]

R = PolynomialRing(ZZ, 'x')
X = R.gen()


def run(name, q, logn, table, seed):
    n = 1 << logn
    stream = Stream(DOMAIN + q.to_bytes(4, "little") + bytes([logn]) + seed)
    Fq = GF(q)
    S = PolynomialRing(Fq, 'y')
    modulus = S.gen() ** n + 1
    counts = dict(norm_fg=0, norm_ortho=0, not_invertible=0, solve=0)
    near = []
    cand = 0
    while True:
        cand += 1
        f = sample(table, stream, n)
        g = sample(table, stream, n)
        n1 = sum(v * v for v in f) + sum(v * v for v in g)
        if 10000 * n1 > 13689 * q:
            counts["norm_fg"] += 1
            continue
        n2 = _fx.n2_highprec(f, g, q)
        bound = 13689 * q / 10000
        rel = abs(float(n2 / bound) - 1.0)
        if rel < 2.0 ** -30:
            near.append(cand)
        if n2 > bound:
            counts["norm_ortho"] += 1
            continue
        fq = S(f)
        if fq.gcd(modulus) != 1:
            counts["not_invertible"] += 1
            continue
        res_f = R(f).resultant(X ** n + 1)
        res_g = R(g).resultant(X ** n + 1)
        if res_f.gcd(res_g) != 1:
            counts["solve"] += 1
            continue
        h = (S(g) * fq.inverse_mod(modulus)) % modulus
        hc = [int(c) for c in h.list()] + [0] * (n - len(h.list()))
        return cand, counts, near, f, g, hc


def main():
    tables = gauss_tables.tables()
    lines = ["# K-ORACLE-1: tools/keygen/oracle.sage.py (independent TrapGen candidate loop)",
             "# per case: set, seed, candidates, counters, near-boundary candidates, f, g, h"]
    t0 = time.time()
    for name, (q, logn, tname) in SETS.items():
        for seed in SEEDS:
            cand, counts, near, f, g, h = run(name, q, logn, tables[tname], seed)
            lines += [f"case {name}", f"seed {seed.hex()}", f"candidates {cand}",
                      "counts " + ",".join(f"{k}={v}" for k, v in counts.items()),
                      "near " + ",".join(map(str, near)),
                      "f " + ",".join(map(str, f)), "g " + ",".join(map(str, g)),
                      "h " + ",".join(map(str, h))]
            print(f"{name} seed {seed.hex()[:8]}: candidate {cand}, {counts}, near {near} "
                  f"({time.time() - t0:.1f} s)", file=sys.stderr)
    out = ROOT / "tests" / "fixtures" / "keygen" / "trapgen_oracle.txt"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
