#!/usr/bin/env python3
"""Known-answer vectors for the f, g sampler (`sample_fg`) and the key-generation stream:
test K-CDT-3 of design.md section 7.

An independent re-derivation in Python: hashlib.shake_256 for the stream and the tables of
tools/gauss_tables.py (computed from their definition, not read from src/tables.rs).

The key-generation stream of a 32-byte seed is
    SHAKE256("ntru-trapdoor/v1/keygen" || LE32(q) || [logn] || seed)        (design.md 5.1)
consumed in order. One coefficient is K = #{j : V < T_j} for a uniform 128-bit V, with a sign
bit b: return -K if b = 0, K if b = 1 and K != 0, and draw again if b = 1 and K = 0. The draw
is lazy (src/keygen/gauss.rs): w = the next 8 bytes (little-endian), b = w & 1, H = w >> 1 the
top 63 bits of V = H 2^65 + L. If H equals the top 63 bits T_j >> 65 of no entry, then
K = #{j : H < T_j >> 65}; otherwise L is read (the next 8 bytes, little-endian, then the low bit
of one byte: L = 2 lo + bit) and K = #{j : V < T_j}. Both counts are full scans here (the crate
uses binary searches).

Writes tests/fixtures/keygen/cdt_kat.txt: per (parameter set, seed), the first candidate's f and
g (all n coefficients each; f is drawn first) and the number of stream bytes they consumed.

Usage: python3 tools/keygen/cdt_kat.py   (needs mpmath for tools/gauss_tables.py)
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import gauss_tables  # noqa: E402

DOMAIN = b"ntru-trapdoor/v1/keygen"
SETS = {
    # name: (q, logn, table)
    "PCS": (1048573, 10, "FG_PCS_128"),
    "FALCON_1024": (12289, 10, "FG_FALCON1024_128"),
}
SEEDS = [bytes(32), bytes(range(1, 33)), hashlib.sha256(b"ntru-trapdoor cdt kat").digest()]


class Stream:
    """The SHAKE256 output stream of `data`, read sequentially."""

    def __init__(self, data):
        self.data = data
        self.buf = b""
        self.pos = 0

    def take(self, k):
        if self.pos + k > len(self.buf):
            self.buf = hashlib.shake_256(self.data).digest(max(2 * len(self.buf), self.pos + k, 1 << 16))
        out = self.buf[self.pos:self.pos + k]
        self.pos += k
        return out


def draw_half(table, tops, stream):
    """One lazy draw (K, b); `tops` = [t >> 65 for t in table]."""
    w = int.from_bytes(stream.take(8), "little")
    b = w & 1
    h = w >> 1
    if h in tops:
        lo = int.from_bytes(stream.take(8), "little")
        bit = stream.take(1)[0] & 1
        v = (h << 65) | (lo << 1) | bit
        return sum(1 for t in table if v < t), b
    return sum(1 for x in tops if h < x), b


def sample(table, stream, count):
    tops = [t >> 65 for t in table]
    out = []
    while len(out) < count:
        k, b = draw_half(table, tops, stream)
        if b == 0:
            out.append(-k)
        elif k != 0:
            out.append(k)
    return out


def main():
    tables = gauss_tables.tables()
    lines = ["# K-CDT-3: tools/keygen/cdt_kat.py (hashlib.shake_256 + tools/gauss_tables.py)",
             "# per case: set, seed (hex), f and g of the first candidate, bytes consumed"]
    for name, (q, logn, tname) in SETS.items():
        n = 1 << logn
        for seed in SEEDS:
            s = Stream(DOMAIN + q.to_bytes(4, "little") + bytes([logn]) + seed)
            f = sample(tables[tname], s, n)
            g = sample(tables[tname], s, n)
            lines += [f"case {name}", f"seed {seed.hex()}", "f " + ",".join(map(str, f)),
                      "g " + ",".join(map(str, g)), f"bytes {s.pos}"]
    out = ROOT / "tests" / "fixtures" / "keygen" / "cdt_kat.txt"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
