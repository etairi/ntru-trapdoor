#!/usr/bin/env python3
"""The sampling tables of ntru-trapdoor, `src/tables.rs`, computed from their definition.

A port of Jali's `tools/kat/half_gaussian_cdf.py` (https://github.com/etairi/jali, commit
0cd24f3; same author, MIT), generalised to any width and precision.

Every table is the reverse CDT of a half-Gaussian K on {0, 1, 2, ...} with
Pr[K = i] proportional to rho(i) = exp(-(num/den) i^2), that is of width sigma with
1/(2 sigma^2) = num/den (the standard-deviation parameter of Falcon and fn-dsa):

    name                num/den                 width              bits  rows  use
    BASE_3_1_192        50/961                  sigma0 = 3.1       192   41    SamplerZ base, PCS set
    GAUSS0_1_8205_79    2000000/13256881        sigma0 = 1.8205     79   all   SamplerZ base, Falcon-1024
    FG_PCS_128          10240000/14353915797    sigma_fg = 26.474  128   all   f, g of the PCS set
    FG_FALCON1024_128   10240000/168224121      sigma_fg = 2.866   128   all   f, g at Falcon-1024's q

The f, g widths are sigma_fg = 1.17 sqrt(q / (2 n)) with n = 1024, so that
1/(2 sigma_fg^2) = 10000 n / (13689 q) exactly. GAUSS0_1_8205_79 must equal fn-dsa's
`GAUSS0` (fn-dsa-sign/src/sampler.rs:78-97, commit 0629bb1) entry for entry; the script
checks it.

Entry j of a table is round(2^bits Pr[K > j]); a table keeps the entries that do not round
to 0 ("all"), or only its first `rows` entries. A sampler draws V uniform in [0, 2^bits) and
sets K = #{j : V < entry j}. The entries are non-increasing (equal neighbours occur in the far
tail, where both round to the same integer), so Pr[K > j] = entry_j / 2^bits exactly and K never
exceeds the table length.

BASE_3_1_192 keeps the 41 rows of the 128-bit table, each rounded at 192 bits, so that the
relative error of every row is at most about 2^-64 except the last one, which also carries the
omitted tail Pr[K > 41] (about 1.4% of Pr[K = 41]). A full table at 128 or 192 bits ends with a
row of probability about 2^-bits and a relative rounding error of tens of percent, which
dominates a Renyi divergence of order 256 (finding F2 in docs/precision.md).

Every entry is enclosed in interval arithmetic (mpmath.iv, 1024-bit endpoints). The sums are
cut after TERMS terms and the rest is bounded by a geometric series:
(N + m)^2 >= N^2 + 2 N m gives sum_{i >= N} rho(i) <= rho(N) / (1 - exp(-2 N num/den)).
The script fails if an enclosure contains a rounding boundary, so every rounding is certain.

Usage:
    python3 tools/gauss_tables.py              print the Rust file
    python3 tools/gauss_tables.py --check      compare it with src/tables.rs
    python3 tools/gauss_tables.py --json       print the tables as a JSON object
    python3 tools/gauss_tables.py --digest     print the digests that the crate's tests pin
    python3 tools/gauss_tables.py --jali PATH  regression: recompute Jali's HALF_3_1 (256 bits)
                                               and compare it with PATH (jali src/rand/cdf.rs)
"""
import argparse
import hashlib
import json
import re
import sys
import textwrap
from pathlib import Path

import mpmath as mp

PRECISION = 1024
RUST = Path(__file__).resolve().parents[1] / "src" / "tables.rs"

# name: (num, den, width as the documentation writes it, bits, TERMS, rows (None = every entry
# that does not round to 0), purpose)
TABLES = {
    "BASE_3_1_192": (50, 961, "3.1", 192, 160, 41,
                     "the base half-Gaussian of `SamplerZ` for the PCS set "
                     "(`BaseSampler::HalfGauss3_1`)"),
    "GAUSS0_1_8205_79": (2000000, 13256881, "1.8205", 79, 160, None,
                         "fn-dsa's `GAUSS0` (fn-dsa-sign/src/sampler.rs:78-97), the base "
                         "half-Gaussian of `SamplerZ` at Falcon-1024 (`BaseSampler::Falcon1_8205`)"),
    "FG_PCS_128": (10240000, 14353915797, "26.474", 128, 520, None,
                   "f and g of the PCS set: sigma_fg = 1.17 sqrt(q/(2n)), q = 1048573, n = 1024"),
    "FG_FALCON1024_128": (10240000, 168224121, "2.866", 128, 160, None,
                          "f and g at Falcon-1024's modulus: sigma_fg = 1.17 sqrt(q/(2n)), "
                          "q = 12289, n = 1024"),
}

# fn-dsa-sign/src/sampler.rs:78-97 at commit 0629bb1 (The Unlicense): 31 + 24 + 24-bit limbs,
# high limb first.
FN_DSA_GAUSS0 = [
    [1375468055, 6936092, 9176346], [711562636, 1023934, 15582455],
    [289335016, 4826132, 14746371], [90749601, 12313548, 10843417],
    [21676598, 5732767, 9419414], [3908963, 11171514, 6206010],
    [529006, 11146683, 11891888], [53503, 15610582, 11661180],
    [4032, 7095613, 11671091], [225, 16701698, 407118],
    [9, 6787688, 1638204], [0, 4870085, 8687822],
    [0, 111406, 13946073], [0, 1887, 12017202],
    [0, 23, 11452285], [0, 0, 3689579],
    [0, 0, 25354], [0, 0, 129],
]


def _rho(i, num, den):
    """An enclosure of exp(-num i^2 / den)."""
    return mp.iv.exp(-mp.iv.mpf(num * i * i) / den)


def _suffixes(num, den, terms):
    """Enclosures of sum_{i >= j} rho(i) for j = 0..terms (the tail beyond `terms` bounded)."""
    iv = mp.iv
    rho = [_rho(i, num, den) for i in range(terms)]
    # The omitted terms i >= terms lie in [0, rest].
    rest = _rho(terms, num, den) / (1 - iv.exp(-iv.mpf(2 * num * terms) / den))
    omitted = iv.mpf([0, rest.b])
    # suffix[j] encloses sum_{i >= j} rho(i); accumulated from the tail, no cancellation.
    suffix = [None] * (terms + 1)
    suffix[terms] = omitted
    for i in range(terms - 1, -1, -1):
        suffix[i] = suffix[i + 1] + rho[i]
    return suffix


def table(num, den, bits, terms, rows=None):
    """The entries of one table, as Python integers: every entry that does not round to 0, or
    the first `rows` entries (each of which must be nonzero)."""
    iv = mp.iv
    saved = iv.prec, mp.mp.prec
    try:
        iv.prec = mp.mp.prec = PRECISION
        suffix = _suffixes(num, den, terms)
        total = suffix[0]
        scale = iv.mpf(2) ** bits
        out = []
        for j in range(terms - 1):
            if rows is not None and j == rows:
                return out
            x = suffix[j + 1] / total * scale
            lo, hi = mp.mpf(x.a), mp.mpf(x.b)
            r = int(mp.nint((lo + hi) / 2))
            if not (r - mp.mpf(1) / 2 < lo and hi < r + mp.mpf(1) / 2):
                raise ArithmeticError(f"entry {j}: the enclosure [{lo}, {hi}] does not fix "
                                      "the rounding")
            if r == 0:
                if rows is not None:
                    raise ArithmeticError(f"entry {j} rounds to 0 before row {rows}")
                return out
            out.append(r)
        raise ArithmeticError("TERMS too small: no entry rounds to 0")
    finally:
        iv.prec, mp.mp.prec = saved


def cut_tail(num, den, terms, rows):
    """For a table cut at `rows` entries: log2 Pr[K > rows] and log2 of its ratio to
    Pr[K = rows], the relative excess of the last row (midpoints of the enclosures)."""
    iv = mp.iv
    saved = iv.prec, mp.mp.prec
    try:
        iv.prec = mp.mp.prec = PRECISION
        suffix = _suffixes(num, den, terms)
        mid = lambda x: (mp.mpf(x.a) + mp.mpf(x.b)) / 2
        tail = mid(suffix[rows + 1] / suffix[0])
        last = mid((suffix[rows] - suffix[rows + 1]) / suffix[0])
        return float(mp.log(tail, 2)), float(mp.log(tail / last, 2))
    finally:
        iv.prec, mp.mp.prec = saved


def limbs79(v):
    """A 79-bit entry as fn-dsa's [31-bit, 24-bit, 24-bit] limbs, high limb first."""
    assert 0 <= v < 1 << 79
    return [v >> 48, (v >> 24) & 0xFFFFFF, v & 0xFFFFFF]


def limbs192(v):
    """A 192-bit entry as three 64-bit limbs, high limb first."""
    assert 0 <= v < 1 << 192
    return [v >> 128, (v >> 64) & ((1 << 64) - 1), v & ((1 << 64) - 1)]


def tables():
    """Every table by name, in the order of TABLES; checks GAUSS0 against fn-dsa."""
    out = {name: table(num, den, bits, terms, rows)
           for name, (num, den, _, bits, terms, rows, _) in TABLES.items()}
    mine = [limbs79(v) for v in out["GAUSS0_1_8205_79"]]
    if mine != FN_DSA_GAUSS0:
        raise ArithmeticError("GAUSS0_1_8205_79 differs from fn-dsa's GAUSS0")
    return out


HEADER = r"""//! Sampling tables, written by `tools/gauss_tables.py` from their definition; do not edit by
//! hand (`python3 tools/gauss_tables.py --check` compares this file with the definition).
//!
//! Every table is the reverse CDT of a half-Gaussian $`K\in\{0,1,2,\dots\}`$ with
//! $`\Pr[K=i]\propto e^{-(a/b)i^2}`$, i.e. of width $`\sigma`$ with $`1/(2\sigma^2)=a/b`$ (the
//! standard-deviation parameter of Falcon and fn-dsa, not the GPV $`s=\sigma\sqrt{2\pi}`$).
//! Entry $`j`$ is $`2^{p}\Pr[K>j]`$ rounded to the nearest integer at precision $`p`$; a table
//! keeps the entries that do not round to 0, or (`BASE_3_1_192`) a fixed number of rows. A
//! sampler draws $`V`$ uniform in $`[0,2^p)`$ and returns $`K=\#\{j:V<T_j\}`$, so that
//! $`\Pr[K>j]=T_j/2^p`$ exactly. The entries are non-increasing. Every entry is enclosed in
//! 1024-bit interval arithmetic and its rounding is certain (the script fails otherwise).
//!
//! Generator: a port of Jali's `tools/kat/half_gaussian_cdf.py` (same author, MIT).
"""


def rust(values):
    """The text of `src/tables.rs`."""
    parts = [HEADER]
    for name, (num, den, width, bits, terms, rows, purpose) in TABLES.items():
        entries = values[name]
        if rows is None:
            extent = f"{len(entries)} rows (the next one rounds to 0)"
        else:
            tail, excess = cut_tail(num, den, terms, rows)
            extent = (f"{len(entries)} rows, cut there: $`K\\le{rows}`$, and the last row also "
                      f"carries the tail $`\\Pr[K>{rows}]\\approx2^{{{tail:.1f}}}`$, "
                      f"$`2^{{{excess:.1f}}}`$ of $`\\Pr[K={rows}]`$")
        doc = (f"Width {width}: $`\\Pr[K=i]\\propto e^{{-{num}i^2/{den}}}`$, entries "
               f"$`\\mathrm{{round}}(2^{{{bits}}}\\Pr[K>j])`$, {extent}. Used for {purpose}.")
        parts.append("\n" + "".join(f"/// {line}\n" for line in textwrap.wrap(
            doc, 96, break_long_words=False, break_on_hyphens=False)))
        if bits == 79:
            parts.append("/// Layout as in fn-dsa: 31-, 24- and 24-bit limbs, high limb first.\n")
            parts.append(f"pub const {name}: [[u32; 3]; {len(entries)}] = [\n")
            parts.extend("    [{}, {}, {}],\n".format(*limbs79(v)) for v in entries)
        elif bits == 192:
            parts.append("/// Layout: three 64-bit limbs, high limb first.\n")
            parts.append(f"pub const {name}: [[u64; 3]; {len(entries)}] = [\n")
            parts.extend("    [0x{:016x}, 0x{:016x}, 0x{:016x}],\n".format(*limbs192(v))
                         for v in entries)
        else:
            assert bits == 128
            parts.append(f"pub const {name}: [u128; {len(entries)}] = [\n")
            parts.extend(f"    0x{v:032x},\n" for v in entries)
        parts.append("];\n")
    return "".join(parts)


def digest(values):
    """SHAKE256, 32 bytes in hex, per table: LE64(length), then every entry as bits/8
    little-endian bytes (128- and 192-bit tables) or as its three limbs, each 4 little-endian
    bytes (79-bit table)."""
    out = {}
    for name, (_, _, _, bits, _, _, _) in TABLES.items():
        entries = values[name]
        data = len(entries).to_bytes(8, "little")
        if bits == 79:
            data += b"".join(x.to_bytes(4, "little") for v in entries for x in limbs79(v))
        else:
            data += b"".join(v.to_bytes(bits // 8, "little") for v in entries)
        out[name] = hashlib.shake_256(data).hexdigest(32)
    return out


def check_jali(path):
    """Recompute Jali's HALF_3_1 (256 bits, with its terminating 0) and compare with `path`."""
    text = Path(path).read_text()
    block = text.split("pub(super) const HALF_3_1:")[1].split("];")[0]
    theirs = [int(h, 16) for h in re.findall(r'from_be_hex\("([0-9a-f]{64})"\)', block)]
    mine = table(50, 961, 256, 160) + [0]
    same = mine == theirs
    print(f"Jali HALF_3_1: {len(theirs)} entries in {path}; recomputed {len(mine)}: "
          f"{'IDENTICAL' if same else 'DIFFERENT'}")
    return same


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help=f"compare with {RUST.name}")
    ap.add_argument("--json", action="store_true", help="print the tables as JSON")
    ap.add_argument("--digest", action="store_true", help="print the pinned digests")
    ap.add_argument("--jali", metavar="PATH", help="compare HALF_3_1 with Jali's cdf.rs")
    a = ap.parse_args()
    if a.jali:
        sys.exit(0 if check_jali(a.jali) else 1)
    values = tables()
    if a.json:
        print(json.dumps(values, indent=2))
        return
    if a.digest:
        for name, d in digest(values).items():
            print(f"{name} {len(values[name])} {d}")
        return
    text = rust(values)
    if a.check:
        same = RUST.read_bytes() == text.encode()
        print(f"src/{RUST.name}: {'identical' if same else 'DIFFERS from the computed tables'}")
        sys.exit(0 if same else 1)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
