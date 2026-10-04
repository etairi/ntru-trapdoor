"""Ducas and van Woerden's estimator ("NTRU Fatigue: How Stretched is Overstretched?", ASIACRYPT 2021)
at the parameter sets of this crate (docs/security.md, "Key distribution").

Usage: sage -python tools/stats/ntru_fatigue.py <NTRUFatigue checkout> [pcs | pcs-worst | falcon1024 | <q> <sigma^2>]

NTRUFatigue is the authors' code, https://github.com/WvanWoerden/NTRUFatigue (checked at commit
02971b3). It has no licence file, so it is not copied here: this script reads estimator.sage from
your checkout, changes one line, max_n_cache = 10000 -> 1100 (the size of the precomputed tables;
the values for d <= 1100 are the same and the precomputation is faster), preparses it in memory and
runs

    combined_attack_prob(q, d, sigma^2, ntru="circulant", fixed_tours=8)

that is, progressive BKZ with 8 tours per block size, as in the authors' Figures 6-8, with the
circulant volume model of their Figure 7 (for x^d + 1 the volume model differs by (1 - ln 2)/2 nats
in total). It prints the expected successful block size and the probabilities that key recovery
(SKR) or dense-sublattice discovery (DSD) happens first. The estimator scans block sizes below d
and ignores per-block-size probabilities below 10^-7.

Instances (d = 1024):
- pcs: q = 1048573, sigma^2 = sigma_fg^2 = 1.17^2 q / (2d) (the width before the norm tests);
- pcs-worst: the worst key that the norm tests admit, ||(g, f)||^2 = q / 1.17^2 and
  ln vol(R(g, -f)) = d (ln q / 2 - ln 1.17);
- falcon1024: q = 12289 and its sigma_fg^2;
- or any q and sigma^2.
"""

import math
import os
import re
import sys
import time

USAGE = ("usage: sage -python tools/stats/ntru_fatigue.py <NTRUFatigue checkout> "
         "[pcs | pcs-worst | falcon1024 | <q> <sigma^2>]")


def instance(args, d):
    """(label, q, sigma^2, ln vol override or None) from the command line."""
    if args in ([], ["pcs"]):
        q = 1048573
        return "pcs", q, 1.17 ** 2 * q / (2 * d), None
    if args == ["pcs-worst"]:
        q = 1048573
        return "pcs-worst", q, q / (1.17 ** 2 * 2 * d), d * (math.log(q) / 2 - math.log(1.17))
    if args == ["falcon1024"]:
        q = 12289
        return "falcon1024", q, 1.17 ** 2 * q / (2 * d), None
    if len(args) == 2:
        try:
            return "custom", int(args[0]), float(args[1]), None
        except ValueError:
            pass
    sys.exit(USAGE)


def main():
    if len(sys.argv) < 2:
        sys.exit(USAGE)
    d = 1024
    label, q, s2, dsl_logvol = instance(sys.argv[2:], d)
    path = os.path.join(sys.argv[1], "estimator.sage")
    try:
        with open(path) as fh:
            src = fh.read()
    except OSError as e:
        sys.exit(f"{path}: {e}")
    src, count = re.subn(r"(?m)^max_n_cache = 10000$", "max_n_cache = 1100", src)
    if count != 1:
        sys.exit(f"{path}: expected one line 'max_n_cache = 10000' (checked at commit 02971b3)")

    from sage.repl.preparse import preparse_file

    ns = {}
    exec("from sage.all import *", ns)
    exec(compile(preparse_file(src), path, "exec"), ns)

    t0 = time.time()
    avg, p_skr, p_dsd, _ = ns["combined_attack_prob"](
        q, d, s2, ntru="circulant", fixed_tours=8, verbose=False, dsl_logvol=dsl_logvol)
    print(f"{label}: d={d} q={q} sigma^2={s2:.4f}: expected beta {avg:.2f}  "
          f"P[SKR first] {p_skr:.6f}  P[DSD first] {p_dsd:.3e}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
