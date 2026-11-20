#!/usr/bin/env python3
"""Checks that the compiled base sampler does not branch on the draw.

The width-3.1 base sampler compares the top 63 bits H of a draw with the top bits H_j of the
nonzero entries of `BASE_3_1_192`, in a scan of fixed length (docs/security.md, "Side
channels"). The source has no branch there, but a compiler may add one: an OR of the equalities
H = H_j, for example, compiles to a tree of branches on H. This script builds the library to
LLVM IR (optimized, one codegen unit) and fails if a `switch` has at least four of the H_j among
its cases, or if a conditional branch tests an `icmp` against one of the H_j of at least 2^40
(the smaller ones also occur as unrelated constants). Both are what such a tree looks like before
code generation. A check of the IR, not of the machine code: the code generator may still turn
selects into branches, which this script does not see.

    python3 tools/sampler_branches.py [--target TRIPLE] [--features LIST]

Run from the repository root. The build goes to `target/sampler-branches/<target>-<features>`.
"""

import argparse
import glob
import os
import re
import subprocess
import sys


def table_tops(path):
    """The top 63 bits of the nonzero entries of BASE_3_1_192 (as `sampler.rs` derives them)."""
    src = open(path).read()
    body = src[src.index("pub const BASE_3_1_192"):]
    body = body[body.index("= [") + 3:]
    body = body[:body.index("];")]
    rows = re.findall(r"\[(0x[0-9a-f]+),\s*0x[0-9a-f]+,\s*0x[0-9a-f]+\]", body)
    if len(rows) != 41:
        sys.exit(f"expected 41 rows in BASE_3_1_192, found {len(rows)}")
    tops = [int(r, 16) >> 1 for r in rows]
    return {t for t in tops if t != 0}


def ir_files(target_dir):
    return glob.glob(os.path.join(target_dir, "**", "ntru_trapdoor-*.ll"), recursive=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="target triple (default: the host)")
    ap.add_argument("--features", default="", help="cargo features, e.g. simd")
    args = ap.parse_args()

    tops = table_tops("src/tables.rs")
    config = f"{args.target or 'host'}-{args.features.replace(',', '+') or 'default'}"
    target_dir = os.path.join("target", "sampler-branches", config)
    cmd = ["cargo", "rustc", "--quiet", "--locked", "--release", "--lib"]
    if args.target:
        cmd += ["--target", args.target]
    if args.features:
        cmd += ["--features", args.features]
    cmd += ["--", "--emit", "llvm-ir", "-C", "codegen-units=1"]
    subprocess.run(cmd, check=True, env={**os.environ, "CARGO_TARGET_DIR": target_dir})
    # A fresh build emits nothing new: its IR file from the last build is that of these sources.
    files = sorted(ir_files(target_dir), key=os.path.getmtime)
    if not files:
        sys.exit(f"no IR file under {target_dir}")
    ir = open(files[-1]).read()

    bad = []
    for fn in re.split(r"\ndefine ", ir)[1:]:
        name = re.search(r"@(\S+?)\(", fn.split("\n", 1)[0])
        name = name.group(1) if name else "?"
        # Switches with entries among their cases.
        for cases in re.findall(r"switch i64 [^,]+, label %[^\s]+ \[(.*?)\]", fn, re.S):
            vals = {int(v) & (2**64 - 1) for v in re.findall(r"i64 (-?\d+), label", cases)}
            if len(vals & tops) >= 4:
                bad.append(f"{name}: switch with {len(vals & tops)} table entries among its cases")
        # Conditional branches on a comparison with an entry.
        defs = {}
        for m in re.finditer(r"^\s*(%[\w.]+) = icmp \w+ i64 ([^,]+), ([^\s,]+)", fn, re.M):
            ops = {m.group(2).strip(), m.group(3).strip()}
            consts = {int(o) & (2**64 - 1) for o in ops if re.fullmatch(r"-?\d+", o)}
            defs[m.group(1)] = consts
        for m in re.finditer(r"br i1 (%[\w.]+),", fn):
            if {c for c in defs.get(m.group(1), set()) if c >= 2**40} & tops:
                bad.append(f"{name}: conditional branch on a comparison with a table entry")
    where = f"{args.target or 'host'}{', features ' + args.features if args.features else ''}"
    if bad:
        print(f"branches on the draw ({where}):")
        for b in sorted(set(bad)):
            print("  " + b)
        sys.exit(1)
    print(f"no branch on the base draw's table entries ({where})")


if __name__ == "__main__":
    main()
