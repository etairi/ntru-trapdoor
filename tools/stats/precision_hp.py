#!/usr/bin/env sage -python
"""The binary64 leaf centres and widths of the crate's ffSampling
against a 200-bit reference, on real PCS keys and many targets; and the Renyi budget.

Input: the recordings of examples/stats_precision.rs (DIR/key<k>.txt, DIR/key<k>.bin): for
TrapGen keys at the PCS set and, per key, structured and uniform targets, every leaf call of
SampPre's ffSampling as the production sampler made it: the binary64 centre mu_i, the leaf value
1/sigma'_i and the decision z_i, in sampling order; plus the output s.

Reference: the same ffLDL + ffSampling at PREC = 200 bits (mpmath), Falcon's Alg. 9/11 over all
complex evaluation points, with the functions of the crate's tools/sign/precision_fixture.sage.py
(sign stage; its leaf order, validated by the crate's test S-PREC-1, is fn-dsa's). The reference
replays the recorded decisions z_i, so both runs follow the same path; the exact centre c_i of
leaf i is a function of the key, the target and z_1 .. z_{i-1} only. Per leaf:
    Delta_i = mu_i - c_i, reported as |Delta_i| / sigma'_i;  and |1/isigma_i - sigma'_i| / sigma'_i.
The final s is recomputed exactly over Z from the decisions (s0 = t - (z0 g + z1 G),
s1 = z0 f + z1 F) and must equal the crate's output.

Renyi budget (a = 256), per call [the task's formula]:
    delta_call = sum over the 2d leaves of a Delta_i^2 / (2 sigma'_i^2),
plus the width term sum_i a (dsigma_i / sigma'_i)^2 (ln R_a of two centred Gaussians whose widths
differ by a relative d is a d^2 + O(d^3); checked numerically in renyi_check.py), and the
conservative variant 2d * a * max_i (Delta_i / sigma'_i)^2 / 2 with the largest error seen over
all calls. Totals over Q calls: Q * delta (nats).

Run (about 2-4 minutes on 4 processes):
    sage -python precision_hp.py data/precision > precision_hp.out
"""
import math
import multiprocessing as mpr
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

PREC = 200
A = 256
Q_LIST = (48, 57)


def hp():
    import mpmath
    mpmath.mp.prec = PREC
    return mpmath


# --- 200-bit ffLDL / ffSampling: from tools/sign/precision_fixture.sage.py (sign stage) ---------


class HP:
    def __init__(self, mp):
        self.mp = mp
        self.cache = {}

    def zeta(self, n):
        if n not in self.cache:
            self.cache[n] = [self.mp.expjpi(self.mp.mpf(2 * j + 1) / n) for j in range(n // 2)]
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
    mp = B.mp
    if T[0] == "leaf":
        # fn-dsa's order: the t0 side (even part) first; l10 = 0 here.
        _, L10, D00, D11 = T
        z0 = sampler(t0[0].real, sigma / mp.sqrt(D00))
        z1 = sampler(t1[0].real, sigma / mp.sqrt(D11))
        return [mp.mpc(z0, 0)], [mp.mpc(z1, 0)]
    _, L10, T0, T1 = T
    a, b = split(B, t1)
    za, zb = ffsamp(B, a, b, T1, sigma, sampler)
    z1 = merge(B, za, zb)
    t0p = [t0[j] + (t1[j] - z1[j]) * L10[j] for j in range(len(t0))]
    a, b = split(B, t0p)
    za, zb = ffsamp(B, a, b, T0, sigma, sampler)
    z0 = merge(B, za, zb)
    return z0, z1


# --- Data -----------------------------------------------------------------------------------------


def f64(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def load_key(txt):
    d = {}
    for line in Path(txt).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        k, *v = line.split()
        d[k] = v
    key = {k: [int(x) for x in d[k]] for k in ("f", "g", "F", "G", "h")}
    key["isigma"] = [f64(int(x, 16)) for x in d["isigma"]]
    key["targets"] = int(d["targets"][0])
    return key


def load_targets(binf, n, count):
    raw = Path(binf).read_bytes()
    rec = 4 + 4 * n * 3 + 8 * 2 * n + 4 * 2 * n
    assert len(raw) == rec * count, (len(raw), rec, count)
    out = []
    for i in range(count):
        b = raw[i * rec:(i + 1) * rec]
        kind = struct.unpack_from("<I", b, 0)[0]
        o = 4
        t = np.frombuffer(b, dtype="<u4", count=n, offset=o).astype(np.int64)
        o += 4 * n
        s = np.frombuffer(b, dtype="<i4", count=2 * n, offset=o).astype(np.int64)
        o += 8 * n
        mu = np.frombuffer(b, dtype="<f8", count=2 * n, offset=o)
        o += 16 * n
        z = np.frombuffer(b, dtype="<i4", count=2 * n, offset=o).astype(np.int64)
        out.append((kind, t, s[:n], s[n:], mu, z))
    return out


def negacyclic(a, b):
    n = len(a)
    c = np.convolve(a, b)
    out = c[:n].copy()
    out[: n - 1] -= c[n:]
    return out


# --- Worker ---------------------------------------------------------------------------------------


def run_key(args):
    d, k, limit = args
    mp = hp()
    t0 = time.time()
    key = load_key(Path(d) / f"key{k}.txt")
    n = len(key["f"])
    q = 1048573
    B = HP(mp)
    sigma_min = mp.sqrt(mp.log(4 * n * (1 + mp.mpf(2) ** 128)) / mp.pi) / mp.sqrt(2 * mp.pi)
    sigma = mp.mpf("1.17") * mp.sqrt(q) * sigma_min
    f, g, F, G = (fft(B, [mp.mpc(v, 0) for v in key[c]]) for c in ("f", "g", "F", "G"))
    G00 = [g[j] * g[j].conjugate() + f[j] * f[j].conjugate() for j in range(n)]
    G01 = [g[j] * G[j].conjugate() + f[j] * F[j].conjugate() for j in range(n)]
    G11 = [G[j] * G[j].conjugate() + F[j] * F[j].conjugate() for j in range(n)]
    T = ffldl(B, G00, G01, G11)
    t_tree = time.time() - t0
    targets = load_targets(Path(d) / f"key{k}.bin", n, key["targets"])[:limit]
    fz, gz, Fz, Gz = (np.array(key[c], dtype=np.int64) for c in ("f", "g", "F", "G"))
    rows = []
    rel_all = []
    width_rel = None
    for ti, (kind, t, s0, s1, mu, z) in enumerate(targets):
        hf = fft(B, [mp.mpc(int(v), 0) for v in t])
        qq = mp.mpc(q, 0)
        t0v = [-(hf[j] * F[j]) / qq for j in range(n)]
        t1v = [(hf[j] * f[j]) / qq for j in range(n)]
        rec = []
        pos = [0]

        def samp(c, s):
            i = pos[0]
            rec.append((c, s))
            pos[0] += 1
            return int(z[i])

        z0h, z1h = ffsamp(B, t0v, t1v, T, sigma, samp)
        assert pos[0] == 2 * n
        # Exact s from the decisions.
        z0 = np.array([int(mp.nint(v.real)) for v in ifft(B, z0h)], dtype=np.int64)
        z1 = np.array([int(mp.nint(v.real)) for v in ifft(B, z1h)], dtype=np.int64)
        s0x = t - (negacyclic(z0, gz) + negacyclic(z1, Gz))
        s1x = negacyclic(z0, fz) + negacyclic(z1, Fz)
        exact = bool(np.array_equal(s0x, s0) and np.array_equal(s1x, s1))
        # Centres and widths.
        rel = np.empty(2 * n)
        cmag = 0.0
        for i, (c, s) in enumerate(rec):
            delta = mp.mpf(float(mu[i])) - c
            rel[i] = float(abs(delta) / s)
            cmag = max(cmag, abs(float(c)))
        if width_rel is None:
            width_rel = np.array([float(abs(1 / mp.mpf(key["isigma"][i]) - s) / s) for i, (c, s) in enumerate(rec)])
            widths = np.array([float(s) for _, s in rec])
        sumsq = float(np.sum(rel * rel))
        imax = int(np.argmax(rel))
        rows.append(dict(kind=kind, ti=ti, max=float(rel[imax]), imax=imax, rms=math.sqrt(sumsq / (2 * n)),
                         sumsq=sumsq, cmag=cmag, exact=exact,
                         half=(float(np.max(rel[:n])), float(np.max(rel[n:])))))
        rel_all.append(rel.astype(np.float32))
    return dict(k=k, rows=rows, rel=np.concatenate(rel_all), width_rel=width_rel, widths=widths,
                t_tree=t_tree, t_total=time.time() - t0)


def lg(x):
    return math.log2(x) if x > 0 else float("-inf")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "data/precision"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    nkeys = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    keys = sorted(int(re.findall(r"\d+", p.stem)[0]) for p in Path(d).glob("key*.txt"))[:nkeys]
    import mpmath
    print(f"# precision_hp.py: {PREC}-bit reference (mpmath {mpmath.__version__}, {mpmath.libmp.BACKEND}); "
          f"numpy {np.__version__}; python {sys.version.split()[0]}; data {d}; keys {keys}")
    ctx = mpr.get_context("spawn")
    with ctx.Pool(4) as pool:
        res = pool.map(run_key, [(d, k, limit) for k in keys])
    kinds = {0: "zero", 1: "all q-1", 2: "all (q-1)/2", 3: "aligned F", 4: "aligned f", 10: "uniform"}
    all_rel = []
    print("# per key: tree build time, leaf-width error, and per-target-kind maxima of |Delta|/sigma'")
    for r in res:
        rows = r["rows"]
        wr = r["width_rel"]
        print(f"key {r['k']}: tree {r['t_tree']:.1f} s, total {r['t_total']:.1f} s; leaf widths "
              f"[{r['widths'].min():.6f}, {r['widths'].max():.6f}]; max |dsigma'|/sigma' = 2^{lg(wr.max()):.2f}, "
              f"rms 2^{lg(math.sqrt(float(np.mean(wr * wr)))):.2f}; s exact on {sum(x['exact'] for x in rows)}/{len(rows)} calls")
        for kd in sorted(set(x["kind"] for x in rows)):
            sel = [x for x in rows if x["kind"] == kd]
            m = max(sel, key=lambda x: x["max"])
            print(f"   {kinds[kd]:12s} ({len(sel):3d} calls): max |Delta|/sigma' 2^{lg(m['max']):.2f} (leaf {m['imax']}), "
                  f"rms over leaves 2^{lg(max(x['rms'] for x in sel)):.2f} (worst call), "
                  f"sum (Delta/sigma')^2 up to 2^{lg(max(x['sumsq'] for x in sel)):.2f}; max |centre| 2^{lg(max(x['cmag'] for x in sel)):.2f}; "
                  f"first/second half max 2^{lg(max(x['half'][0] for x in sel)):.2f} / 2^{lg(max(x['half'][1] for x in sel)):.2f}")
        all_rel.append(r["rel"])
    rel = np.concatenate(all_rel).astype(np.float64)
    rows_all = [x for r in res for x in r["rows"]]
    wr_all = np.concatenate([r["width_rel"] for r in res])
    n_leaves = rel.size
    print(f"# ALL: {len(rows_all)} calls on {len(res)} keys, {n_leaves:,} leaves; s exact on "
          f"{sum(x['exact'] for x in rows_all)}/{len(rows_all)} calls")
    print(f"#   max |Delta|/sigma' = {rel.max():.4e} = 2^{lg(rel.max()):.2f}; rms 2^{lg(math.sqrt(float(np.mean(rel * rel)))):.2f}; "
          f"median 2^{lg(float(np.median(rel))):.2f}")
    qs = [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999]
    print("#   quantiles of |Delta|/sigma': " + ", ".join(f"{q}: 2^{lg(float(np.quantile(rel, q))):.2f}" for q in qs))
    edges = np.arange(-60, -33, 1.0)
    lr = np.log2(np.maximum(rel, 1e-300))
    hist, _ = np.histogram(lr, bins=np.concatenate([[-1e9], edges, [1e9]]))
    print("#   histogram of log2(|Delta|/sigma') (bin upper edges): " +
          ", ".join(f"<{e:.0f}: {c}" for e, c in zip(list(edges) + ["inf"], hist) if c))
    print(f"#   leaf widths: max |dsigma'|/sigma' = 2^{lg(wr_all.max()):.2f}")
    # Renyi budget.
    a = A
    deltas = np.array([a * x["sumsq"] / 2 for x in rows_all])
    width_terms = [a * float(np.sum(r["width_rel"] ** 2)) for r in res]
    wmax = max(width_terms)
    dmax = float(deltas.max())
    dmean = float(deltas.mean())
    cons = 2 * 1024 * a * float(rel.max()) ** 2 / 2
    unif = np.array([a * x["sumsq"] / 2 for x in rows_all if x["kind"] == 10] or [0.0])
    print(f"# RENYI (a = {a}), per call, nats:")
    print(f"#   centre term delta = sum_i a Delta_i^2/(2 sigma'_i^2): max over calls 2^{lg(dmax):.2f}, mean 2^{lg(dmean):.2f}, "
          f"uniform targets max 2^{lg(float(unif.max())):.2f} mean 2^{lg(float(unif.mean())):.2f}")
    print(f"#   conservative 2d * a * max(Delta/sigma')^2 / 2 = 2^{lg(cons):.2f}")
    print(f"#   width term sum_i a (dsigma/sigma)^2: max over keys 2^{lg(wmax):.2f}")
    for name, v in (("max over calls (centre + width)", dmax + wmax), ("conservative (centre + width)", cons + wmax)):
        print(f"#   {name}: 2^{lg(v):.2f} per call; " +
              "; ".join(f"Q = 2^{qq}: Q delta = 2^{lg(v) + qq:.2f} nats" for qq in Q_LIST))


if __name__ == "__main__":
    main()
