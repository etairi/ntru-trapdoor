#!/usr/bin/env sage -python
"""The output distribution of SampPre at the PCS set.

Input: data/samppre/ from examples/stats_samppre.rs: 8 TrapGen keys x 125,000 preimages of fresh
uniform targets (one SampPre attempt each, without the norm check), 200,000 preimages of one fixed
target on key 0, and 100,000 calls of the public samp_pre on key 0.

Reference: the discrete Gaussian of standard deviation sigma = 2656.4220405 (s-parameter
varsigma = sigma sqrt(2 pi) = 6658.66) on the coset {s : s0 + h s1 = t}. Above the smoothing
parameter its moments are those of the continuous Gaussian N(0, sigma^2 I_{2d}) up to relative
O(eps) terms, so (approximations used for the tests, all far below the tests' resolution):
- ||s||^2 / sigma^2 ~ chi-square with 2d = 2048 degrees of freedom;
- every coordinate and every unit direction u: <s, u> ~ N(0, sigma^2), E<s,u>^4 = 3 sigma^4;
- key-adapted directions (FFT points k < d/2): |p|^2 / (d sigma^2) ~ Exp(1) for p_k on w0_k (the
  direction of the basis row (g, -f)) and on w1_k (its complement, the first Gram-Schmidt step);
- for one fixed target, E s_j = 0 for every coordinate (the coset is smoothed).
p-values: two-sided normal for z-scores, chi-square for sums of squared z-scores, and
Kolmogorov-Smirnov for the norm law.

Tail: P(||s||^2 > beta^2) for beta^2 = varsigma^2 * 2d = 2 pi * 2d * sigma^2, from the
chi-square law (mpmath), and the rigorous bound of Micciancio-Regev's Lemma 4.4 [recalled:
Pr[||x|| > s sqrt(m)] <= (1+eps)/(1-eps) 2^-m for x ~ D_{Lambda+c, s}, s >= eta_eps(Lambda)].

Run: sage -python samppre_analysis.py data/samppre > samppre_analysis.out
"""
import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy import stats

D = 1024
DIM = 2 * D


def f64(bits):
    import struct
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def load_acc(path):
    acc = {"dir": [], "fft": []}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            continue
        f = line.split()
        if f[0] == "n":
            acc["sigma"] = f64(int(f[5], 16))
            acc["beta_sq"] = int(f[7])
        elif f[0] == "calls":
            acc["calls"], acc["exceed"], acc["max_norm"] = int(f[1]), int(f[3]), int(f[5])
        elif f[0] == "sum":
            acc["sum"] = np.array([int(x) for x in f[1:]], dtype=np.float64)
        elif f[0] == "sumsq":
            acc["sumsq"] = np.array([int(x) for x in f[1:]], dtype=np.float64)
        elif f[0] == "sum4":
            acc["sum4"] = int(f[1])
        elif f[0] == "dir":
            acc["dir"].append([float(x) for x in f[1:]])
        elif f[0] == "fftdir":
            acc["fft"].append([float(x) for x in f[2:]])
    acc["dir"] = np.array(acc["dir"])
    acc["fft"] = np.array(acc["fft"])
    return acc


def p2(z):
    return float(math.erfc(abs(z) / math.sqrt(2)))


def chi2_p(x, df):
    return float(stats.chi2.sf(x, df))


def norm_tests(name, norms, sigma, beta_sq):
    x = norms.astype(np.float64) / (sigma * sigma)
    n = x.size
    k = DIM
    mean = x.mean()
    z_mean = (mean - k) / math.sqrt(2 * k / n)
    var = x.var(ddof=1)
    z_var = (var - 2 * k) / math.sqrt((8 * k * k + 48 * k) / n)
    ks = stats.kstest(x, stats.chi2(k).cdf)
    # 100 equiprobable bins under chi2_2048.
    edges = stats.chi2(k).ppf(np.linspace(0, 1, 101))
    obs, _ = np.histogram(x, bins=edges)
    x2 = float(np.sum((obs - n / 100) ** 2 / (n / 100)))
    sig_eff = math.sqrt(mean / k) * sigma
    print(f"  {name}: N = {n:,}; mean ||s||^2/(2d sigma^2) = {mean / k:.6f} (z {z_mean:+.2f}, p {p2(z_mean):.3f}); "
          f"var/(2*2d) = {var / (2 * k):.4f} (z {z_var:+.2f}, p {p2(z_var):.3f}); KS D = {ks.statistic:.5f}, p = {ks.pvalue:.3f}; "
          f"100-bin chi2 {x2:.1f}/99, p = {chi2_p(x2, 99):.3f}")
    print(f"     implied sigma = {sig_eff:.3f} (s-parameter {sig_eff * math.sqrt(2 * math.pi):.2f}); "
          f"max ||s||^2 / beta^2 = {norms.max() / beta_sq:.4f}; exceed beta^2: {int(np.sum(norms > beta_sq))}")
    return dict(n=n, mean=mean / k, z=z_mean, ks_p=ks.pvalue, x2_p=chi2_p(x2, 99), max_ratio=norms.max() / beta_sq,
                exceed=int(np.sum(norms > beta_sq)))


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data/samppre")
    keys = sorted(int(p.stem.replace("samppre_key", "")) for p in d.glob("samppre_key*.txt"))
    print(f"# samppre_analysis.py on {d}: keys {keys}; numpy {np.__version__}, scipy "
          f"{__import__('scipy').__version__}, mpmath {mp.__version__}")
    accs = {k: load_acc(d / f"samppre_key{k}.txt") for k in keys}
    sigma = accs[keys[0]]["sigma"]
    beta_sq = accs[keys[0]]["beta_sq"]
    s2 = sigma * sigma
    print(f"# sigma = {sigma!r}, beta^2 = {beta_sq}, beta^2 / (2d sigma^2) = {beta_sq / (DIM * s2):.6f} (2 pi = {2 * math.pi:.6f})")

    print("# (A) ||s||^2 against sigma^2 chi-square(2d), per key and pooled")
    alln = []
    summary = []
    for k in keys:
        nr = np.fromfile(d / f"norms_key{k}.bin", dtype="<u8")
        alln.append(nr)
        summary.append(norm_tests(f"key {k}", nr, sigma, beta_sq))
    pooled = np.concatenate(alln)
    pr = norm_tests("pooled", pooled, sigma, beta_sq)
    api = np.fromfile(d / "api_norms_key0.bin", dtype="<u8") if (d / "api_norms_key0.bin").exists() else np.zeros(0, dtype=np.uint64)
    if api.size:
        norm_tests("public samp_pre (key 0, hedged seeds)", api, sigma, beta_sq)
    fixed = np.fromfile(d / "fixed_norms_key0.bin", dtype="<u8") if (d / "fixed_norms_key0.bin").exists() else np.zeros(0, dtype=np.uint64)
    if fixed.size:
        norm_tests("one fixed target (key 0)", fixed, sigma, beta_sq)
    print(f"  per-key KS p-values: {', '.join(f'{x['ks_p']:.3f}' for x in summary)}; per-key mean z: "
          f"{', '.join(f'{x['z']:+.2f}' for x in summary)}")
    # Tail of the norm.
    mp.mp.prec = 128
    kk = mp.mpf(DIM)
    xb = mp.mpf(beta_sq) / mp.mpf(s2)
    tail = mp.gammainc(kk / 2, xb / 2, mp.inf, regularized=True)
    print(f"  tail: P(chi2_2048 > beta^2/sigma^2 = {mp.nstr(xb, 8)}) = 2^{float(mp.log(tail, 2)):.1f} per attempt "
          f"[computed, continuous approximation]; Micciancio-Regev Lemma 4.4 bound (1+eps)/(1-eps) 2^-2048 [recalled]; "
          f"observed: {int(np.sum(pooled > beta_sq))} of {pooled.size + api.size + fixed.size:,} attempts")

    print(f"# (B) per-coordinate moments, random targets (each key: 2048 coordinates x {accs[keys[0]]['calls']:,} samples)")
    zsq_all, var_all, kurt_z = [], [], []
    for k in keys:
        a = accs[k]
        n = a["calls"]
        m = a["sum"] / n
        v = a["sumsq"] / n - m * m
        zm = m / (sigma / math.sqrt(n))
        zv = (v / s2 - 1) / math.sqrt(2 / n)
        pooled_var = float(np.mean(v)) / s2
        se_pool = math.sqrt(2 / (n * DIM))
        e4 = a["sum4"] / (n * DIM) / (s2 * s2)
        zk = (e4 - 3) / math.sqrt(96 / (n * DIM))
        kurt_z.append(zk)
        var_all.append(pooled_var)
        print(f"  key {k}: means: sum z^2 = {float(np.sum(zm * zm)):.0f}/2048 (p {chi2_p(float(np.sum(zm * zm)), DIM):.3f}), "
              f"max |z| {float(np.max(np.abs(zm))):.2f}; variances: pooled var/sigma^2 = {pooled_var:.6f} "
              f"(z {(pooled_var - 1) / se_pool:+.2f}), sum z^2 = {float(np.sum(zv * zv)):.0f}/2048 "
              f"(p {chi2_p(float(np.sum(zv * zv)), DIM):.3f}), max |z| {float(np.max(np.abs(zv))):.2f}; "
              f"E s^4 / (3 sigma^4) = {e4 / 3:.6f} (z {zk:+.2f})")
    n_tot = sum(accs[k]["calls"] for k in keys)
    pv = float(np.mean(var_all))
    print(f"  all keys: pooled var/sigma^2 = {pv:.6f} (z {(pv - 1) / math.sqrt(2 / (n_tot * DIM)):+.2f}); "
          f"kurtosis z-scores {', '.join(f'{x:+.2f}' for x in kurt_z)}")

    if accs[keys[0]]["dir"].size == 0:
        print("# (C) no random directions in this run")
    else:
        directions(accs, keys, s2, n_tot)
    fft_and_fixed(d, accs, keys, sigma, s2)


def directions(accs, keys, s2, n_tot):
    print("# (C) 32 random unit directions u (shared by all keys): Var <s,u>, E<s,u>^3, E<s,u>^4")
    zv_all, z3_all, z4_all = [], [], []
    pooled_dir = np.zeros((32, 4))
    for k in keys:
        a = accs[k]
        n = a["calls"]
        dr = a["dir"]
        pooled_dir += dr
        m1 = dr[:, 0] / n
        var = dr[:, 1] / n - m1 * m1
        zv = (var / s2 - 1) / math.sqrt(2 / n)
        z3 = (dr[:, 2] / n) / math.sqrt(15 * s2 ** 3 / n)
        z4 = (dr[:, 3] / n / (s2 * s2) - 3) / math.sqrt(96 / n)
        zv_all.extend(zv)
        z3_all.extend(z3)
        z4_all.extend(z4)
    for name, zz in (("variance", zv_all), ("third moment", z3_all), ("fourth moment", z4_all)):
        zz = np.array(zz)
        print(f"  {name}: {zz.size} z-scores (8 keys x 32 directions): sum z^2 = {float(np.sum(zz * zz)):.1f}/{zz.size} "
              f"(p {chi2_p(float(np.sum(zz * zz)), zz.size):.3f}); max |z| {float(np.max(np.abs(zz))):.2f}; "
              f"mean z {float(np.mean(zz)):+.3f}")
    m1 = pooled_dir[:, 0] / n_tot
    var = pooled_dir[:, 1] / n_tot - m1 * m1
    rv = var / s2
    print(f"  pooled over keys: Var<s,u>/sigma^2 in [{rv.min():.5f}, {rv.max():.5f}], mean {rv.mean():.6f} "
          f"(SE of one direction {math.sqrt(2 / n_tot):.5f}; of the mean {math.sqrt(2 / (n_tot * 32)):.6f})")


def fft_and_fixed(d, accs, keys, sigma, s2):

    print("# (D) key-adapted directions at the d/2 FFT points: X = |p|^2/(d sigma^2) ~ Exp(1)")
    for k in keys:
        a = accs[k]
        n = a["calls"]
        fd = a["fft"]
        out = []
        for col, name in ((0, "w0 (row (g,-f))"), (2, "w1 (first GS step)")):
            m = fd[:, col] / (n * D * s2)
            m2 = fd[:, col + 1] / (n * (D * s2) ** 2)
            zk = (m - 1) * math.sqrt(n)
            zpool = (float(np.mean(m)) - 1) * math.sqrt(n * len(m))
            z2 = (float(np.mean(m2)) - 2) / math.sqrt(20 / (n * len(m)))
            out.append(f"{name}: mean {float(np.mean(m)):.6f} (z {zpool:+.2f}), per-point sum z^2 = "
                       f"{float(np.sum(zk * zk)):.0f}/{len(m)} (p {chi2_p(float(np.sum(zk * zk)), len(m)):.3f}), "
                       f"max |z| {float(np.max(np.abs(zk))):.2f}, E X^2/2 = {float(np.mean(m2)) / 2:.5f} (z {z2:+.2f})")
        print(f"  key {k}: " + "; ".join(out))

    if not (d / "fixed_key0.txt").exists():
        return
    print("# (E) one fixed target on key 0 (one coset): per-coordinate means and variances")
    fa = load_acc(d / "fixed_key0.txt")
    n = fa["calls"]
    m = fa["sum"] / n
    v = fa["sumsq"] / n - m * m
    zm = m / (sigma / math.sqrt(n))
    zv = (v / s2 - 1) / math.sqrt(2 / n)
    print(f"  N = {n:,}: means sum z^2 = {float(np.sum(zm * zm)):.1f}/2048 (p {chi2_p(float(np.sum(zm * zm)), DIM):.3f}), "
          f"max |z| {float(np.max(np.abs(zm))):.2f}; variances: pooled {float(np.mean(v)) / s2:.6f} "
          f"(z {(float(np.mean(v)) / s2 - 1) / math.sqrt(2 / (n * DIM)):+.2f}), sum z^2 = {float(np.sum(zv * zv)):.1f}/2048 "
          f"(p {chi2_p(float(np.sum(zv * zv)), DIM):.3f})")


if __name__ == "__main__":
    main()
