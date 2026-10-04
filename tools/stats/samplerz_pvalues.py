#!/usr/bin/env sage -python
"""Chi-square, G and moment tests of the crate's
SamplerZ against the exact D_{Z, sigma', mu}, with p-values.

Input: the histograms written by `examples/stats_samplerz.rs` (default data/samplerz_hist.txt):
70 (sigma', mu) pairs, sigma' over the whole PCS leaf range [sigma_min, 1.17^2 sigma_min], ten
centres each, DRAWS draws per pair.

Reference: D_{Z, sigma', mu}(z) = exp(-(z - mu)^2 / (2 sigma'^2)) / sum_k exp(-(k - mu)^2 / ...),
with sigma' = 1/isigma and mu the exact reals of the f64 inputs, in mpmath at 128 bits.

Per pair:
- chi-square and G (likelihood-ratio) statistics over the values with expected count >= 10,
  the two tails merged into one bin each (each with expected count >= 10); p-values from the
  regularised upper incomplete gamma (exact chi-square law with bins - 1 degrees of freedom);
- the same chi-square against the implementation's own exact law (samplerz_exact.law: table,
  FACCT and binary64 emulated), which differs from the ideal by far less than the test resolves;
- raw moments about mu, E(z - mu)^k for k = 1..4: z-score against the exact value with the exact
  standard error sqrt((E(z - mu)^{2k} - E(z - mu)^k ^2) / N), two-sided normal p-value;
- the acceptance rate draws/trials against the exact rate of the implementation, 0.63368744...

Summary: the distribution of the 70 chi-square p-values (Kolmogorov-Smirnov against U(0, 1) and
Fisher's combination), and of all moment p-values.

Run: sage -python samplerz_pvalues.py data/samplerz_hist.txt > samplerz_pvalues.out
"""
import math
import sys

import mpmath as mp
from scipy import stats

import samplerz_exact as se

mp.mp.prec = 128


def chi2_sf(x, df):
    return mp.gammainc(mp.mpf(df) / 2, mp.mpf(x) / 2, mp.inf, regularized=True)


def norm_p(z):
    return mp.erfc(abs(mp.mpf(z)) / mp.sqrt(2))


def parse(path):
    lines = open(path).read().splitlines()
    out = []
    header = lines[0]
    hdr = None
    for line in lines[1:]:
        f = line.split()
        if f[0] == "pair":
            hdr = (int(f[1]), se.f64(int(f[2], 16)), se.f64(int(f[3], 16)), int(f[4]), int(f[5]), int(f[6]))
        elif f[0] == "hist":
            out.append((hdr, [int(x) for x in f[1:]]))
    return header, out


def ideal_pmf(isigma, mu, zs):
    sig = 1 / mp.mpf(isigma)
    m = mp.mpf(mu)
    c = 1 / (2 * sig * sig)
    s = math.floor(mu)
    tot = mp.fsum(mp.exp(-((mp.mpf(k) - m) ** 2) * c) for k in range(s - 120, s + 121))
    return {z: mp.exp(-((mp.mpf(z) - m) ** 2) * c) / tot for z in zs}


def binned(obs, pmf, n):
    """Bins: values with expected >= 10; left and right tails merged (expected >= 10 each)."""
    zs = sorted(pmf)
    exp = {z: pmf[z] * n for z in zs}
    core = [z for z in zs if exp[z] >= 10]
    lo, hi = core[0], core[-1]
    bins = []
    left_e = mp.fsum(exp[z] for z in zs if z < lo)
    left_o = sum(obs.get(z, 0) for z in zs if z < lo)
    right_e = mp.fsum(exp[z] for z in zs if z > hi)
    right_o = sum(obs.get(z, 0) for z in zs if z > hi)
    # Fold small tails into the neighbouring core bin.
    core_bins = [[exp[z], obs.get(z, 0)] for z in core]
    if left_e >= 10:
        bins.append([left_e, left_o])
    else:
        core_bins[0][0] += left_e
        core_bins[0][1] += left_o
    bins.extend(core_bins)
    if right_e >= 10:
        bins.append([right_e, right_o])
    else:
        bins[-1][0] += right_e
        bins[-1][1] += right_o
    return bins


def chi2_and_g(bins):
    x2 = mp.fsum((o - e) ** 2 / e for e, o in bins)
    g = 2 * mp.fsum(o * mp.log(o / e) for e, o in bins if o > 0)
    df = len(bins) - 1
    return x2, g, df


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/samplerz_hist.txt"
    header, pairs = parse(path)
    base = se.pmf_from_rcdt(se.load_base_table(), 128)
    print(f"# samplerz_pvalues.py on {path}")
    print(f"# {header}")
    print(f"# mpmath {mp.__version__} ({mp.libmp.BACKEND}), scipy {stats.__name__ and __import__('scipy').__version__}")
    print("# pair | sigma' | mu | draws | chi2/df, p | G p | chi2 vs exact impl law, p | "
          "moments k=1..4: z (p) | acceptance (z)")
    p_chi, p_g, p_mom, p_acc, p_impl = [], [], [], [], []
    rows = []
    for (idx, isg, mu, n, trials, zmin), hist in pairs:
        obs = {zmin + i: c for i, c in enumerate(hist) if c}
        s = math.floor(mu)
        zs = list(range(s - 64, s + 65))
        pmf = ideal_pmf(isg, mu, zs)
        outside = sum(c for z, c in obs.items() if z not in pmf)
        assert outside == 0
        x2, g, df = chi2_and_g(binned(obs, pmf, n))
        pc, pg = chi2_sf(x2, df), chi2_sf(g, df)
        # Against the implementation's exact law.
        zrel, impl, _, C, _, _, _ = se.law(isg, mu, base)
        pim = {s + z: p for z, p in zip(zrel, impl)}
        pim_full = {z: pim.get(z, mp.mpf(0)) for z in zs}
        x2i, _, dfi = chi2_and_g(binned(obs, {z: v for z, v in pim_full.items() if v > 0}, n))
        pci = chi2_sf(x2i, dfi)
        # Moments about mu.
        m = mp.mpf(mu)
        mom = []
        for k in range(1, 5):
            ek = mp.fsum(pmf[z] * (z - m) ** k for z in zs)
            e2k = mp.fsum(pmf[z] * (z - m) ** (2 * k) for z in zs)
            sk = mp.fsum(c * (mp.mpf(z) - m) ** k for z, c in obs.items()) / n
            se_k = mp.sqrt((e2k - ek * ek) / n)
            zk = (sk - ek) / se_k
            mom.append((float(zk), float(norm_p(zk))))
            p_mom.append(float(norm_p(zk)))
        rate = mp.mpf(n) / trials
        se_acc = mp.sqrt(C * (1 - C) / trials)
        za = (rate - C) / se_acc
        p_acc.append(float(norm_p(za)))
        p_chi.append(float(pc))
        p_g.append(float(pg))
        p_impl.append(float(pci))
        rows.append((idx, 1 / isg, mu, n, float(x2), df, float(pc), float(pg), float(x2i), dfi, float(pci), mom,
                     float(rate), float(za)))
        momtxt = " ".join(f"{z:+.2f}({p:.3f})" for z, p in mom)
        print(f"{idx:2d} {1/isg:.6f} {mu:>18.12f} {n} chi2 {float(x2):8.2f}/{df:2d} p {float(pc):.4f} | "
              f"G p {float(pg):.4f} | impl p {float(pci):.4f} | {momtxt} | acc {float(rate):.6f} ({float(za):+.2f})",
              flush=True)
    print("# SUMMARY")
    for name, ps in (("chi-square vs ideal", p_chi), ("G vs ideal", p_g), ("chi-square vs exact impl law", p_impl),
                     ("moments (4 per pair)", p_mom), ("acceptance", p_acc)):
        ks = stats.kstest(ps, "uniform")
        fisher = stats.combine_pvalues(ps, method="fisher")
        print(f"#   {name}: {len(ps)} p-values, min {min(ps):.4f}, median {sorted(ps)[len(ps) // 2]:.4f}, "
              f"max {max(ps):.4f}; #p<0.01: {sum(1 for p in ps if p < 0.01)}, #p<0.05: {sum(1 for p in ps if p < 0.05)}; "
              f"KS vs U(0,1): D = {ks.statistic:.4f}, p = {ks.pvalue:.4f}; Fisher combined p = {fisher.pvalue:.4f}")
    n_total = sum(r[3] for r in rows)
    print(f"#   draws in total: {n_total:,}")


if __name__ == "__main__":
    main()
