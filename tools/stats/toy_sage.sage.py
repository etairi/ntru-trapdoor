#!/usr/bin/env sage -python
"""The crate's SampPre against Sage's
DiscreteGaussianDistributionLatticeSampler on the same coset, at toy sizes.

Toys (the lattice is {(s0, s1) : s0 + h s1 = 0 mod q} in Z^{2n}, the coset s0 + h s1 = t):
  A: n = 2, q = 17 (the key of the crate's tests/sign_toy_exact.rs), sigma = 11, sigma_min = 2.0;
     leaves 2.506 and 2.840. Dimension 4: the coset Gaussian is enumerated exactly.
  B: n = 4, q = 13, C: n = 8, q = 29 (q = 5 mod 8 as at the PCS set, so x^n + 1 has two
     irreducible factors mod q, as x^1024 + 1 mod 1048573): f, g from Sage's
     DiscreteGaussianDistributionIntegerSampler at 1.17 sqrt(q/(2n)); NTRUSolve by the
     independent Sage solver of tools/ntru_sizes.py; the
     ffLDL leaves are computed at 200 bits (precision_hp.py); sigma = 3.0 sqrt(min leaf value),
     so the leaf widths lie in [~2.07, 3.0]. These exercise the recursive path of ffSampling
     (split, merge, l10 products), which logn = 1 does not.

Sage's sampler (Albrecht's GPV/Klein sampler, in the standard-deviation convention
rho(x) = exp(-|x - c|^2 / (2 sigma^2))) is given the NTRU basis rows x^k (g, -f), x^k (G, -F) and
the centre c = (t, 0); a sample v is a lattice vector and s = c - v. Its integer sampler cuts the
tail at 6 sigma' per coordinate (Sage's default tau = 6, probability about 2e-9 per coordinate).

Subcommands:
  gen DIR                 write DIR/toy{A,B,C}.spec (for examples/stats_toy.rs) and toy*.json
  sage DIR TOY N WORKERS [BATCH]  N Sage samples of toy TOY -> DIR/toyTOY_sage.npy (BATCH > 0:
                          other seeds, DIR/toyTOY_sage_bBATCH.npy)
  analyze DIR [TOYS]      compare DIR/toyX_crate.bin (stats_toy) with Sage and, for A, the exact law
"""
import json
import math
import multiprocessing as mpr
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NS = str(HERE.parent)  # tools/, which holds ntru_sizes.py


def rot(a, k):
    n = len(a)
    out = [0] * n
    for i, c in enumerate(a):
        j = i + k
        if j < n:
            out[j] += c
        else:
            out[j - n] -= c
    return out


def basis_rows(f, g, F, G):
    n = len(f)
    rows = [rot(g, k) + [-v for v in rot(f, k)] for k in range(n)]
    rows += [rot(G, k) + [-v for v in rot(F, k)] for k in range(n)]
    return rows


def ffldl_leaves(f, g, F, G):
    """The 2n ffLDL leaf values (squared Gram-Schmidt norms in ffSampling's order), 200-bit."""
    sys.path.insert(0, str(HERE))
    import precision_hp as ph
    mp = ph.hp()
    B = ph.HP(mp)
    n = len(f)
    fh, gh, Fh, Gh = (ph.fft(B, [mp.mpc(v, 0) for v in c]) for c in (f, g, F, G))
    G00 = [gh[j] * gh[j].conjugate() + fh[j] * fh[j].conjugate() for j in range(n)]
    G01 = [gh[j] * Gh[j].conjugate() + fh[j] * Fh[j].conjugate() for j in range(n)]
    G11 = [Gh[j] * Gh[j].conjugate() + Fh[j] * Fh[j].conjugate() for j in range(n)]
    T = ph.ffldl(B, G00, G01, G11)
    out = []

    def walk(t):
        if t[0] == "leaf":
            out.extend([float(t[2]), float(t[3])])
            return
        walk(t[3])
        walk(t[2])

    walk(T)
    return out


def gen(d):
    from sage.all import GF, PolynomialRing, matrix, ZZ, set_random_seed
    from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler
    sys.path.insert(0, NS)
    import ntru_sizes as ns
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    toys = {}
    # Toy A: the crate's tests/sign_toy_exact.rs.
    toys["A"] = dict(logn=1, q=17, sigma=11.0, sigma_min=2.0, f=[-3, -2], g=[-1, -1], F=[2, -2], G=[-3, 2], t=[5, 11])
    for name, logn, q, seed in (("B", 2, 13, 7100), ("C", 3, 29, 7200)):
        n = 1 << logn
        D = DiscreteGaussianDistributionIntegerSampler(sigma=1.17 * math.sqrt(q / (2 * n)))
        Rq = PolynomialRing(GF(q), "y")
        M = Rq.gen() ** n + 1
        found = None
        for attempt in range(200000):
            set_random_seed(seed + attempt)
            fc = [int(D()) for _ in range(n)]
            gc = [int(D()) for _ in range(n)]
            if Rq(fc).gcd(M) != 1:
                continue
            ns.N = n
            r = ns.solve(ns.R(fc), ns.R(gc), q, n, 0, {})
            if r is None:
                continue
            Fc, Gc = ns.coeffs(r[0], n), ns.coeffs(r[1], n)
            assert ns.modneg(ns.R(fc) * ns.R(Gc) - ns.R(gc) * ns.R(Fc), n) == ns.R(q)
            leaves = ffldl_leaves(fc, gc, Fc, Gc)
            ratio = math.sqrt(max(leaves) / min(leaves))
            if ratio > 1.45:
                continue
            found = (fc, gc, Fc, Gc, leaves, attempt)
            break
        fc, gc, Fc, Gc, leaves, attempt = found
        sigma = 3.0 * math.sqrt(min(leaves))
        widths = [sigma / math.sqrt(v) for v in leaves]
        sigma_min = min(widths) * (1 - 1e-9)
        set_random_seed(seed + 99999)
        t = [int(x) for x in np.random.default_rng(seed).integers(0, q, n)]
        toys[name] = dict(logn=logn, q=q, sigma=sigma, sigma_min=sigma_min, f=fc, g=gc, F=Fc, G=Gc, t=t,
                          seed=seed + attempt, leaf_widths=widths)
    for name, ty in toys.items():
        n = 1 << ty["logn"]
        q = ty["q"]
        Rq = PolynomialRing(GF(q), "y")
        M = Rq.gen() ** n + 1
        h = (Rq(ty["g"]) * Rq(ty["f"]).inverse_mod(M)) % M
        ty["h"] = [int(c) for c in h.list()] + [0] * (n - len(h.list()))
        Bm = matrix(ZZ, basis_rows(ty["f"], ty["g"], ty["F"], ty["G"]))
        Gs, _ = Bm.gram_schmidt()
        ty["sage_gso_widths"] = [ty["sigma"] / float(r.norm()) for r in Gs]
        ty["det"] = int(abs(Bm.det()))
        spec = [f"# toy {name} for examples/stats_toy.rs (written by toy_sage.sage.py gen)",
                f"logn {ty['logn']}", f"q {q}", f"sigma {ty['sigma']!r}", f"sigma_min {ty['sigma_min']!r}"]
        for k in ("f", "g", "F", "G", "t"):
            spec.append(f"{k} " + " ".join(str(v) for v in ty[k]))
        (d / f"toy{name}.spec").write_text("\n".join(spec) + "\n")
        (d / f"toy{name}.json").write_text(json.dumps(ty, indent=1))
        print(f"toy {name}: n {n} q {q} sigma {ty['sigma']:.6f} sigma_min {ty['sigma_min']:.6f} det {ty['det']} "
              f"(q^n = {q ** n}); f {ty['f']} g {ty['g']} F {ty['F']} G {ty['G']} t {ty['t']} h {ty['h']}")
        if "leaf_widths" in ty:
            print(f"   ffLDL leaf widths {[round(w, 4) for w in ty['leaf_widths']]}")
        print(f"   Sage's Klein widths (row order) {[round(w, 4) for w in ty['sage_gso_widths']]}")


def sage_worker(args):
    d, name, count, wid, batch = args
    from sage.all import matrix, ZZ, QQ, RR, vector, set_random_seed
    from sage.stats.distributions.discrete_gaussian_lattice import DiscreteGaussianDistributionLatticeSampler as DGL
    ty = json.loads((Path(d) / f"toy{name}.json").read_text())
    n = 1 << ty["logn"]
    Bm = matrix(ZZ, basis_rows(ty["f"], ty["g"], ty["F"], ty["G"]))
    c = vector(QQ, ty["t"] + [0] * n)
    set_random_seed(424242 + 1000 * ord(name) + wid + 100000 * batch)
    sampler = DGL(Bm, RR(ty["sigma"]), c)
    out = np.empty((count, 2 * n), dtype=np.int16)
    cz = np.array(ty["t"] + [0] * n, dtype=np.int64)
    for i in range(count):
        v = np.array([int(x) for x in sampler()], dtype=np.int64)
        out[i] = cz - v
    return out


def sage_cmd(d, name, n, workers, batch=0):
    t0 = time.time()
    per = n // workers
    ctx = mpr.get_context("spawn")
    with ctx.Pool(workers) as pool:
        parts = pool.map(sage_worker, [(d, name, per + (1 if w < n % workers else 0), w, batch) for w in range(workers)])
    arr = np.concatenate(parts)
    suffix = "" if batch == 0 else f"_b{batch}"
    np.save(Path(d) / f"toy{name}_sage{suffix}.npy", arr)
    print(f"toy {name}: {arr.shape[0]} Sage samples in {time.time() - t0:.0f} s")


# --- Analysis ---------------------------------------------------------------------------------------


def check_coset(ty, s):
    """Every row satisfies s0 + h s1 = t mod q."""
    n = 1 << ty["logn"]
    q = ty["q"]
    h = np.array(ty["h"], dtype=np.int64)
    s0, s1 = s[:, :n].astype(np.int64), s[:, n:].astype(np.int64)
    acc = s0.copy()
    for j in range(n):
        for i in range(n):
            k = i + j
            if k < n:
                acc[:, k] += h[i] * s1[:, j]
            else:
                acc[:, k - n] -= h[i] * s1[:, j]
    return bool(np.all(np.mod(acc - np.array(ty["t"]), q) == 0))


def chi2_p(x, df):
    from scipy import stats
    return float(stats.chi2.sf(x, df))


def exact_toyA(ty):
    """All coset points within 8 sigma of 0 and their exact (f64) probabilities."""
    q, sigma = ty["q"], ty["sigma"]
    h0, h1 = ty["h"]
    t0, t1 = ty["t"]
    r = int(math.ceil(8 * sigma))
    pts, w = [], []
    for a in range(-r, r + 1):
        for b in range(-r, r + 1):
            c0 = (t0 - (h0 * a - h1 * b)) % q
            c1 = (t1 - (h0 * b + h1 * a)) % q
            xs = np.arange(c0 - q * ((c0 + r) // q), r + 1, q)
            ys = np.arange(c1 - q * ((c1 + r) // q), r + 1, q)
            xs = xs[xs >= -r]
            ys = ys[ys >= -r]
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            n2 = X * X + Y * Y + a * a + b * b
            for x, y, nn in zip(X.ravel(), Y.ravel(), n2.ravel()):
                pts.append((int(x), int(y), a, b))
                w.append(math.exp(-nn / (2 * sigma * sigma)))
    w = np.array(w)
    return pts, w / w.sum()


def counts_of(s):
    keys, cnt = np.unique(s, axis=0, return_counts=True)
    return {tuple(int(v) for v in k): int(c) for k, c in zip(keys, cnt)}


def gof_exact(pts, prob, obs, n, nmin):
    """Chi-square of observed counts against exact probabilities: points with expected >= 10
    on their own, the rest grouped by squared norm into bins of expected >= 10."""
    bins = []
    shells = {}
    seen = 0
    for p, pr in zip(pts, prob):
        e = pr * n
        o = obs.get(p, 0)
        seen += o
        if pr * nmin >= 10:
            bins.append([e, o])
        else:
            key = sum(v * v for v in p)
            sh = shells.setdefault(key, [0.0, 0])
            sh[0] += e
            sh[1] += o
    e_acc, o_acc = 0.0, 0
    for key in sorted(shells):
        e_acc += shells[key][0]
        o_acc += shells[key][1]
        if e_acc * nmin / n >= 10:
            bins.append([e_acc, o_acc])
            e_acc, o_acc = 0.0, 0
    bins[-1][0] += e_acc
    bins[-1][1] += o_acc
    x2 = sum((o - e) ** 2 / e for e, o in bins)
    return x2, len(bins) - 1, seen


def two_sample(c1, c2, n1, n2, keys):
    """Two-sample chi-square over the given bins (lists of counts)."""
    k1, k2 = math.sqrt(n2 / n1), math.sqrt(n1 / n2)
    x2 = 0.0
    df = 0
    for a, b in zip(c1, c2):
        if a + b == 0:
            continue
        x2 += (k1 * a - k2 * b) ** 2 / (a + b)
        df += 1
    return x2, df - 1


def binned_two_sample(x1, x2, nbins=50):
    edges = np.unique(np.quantile(np.concatenate([x1, x2]), np.linspace(0, 1, nbins + 1)))
    edges[0] -= 1
    edges[-1] += 1
    c1, _ = np.histogram(x1, bins=edges)
    c2, _ = np.histogram(x2, bins=edges)
    return two_sample(c1, c2, len(x1), len(x2), None)


def marginal_two_sample(v1, v2, pooled_min=200):
    lo, hi = min(v1.min(), v2.min()), max(v1.max(), v2.max())
    vals = np.arange(lo, hi + 1)
    c1 = np.array([np.sum(v1 == v) for v in vals])
    c2 = np.array([np.sum(v2 == v) for v in vals])
    # merge neighbouring values until each bin has >= pooled_min pooled counts (with a 10:1
    # sample-size ratio, 200 pooled counts give the smaller sample about 18 expected)
    b1, b2, a1, a2 = [], [], 0, 0
    for x, y in zip(c1, c2):
        a1 += x
        a2 += y
        if a1 + a2 >= pooled_min:
            b1.append(a1)
            b2.append(a2)
            a1 = a2 = 0
    b1[-1] += a1
    b2[-1] += a2
    return two_sample(b1, b2, len(v1), len(v2), None)


def marginal_vs_dz(v, sigma):
    """One-sample chi-square of a coordinate marginal against D_{Z, sigma, 0} (bins: values with
    expected >= 10, tails merged)."""
    import mpmath as mpm
    mpm.mp.prec = 100
    lo, hi = int(-14 * sigma), int(14 * sigma)
    w = {x: mpm.exp(-mpm.mpf(x * x) / (2 * mpm.mpf(sigma) ** 2)) for x in range(lo, hi + 1)}
    tot = mpm.fsum(w.values())
    n = len(v)
    vals, cnt = np.unique(v, return_counts=True)
    obs = dict(zip(vals.tolist(), cnt.tolist()))
    assert all(lo <= x <= hi for x in obs)
    bins, e_acc, o_acc = [], 0.0, 0
    for x in range(lo, hi + 1):
        e_acc += float(w[x] / tot) * n
        o_acc += obs.get(x, 0)
        if e_acc >= 10:
            bins.append([e_acc, o_acc])
            e_acc, o_acc = 0.0, 0
    bins[-1][0] += e_acc
    bins[-1][1] += o_acc
    x2 = sum((o - e) ** 2 / e for e, o in bins)
    return x2, len(bins) - 1


def key_directions(ty):
    """Orthonormal key-adapted directions in R^{2n}: per evaluation point (n/2 of them, one per
    conjugate pair), w0 = (g, -f)/|.| and w1 = (conj f, conj g)/|.|, real and imaginary parts."""
    n = 1 << ty["logn"]
    zeta = [np.exp(1j * np.pi * (2 * k + 1) / n) for k in range(n // 2)]
    dirs = []
    for z in zeta:
        powers = np.array([z ** j for j in range(n)])
        fh = np.dot(ty["f"], powers)
        gh = np.dot(ty["g"], powers)
        nk = math.sqrt(abs(fh) ** 2 + abs(gh) ** 2)
        for (a, b) in ((gh / nk, -fh / nk), (np.conj(fh) / nk, np.conj(gh) / nk)):
            # p = <s_hat, w> = sum_j s0_j z^j conj(a) + s1_j z^j conj(b): linear in s, as a complex row.
            row = np.concatenate([powers * np.conj(a), powers * np.conj(b)])
            dirs.append(np.real(row) * math.sqrt(2 / n))
            dirs.append(np.imag(row) * math.sqrt(2 / n))
    W = np.array(dirs)
    return W


def analyze(d, toys="ABC"):
    from scipy import stats
    d = Path(d)
    print(f"# toy_sage.sage.py analyze: numpy {np.__version__}, scipy {__import__('scipy').__version__}")
    for name in toys:
        ty = json.loads((d / f"toy{name}.json").read_text())
        n = 1 << ty["logn"]
        dim = 2 * n
        sigma = ty["sigma"]
        crate = np.fromfile(d / f"toy{name}_crate.bin", dtype="<i2").reshape(-1, dim)
        sage = np.load(d / f"toy{name}_sage.npy")
        n1, n2 = crate.shape[0], sage.shape[0]
        print(f"== toy {name}: n {n}, q {ty['q']}, sigma {sigma:.6f}; crate {n1:,} samples, Sage {n2:,} samples")
        print(f"   coset check (s0 + h s1 = t): crate {check_coset(ty, crate[:200000])} (first 200,000), "
              f"Sage {check_coset(ty, sage)}")
        if name == "A":
            pts, prob = exact_toyA(ty)
            oc, osg = counts_of(crate), counts_of(sage)
            x2c, dfc, seen_c = gof_exact(pts, prob, oc, n1, n2)
            x2s, dfs, seen_s = gof_exact(pts, prob, osg, n2, n2)
            print(f"   exact law: {len(pts):,} coset points within 8 sigma (the rest is below 1e-13)")
            print(f"   crate vs exact: chi2 {x2c:.1f}/{dfc}, p = {chi2_p(x2c, dfc):.4f} (all {seen_c} samples inside)")
            print(f"   Sage  vs exact: chi2 {x2s:.1f}/{dfs}, p = {chi2_p(x2s, dfs):.4f} (all {seen_s} samples inside)")
            # Two-sample on the same bins as for the Sage sample.
            bins_c, bins_s = [], []
            shells = {}
            for p, pr in zip(pts, prob):
                if pr * n2 >= 10:
                    bins_c.append(oc.get(p, 0))
                    bins_s.append(osg.get(p, 0))
                else:
                    key = sum(v * v for v in p)
                    sh = shells.setdefault(key, [0.0, 0, 0])
                    sh[0] += pr
                    sh[1] += oc.get(p, 0)
                    sh[2] += osg.get(p, 0)
            acc = [0.0, 0, 0]
            for key in sorted(shells):
                for i in range(3):
                    acc[i] += shells[key][i]
                if acc[0] * n2 >= 10:
                    bins_c.append(acc[1])
                    bins_s.append(acc[2])
                    acc = [0.0, 0, 0]
            bins_c[-1] += acc[1]
            bins_s[-1] += acc[2]
            x2, df = two_sample(bins_c, bins_s, n1, n2, None)
            print(f"   crate vs Sage (two-sample, joint law): chi2 {x2:.1f}/{df}, p = {chi2_p(x2, df):.4f}")
        # Moments and directions for every toy.
        W = key_directions(ty)
        for lab, smp in (("crate", crate), ("Sage", sage)):
            x = smp.astype(np.float64)
            m = x.mean(axis=0)
            zm = m / (sigma / math.sqrt(len(x)))
            cov = np.cov(x, rowvar=False) / (sigma * sigma)
            iu = np.triu_indices(dim, 1)
            zc = cov[iu] * math.sqrt(len(x))
            zv = (np.diag(cov) - 1) / math.sqrt(2 / len(x))
            P = x @ W.T
            pv = P.var(axis=0) / (sigma * sigma)
            zp = (pv - 1) / math.sqrt(2 / len(x))
            nrm = (x * x).sum(axis=1) / (sigma * sigma)
            zn = (nrm.mean() - dim) / math.sqrt(2 * dim / len(x))
            print(f"   {lab:5s}: E||s||^2/(2n sigma^2) = {nrm.mean() / dim:.5f} (z {zn:+.2f}); means sum z^2 = "
                  f"{float(np.sum(zm * zm)):.1f}/{dim} (p {chi2_p(float(np.sum(zm * zm)), dim):.3f}); coordinate variances "
                  f"sum z^2 = {float(np.sum(zv * zv)):.1f}/{dim} (p {chi2_p(float(np.sum(zv * zv)), dim):.3f}); correlations "
                  f"sum z^2 = {float(np.sum(zc * zc)):.1f}/{len(zc)} (p {chi2_p(float(np.sum(zc * zc)), len(zc)):.3f}); "
                  f"key-adapted directions: Var/sigma^2 in [{pv.min():.4f}, {pv.max():.4f}], sum z^2 = "
                  f"{float(np.sum(zp * zp)):.1f}/{len(zp)} (p {chi2_p(float(np.sum(zp * zp)), len(zp)):.3f})")
        # Two-sample tests.
        xs = []
        for j in range(dim):
            x2, df = marginal_two_sample(crate[:, j].astype(np.int64), sage[:, j].astype(np.int64))
            xs.append((x2, df))
        tot = sum(x for x, _ in xs)
        tdf = sum(df for _, df in xs)
        pm = [chi2_p(x, df) for x, df in xs]
        print(f"   crate vs Sage, coordinate marginals (bins of >= 200 pooled counts): {dim} two-sample chi2 tests, "
              f"p in [{min(pm):.3f}, {max(pm):.3f}]; sum {tot:.1f}/{tdf} (p {chi2_p(tot, tdf):.3f}); "
              f"KS of the {dim} p-values vs U(0,1): p {stats.kstest(pm, 'uniform').pvalue:.3f}")
        xs20 = [marginal_two_sample(crate[:, j].astype(np.int64), sage[:, j].astype(np.int64), 20) for j in range(dim)]
        print(f"   (same with bins of >= 20 pooled counts, too small for the 10:1 ratio: sum "
              f"{sum(x for x, _ in xs20):.1f}/{sum(d for _, d in xs20)}, p {chi2_p(sum(x for x, _ in xs20), sum(d for _, d in xs20)):.3f})")
        if name != "A":
            for lab, smp in (("crate", crate), ("Sage", sage)):
                one = [marginal_vs_dz(smp[:, j].astype(np.int64), sigma) for j in range(dim)]
                p1 = [chi2_p(x, d) for x, d in one]
                print(f"   {lab:5s} coordinate marginals vs D_(Z,sigma): {dim} chi2 tests, p in [{min(p1):.3f}, {max(p1):.3f}]; "
                      f"sum {sum(x for x, _ in one):.1f}/{sum(d for _, d in one)} (p {chi2_p(sum(x for x, _ in one), sum(d for _, d in one)):.3f})")
        n1v = (crate.astype(np.float64) ** 2).sum(axis=1)
        n2v = (sage.astype(np.float64) ** 2).sum(axis=1)
        x2, df = binned_two_sample(n1v, n2v)
        ks = stats.ks_2samp(n1v, n2v)
        print(f"   crate vs Sage, ||s||^2: 50-bin two-sample chi2 {x2:.1f}/{df} (p {chi2_p(x2, df):.3f}); "
              f"KS two-sample p {ks.pvalue:.3f}")
        P1 = crate.astype(np.float64) @ W.T
        P2 = sage.astype(np.float64) @ W.T
        pks = [stats.ks_2samp(P1[:, i], P2[:, i]).pvalue for i in range(W.shape[0])]
        print(f"   crate vs Sage, key-adapted projections: {W.shape[0]} two-sample KS tests, p in "
              f"[{min(pks):.3f}, {max(pks):.3f}], Fisher combined p {stats.combine_pvalues(pks).pvalue:.3f}")


def sagecheck(d, name):
    """Every Sage batch of toy `name` alone and pooled: coordinate marginals vs D_{Z,sigma}, and the
    pooled batches against the crate (two-sample marginals, bins of >= 200 pooled counts)."""
    from scipy import stats
    d = Path(d)
    ty = json.loads((d / f"toy{name}.json").read_text())
    n = 1 << ty["logn"]
    dim = 2 * n
    sigma = ty["sigma"]
    files = sorted(d.glob(f"toy{name}_sage*.npy"))
    batches = [(f.name, np.load(f)) for f in files]
    batches.append(("pooled", np.concatenate([b for _, b in batches])))
    print(f"# toy_sage.sage.py sagecheck {name}: Sage batches {[f.name for f in files]}")
    for lab, smp in batches:
        one = [marginal_vs_dz(smp[:, j].astype(np.int64), sigma) for j in range(dim)]
        p1 = [chi2_p(x, k) for x, k in one]
        tot, df = sum(x for x, _ in one), sum(k for _, k in one)
        print(f"   {lab:18s} N = {len(smp):,}: marginals vs D_(Z,sigma): p in [{min(p1):.3f}, {max(p1):.3f}], "
              f"sum {tot:.1f}/{df} (p {chi2_p(tot, df):.3f})")
    crate = np.fromfile(d / f"toy{name}_crate.bin", dtype="<i2").reshape(-1, dim)
    pooled = batches[-1][1]
    xs = [marginal_two_sample(crate[:, j].astype(np.int64), pooled[:, j].astype(np.int64)) for j in range(dim)]
    pm = [chi2_p(x, k) for x, k in xs]
    tot, df = sum(x for x, _ in xs), sum(k for _, k in xs)
    print(f"   crate ({len(crate):,}) vs pooled Sage ({len(pooled):,}), marginals two-sample: p in [{min(pm):.3f}, {max(pm):.3f}], "
          f"sum {tot:.1f}/{df} (p {chi2_p(tot, df):.3f}); KS of the p-values {stats.kstest(pm, 'uniform').pvalue:.3f}")


def main():
    cmd = sys.argv[1]
    if cmd == "gen":
        gen(sys.argv[2])
    elif cmd == "sage":
        sage_cmd(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
                 int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    elif cmd == "sagecheck":
        sagecheck(sys.argv[2], sys.argv[3])
    elif cmd == "analyze":
        analyze(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "ABC")


if __name__ == "__main__":
    main()
