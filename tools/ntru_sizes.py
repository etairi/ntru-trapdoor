#!/usr/bin/env sage -python
"""
Coefficient sizes inside a Pornin-style NTRUSolve, per recursion depth.

Purpose: decide whether pqe-hawk's NTRU solver (a port of Pornin's ntrugen,
src/keygen/ntru.rs) can be re-profiled for the PCS parameters
(n = 1024, q = 1048573, sigma_fg = 26.474), and how large its RNS buffers
and prime table must be.  Calibration: Falcon-1024 (q = 12289,
sigma_fg = 1.17*sqrt(q/2n) = 2.8661), whose profile is known from fn-dsa
(fn-dsa-kgen/src/ntru.rs:56-57).

Method (exact integer arithmetic, Sage ZZ[x]; float only for the Babai
rounding, as in Pornin/Prest):
  depth d: (f_d, g_d) with f_{d+1}(y) = N(f_d), N(f)(x^2) = f(x) f(-x).
  deepest: xgcd of the resultants, F_L = -v q, G_L = u q.
  lift:    F_d = F_{d+1}(x^2) g_d(-x), G_d = G_{d+1}(x^2) f_d(-x)  (unreduced)
  reduce:  F_d -= k f_d, G_d -= k g_d with k = round((F f* + G g*)/(f f* + g g*)),
           iterated on 53-bit top windows until no progress.
Recorded per depth: max bit length of |f_d|,|g_d|, of the unreduced (F_d,G_d)
and of the reduced (F_d,G_d); words = ceil((bits+1)/31) (signed 31-bit limbs,
the ntrugen representation).

f, g are drawn with Sage's DiscreteGaussianDistributionIntegerSampler
(sigma = standard-deviation parameter, rho = exp(-x^2/(2 sigma^2))).
"""
import sys, json, math, time
import numpy as np
from sage.all import ZZ, PolynomialRing, xgcd, set_random_seed
from sage.stats.distributions.discrete_gaussian_integer import \
    DiscreteGaussianDistributionIntegerSampler

R = PolynomialRing(ZZ, 'x')
X = R.gen()

def coeffs(p, m):
    c = p.list()
    return [int(v) for v in c] + [0] * (m - len(c))

def bits(cs):
    return max((abs(int(v)).bit_length() for v in cs), default=0)

def words(b):
    return (b + 1 + 30) // 31  # signed, 31-bit limbs

def modneg(p, m):
    # reduce p mod x^m + 1
    c = p.list()
    out = [0] * m
    for i, v in enumerate(c):
        if (i // m) % 2 == 0:
            out[i % m] += v
        else:
            out[i % m] -= v
    return R(out)

def field_norm(f, m):
    # N(f)(y) = fe(y)^2 - y fo(y)^2 mod (y^(m/2) + 1)
    c = coeffs(f, m)
    fe = R(c[0::2]); fo = R(c[1::2])
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

KBITS = 30  # bits of k computed per reduction round (float64 keeps ~45 good bits)

def reduce(f, g, F, G, m):
    """Babai round-off of (F,G) against (f,g), in rounds of <= KBITS bits.

    k = round((F f* + G g*)/(f f* + g g*) * 2^-s) from 53-bit top windows,
    then (F,G) -= 2^s k (f,g); s shrinks to 0, where one exact-scale Babai
    step finishes the reduction.
    """
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
            continue  # cannot happen: k has ~KBITS bits while s > 0
        kp = R(k)
        F = F - modneg(kp * f, m) * (1 << s)
        G = G - modneg(kp * g, m) * (1 << s)
        rounds += 1
        if rounds > 100000:
            raise RuntimeError("reduce did not converge")
    return F, G

def solve(f, g, q, m, depth, rec):
    if m == 1:
        fL, gL = coeffs(f, 1)[0], coeffs(g, 1)[0]
        d, u, v = xgcd(fL, gL)
        if d != 1:
            return None
        F, G = R(-v * q), R(u * q)
        rec[depth] = dict(fg=max(bits([fL]), bits([gL])), FG_unred=None,
                          FG_red=max(bits([-v * q]), bits([u * q])))
        return F, G
    fn, gn = field_norm(f, m), field_norm(g, m)
    r = solve(fn, gn, q, m // 2, depth + 1, rec)
    if r is None:
        return None
    Fd, Gd = r
    F = modneg(lift(Fd, m // 2) * galois(g, m), m)
    G = modneg(lift(Gd, m // 2) * galois(f, m), m)
    unred = max(bits(coeffs(F, m)), bits(coeffs(G, m)))
    extra = {}
    if depth == 0:
        # Quantities that pqe-hawk's solve_ntru_depth0 holds modulo ONE
        # 31-bit prime and then normalises to (-p/2, p/2) with mp_norm
        # (src/keygen/ntru.rs:869-923): kn = F adj(f) + G adj(g) and
        # kd = f adj(f) + g adj(g), with (F, G) unreduced.
        fadj, gadj = adj(f, m), adj(g, m)
        kn = modneg(F * fadj + G * gadj, m)
        kd = modneg(f * fadj + g * gadj, m)
        extra = dict(kn_bits=bits(coeffs(kn, m)), kd_bits=bits(coeffs(kd, m)))
    F, G = reduce(f, g, F, G, m)
    rec[depth] = dict(fg=max(bits(coeffs(f, m)), bits(coeffs(g, m))),
                      FG_unred=unred,
                      FG_red=max(bits(coeffs(F, m)), bits(coeffs(G, m))), **extra)
    return F, G

def adj(f, m):
    # adjoint f*(x) = f(1/x) mod x^m + 1: c0, -c_{m-1}, ..., -c_1
    c = coeffs(f, m)
    return R([c[0]] + [-c[m - i] for i in range(1, m)])

def gs_norm_sq(fc, gc, q):
    # Falcon/DLP14 Gram-Schmidt norm: max(||(g,-f)||, ||q (f*, g*)/(ff*+gg*)||)
    m = len(fc)
    fa, ga = nfft(fc), nfft(gc)
    d = (np.abs(fa) ** 2 + np.abs(ga) ** 2)
    n1 = sum(v * v for v in fc) + sum(v * v for v in gc)
    n2 = (q * q) * float(np.sum(1.0 / d)) / m
    return max(n1, n2)

def run(label, n, q, sigma, keys, seed):
    set_random_seed(seed)
    D = DiscreteGaussianDistributionIntegerSampler(sigma=sigma)
    out = []
    tried = gs_ok = 0
    t0 = time.time()
    while len(out) < keys:
        fc = [int(D()) for _ in range(n)]
        gc = [int(D()) for _ in range(n)]
        tried += 1
        gs = gs_norm_sq(fc, gc, q)
        gs_pass = gs <= (1.17 ** 2) * q
        gs_ok += int(gs_pass)
        f, g = R(fc), R(gc)
        rec = {}
        r = solve(f, g, q, n, 0, rec)
        if r is None:
            continue
        F, G = r
        # exact check of the NTRU equation
        lhs = modneg(f * G - g * F, n)
        assert lhs == R(q), "NTRU equation failed"
        Fc, Gc = coeffs(F, n), coeffs(G, n)
        out.append(dict(rec={int(k): v for k, v in rec.items()},
                        max_fg=max(abs(v) for v in fc + gc),
                        max_FG=max(abs(v) for v in Fc + Gc),
                        gs_ratio=math.sqrt(gs) / math.sqrt(q), gs_pass=bool(gs_pass)))
    dt = time.time() - t0
    L = int(math.log2(n))
    print(f"== {label}: n={n} q={q} sigma={sigma} solved keys={keys} seed={seed} "
          f"(drawn {tried}; GS<=1.17sqrt(q): {gs_ok}/{tried}; {dt:.1f}s)")
    agg_all = {}
    for name, sel in (("all solved keys", out),
                      ("GS-accepted keys only", [o for o in out if o['gs_pass']])):
        if not sel:
            print(f" [{name}: none]")
            continue
        print(f" [{name}: {len(sel)}]")
        print(" depth  max bits f,g -> words | unreduced F,G -> words | reduced F,G -> words")
        agg = {}
        for d in range(L + 1):
            fg = max(o['rec'][d]['fg'] for o in sel)
            un = [o['rec'][d]['FG_unred'] for o in sel if o['rec'][d]['FG_unred'] is not None]
            un = max(un) if un else None
            rd = max(o['rec'][d]['FG_red'] for o in sel)
            agg[d] = dict(fg_bits=fg, fg_words=words(fg),
                          unred_bits=un, unred_words=(words(un) if un is not None else None),
                          red_bits=rd, red_words=words(rd))
            print(f" {d:5d}  {fg:6d} -> {words(fg):3d}       | "
                  f"{(un if un is not None else '-'):>6} -> {(words(un) if un is not None else '-'):>3}"
                  f"          | {rd:6d} -> {words(rd):3d}")
        kn = max(o['rec'][0]['kn_bits'] for o in sel)
        kd = max(o['rec'][0]['kd_bits'] for o in sel)
        print(f" depth 0: max bits kn = F adj f + G adj g (unreduced): {kn}; "
              f"kd = f adj f + g adj g: {kd}  (single 31-bit prime needs <= 30)")
        print(f" max |f|,|g| coefficient: {max(o['max_fg'] for o in sel)}; "
              f"max |F|,|G| (reduced, depth 0): {max(o['max_FG'] for o in sel)}")
        agg_all[name] = dict(per_depth=agg, kn_bits=kn, kd_bits=kd,
                             max_fg=max(o['max_fg'] for o in sel),
                             max_FG=max(o['max_FG'] for o in sel), keys=len(sel))
    r = [o['gs_ratio'] for o in out]
    print(f" GS/sqrt(q) over solved keys: min {min(r):.3f}, median "
          f"{sorted(r)[len(r)//2]:.3f}, max {max(r):.3f}")
    return dict(label=label, n=n, q=q, sigma=sigma, keys=keys, seed=seed,
                drawn=tried, gs_pass=gs_ok, seconds=dt, aggregates=agg_all,
                gs_ratios=r)

if __name__ == "__main__":
    keys = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    res = []
    q_f = 12289
    res.append(run("Falcon-1024 (calibration)", 1024, q_f,
                   1.17 * math.sqrt(q_f / 2048), keys, 1))
    q_p = 1048573
    res.append(run("PCS (Sigma-vSIS)", 1024, q_p,
                   1.17 * math.sqrt(q_p / 2048), keys, 2))
    with open(sys.argv[2] if len(sys.argv) > 2 else "ntru_sizes.json", "w") as fh:
        json.dump(res, fh, indent=1, default=str)
