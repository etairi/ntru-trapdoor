#!/usr/bin/env python3
"""Every numeric constant of the ntru-trapdoor design, recomputed at high precision.

The f64 constants of the parameter presets, computed at 256 bits.

Run:  <venv>/bin/python constants.py > constants.out
      (mpmath 1.3.0 or 1.4.1)

Conventions (checked against DKLW25 p. 5 and p. 17, and against fn-dsa sampler.rs:32-73):
  * GPV s-parameter: rho_s(x) = exp(-pi |x|^2 / s^2).  The paper's varsigma is an s-parameter.
  * Falcon/fn-dsa standard-deviation parameter: rho(x) = exp(-x^2 / (2 sigma^2)), so
    sigma = s / sqrt(2 pi).
  * eta_eps(Z^m) <= sqrt(ln(2 m (1 + 1/eps)) / pi)  (s-convention); m = 2d.
  * sigma_min = eta_eps(Z^{2d}) / sqrt(2 pi)  (Falcon's smoothz2n).
  * sigma = 1.17 sqrt(q) sigma_min;  varsigma = sigma sqrt(2 pi).
  * leaves of the ffLDL tree: sigma'_i = sigma / ||b~_i|| in [sigma_min, 1.17^2 sigma_min].
"""
import math
from fractions import Fraction

import mpmath as mp

mp.mp.prec = 256


def is_prime(n):
    if n < 2:
        return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:  # deterministic for n < 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def f64(x):
    """Nearest binary64 of an mpf (round to nearest even) and its bit pattern."""
    v = float(mp.mpf(x))  # mpmath converts with correct rounding
    import struct
    bits = struct.unpack("<Q", struct.pack("<d", v))[0]
    return v, bits


def scaled(j, sc):
    """fn-dsa FLR::scaled(j, sc) = j * 2^sc, exactly."""
    return mp.mpf(j) * mp.mpf(2) ** sc


def show(name, x, digits=20):
    v, bits = f64(x)
    print(f"  {name:<34} = {mp.nstr(mp.mpf(x), digits)}   f64 0x{bits:016x} ({v!r})")
    return v, bits


def eta(m, eps):
    return mp.sqrt(mp.log(2 * m * (1 + 1 / eps)) / mp.pi)


def half_gauss_mass(inv2s2, zmax=4000):
    """sum_{z >= 0} exp(-z^2 * inv2s2)."""
    return mp.fsum(mp.e ** (-(z * z) * inv2s2) for z in range(0, zmax))


def acceptance(sigma_leaf, r, sigma0, sigma_min):
    """Per-trial acceptance of fn-dsa's SamplerZ (bimodal base at sigma0, ccs = sigma_min/sigma')."""
    s2 = 2 * mp.mpf(sigma_leaf) ** 2
    zs = range(-200, 201)
    num = mp.fsum(mp.e ** (-((z - r) ** 2) / s2) for z in zs)
    den = 2 * half_gauss_mass(1 / (2 * mp.mpf(sigma0) ** 2), 400)
    return (mp.mpf(sigma_min) / sigma_leaf) * num / den


def table_rows(inv2s2, bits, zmax):
    """Number of nonzero entries round(2^bits Pr[K > j]) of a half-Gaussian K (exact mp sums)."""
    # Suffix sums accumulated from the far tail (no cancellation), at 1024 bits.
    with mp.workprec(1024):
        rho = [mp.e ** (-(z * z) * inv2s2) for z in range(zmax)]
        suffix = [mp.mpf(0)] * (zmax + 1)
        for z in range(zmax - 1, -1, -1):
            suffix[z] = suffix[z + 1] + rho[z]
        tot = suffix[0]
        scale = mp.mpf(2) ** bits
        rows = 0
        for j in range(zmax - 1):
            if mp.nint(suffix[j + 1] / tot * scale) == 0:
                return rows
            rows += 1
    raise RuntimeError("zmax too small")


def section(t):
    print()
    print("== " + t)


def params(name, logn, q, eps, sigma0, inv2s0_frac):
    n = 1 << logn
    d = n
    section(f"{name}: logn {logn}, n = d = {n}, q = {q}, eps = 2^{mp.nstr(mp.log(eps, 2), 6)}")
    print(f"  q prime: {is_prime(q)}; q mod 8 = {q % 8}; q mod 2n = {q % (2 * n)}; bitlen(q-1) = {(q - 1).bit_length()}")
    e = eta(2 * d, eps)
    smin = e / mp.sqrt(2 * mp.pi)
    sig = mp.mpf(117) / 100 * mp.sqrt(q) * smin
    vs = sig * mp.sqrt(2 * mp.pi)
    sfg = mp.mpf(117) / 100 * mp.sqrt(mp.mpf(q) / (2 * d))
    show("eta_eps(Z^{2d}) (s-convention)", e)
    show("sigma_min = eta/sqrt(2pi)", smin)
    show("sigma = 1.17 sqrt(q) sigma_min", sig)
    show("varsigma = sigma sqrt(2pi)", vs)
    show("1/sigma (inv_sigma)", 1 / sig)
    smax = mp.mpf(117) ** 2 / 10000 * smin
    show("sigma_max = 1.17^2 sigma_min", smax)
    show("sigma_fg = 1.17 sqrt(q/(2d))", sfg)
    inv2fg = Fraction(10000 * 2 * d, 2 * 13689 * q)  # 1/(2 sigma_fg^2) = 10000*2d/(2*13689*q)
    print(f"  1/(2 sigma_fg^2) as a fraction     = {inv2fg.numerator}/{inv2fg.denominator}")
    gs2 = Fraction(13689 * q, 10000)
    print(f"  GS bound (1.17 sqrt q)^2           = {gs2.numerator}/{gs2.denominator} = {float(gs2)!r}")
    print(f"    integer test on N1=||(g,-f)||^2 : 10000*N1 <= {13689 * q}  (N1 <= {gs2.numerator // gs2.denominator})")
    show("GS bound as f64", mp.mpf(gs2.numerator) / gs2.denominator)
    show("1.17 sqrt(q)", mp.mpf(117) / 100 * mp.sqrt(q))
    b2 = vs * vs * 2 * d
    print(f"  beta_s^2 = varsigma^2 * 2d          = {mp.nstr(b2, 25)}  floor {int(mp.floor(b2))}")
    print(f"  beta_s = varsigma sqrt(2d)          = {mp.nstr(mp.sqrt(b2), 15)}  (2^{mp.nstr(mp.log(mp.sqrt(b2), 2), 6)})")
    es2 = 2 * d * sig * sig
    print(f"  E||s||^2 = 2d sigma^2               = {mp.nstr(es2, 15)};  beta^2 / E||s||^2 = {mp.nstr(b2 / es2, 10)} (= 2 pi)")
    show("1/q (inv_q)", mp.mpf(1) / q)
    show(f"1/(2 sigma0^2), sigma0 = {sigma0}", mp.mpf(inv2s0_frac.numerator) / inv2s0_frac.denominator)
    print(f"  1/(2 sigma0^2) as a fraction       = {inv2s0_frac.numerator}/{inv2s0_frac.denominator}")
    print(f"  leaf range [sigma_min, 1.17^2 sigma_min] = [{mp.nstr(smin, 8)}, {mp.nstr(smax, 8)}]; sigma0 = {sigma0}; "
          f"margin sigma0/sigma_max = {mp.nstr(mp.mpf(sigma0) / smax, 8)}")
    lo_leaf = (sig / sigma0) ** 2
    hi_leaf = mp.mpf(gs2.numerator) / gs2.denominator
    print(f"  tree leaf values d_i (sigma' = sigma/sqrt(d_i)) allowed in [(sigma/sigma0)^2, (1.17 sqrt q)^2] = "
          f"[{mp.nstr(lo_leaf, 12)}, {mp.nstr(hi_leaf, 12)}];  q/1.17^2 = {mp.nstr(mp.mpf(q) * 10000 / 13689, 12)}")
    for sl in (smin, (smin + smax) / 2, smax):
        accs = [acceptance(sl, r, sigma0, smin) for r in (0, mp.mpf('0.3'), mp.mpf('0.5'), mp.mpf('0.99'))]
        print(f"  SamplerZ acceptance at sigma' = {mp.nstr(sl, 7)}: r = 0, .3, .5, .99 -> "
              + ", ".join(mp.nstr(a, 12) for a in accs))
    return dict(n=n, q=q, smin=smin, sig=sig, vs=vs, sfg=sfg, smax=smax, inv2fg=inv2fg, b2=b2)


def main():
    print("mpmath", mp.__version__, "precision", mp.mp.prec, "bits")

    # ---- PCS ---------------------------------------------------------------------------
    pcs = params("PCS", 10, 1048573, mp.mpf(2) ** -128, 3.1, Fraction(50, 961))
    q = 1048573
    r = pow(2, (q - 1) // 4, q)
    r = min(r, q - r)
    print(f"  sqrt(-1) mod q (2 is a non-residue since q = 5 mod 8): r = {r}; r^2 + 1 mod q = {(r * r + 1) % q}")
    print(f"  ord of q mod 2048: {next(k for k in range(1, 2048) if pow(q, k, 2048) == 1)} "
          "(512 => X^1024+1 = product of 2 irreducible factors of degree 512)")

    # ---- Falcon-1024 (fn-dsa constants must be reproduced bit for bit) ------------------
    fal = params("Falcon-1024", 10, 12289, 1 / mp.sqrt(mp.mpf(256) * mp.mpf(2) ** 64), 1.8205,
                 Fraction(10 ** 8, 2 * 331422025))
    section("fn-dsa constants (fn-dsa-sign/src/sampler.rs:30, :59, :72; lib.rs:480; :100, :103)")
    checks = [
        ("INV_SIGMA[10]", scaled(6846791885593314, -60), 1 / fal["sig"]),
        ("SIGMA_MIN[10]", scaled(5846934829975396, -52), fal["smin"]),
        ("INV_2SQRSIGMA0", scaled(5435486223186882, -55), 1 / (2 * mp.mpf("1.8205") ** 2)),
        ("INV_Q (1/12289)", scaled(6004310871091074, -66), mp.mpf(1) / 12289),
        ("LOG2", scaled(6243314768165359, -53), mp.log(2)),
        ("INV_LOG2", scaled(6497320848556798, -52), 1 / mp.log(2)),
    ]
    for name, const, exact in checks:
        cv, cb = f64(const)
        ev, eb = f64(exact)
        print(f"  {name:<16} fn-dsa 0x{cb:016x}  nearest(exact) 0x{eb:016x}  {'EQUAL' if cb == eb else 'DIFFERENT'}"
              f"  ({cv!r})")
    print(f"  SQBETA[10] = 70265242; floor((1.1 sigma sqrt(2n))^2) = "
          f"{int(mp.floor((mp.mpf('1.1') * fal['sig'] * mp.sqrt(2048)) ** 2))}")

    # ---- base samplers and BerExp saturation ------------------------------------------
    section("Base half-Gaussian tables: nonzero rows of round(2^p Pr[K > j])")
    for s0, frac in ((1.8205, Fraction(2000000, 13256881)), (3.1, Fraction(50, 961))):
        inv = mp.mpf(frac.numerator) / frac.denominator
        rows = {p: table_rows(inv, p, 200) for p in (79, 128, 256)}
        print(f"  sigma0 = {s0}: rows at 79 / 128 / 256 bits = {rows[79]} / {rows[128]} / {rows[256]}")
    for name, frac in (("PCS sigma_fg", pcs["inv2fg"]), ("Falcon-1024 sigma_fg", fal["inv2fg"])):
        inv = mp.mpf(frac.numerator) / frac.denominator
        print(f"  {name} = {mp.nstr(1 / mp.sqrt(2 * inv), 12)}: rows at 64 / 128 bits = "
              f"{table_rows(inv, 64, 2000)} / {table_rows(inv, 128, 2000)}")

    section("BerExp saturation (fn-dsa sampler.rs:271-281) at sigma0 = 3.1")
    s0 = mp.mpf("3.1")
    inv0 = 1 / (2 * s0 * s0)
    mass = half_gauss_mass(inv0, 400)
    smin = pcs["smin"]
    for z0 in range(20, 36):
        # worst case: b = 1, r -> 0 gives z - r -> z0 + 1
        x = (z0 + 1) ** 2 / (2 * smin * smin) - z0 * z0 * inv0
        if x >= 64 * mp.log(2):
            tail = mp.fsum(mp.e ** (-(k * k) * inv0) for k in range(z0, 400)) / mass
            print(f"  s = floor(x/ln2) >= 64 first possible at z0 = {z0} (x = {mp.nstr(x, 8)}); "
                  f"P(z0 >= {z0}) = 2^{mp.nstr(mp.log(tail, 2), 6)}")
            break

    section("Work per SampPre call (2n = 2048 leaves)")
    for name, d, s0, smin_, bytes_per_draw in (("PCS", pcs, 3.1, pcs["smin"], 17),
                                               ("Falcon-1024", fal, 1.8205, fal["smin"], 10)):
        acc = acceptance(smin_, 0, s0, smin_)
        trials = 1 / acc
        ber_bytes = mp.mpf(256) / 255  # lazy comparison: expected bytes until the first differing byte
        tot = 2048 * trials * (bytes_per_draw + ber_bytes)
        print(f"  {name}: acceptance {mp.nstr(acc, 8)}, trials/leaf {mp.nstr(trials, 6)}, PRNG bytes/call "
              f"~{mp.nstr(tot, 7)} (~{mp.nstr(tot / 136, 5)} Keccak-f at rate 136)")

    section("Two NTT primes for exact R_q products (fn-dsa-kgen mp31.rs:343-347, PRIMES[0..2])")
    p1, p2 = 2147473409, 2147389441
    for p in (p1, p2):
        print(f"  p = {p}: prime {is_prime(p)}, p mod 2048 = {p % 2048}, p < 2^31 {p < 2 ** 31}")
    P = p1 * p2
    print(f"  p1 p2 = {P} = 2^{math.log2(P):.4f}")
    for qq, nn in ((1048573, 1024), (12289, 1024), (2 ** 25 - 39, 1024)):
        bound = 2 * nn * (qq - 1) ** 2
        print(f"  q = {qq}, n = {nn}: 2 n (q-1)^2 = 2^{math.log2(bound):.4f} < p1 p2: {bound < P}")
    print(f"  largest q with 2*1024*(q-1)^2 < p1 p2: {math.isqrt(P // 2048)} (2^{math.log2(math.isqrt(P // 2048)):.3f})")

    section("Sizes")
    for name, qq, fgb, FGb in (("PCS", 1048573, 10, 13), ("Falcon-1024", 12289, 7, 9)):
        b = (qq - 1).bit_length()
        print(f"  {name}: R_q element {1024 * b // 8} B ({b} bits/coef); secret key 1 + 1024*(2*{fgb}+2*{FGb})/8 = "
              f"{1 + 1024 * (2 * fgb + 2 * FGb) // 8} B")
    print(f"  ffLDL tree (logn+1) n f64 = {11 * 1024} f64 = {11 * 1024 * 8} B; basis 4n f64 = {4 * 1024 * 8} B; "
          f"expanded key {15 * 1024 * 8} B")


if __name__ == "__main__":
    main()
