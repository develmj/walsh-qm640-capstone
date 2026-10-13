"""Reproduce the sample-size calculations in Table 1 of the synopsis.

RQ1: Hanley-McNeil CI for a single AUC plus paired DeLong power.
RQ2: two-proportion z test with unequal allocation (Fleiss et al., 2003).
RQ3: two-proportion z test on the false-decline rate within non-defaulters.
RQ4: one-sided non-inferiority of correlated AUCs."""
from math import asin, sqrt

from scipy.stats import chi2, ncx2, norm

z975, z95, z80 = norm.ppf(0.975), norm.ppf(0.95), norm.ppf(0.80)
PREV = 6636 / 30000
N_MALE, N_FEMALE = 11888, 18112
DEF_MALE, DEF_FEMALE = 2873, 3763


def hanley_mcneil_constant(A, prev=PREV):
    Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
    return (prev * (Q1 - A * A) + (1 - prev) * (Q2 - A * A)) / (prev * (1 - prev))


def hanley_mcneil_exact(A, half_width, prev=PREV):
    """Smallest n for which the full Hanley-McNeil SE (minus-one terms kept)
    meets the half-width target."""
    Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
    target = (half_width / z975) ** 2
    n = 1000
    while True:
        nd, nn = prev * n, (1 - prev) * n
        se2 = (A * (1 - A) + (nd - 1) * (Q1 - A * A) + (nn - 1) * (Q2 - A * A)) / (nd * nn)
        if se2 <= target:
            return n
        n += 1


def rq1(A=0.78, half_width=0.01, delta=0.02, r=0.70):
    c = hanley_mcneil_constant(A)
    n_precision = hanley_mcneil_exact(A, half_width)
    n_paired = 2 * (1 - r) * c * (z975 + z80) ** 2 / delta ** 2
    return n_precision, n_paired


def two_prop(p1, p2, k, alpha=0.05, power=0.80):
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p1 + k * p2) / (1 + k)
    n1 = (za * sqrt((1 + 1 / k) * pbar * (1 - pbar)) + zb * sqrt(p1 * (1 - p1) + p2 * (1 - p2) / k)) ** 2 / (p1 - p2) ** 2
    return n1, k * n1


def rq2(p1=0.24, p2=0.21):
    n1, n2 = two_prop(p1, p2, N_FEMALE / N_MALE)
    h = 2 * asin(sqrt(p1)) - 2 * asin(sqrt(p2))
    crit = chi2.ppf(0.95, 3)
    n_chi = 100
    while ncx2.sf(crit, 3, n_chi * 0.01) < 0.8:
        n_chi += 1
    return n1, n2, n1 / (N_MALE / 30000), h, n_chi


def rq3(fdr_m=0.19, fdr_f=0.14):
    nd_m, nd_f = N_MALE - DEF_MALE, N_FEMALE - DEF_FEMALE
    n1, n2 = two_prop(fdr_m, fdr_f, nd_f / nd_m)
    share = (N_MALE / 30000) * (nd_m / N_MALE)
    return n1, n2, n1 / share


def rq4(A=0.78, margin=0.01, r=0.70):
    c = hanley_mcneil_constant(A)
    se_diff = margin / (z95 + z80)
    se_single = se_diff / sqrt(2 * (1 - r))
    return hanley_mcneil_exact(A, z975 * se_single)


if __name__ == "__main__":
    n_prec, n_pair = rq1()
    print(f"RQ1 precision n = {n_prec:,.0f}; paired-contrast power n = {n_pair:,.0f}")
    n1, n2, N, h, n_chi = rq2()
    print(f"RQ2 n_male = {n1:,.0f}, n_female = {n2:,.0f}, N = {N:,.0f}, Cohen h = {h:.3f}, chi-square N = {n_chi:,}")
    n1, n2, N = rq3()
    print(f"RQ3 male non-defaulters = {n1:,.0f}, female = {n2:,.0f}, N = {N:,.0f}")
    for r in (0.50, 0.70, 0.85):
        print(f"RQ4 r = {r:.2f}: n = {rq4(r=r):,.0f}")
    print(f"Final N_min = {max(n_prec, rq2()[2], rq3()[2], rq4()):,.0f}")
