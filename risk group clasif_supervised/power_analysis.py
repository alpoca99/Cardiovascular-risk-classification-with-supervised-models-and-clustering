#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
power_analysis.py
=================
Post-hoc statistical-power / sample-size analysis for the paired McNemar
comparison between the UMAP- and PCA-based pipelines on the external cohort
(n = 219). Prepared in response to Reviewer 2 (Major issue #2), which asked
how large a sample McNemar's test would need to detect an ~4-percentage-point
accuracy difference at 80% power.

Method
------
Paired McNemar test, normal (Lachin, 1992) approximation. For a two-sided
test at level alpha and power (1 - beta), the required number of *paired*
subjects is

    n = [ z_{1-alpha/2} * sqrt(p_d) + z_{1-beta} * sqrt(p_d - d^2) ]^2 / d^2

where
    p_d = p01 + p10   is the proportion of DISCORDANT pairs, and
    d   = |p01 - p10| is the marginal (accuracy) difference between the two
                       classifiers on the same patients.

The achieved power at a given n is obtained by inverting the same expression:

    power = Phi( ( d*sqrt(n) - z_{1-alpha/2}*sqrt(p_d) ) / sqrt(p_d - d^2) )

Reference
---------
Lachin, J.M. (1992). Power and sample size evaluation for the McNemar test
with application to matched case-control studies. Statistics in Medicine,
11(9), 1239-1251.

Reproducibility
---------------
Self-contained: depends only on numpy and scipy, needs no data files, and is
fully deterministic. Run with:  python power_analysis.py
"""

from math import sqrt
from scipy.stats import norm

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ALPHA = 0.05          # two-sided significance level
POWER_TARGET = 0.80   # target power for sample-size calculations
N_EXTERNAL = 219      # size of the external validation cohort

z_alpha = norm.ppf(1 - ALPHA / 2)   # 1.9600 for alpha = 0.05
z_beta  = norm.ppf(POWER_TARGET)    # 0.8416 for 80% power


# ----------------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------------
def required_n(d, p_d, power=POWER_TARGET):
    """Paired subjects needed to detect a marginal difference d given a
    discordant-pair proportion p_d, at the requested power and ALPHA.

    d, p_d are proportions (e.g. 0.04 for 4 pp). Requires p_d > d.
    """
    if d <= 0:
        return float("inf")
    if p_d <= d * d:
        raise ValueError("Discordant proportion p_d must exceed d^2.")
    zb = norm.ppf(power)
    return (z_alpha * sqrt(p_d) + zb * sqrt(p_d - d * d)) ** 2 / (d * d)


def achieved_power(n, d, p_d):
    """Power of the paired McNemar test at sample size n for marginal
    difference d and discordant-pair proportion p_d (two-sided, ALPHA)."""
    if d <= 0 or p_d <= d * d:
        return float("nan")
    z = (d * sqrt(n) - z_alpha * sqrt(p_d)) / sqrt(p_d - d * d)
    return norm.cdf(z)


# ----------------------------------------------------------------------------
# 1) Observed discordant pairs on the external cohort (n = 219)
#    b01 / b10 are the off-diagonal counts of the paired 2x2 table for each
#    UMAP-vs-PCA comparison, taken from run_validation.py (McNemar section).
# ----------------------------------------------------------------------------
OBSERVED = {
    "LightGBM  (UMAP vs PCA)": (10, 7),
    "XGBoost   (UMAP vs PCA)": (9, 13),
    "RandomForest (UMAP vs PCA)": (9, 7),
}

print("=" * 74)
print(f"McNemar power analysis  (alpha = {ALPHA}, target power = {POWER_TARGET:.0%})")
print("=" * 74)

print(f"\n1) Observed effects on the external cohort (n = {N_EXTERNAL}):")
print(f"   {'Comparison':<28}{'diff (pp)':>10}{'discordant':>13}{'n for 80%':>13}")
for name, (b01, b10) in OBSERVED.items():
    d = abs(b01 - b10) / N_EXTERNAL          # marginal difference
    p_d = (b01 + b10) / N_EXTERNAL           # discordant proportion
    n80 = required_n(d, p_d)
    print(f"   {name:<28}{d * 100:>9.2f}{p_d * 100:>12.1f}%{n80:>13,.0f}")

# ----------------------------------------------------------------------------
# 2) Reviewer scenario: how many patients to detect a 4-pp difference at 80%?
#    (reported across a plausible range of discordant-pair proportions)
# ----------------------------------------------------------------------------
D_TARGET = 0.04   # 4 percentage points
print(f"\n2) Sample size for a {D_TARGET*100:.0f}-pp difference at {POWER_TARGET:.0%} power:")
print(f"   {'discordant prop.':<20}{'required n':>12}")
for p_d in (0.06, 0.08, 0.10, 0.12):
    print(f"   {p_d*100:>6.0f}%{'':<13}{required_n(D_TARGET, p_d):>12,.0f}")

# ----------------------------------------------------------------------------
# 3) Achieved power at the actual cohort size (n = 219)
# ----------------------------------------------------------------------------
print(f"\n3) Achieved power at n = {N_EXTERNAL}:")
print(f"   {'effect (pp)':<14}{'discordant':>12}{'power':>10}")
for d_pp, p_d in ((2, 0.10), (4, 0.10), (6, 0.10), (8, 0.10)):
    pw = achieved_power(N_EXTERNAL, d_pp / 100, p_d)
    print(f"   {d_pp:>6} pp{'':<4}{p_d*100:>10.0f}%{pw*100:>9.0f}%")

# ----------------------------------------------------------------------------
# Summary sentence (matches the manuscript wording)
# ----------------------------------------------------------------------------
pw_4 = achieved_power(N_EXTERNAL, 0.04, 0.10) * 100
n_4  = required_n(0.04, 0.10)
print("\n" + "-" * 74)
print("Summary (as reported in the manuscript):")
print(f"  With n = {N_EXTERNAL} and a discordant-pair proportion of ~8-10%, the paired")
print(f"  McNemar test has only ~{pw_4:.0f}% power to detect even a 4-percentage-point")
print(f"  accuracy difference; reaching {POWER_TARGET:.0%} power for a 4-point difference")
print(f"  would require roughly n ~ {n_4:,.0f} patients, and detecting the differences")
print("  actually observed (<= 2 pp) would require several thousand.")
print("-" * 74)
