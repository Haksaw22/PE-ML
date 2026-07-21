"""
Spearhead B2 -- PAIRED robust-vs-oracle comparison (settles the verdict).

B1 found that, AVERAGED over draws, the robust (minimax-over-misspecification) and
the oracle ("misspecification-complement") designers put a nearly IDENTICAL amount of
excitation into the unrepresentable complement, while differing in DIRECTION (oracle
targets the true error theta_c; robust guards a worst-case direction). But an equality
of two AVERAGES can hide large PER-DRAW differences. This script computes, for each
draw, the PAIRED quantities:

    |comp_robust - comp_oracle|            (magnitude separation, per draw)
    cos^2(dir_robust, dir_oracle)          (direction separation, per draw)
    cos^2(dir_oracle, theta_c)             (does oracle target the true error?)
    cos^2(dir_robust, theta_c)             (does robust target the true error?)

and reports mean +/- std over many draws, for the correlated test (where the effect
lives). Robust gets a larger restart budget here to make sure we compare GLOBAL optima
(the robust objective has a shallow spurious local min at comp ~ 0.01).

Verdict logic printed at the end:
  * magnitude coincides (small |comp_r - comp_o| relative to the values) AND
    direction separates (cos^2(dir_r,dir_o) well below 1)  =>  designer 4 is the
    average-case member of the SAME robust-OED family as designer 3: SUBSUMED (b),
    separating from designer 3 only along the classical minimax-vs-Bayes axis.
"""
from __future__ import annotations
import numpy as np
import exp_b1_misspec as e

e.RESTARTS = {"bayes": 3, "minimax": 4, "robust": 60, "oracle": 40}  # heavier for paired opt

D, r, N, sigma2, kappa, rho = 6, 3, 100.0, 1.0, 1.0, 1.0
NDRAW = 12


def paired(mode, rho=rho, N=N):
    rows = []
    for s in range(NDRAW):
        prob = e.build(D, r, mode, N, sigma2, kappa, rho, seed=500 + s)
        res = e.solve_all(prob, seed=500 + s)
        cr, co = res["robust"]["comp"], res["oracle"]["comp"]
        dr, do = res["robust"]["top"], res["oracle"]["top"]
        tc = prob.theta_c.numpy()
        tcn = tc / np.linalg.norm(tc)

        def c2(a, b):
            if a is None or b is None:
                return np.nan
            return float(np.dot(a, b)) ** 2

        rows.append(dict(
            cr=cr, co=co, dmag=abs(cr - co),
            reldmag=abs(cr - co) / (0.5 * (cr + co) + 1e-9),
            cos2_ro=c2(dr, do),
            cos2_o_tc=c2(do, tcn),
            cos2_r_tc=c2(dr, tcn),
        ))
    return rows


def summ(rows, key):
    a = np.array([r[key] for r in rows], float)
    return np.nanmean(a), np.nanstd(a)


if __name__ == "__main__":
    print("#" * 82)
    print("SPEARHEAD B2 -- paired robust-vs-oracle (correlated test)")
    print(f"D={D} r={r} N={N:.0f} rho={rho}  NDRAW={NDRAW}  robust restarts={e.RESTARTS['robust']}")
    print("#" * 82)

    for mode in ["correlated"]:
        rows = paired(mode)
        print(f"\n--- per-draw table (test={mode}) ---")
        print(f"{'draw':>4s} {'comp_rob':>9s} {'comp_orc':>9s} {'|dmag|':>8s} "
              f"{'cos2(r,o)':>10s} {'cos2(o,tc)':>11s} {'cos2(r,tc)':>11s}")
        for i, rr in enumerate(rows):
            print(f"{i:4d} {rr['cr']:9.4f} {rr['co']:9.4f} {rr['dmag']:8.4f} "
                  f"{rr['cos2_ro']:10.3f} {rr['cos2_o_tc']:11.3f} {rr['cos2_r_tc']:11.3f}")

        print("\n--- summary (mean +/- std) ---")
        for key, lab in [("cr", "comp robust"), ("co", "comp oracle"),
                         ("dmag", "|comp_rob - comp_orc|  (paired magnitude gap)"),
                         ("reldmag", "relative paired magnitude gap"),
                         ("cos2_ro", "cos2(dir_robust, dir_oracle)  (1=same dir)"),
                         ("cos2_o_tc", "cos2(dir_oracle, theta_c)     (1=targets truth)"),
                         ("cos2_r_tc", "cos2(dir_robust, theta_c)     (1=targets truth)")]:
            m, sd = summ(rows, key)
            print(f"  {lab:48s} {m:7.4f} +/- {sd:6.4f}")

        # verdict signal
        m_dmag, _ = summ(rows, "reldmag")
        m_cos, _ = summ(rows, "cos2_ro")
        m_otc, _ = summ(rows, "cos2_o_tc")
        m_rtc, _ = summ(rows, "cos2_r_tc")
        print("\n--- verdict signal ---")
        print(f"  magnitude: relative paired gap = {m_dmag:.2f} "
              f"({'COINCIDE' if m_dmag < 0.5 else 'SEPARATE'} in magnitude)")
        print(f"  direction: cos2(dir_r,dir_o) = {m_cos:.2f} "
              f"({'SEPARATE' if m_cos < 0.7 else 'COINCIDE'} in direction);"
              f"  oracle->tc={m_otc:.2f}, robust->tc={m_rtc:.2f}")
