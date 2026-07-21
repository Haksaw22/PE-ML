"""
A3 stage 3 — statistics done properly (repairs: k-confounded quartile ratio, no CIs,
no paired tests, "4.88x" headline).

Everything is computed from the saved per-trial arrays (a3_trials.npz).
  - Spearman with bootstrap 95% CIs, pooled AND at fixed k.
  - PAIRED bootstrap for correlation differences (same resamples for both diagnostics —
    the dependent-samples test Audit 4 said was impossible from A2's aggregates).
  - Quartile MSE ratios computed WITHIN each k (the deconfounded version of "4.88x"),
    alongside the pooled number reported as the confounded upper bound it is.
  - Excess-error analysis: does the diagnostic predict the transformer's error ABOVE the
    Bayes-ridge floor on the same context (ICL-specific failure), or only the floor itself?

Run:  python experiments/spearhead_a/exp_a3_analysis.py
"""
import numpy as np
from scipy.stats import spearmanr

BOOT = 2000
RNG = np.random.default_rng(0)


def rho(x, y):
    return spearmanr(x, y).correlation


def boot_ci(x, y, n=BOOT):
    idx = np.arange(len(x))
    vals = [rho(x[s], y[s]) for s in (RNG.choice(idx, len(idx)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def paired_diff_ci(a, b, err, n=BOOT):
    """CI for |rho(a,err)| - |rho(b,err)| using the SAME bootstrap resamples."""
    idx = np.arange(len(err))
    vals = []
    for _ in range(n):
        s = RNG.choice(idx, len(idx))
        vals.append(abs(rho(a[s], err[s])) - abs(rho(b[s], err[s])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi, float(np.mean(vals))


def main():
    Z = np.load("experiments/spearhead_a/a3_trials.npz")
    k, err, gap = Z["k"], Z["err_tx"], Z["err_gap"]
    diags = {n: Z[n] for n in ["lmp", "lmf", "lmp_att", "lmp_est", "trace_proj",
                               "predvar_q", "predvar_tr"]}
    L = ["# Spearhead A / Stage A3 — repaired construct-validity results", ""]
    L.append(f"n = {len(err)} trials | mean transformer MSE {err.mean():.4f} | "
             f"mean Bayes-ridge floor {Z['err_ridge'].mean():.4f}")

    L.append("\n## Pooled Spearman vs transformer query MSE (95% bootstrap CI)")
    L.append("(excitation diagnostics: negative = predicts error; variance baselines: positive)")
    for name, x in diags.items():
        lo, hi = boot_ci(x, err)
        L.append(f"  {name:11s}: rho = {rho(x, err):+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    lo, hi = boot_ci(k, err)
    L.append(f"  {'k':11s}: rho = {rho(k, err):+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    L.append("\n## Fixed-k slices (the deconfounded comparison)")
    for k0 in (8, 12, 16):
        m = k == k0
        if m.sum() < 60:
            continue
        cells = "  ".join(f"{n}={rho(x[m], err[m]):+.3f}" for n, x in diags.items())
        L.append(f"  k={k0:2d} (n={int(m.sum())}): {cells}")

    L.append("\n## Paired bootstrap: |rho| differences at fixed k (positive = first wins)")
    pairs = [("lmp", "lmf"), ("lmp_att", "lmp"), ("lmp_est", "lmp"), ("lmp", "predvar_q")]
    for a, b in pairs:
        for k0 in (8, 12, 16):
            m = k == k0
            if m.sum() < 60:
                continue
            lo, hi, mean = paired_diff_ci(diags[a][m], diags[b][m], err[m])
            sig = "SIG" if lo > 0 or hi < 0 else "ns "
            L.append(f"  {a:8s} vs {b:9s} @k={k0:2d}: {mean:+.3f} [{lo:+.3f},{hi:+.3f}] {sig}")

    L.append("\n## Quartile MSE ratio (low-excitation / high-excitation, by lmp)")
    lo_m = Z["lmp"] < np.quantile(Z["lmp"], 0.25)
    hi_m = Z["lmp"] > np.quantile(Z["lmp"], 0.75)
    L.append(f"  POOLED (k-confounded, upper bound): {err[lo_m].mean() / err[hi_m].mean():.2f}x")
    per_k = []
    for k0 in (8, 12, 16):
        m = k == k0
        if m.sum() < 60:
            continue
        x = Z["lmp"][m]
        lo_i = x < np.quantile(x, 0.25)
        hi_i = x > np.quantile(x, 0.75)
        r = err[m][lo_i].mean() / err[m][hi_i].mean()
        per_k.append(r)
        L.append(f"  k={k0:2d} (deconfounded): {r:.2f}x")
    if per_k:
        L.append(f"  fixed-k mean: {np.mean(per_k):.2f}x  <-- honest headline")

    L.append("\n## Excess-error analysis (ICL-specific failure vs 'hard for anyone')")
    L.append(f"  rho(err_ridge, err_tx)      = {rho(Z['err_ridge'], err):+.3f}  "
             "(how much of tx error is just problem hardness)")
    for n in ("lmp", "lmp_att", "predvar_q"):
        L.append(f"  rho({n:9s}, err_gap)    = {rho(diags[n], gap):+.3f}  (gap = tx - ridge floor)")
    for k0 in (8, 12, 16):
        m = k == k0
        if m.sum() < 60:
            continue
        L.append(f"    k={k0:2d}: rho(lmp, gap) = {rho(Z['lmp'][m], gap[m]):+.3f}")

    out = "\n".join(L)
    print(out)
    with open("experiments/spearhead_a/A3_RESULTS.md", "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
