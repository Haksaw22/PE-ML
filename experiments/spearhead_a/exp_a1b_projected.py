"""
Spearhead A / Stage A1b — the TASK-PROJECTED ("corrected") object.

A1 found the naive lam_min(G) predicts error, but the prior-regularized correction was inert in
the isotropic case. The real claim is that the correct object is TASK-PROJECTED: project the
in-context Gram onto the subspace the task+query actually care about (P), because naive lam_min(G)
can be hijacked by a direction with tiny prior/query weight that does not affect prediction.

Test: with an ANISOTROPIC prior (task lives in an R-dim subspace) and a query distribution aligned
with it, does lam_min(P^T G P) predict the (query-weighted) prediction error BETTER than the naive
full-space lam_min(G)? If yes, the projection is the value-add and the "which-modes-recoverable"
instrument is justified.
"""
import numpy as np
from scipy.stats import spearmanr

RNG = np.random.default_rng(1)
D, R, SIGMA = 12, 4, 0.1
EPS = 1e-3  # prior/query weight on unimportant directions


def make_context(k, spread, rng):
    eig = np.exp(rng.uniform(-spread, spread, size=D)); eig /= eig.mean()
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    L = np.linalg.cholesky(Q @ np.diag(eig) @ Q.T + 1e-9 * np.eye(D))
    return rng.standard_normal((k, D)) @ L.T


def trial(k, spread, rng):
    # task basis: first R coords important (prior var 1), rest unimportant (prior var EPS)
    Qtask, _ = np.linalg.qr(rng.standard_normal((D, D)))
    prior_var = np.where(np.arange(D) < R, 1.0, EPS)
    query_w = np.where(np.arange(D) < R, 1.0, EPS)     # query cares about important dirs
    w = rng.standard_normal(D) * np.sqrt(prior_var)     # task-coord truth
    X = make_context(k, spread, rng)                    # ambient design
    Xt = X @ Qtask                                      # design in task coords
    y = Xt @ w + SIGMA * rng.standard_normal(k)
    lam_vec = SIGMA**2 / prior_var                      # per-direction Bayes ridge
    G = Xt.T @ Xt
    what = np.linalg.solve(G + np.diag(lam_vec), Xt.T @ y)
    err = float(np.sum(query_w * (what - w) ** 2))      # query-weighted prediction error

    ev_full = np.clip(np.linalg.eigvalsh(G), 0, None)
    Gp = G[:R, :R]                                      # projected onto important subspace P
    ev_p = np.clip(np.linalg.eigvalsh(Gp), 0, None)
    diag = {
        "lam_min_full": ev_full[0],                    # naive object
        "lam_min_proj": ev_p[0],                       # CORRECTED: projected
        "eff_rank_proj": ev_p.sum() / (ev_p[-1] + 1e-12),
        "trace_full": ev_full.sum(),
        "k": float(k),
    }
    return diag, err


N = 8000
recs = []
for _ in range(N):
    k = int(RNG.integers(5, 25))
    d, e = trial(k, RNG.uniform(0, 3.0), RNG)
    d["err"] = e
    recs.append(d)

KEYS = ["lam_min_full", "lam_min_proj", "eff_rank_proj", "trace_full", "k"]
A = {key: np.array([r[key] for r in recs]) for key in KEYS + ["err"]}

print("=" * 64)
print(f"Spearhead A / Stage A1b : task-projected object  (D={D}, R={R})")
print("=" * 64)
print("\nPOOLED Spearman(diagnostic, query-weighted error)  [negative = good]:")
for key in KEYS:
    print(f"   {key:14s}: rho = {spearmanr(A[key], A['err']).correlation:+.3f}")

print("\nFIXED shot count (isolates conditioning):  naive lam_min_full vs CORRECTED lam_min_proj")
print(f"   {'k':>3s} {'n':>5s} {'full':>8s} {'proj':>8s}  winner")
for kk in (8, 12, 16, 20):
    m = A["k"] == kk
    if m.sum() < 40:
        continue
    rf = spearmanr(A["lam_min_full"][m], A["err"][m]).correlation
    rp = spearmanr(A["lam_min_proj"][m], A["err"][m]).correlation
    win = "PROJ" if abs(rp) > abs(rf) else "full"
    print(f"   {kk:3d} {m.sum():5d} {rf:+8.3f} {rp:+8.3f}  {win}")
print("\nDone.")
