"""
Spearhead A / Stage A1 — ICL = Bayes-ridge surrogate.

Tests the core value proposition of the excitation-diagnostic instrument in the idealized
(analytically tractable) case, BEFORE touching a real transformer:

  Q1. Does a conditioning diagnostic of the in-context Gram predict query prediction error
      at FIXED shot count (i.e. is it more than "more demos = better")?
  Q2. Does the PRIOR-REGULARIZED ("corrected") object lambda_min(G + lam I) predict better
      than the naive lambda_min(G)?
  Q3. Prior gain: does the prior rescue prediction below the identification (OLS) threshold
      (the "prediction-below-identification" delta effect)?

Setup: in-context linear regression. Context = k pairs (x_i, y_i), y_i = <w,x_i> + noise,
w ~ N(0, tau^2 I). We VARY context conditioning (anisotropic input covariance) so excitation
decouples from shot count. ICL learner = Bayes-optimal ridge, lam = sigma^2 / tau^2.
Query error (x_q ~ N(0,I)) = ||what - w||^2.
"""
import numpy as np
from scipy.stats import spearmanr

RNG = np.random.default_rng(0)
D, SIGMA, TAU = 8, 0.1, 1.0
LAM = SIGMA**2 / TAU**2


def make_context(k, spread, rng):
    """k x D design with controlled anisotropy: eigenvalues of input cov span exp(+/-spread)."""
    eig = np.exp(rng.uniform(-spread, spread, size=D))
    eig /= eig.mean()
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    L = np.linalg.cholesky(Q @ np.diag(eig) @ Q.T + 1e-9 * np.eye(D))
    return rng.standard_normal((k, D)) @ L.T


def trial(k, spread, rng):
    X = make_context(k, spread, rng)
    w = rng.standard_normal(D) * TAU
    y = X @ w + SIGMA * rng.standard_normal(k)
    G = X.T @ X
    w_ridge = np.linalg.solve(G + LAM * np.eye(D), X.T @ y)      # prior (Bayes)
    w_ols = np.linalg.pinv(X) @ y                                # prior-free (min-norm)
    ev = np.clip(np.linalg.eigvalsh(G), 0, None)
    diag = {
        "lam_min_G": ev[0],                       # naive object
        "lam_min_reg": ev[0] + LAM,               # corrected (prior-regularized) object
        "eff_rank": ev.sum() / (ev[-1] + 1e-12),  # scale-invariant effective rank
        "cond": (ev[-1] + 1e-12) / (ev[0] + 1e-12),
        "trace": ev.sum(),                        # coverage baseline
        "logdet_reg": np.sum(np.log(ev + LAM)),   # DPP/log-det baseline
        "k": float(k),                            # shot-count baseline
    }
    return diag, float(np.sum((w_ridge - w) ** 2)), float(np.sum((w_ols - w) ** 2))


N = 8000
recs = []
for _ in range(N):
    k = int(RNG.integers(3, 20))
    diag, er, eo = trial(k, RNG.uniform(0, 3.0), RNG)
    diag["err_ridge"], diag["err_ols"] = er, eo
    recs.append(diag)

KEYS = ["lam_min_G", "lam_min_reg", "eff_rank", "cond", "trace", "logdet_reg", "k"]
A = {key: np.array([r[key] for r in recs]) for key in KEYS + ["err_ridge", "err_ols"]}

print("=" * 68)
print("Spearhead A / Stage A1 : ICL = Bayes-ridge surrogate")
print(f"D={D} sigma={SIGMA} tau={TAU} lam={LAM:g}  N={N}  k~U[3,20)")
print("=" * 68)

print("\n[Q1/Q2] POOLED Spearman(diagnostic, ridge query MSE):")
print("        (excitation up -> error down, so negative rho = good predictor)")
for key in KEYS:
    rho = spearmanr(A[key], A["err_ridge"]).correlation
    print(f"   {key:12s}: rho = {rho:+.3f}")

print("\n[Q1] FIXED shot count (isolates conditioning from k):")
for kk in (5, 8, 12, 16):
    m = A["k"] == kk
    if m.sum() < 40:
        continue
    print(f"   k={kk:2d} (n={m.sum():4d}):  " + "  ".join(
        f"{key}={spearmanr(A[key][m], A['err_ridge'][m]).correlation:+.2f}"
        for key in ("lam_min_G", "lam_min_reg", "eff_rank", "cond", "trace")))

print("\n[Q3] PRIOR GAIN (prediction-below-identification), binned by lam_min_G:")
print(f"   {'lam_min_G':>10s} {'ridge MSE':>10s} {'OLS MSE':>10s} {'OLS/ridge':>10s}")
order = np.argsort(A["lam_min_G"])
for b in np.array_split(order, 5):
    print(f"   {A['lam_min_G'][b].mean():10.3f} {A['err_ridge'][b].mean():10.3f} "
          f"{A['err_ols'][b].mean():10.3f} {A['err_ols'][b].mean()/A['err_ridge'][b].mean():10.2f}")

# Under-determined slice (k < D): the regime where identification is impossible but prediction may not be.
und = A["k"] < D
print(f"\n[Q3] Under-determined (k<D={D}, n={und.sum()}): "
      f"ridge MSE={A['err_ridge'][und].mean():.3f}  OLS MSE={A['err_ols'][und].mean():.3f}  "
      f"ratio={A['err_ols'][und].mean()/A['err_ridge'][und].mean():.1f}x")
print("\nDone.")
