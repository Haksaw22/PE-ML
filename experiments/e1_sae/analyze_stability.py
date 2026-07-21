"""
E1 stage 2 — the pre-registered test (DESIGN.md): does per-feature excitation alpha_j
predict cross-seed instability BEYOND firing frequency?

Steps: (1) cross-seed decoder matching -> instab_j for seed-0 features;
(2) per-feature conditional Gram in the corpus top-64 PC space -> alpha_j (lam_min),
    accumulated with the einsum-batch trick;
(3) frequency-matched random-half-space NULL features -> the generic-geometry effect;
(4) partial Spearman(alpha, instab | log freq, n) with bootstrap CI + decile bins.

  python analyze_stability.py [--smoke]
"""
import argparse
import numpy as np
import torch
from scipy.stats import spearmanr

p = argparse.ArgumentParser()
p.add_argument("--smoke", action="store_true")
p.add_argument("--outdir", default=".")
p.add_argument("--seeds", type=int, default=5)
a = p.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, R_PC, MIN_N = 768, 64, (100 if a.smoke else 500)
N_EVAL = 200_000 if a.smoke else 1_000_000
N_NULL = 200 if a.smoke else 1000
BATCH = 8192

saes = [torch.load(f"{a.outdir}/sae_seed{s}.pt", weights_only=True) for s in range(a.seeds)]
W = saes[0]["width"]
scale = saes[0]["scale"]

# ---- (1) instability: seed-0 decoder rows vs best match in each other seed ----
D0 = torch.nn.functional.normalize(saes[0]["W_dec"], dim=1)
sims = []
for s in range(1, a.seeds):
    Ds = torch.nn.functional.normalize(saes[s]["W_dec"], dim=1)
    sims.append((D0 @ Ds.T).max(1).values)
instab = (1 - torch.stack(sims).mean(0)).numpy()

# ---- (2)+(3) excitation accumulation over held-out corpus ----
Xh = np.load(f"{a.outdir}/acts_held.npy", mmap_mode="r")
N = min(len(Xh), N_EVAL)
# corpus top PCs (seed-independent)
sample = torch.tensor(np.asarray(Xh[:200_000], np.float32), device=DEV) * scale
mu = sample.mean(0)
cov = (sample - mu).T @ (sample - mu) / len(sample)
Pc = torch.linalg.eigh(cov).eigenvectors[:, -R_PC:]          # (D, R_PC)

W_enc, b_enc, b_dec = (saes[0][k].to(DEV) for k in ("W_enc", "b_enc", "b_dec"))
rngn = np.random.default_rng(0)
V_null = torch.nn.functional.normalize(
    torch.tensor(rngn.standard_normal((N_NULL, D)), dtype=torch.float32, device=DEV), dim=1)
# null thresholds: match null firing freq to the freq DISTRIBUTION of real features:
# quantiles assigned per null from a first pass
proj_sample = sample @ V_null.T
S_real = torch.zeros(W, R_PC * R_PC, device=DEV)
n_real = torch.zeros(W, device=DEV)
S_null = torch.zeros(N_NULL, R_PC * R_PC, device=DEV)
n_null = torch.zeros(N_NULL, device=DEV)
freq_real_pass = torch.zeros(W, device=DEV)

with torch.no_grad():
    # first pass on sample: real firing freqs -> null thresholds matched to their quantiles
    f = torch.relu((sample - b_dec) @ W_enc + b_enc)
    freq_s = (f > 0).float().mean(0).clamp_min(1e-6)
    q = freq_s[torch.randint(W, (N_NULL,), generator=torch.Generator().manual_seed(1))]
    qv = (1 - q).clamp(0.001, 0.999)
    thr = torch.empty(N_NULL, device=DEV)
    for c0 in range(0, N_NULL, 50):                      # chunked (torch.quantile size limit)
        cols = proj_sample[:, c0:c0 + 50]
        thr[c0:c0 + 50] = torch.quantile(cols, qv[c0:c0 + 50], dim=0).diagonal()

    for i0 in range(0, N, BATCH):
        xb = torch.tensor(np.asarray(Xh[i0:i0 + BATCH], np.float32), device=DEV) * scale
        z = (xb - mu) @ Pc                                   # (B, R_PC)
        zz = (z.unsqueeze(2) * z.unsqueeze(1)).reshape(len(z), -1)  # (B, R_PC^2)
        f = torch.relu((xb - b_dec) @ W_enc + b_enc)
        m = (f > 0).float()                                  # (B, W)
        S_real += m.T @ zz
        n_real += m.sum(0)
        freq_real_pass += m.sum(0)
        mn = ((xb @ V_null.T) > thr).float()
        S_null += mn.T @ zz
        n_null += mn.sum(0)

def alphas(S, n):
    out = np.full(len(S), np.nan)
    S = S.reshape(len(S), R_PC, R_PC).cpu().numpy()
    n_ = n.cpu().numpy()
    for j in range(len(S)):
        if n_[j] >= MIN_N:
            out[j] = np.clip(np.linalg.eigvalsh(S[j] / n_[j]), 0, None)[0]
    return out

al_real, al_null = alphas(S_real, n_real), alphas(S_null, n_null)
freq = (freq_real_pass / N).cpu().numpy()
nreal = n_real.cpu().numpy()

# ---- (4) statistics ----
def partial_spear(x, y, covars):
    """Spearman of rank-residuals after regressing out covariates (rank scale)."""
    def resid(v):
        r = np.argsort(np.argsort(v)).astype(float)
        A = np.column_stack([np.argsort(np.argsort(c)).astype(float) for c in covars] +
                            [np.ones(len(v))])
        return r - A @ np.linalg.lstsq(A, r, rcond=None)[0]
    return spearmanr(resid(x), resid(y)).correlation

ok = ~np.isnan(al_real) & (instab > 0)
x, y = al_real[ok], instab[ok]
cov_ = [np.log(freq[ok] + 1e-9), nreal[ok]]
r_raw = spearmanr(x, y).correlation
r_par = partial_spear(x, y, cov_)
boot = []
rngb = np.random.default_rng(2)
idx = np.arange(len(x))
for _ in range(500):
    s = rngb.choice(idx, len(idx))
    boot.append(partial_spear(x[s], y[s], [c[s] for c in cov_]))
lo, hi = np.percentile(boot, [2.5, 97.5])

L = ["# E1 pilot results", "",
     f"seeds={a.seeds} width={W} eval_tokens={N} usable_features={ok.sum()}/{W} "
     f"(min_n={MIN_N}) dead_frac_seed0={saes[0]['dead_frac']:.3f}",
     "",
     f"raw     Spearman(alpha, instability) = {r_raw:+.3f}",
     f"PARTIAL Spearman(alpha, instability | log freq, n) = {r_par:+.3f}  "
     f"[{lo:+.3f}, {hi:+.3f}]  <-- P1/K1 pre-registered test",
     "", "## Frequency-decile bins (within-bin raw Spearman)"]
dec = np.digitize(freq[ok], np.quantile(freq[ok], np.linspace(0.1, 0.9, 9)))
for d in range(10):
    m = dec == d
    if m.sum() > 30:
        L.append(f"  decile {d}: n={int(m.sum()):5d}  rho={spearmanr(x[m], y[m]).correlation:+.3f}")

okn = ~np.isnan(al_null)
L += ["", "## Matched-null (generic geometry) comparison",
      f"  real features:  mean alpha {np.nanmean(al_real):.4f}",
      f"  null halfspaces: mean alpha {np.nanmean(al_null):.4f}  (n={int(okn.sum())})",
      "  (null features have no instability — they calibrate how alpha varies with",
      "   region geometry/frequency alone; a real effect must survive the partial",
      "   correlation AND not be reproducible by frequency-matched geometry)"]

verdict = ("P1 PASS: partial correlation negative, CI excludes 0"
           if hi < 0 else
           "K1 KILL: CI includes 0 — diagnostic does not beat the frequency confound"
           if lo < 0 < hi else
           "UNEXPECTED SIGN: partial correlation positive — investigate before claiming")
L += ["", f"## Pre-registered verdict: {verdict}"]

np.savez(f"{a.outdir}/e1_arrays.npz", alpha=al_real, instab=instab, freq=freq,
         n=nreal, alpha_null=al_null)
out = "\n".join(L)
print(out)
with open(f"{a.outdir}/E1_RESULTS.md", "w") as f:
    f.write(out + "\n")
