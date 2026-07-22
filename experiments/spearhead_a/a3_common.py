"""
Spearhead A / Stage A3 — shared components for the REPAIRED construct-validity suite.

This stage exists because the 2026-07-19 audit (process/process-audit-2026-07-19.md, Audit 4)
found Stage A2's headline results unvalidated:
  - the attention-weighted Gram (the note's actual object) was never computed anywhere,
  - P was oracle-injected (literally the first R raw coordinates),
  - the 4.88x quartile ratio pooled over k and is provably k-confounded (Weyl),
  - no ridge/OLS floor on the same contexts, no predictive-variance baseline,
  - per-trial arrays were never saved, so no paired significance test was possible.

A3 repairs all of the above. Model is hand-rolled so attention maps are first-class outputs.
"""
import numpy as np
import torch
import torch.nn as nn

D, R, SIGMA = 12, 4, 0.1
KMAX = 20
PRIOR_VAR = np.where(np.arange(D) < R, 1.0, 1e-4).astype(np.float64)  # anisotropic prior (var)
DEV = "cpu"


def sample_task(rng):
    return (rng.standard_normal(D) * np.sqrt(PRIOR_VAR)).astype(np.float64)


def sample_context(k, rng, spread_range=(0.0, 2.5)):
    """Anisotropic Gaussian design with random covariance spread (as in A2)."""
    sp = rng.uniform(*spread_range)
    eig = np.exp(rng.uniform(-sp, sp, size=D))
    eig /= eig.mean()
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    L = np.linalg.cholesky(Q @ np.diag(eig) @ Q.T + 1e-9 * np.eye(D))
    return rng.standard_normal((k, D)) @ L.T


def build_tokens(X, y, xq):
    """Token = [x, y, is_query]; demos then query."""
    k = X.shape[0]
    toks = np.zeros((k + 1, D + 2), np.float32)
    toks[:k, :D] = X
    toks[:k, D] = y
    toks[k, :D] = xq
    toks[k, D + 1] = 1.0
    return toks


def gen_batch(B, k, rng):
    toks = np.zeros((B, k + 1, D + 2), np.float32)
    tgt = np.zeros(B, np.float32)
    for b in range(B):
        w = sample_task(rng)
        X = sample_context(k, rng)
        y = X @ w + SIGMA * rng.standard_normal(k)
        xq = rng.standard_normal(D)
        toks[b] = build_tokens(X, y, xq)
        tgt[b] = float(xq @ w)
    return torch.from_numpy(toks), torch.from_numpy(tgt)


class Block(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                nn.Linear(4 * d_model, d_model))

    def forward(self, h, mask=None):
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True,
                         attn_mask=mask)
        h = self.ln1(h + a)
        h = self.ln2(h + self.ff(h))
        return h, w  # w: (B, L, L) attention weights averaged over heads


class ICLModel(nn.Module):
    """Two-block encoder; exposes per-layer attention maps. causal=True masks
    future positions (order sensitivity becomes structural, as in real LLMs)."""

    def __init__(self, d_in=D + 2, d_model=64, nhead=4, nlayers=2, maxlen=KMAX + 2,
                 causal=False):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.zeros(maxlen, d_model))
        self.blocks = nn.ModuleList(Block(d_model, nhead) for _ in range(nlayers))
        self.head = nn.Linear(d_model, 1)
        self.causal = causal

    def forward(self, toks, return_attn=False):
        h = self.embed(toks) + self.pos[: toks.shape[1]]
        mask = None
        if self.causal:
            L = toks.shape[1]
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=toks.device), 1)
        maps = []
        for blk in self.blocks:
            h, w = blk(h, mask)
            maps.append(w)
        out = self.head(h[:, -1]).squeeze(-1)
        return (out, maps) if return_attn else out


def query_attention_weights(maps, k):
    """Attention mass from the query token onto the k demo tokens, averaged over
    layers and renormalized over demo positions. Returns (B, k), rows sum to 1."""
    w = torch.stack([m[:, -1, :k] for m in maps]).mean(0)
    return (w / w.sum(-1, keepdim=True).clamp_min(1e-12)).detach().numpy()


# ---------------------------- diagnostics ----------------------------------

def lam_min(M):
    return float(np.clip(np.linalg.eigvalsh(M), 0, None)[0])


def posterior_cov(X):
    """Bayes-ridge posterior covariance under the TRUE prior/noise (task coords)."""
    A = X.T @ X / SIGMA**2 + np.diag(1.0 / PRIOR_VAR)
    return np.linalg.inv(A)


def ridge_predict(X, y, xq):
    """Posterior-mean prediction under the true prior — the Bayes floor."""
    A = X.T @ X / SIGMA**2 + np.diag(1.0 / PRIOR_VAR)
    what = np.linalg.solve(A, X.T @ y / SIGMA**2)
    return float(xq @ what)


def estimate_P(model_rng, n_probe=8, k_probe=16):
    """Estimate the task subspace from labeled probe TASKS only (no oracle coords):
    ridge-fit each probe task with a weak isotropic prior, take top-R eigenvectors
    of the fitted-coefficient scatter. Returns (D, R) orthonormal basis."""
    W = np.zeros((n_probe, D))
    for i in range(n_probe):
        w = sample_task(model_rng)
        X = sample_context(k_probe, model_rng)
        y = X @ w + SIGMA * model_rng.standard_normal(k_probe)
        W[i] = np.linalg.solve(X.T @ X + 1e-2 * np.eye(D), X.T @ y)
    _, _, Vt = np.linalg.svd(W, full_matrices=False)
    return Vt[:R].T


def trial_diagnostics(X, xq, w_att=None, P_est=None):
    """All per-trial diagnostics. Sign conventions noted in exp_a3_analysis."""
    G = X.T @ X
    Gp = G[:R, :R]
    Spost = posterior_cov(X)
    d = {
        "lmp": lam_min(Gp),                      # projected additive Gram (A2's object)
        "lmf": lam_min(G),                       # full additive Gram
        "trace_proj": float(np.trace(Gp)),
        "predvar_q": float(xq @ Spost @ xq),     # query-specific Bayes predictive variance
        "predvar_tr": float(np.trace(Spost[:R, :R])),  # query-agnostic (G-optimal-style)
    }
    if w_att is not None:                        # attention-weighted Gram — the note's object
        Ga = (X * w_att[:, None]).T @ X
        d["lmp_att"] = lam_min(Ga[:R, :R])
    if P_est is not None:                        # estimated-P variant (kills oracle objection)
        d["lmp_est"] = lam_min(P_est.T @ G @ P_est)
    return d
