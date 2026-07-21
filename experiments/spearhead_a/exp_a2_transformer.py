"""
Spearhead A / Stage A2 — construct validity on a REAL trained transformer.

A1b showed the task-projected diagnostic predicts error in the ridge SURROGATE. The make-or-break
question: does a genuinely TRAINED transformer's in-context error track the same diagnostic, or does
the real learner diverge from the surrogate? And does the diagnostic predict the transformer's
prediction FAILURE (not just recoverability)?

Setup: in-context linear regression, fixed anisotropic prior (first R coords important). Train a small
transformer to predict y_q from (x_1,y_1,...,x_k,y_k, x_q). Then, on held-out contexts with varying
conditioning, correlate the transformer's query MSE with lam_min(P^T G P) [projected] and lam_min(G)
[naive]. Writes results to a2_results.txt.
"""
import numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr

torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)
D, R, SIGMA, DEV = 12, 4, 0.1, "cpu"
PRIOR_STD = np.where(np.arange(D) < R, 1.0, 1e-2).astype(np.float32)  # anisotropic prior
KMAX = 20


def gen_batch(B, k, rng, spread_range=(0.0, 2.5)):
    """Returns tokens (B, k+1, D+2) and targets (B,). Token = [x, y, is_query]."""
    toks = np.zeros((B, k + 1, D + 2), np.float32)
    tgt = np.zeros(B, np.float32)
    for b in range(B):
        w = rng.standard_normal(D).astype(np.float32) * PRIOR_STD
        sp = rng.uniform(*spread_range)
        eig = np.exp(rng.uniform(-sp, sp, size=D)); eig /= eig.mean()
        Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
        L = np.linalg.cholesky(Q @ np.diag(eig) @ Q.T + 1e-9 * np.eye(D)).astype(np.float32)
        X = (rng.standard_normal((k, D)).astype(np.float32)) @ L.T
        y = X @ w + SIGMA * rng.standard_normal(k).astype(np.float32)
        xq = rng.standard_normal(D).astype(np.float32)
        toks[b, :k, :D] = X; toks[b, :k, D] = y
        toks[b, k, :D] = xq; toks[b, k, D + 1] = 1.0
        tgt[b] = xq @ w
    return torch.from_numpy(toks), torch.from_numpy(tgt)


class ICLModel(nn.Module):
    def __init__(self, d_in, d_model=64, nhead=4, nlayers=2, maxlen=KMAX + 2):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.zeros(maxlen, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, toks):
        h = self.embed(toks) + self.pos[:toks.shape[1]]
        return self.head(self.enc(h)[:, -1]).squeeze(-1)


def main():
    rng = np.random.default_rng(0)
    model = ICLModel(D + 2).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    STEPS, B = 4000, 64
    model.train()
    for step in range(STEPS):
        k = int(rng.integers(6, KMAX + 1))
        toks, tgt = gen_batch(B, k, rng)
        pred = model(toks.to(DEV))
        loss = ((pred - tgt.to(DEV)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}", flush=True)

    # ---- Eval: does the transformer's error track the diagnostic? ----
    model.eval()
    rows = []
    with torch.no_grad():
        for _ in range(2500):
            k = int(rng.integers(6, KMAX + 1))
            toks, tgt = gen_batch(1, k, rng)
            pred = model(toks.to(DEV)).item()
            err = (pred - tgt.item()) ** 2
            X = toks[0, :k, :D].numpy()
            G = X.T @ X
            ev = np.clip(np.linalg.eigvalsh(G), 0, None)
            evp = np.clip(np.linalg.eigvalsh(G[:R, :R]), 0, None)
            rows.append((err, ev[0], evp[0], float(k)))
    err, lmf, lmp, kk = (np.array([r[i] for r in rows]) for i in range(4))

    lines = ["Spearhead A / Stage A2 : trained transformer construct validity",
             f"D={D} R={R} steps={STEPS}  eval n={len(rows)}", ""]
    lines.append("Spearman(diagnostic, TRANSFORMER query MSE)  [negative = predicts error]:")
    lines.append(f"   lam_min_proj (P^T G P) : {spearmanr(lmp, err).correlation:+.3f}")
    lines.append(f"   lam_min_full (G)       : {spearmanr(lmf, err).correlation:+.3f}")
    lines.append(f"   k (shot count)         : {spearmanr(kk, err).correlation:+.3f}")
    # fixed-k slices
    lines.append("\nFixed-k (isolates conditioning):  proj vs full vs k")
    for k0 in (8, 12, 16):
        m = kk == k0
        if m.sum() < 30:
            continue
        rp = spearmanr(lmp[m], err[m]).correlation
        rf = spearmanr(lmf[m], err[m]).correlation
        lines.append(f"   k={k0:2d} (n={m.sum():4d}): proj={rp:+.3f}  full={rf:+.3f}")
    # construct-validity gap: does low-excitation predict transformer FAILURE?
    lo = lmp < np.quantile(lmp, 0.25); hi = lmp > np.quantile(lmp, 0.75)
    lines.append(f"\nTransformer MSE: low-excitation quartile={err[lo].mean():.4f}  "
                 f"high-excitation quartile={err[hi].mean():.4f}  "
                 f"ratio={err[lo].mean()/max(err[hi].mean(),1e-9):.2f}x")
    out = "\n".join(lines)
    print("\n" + out)
    with open("experiments/spearhead_a/a2_results.txt", "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
