"""
A3 stage 4 — the ALIASING ARM: the note's signature falsifiable prediction, never before
tested (fixed-point-poe-note.md section 4, prediction 3; Audit 4 repair item 7).

Design. Base context: k0 demos exciting only the first R-1 of the R important directions
(direction R-1 is dark). Then add ONE demo:
  ARM O (orthogonal): a demo along the dark direction  -> raises projected excitation.
  ARM D (duplicate):  a near-copy of an existing demo  -> excitation unchanged.
Same k for both arms, so shot count is controlled by construction.

Predictions (pre-registered):
  P-O: transformer MSE drops far more under ARM O than ARM D, and delta-lmp predicts it.
  P-BUDGET: appending MANY duplicates dilutes query attention on the informative demos
  (softmax budget); if the attention-weighted alpha falls while additive lmp stays flat
  AND MSE degrades, the budget mechanism has empirical content. If MSE stays flat, the
  budget story is decorative at this scale — we publish either way.

Run:  python experiments/spearhead_a/exp_a3_aliasing.py   (needs a3_model.pt)
"""
import numpy as np
import torch
from a3_common import (ICLModel, sample_task, build_tokens, query_attention_weights,
                       lam_min, SIGMA, R, D, DEV)

N_TASKS, K0, N_DUP_MAX = 500, 8, 8
rng = np.random.default_rng(21)


def base_context(k0):
    """Demos spanning only the first R-1 important directions (+ noise dirs)."""
    X = np.zeros((k0, D))
    for i in range(k0):
        v = np.zeros(D)
        v[: R - 1] = rng.standard_normal(R - 1)          # bright important dirs
        v[R:] = 0.3 * rng.standard_normal(D - R)         # unimportant dirs
        X[i] = v
    return X


@torch.no_grad()
def tx_mse(model, X, w, n_q=8):
    y = X @ w + SIGMA * rng.standard_normal(len(X))
    errs = []
    for _ in range(n_q):
        xq = rng.standard_normal(D)
        toks = torch.from_numpy(build_tokens(X, y, xq)[None].astype(np.float32))
        pred, maps = model(toks.to(DEV), return_attn=True)
        errs.append((pred.item() - float(xq @ w)) ** 2)
    w_att = query_attention_weights(maps, len(X))[0]
    Ga = (X * w_att[:, None]).T @ X
    return float(np.mean(errs)), lam_min(Ga[:R, :R])


def main():
    model = ICLModel().to(DEV)
    model.load_state_dict(torch.load("experiments/spearhead_a/a3_model.pt", weights_only=True))
    model.eval()

    res = {a: [] for a in ("base", "orth", "dup")}
    dlmp = {"orth": [], "dup": []}
    for _ in range(N_TASKS):
        w = sample_task(rng)
        X = base_context(K0)
        v_orth = np.zeros(D); v_orth[R - 1] = np.linalg.norm(X[0])   # light the dark direction
        X_o = np.vstack([X, v_orth])
        X_d = np.vstack([X, X[0] + 0.01 * rng.standard_normal(D)])   # near-duplicate
        for arm, Xa in (("base", X), ("orth", X_o), ("dup", X_d)):
            mse, _ = tx_mse(model, Xa, w)
            res[arm].append(mse)
        G = lambda Xa: (Xa.T @ Xa)[:R, :R]
        base_l = lam_min(G(X))
        dlmp["orth"].append(lam_min(G(X_o)) - base_l)
        dlmp["dup"].append(lam_min(G(X_d)) - base_l)
    dlmp = {a: float(np.mean(v)) for a, v in dlmp.items()}

    L = ["# A3 aliasing arm", "",
         f"base k={K0} (dark direction {R-1}), n_tasks={N_TASKS}",
         f"  MSE base: {np.mean(res['base']):.4f}",
         f"  MSE +orthogonal demo: {np.mean(res['orth']):.4f}   (delta-lmp {dlmp['orth']:+.3f})",
         f"  MSE +duplicate demo:  {np.mean(res['dup']):.4f}   (delta-lmp {dlmp['dup']:+.3f})",
         f"  orth improvement / dup improvement: "
         f"{(np.mean(res['base']) - np.mean(res['orth'])) / max(np.mean(res['base']) - np.mean(res['dup']), 1e-9):.1f}x"]

    # budget probe: pile duplicates on, watch attention-weighted alpha + MSE
    L += ["", "## Budget probe (append duplicates of demo 0)"]
    w = None
    mse_c, att_c = [], []
    for nd in range(N_DUP_MAX + 1):
        m_l, a_l = [], []
        rng2 = np.random.default_rng(500)
        for _ in range(200):
            w = sample_task(rng2)
            X = base_context(K0)
            X[:, R - 1] = 0.5 * rng2.standard_normal(K0)   # all dirs lit in this probe
            Xd = np.vstack([X] + [X[0] + 0.01 * rng2.standard_normal(D)] * nd) if nd else X
            mse, a_att = tx_mse(model, Xd, w)
            m_l.append(mse); a_l.append(a_att)
        mse_c.append(np.mean(m_l)); att_c.append(np.mean(a_l))
        L.append(f"  +{nd} dup: MSE {mse_c[-1]:.4f}  attention-weighted alpha {att_c[-1]:.4f}")
    trend = "DEGRADES (budget effect live)" if mse_c[-1] > 1.15 * mse_c[0] else \
            "flat (budget effect decorative at this scale)"
    L.append(f"  verdict: alpha_att {'falls' if att_c[-1] < att_c[0] else 'does not fall'}; "
             f"MSE {trend}")

    out = "\n".join(L)
    print(out)
    with open("experiments/spearhead_a/A3_ALIASING.md", "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
