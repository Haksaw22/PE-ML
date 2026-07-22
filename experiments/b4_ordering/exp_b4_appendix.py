"""
B4 sensitivity appendix — EXPLORATORY, run AFTER the pre-registered primary was
recorded (B4_TIER_A.md: KILL for the pinned U). Permitted by PROPOSAL.md repair 1.
Nothing here is a claim; a variant that predicts strongly becomes a NEW pre-registration
(B4'), not a rescue of the old one.

Candidate order-sensitivity predictors, each scored by mean within-task
Spearman(score, MSE) across permutations (negative = higher score, lower error):
  u_win2 / u_win4 / u_win6 : pinned-style sliding-window attention-weighted lam_min
  u_now4                   : same but UNWEIGHTED (no attention)
  g_disc50 / g_disc80      : lam_min of recency-discounted task Gram, gamma=0.5 / 0.8
                             (the A5 object: causal attention ~ recency weighting)
  g_att_term               : lam_min of attention-weighted TERMINAL Gram (all demos)
  info_pos                 : position (0=first) of the demo with max leverage on the
                             sparsest task direction (is it recency of the crucial demo?)
"""
import sys
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "experiments/spearhead_a")
from a3_common import ICLModel, sample_task, sample_context, build_tokens, SIGMA, R, D

T_TASKS, K, P_PERM, N_Q = 200, 10, 16, 6
rng = np.random.default_rng(23)


def lam_min(G):
    return float(np.clip(np.linalg.eigvalsh(G), 0, None)[0])


def win_score(Xp, w, WIN):
    vals = []
    for j in range(K - WIN + 1):
        Xw, ww = Xp[j:j + WIN, :R], w[j:j + WIN]
        ww = ww / max(ww.sum(), 1e-12)
        vals.append(lam_min((Xw * ww[:, None]).T @ Xw))
    return min(vals)


def main():
    model = ICLModel(causal=True)
    model.load_state_dict(torch.load("experiments/b4_ordering/b4_model_causal.pt",
                                     weights_only=True))
    model.eval()
    scores = {k: [] for k in ["u_win2", "u_win4", "u_win6", "u_now4", "g_disc50",
                              "g_disc80", "g_att_term", "info_pos"]}
    with torch.no_grad():
        for _ in range(T_TASKS):
            w_task = sample_task(rng)
            X = sample_context(K, rng)
            y = X @ w_task + SIGMA * rng.standard_normal(K)
            queries = rng.standard_normal((N_Q, D))
            tgts = queries @ w_task
            # sparsest-excited direction of the unordered set (for info_pos)
            Gfull = X[:, :R].T @ X[:, :R]
            evals, evecs = np.linalg.eigh(Gfull)
            v_sparse = evecs[:, 0]
            lev = (X[:, :R] @ v_sparse) ** 2
            crucial = int(np.argmax(lev))
            per = {k: [] for k in scores}
            errs_all = []
            for p in range(P_PERM):
                perm = rng.permutation(K) if p else np.arange(K)
                Xp, yp = X[perm], y[perm]
                toks = torch.from_numpy(
                    np.stack([build_tokens(Xp, yp, q) for q in queries]).astype(np.float32))
                pred, maps = model(toks, return_attn=True)
                errs_all.append(float(((pred.numpy() - tgts) ** 2).mean()))
                att = maps[-1][:, -1, :K].numpy().mean(0)
                per["u_win2"].append(win_score(Xp, att, 2))
                per["u_win4"].append(win_score(Xp, att, 4))
                per["u_win6"].append(win_score(Xp, att, 6))
                per["u_now4"].append(win_score(Xp, np.ones(K), 4))
                for name, g in (("g_disc50", 0.5), ("g_disc80", 0.8)):
                    dw = g ** (K - 1 - np.arange(K))
                    per[name].append(lam_min((Xp[:, :R] * dw[:, None]).T @ Xp[:, :R]))
                aw = att / max(att.sum(), 1e-12)
                per["g_att_term"].append(lam_min((Xp[:, :R] * aw[:, None]).T @ Xp[:, :R]))
                per["info_pos"].append(float(np.where(perm == crucial)[0][0]))
            e = np.array(errs_all)
            for k in scores:
                v = np.array(per[k])
                if np.std(v) > 0 and np.std(e) > 0:
                    scores[k].append(spearmanr(v, e).correlation)

    L = ["# B4 sensitivity appendix (EXPLORATORY — post-primary, labeled as such)", "",
         f"tasks={T_TASKS} k={K} perms={P_PERM}",
         "", "mean within-task Spearman(score, MSE)  [negative = predicts success]:"]
    for k, v in scores.items():
        v = [x for x in v if not np.isnan(x)]
        se = np.std(v) / np.sqrt(len(v))
        L.append(f"  {k:10s}: {np.mean(v):+.3f}  (+/- {2*se:.3f}, n={len(v)})")
    L += ["", "Reading guide: info_pos positive would mean 'crucial demo early = better';",
          "g_disc* negative would mean recency-discounted excitation is the right object",
          "(the A5 hypothesis). Any strong signal here is a NEW pre-registration, not a rescue."]
    out = "\n".join(L)
    print(out)
    with open("experiments/b4_ordering/B4_APPENDIX.md", "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
