"""
B4 Tier A — the pre-registered test (PROPOSAL.md, protocol items 1-5).

PINNED BEFORE RUNNING (repair 1): window N = 4 demos; attention = mean over heads of
the FINAL layer only; demo feature = input in task coordinates (first R coords).
No post-hoc sweeps of these in the primary result.

Per task: one demo set (k = 10), one fixed query panel, P = 16 permutations.
  U(pi)   = min over windows j of lam_min( sum_{i in window_j} w_i(pi) x_i x_i^T |_R )
            with w = query->demo attention under ordering pi (renormalized).
  MSE(pi) = mean squared error of the causal transformer on the fixed query panel.

Pre-registered readouts:
  EXISTENCE (repair 3): the terminal ADDITIVE Gram is permutation-invariant by
    construction, so any U(pi) spread is order-only signal. Report the fraction of
    tasks with max/min U > 1.2; if ~none, the design is unrunnable -> record and stop.
  PRECONDITION: within-task MSE spread across permutations must exceed query-panel
    noise (order sensitivity must exist at all in this setting; else recorded as such).
  PRIMARY: over within-task permutation pairs with |Delta U| above the median gap,
    sign(U_1 - U_2) predicts sign(MSE_2 - MSE_1); success = accuracy > 0.5 with 95%
    Wilson CI excluding 0.5. Secondary: mean within-task Spearman(U, MSE).

Run from repo root: python experiments/b4_ordering/exp_b4_tier_a.py
"""
import sys
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "experiments/spearhead_a")
from a3_common import ICLModel, sample_task, sample_context, build_tokens, SIGMA, R, D

T_TASKS, K, P_PERM, N_Q, WIN = 300, 10, 16, 6, 4
rng = np.random.default_rng(11)


def U_score(X_ord, w_att):
    """min over sliding windows of lam_min of attention-weighted task-coord Gram."""
    vals = []
    for j in range(K - WIN + 1):
        Xw = X_ord[j:j + WIN, :R]
        ww = w_att[j:j + WIN]
        ww = ww / max(ww.sum(), 1e-12)
        G = (Xw * ww[:, None]).T @ Xw
        vals.append(float(np.clip(np.linalg.eigvalsh(G), 0, None)[0]))
    return min(vals)


def main():
    model = ICLModel(causal=True)
    model.load_state_dict(torch.load("experiments/b4_ordering/b4_model_causal.pt",
                                     weights_only=True))
    model.eval()

    per_task = []
    with torch.no_grad():
        for _ in range(T_TASKS):
            w = sample_task(rng)
            X = sample_context(K, rng)
            y = X @ w + SIGMA * rng.standard_normal(K)
            queries = rng.standard_normal((N_Q, D))
            tgts = queries @ w
            Us, Es = [], []
            for p in range(P_PERM):
                perm = rng.permutation(K) if p else np.arange(K)
                Xp, yp = X[perm], y[perm]
                toks = torch.from_numpy(
                    np.stack([build_tokens(Xp, yp, q) for q in queries]).astype(np.float32))
                pred, maps = model(toks, return_attn=True)
                errs = (pred.numpy() - tgts) ** 2
                # final-layer attention, query token -> demo positions, mean over queries
                att = maps[-1][:, -1, :K].numpy().mean(0)
                Us.append(U_score(Xp, att))
                Es.append(float(errs.mean()))
            per_task.append((np.array(Us), np.array(Es)))

    # existence + precondition
    spread = np.array([u.max() / max(u.min(), 1e-12) for u, _ in per_task])
    frac_spread = float((spread > 1.2).mean())
    mse_rel_range = np.array([(e.max() - e.min()) / max(e.mean(), 1e-12)
                              for _, e in per_task])

    # primary: pairwise sign prediction on large-gap pairs
    gaps, correct, total = [], 0, 0
    for u, e in per_task:
        for a in range(P_PERM):
            for b in range(a + 1, P_PERM):
                gaps.append((abs(u[a] - u[b]), u[a] - u[b], e[b] - e[a]))
    med = np.median([g[0] for g in gaps])
    for g, du, de in gaps:
        if g > med and de != 0:
            total += 1
            correct += int((du > 0) == (de > 0))
    acc = correct / max(total, 1)
    z = 1.96
    ph = acc
    wilson_lo = (ph + z*z/(2*total) - z*np.sqrt(ph*(1-ph)/total + z*z/(4*total**2))) / (1 + z*z/total)
    wilson_hi = (ph + z*z/(2*total) + z*np.sqrt(ph*(1-ph)/total + z*z/(4*total**2))) / (1 + z*z/total)

    rhos = [spearmanr(u, e).correlation for u, e in per_task
            if np.std(u) > 0 and np.std(e) > 0]
    rhos = [r for r in rhos if not np.isnan(r)]

    L = ["# B4 Tier A — pre-registered result", "",
         f"tasks={T_TASKS} k={K} perms={P_PERM} queries={N_Q} window={WIN} "
         "(params pinned in PROPOSAL.md before any run)", "",
         "## Existence check (repair 3)",
         f"  terminal additive Gram: permutation-invariant by construction (order signal is windowed-only)",
         f"  fraction of tasks with U-spread max/min > 1.2: {frac_spread:.2f}",
         f"  median U-spread: {np.median(spread):.2f}",
         "", "## Order-sensitivity precondition",
         f"  within-task MSE relative range across permutations: median {np.median(mse_rel_range):.3f} "
         f"(p10 {np.percentile(mse_rel_range,10):.3f}, p90 {np.percentile(mse_rel_range,90):.3f})",
         "", "## PRIMARY: pairwise sign prediction (large-gap pairs)",
         f"  n_pairs={total}  accuracy={acc:.3f}  Wilson95=[{wilson_lo:.3f}, {wilson_hi:.3f}]",
         f"  pre-registered success: CI excludes 0.5 -> "
         f"{'PASS' if wilson_lo > 0.5 else ('FAIL (below chance!)' if wilson_hi < 0.5 else 'NULL (CI includes 0.5)')}",
         "", "## Secondary: within-task Spearman(U, MSE)",
         f"  mean rho = {np.mean(rhos):+.3f}  (n={len(rhos)} tasks; negative = high-U orderings do better)"]
    out = "\n".join(L)
    print(out)
    with open("experiments/b4_ordering/B4_TIER_A.md", "w") as f:
        f.write(out + "\n")
    np.savez("experiments/b4_ordering/b4_arrays.npz",
             U=np.array([u for u, _ in per_task]), E=np.array([e for _, e in per_task]))


if __name__ == "__main__":
    main()
