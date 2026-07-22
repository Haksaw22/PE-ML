"""
A5 — excitation-preserving memory for streaming ICL (pre-registered in DESIGN.md).

Stream of T=40 demos, context budget C=8, eviction policies compared at equal budget
on the A3 ICL transformer. Regimes: static task; drifting task (w rotates in the task
subspace at rate rho per step — where PERSISTENT excitation should matter).

Policies: fifo | reservoir | excite (undiscounted lam_min-greedy, Chowdhary-Johnson
transplant) | excite_disc (time-discounted Gram, gamma=0.85 — the drift-aware variant)
| oracle (evict to minimize current query MSE; label-using ceiling).

Pre-registered: P1 static: excite > fifo & reservoir. P2 drift: excite_disc > excite,
gap growing with rho. K1/K2: nulls published as such. Oracle gap reported as the
too-easy guard.  Run from repo root:  python experiments/a5_streaming/exp_a5_stream.py
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "experiments/spearhead_a")
from a3_common import ICLModel, sample_task, sample_context, build_tokens, lam_min, SIGMA, R, D

T_STREAM, C, N_TASKS, N_Q, GAMMA, WARM = 40, 8, 120, 4, 0.85, 10
DRIFTS = [0.0, 0.05, 0.15]
POLICIES = ["fifo", "reservoir", "excite", "excite_disc", "oracle"]
rng = np.random.default_rng(31)


def rotate_task(w, rho, rng):
    """Random small rotation within the task subspace (first R coords)."""
    a, b = rng.integers(0, R, 2)
    while b == a:
        b = int(rng.integers(0, R))
    th = rho * (1 + 0.3 * rng.standard_normal())
    w = w.copy()
    wa, wb = w[a], w[b]
    w[a] = np.cos(th) * wa - np.sin(th) * wb
    w[b] = np.sin(th) * wa + np.cos(th) * wb
    return w


def excitation(S, ages, t, gamma=None):
    X = np.array([s[0][:R] for s in S])
    if gamma is None:
        wgt = np.ones(len(S))
    else:
        wgt = gamma ** (t - np.array(ages, float))
    return lam_min((X * wgt[:, None]).T @ X)


@torch.no_grad()
def panel_mse(model, S, queries, w_now):
    X = np.array([s[0] for s in S])
    y = np.array([s[1] for s in S])
    toks = torch.from_numpy(
        np.stack([build_tokens(X, y, q) for q in queries]).astype(np.float32))
    pred = model(toks).numpy()
    return float(((pred - queries @ w_now) ** 2).mean())


def evict(policy, S, ages, t, model, queries, w_now, rng):
    if policy == "fifo":
        return 0
    if policy == "reservoir":
        return int(rng.integers(len(S)))
    if policy in ("excite", "excite_disc"):
        g = GAMMA if policy == "excite_disc" else None
        best, best_v = 0, -np.inf
        for e in range(len(S)):
            keep = [S[i] for i in range(len(S)) if i != e]
            ka = [ages[i] for i in range(len(S)) if i != e]
            v = excitation(keep, ka, t, g)
            if v > best_v:
                best, best_v = e, v
        return best
    if policy == "oracle":
        best, best_v = 0, np.inf
        for e in range(len(S)):
            keep = [S[i] for i in range(len(S)) if i != e]
            v = panel_mse(model, keep, queries, w_now)
            if v < best_v:
                best, best_v = e, v
        return best


def main():
    model = ICLModel()
    model.load_state_dict(torch.load("experiments/spearhead_a/a3_model.pt",
                                     weights_only=True))
    model.eval()
    results = {(rho, p): [] for rho in DRIFTS for p in POLICIES}
    for task_i in range(N_TASKS):
        w0 = sample_task(rng)
        queries = rng.standard_normal((N_Q, D))
        # one shared demo stream per task (same draws for all policies/regimes)
        xs = [sample_context(1, rng)[0] for _ in range(T_STREAM)]
        noises = rng.standard_normal(T_STREAM) * SIGMA
        rot_seed = int(rng.integers(1e9))
        for rho in DRIFTS:
            rrng = np.random.default_rng(rot_seed)
            ws = [w0]
            for t in range(1, T_STREAM):
                ws.append(rotate_task(ws[-1], rho, rrng) if rho > 0 else ws[-1])
            for p in POLICIES:
                prng = np.random.default_rng(1000 + task_i)
                S, ages, errs = [], [], []
                for t in range(T_STREAM):
                    S.append((xs[t], float(xs[t] @ ws[t] + noises[t])))
                    ages.append(t)
                    if len(S) > C:
                        e = evict(p, S, ages, t, model, queries, ws[t], prng)
                        S.pop(e); ages.pop(e)
                    if t >= WARM:
                        errs.append(panel_mse(model, S, queries, ws[t]))
                results[(rho, p)].append(np.mean(errs))
        if task_i % 20 == 0:
            print(f"task {task_i}/{N_TASKS}", flush=True)

    L = ["# A5 results — excitation-preserving memory under streaming",
         "", f"T={T_STREAM} C={C} tasks={N_TASKS} gamma={GAMMA} (per DESIGN.md)", ""]
    for rho in DRIFTS:
        L.append(f"## drift rho = {rho}")
        for p in POLICIES:
            v = np.array(results[(rho, p)])
            se = v.std() / np.sqrt(len(v))
            L.append(f"  {p:12s}: MSE {v.mean():.4f} +/- {2*se:.4f}")
        L.append("")

    def mean(rho, p):
        return np.array(results[(rho, p)]).mean()

    def paired_ci(rho, p1, p2):
        d = np.array(results[(rho, p1)]) - np.array(results[(rho, p2)])
        return d.mean(), 2 * d.std() / np.sqrt(len(d))

    L.append("## Pre-registered readouts (paired per-task differences)")
    d, ci = paired_ci(0.0, "fifo", "excite")
    L.append(f"  P1 static, fifo - excite      : {d:+.4f} +/- {ci:.4f}  "
             f"({'excite wins' if d - ci > 0 else 'NULL' if abs(d) < ci else 'excite LOSES'})")
    d, ci = paired_ci(0.0, "reservoir", "excite")
    L.append(f"  P1 static, reservoir - excite : {d:+.4f} +/- {ci:.4f}")
    for rho in DRIFTS[1:]:
        d, ci = paired_ci(rho, "excite", "excite_disc")
        L.append(f"  P2 rho={rho}, excite - excite_disc: {d:+.4f} +/- {ci:.4f}  "
                 f"({'disc wins' if d - ci > 0 else 'NULL' if abs(d) < ci else 'disc LOSES'})")
    L.append(f"  oracle gap (rho=0): best-policy {min(mean(0.0, p) for p in POLICIES[:-1]):.4f} "
             f"vs oracle {mean(0.0, 'oracle'):.4f}")
    out = "\n".join(L)
    print(out)
    with open("experiments/a5_streaming/A5_RESULTS.md", "w") as f:
        f.write(out + "\n")
    np.savez("experiments/a5_streaming/a5_arrays.npz",
             **{f"{rho}_{p}": np.array(v) for (rho, p), v in results.items()})


if __name__ == "__main__":
    main()
