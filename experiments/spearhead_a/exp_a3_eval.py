"""
A3 stage 2 — evaluation pass. Per trial, records the transformer's query error alongside
EVERY diagnostic and baseline the audit demanded, and SAVES the raw arrays (a3_trials.npz)
so paired statistics are computable (repairs: no saved arrays, no ridge floor, no
predictive-variance baseline, no attention-weighted object, oracle-injected P).

Pre-registration (written before results were seen):
  - predvar_q is the Bayes-optimal per-query uncertainty; if it dominates lmp everywhere
    AND lmp adds nothing at fixed k, the excitation instrument DEMOTES to a derived
    summary of Bayes uncertainty. We publish either way.
  - lmp_att (attention-weighted, the note's actual object) must beat lmp (plain) for the
    attention-normalization story to retain any empirical content.

Run:  python experiments/spearhead_a/exp_a3_eval.py   (needs a3_model.pt)
"""
import numpy as np
import torch
from a3_common import (ICLModel, sample_task, sample_context, build_tokens, ridge_predict,
                       query_attention_weights, trial_diagnostics, estimate_P, SIGMA, KMAX, DEV)

N_TRIALS = 4000
FIELDS = ["k", "err_tx", "err_ridge", "err_gap",
          "lmp", "lmf", "lmp_att", "lmp_est", "trace_proj", "predvar_q", "predvar_tr"]


def main():
    rng = np.random.default_rng(7)
    model = ICLModel().to(DEV)
    model.load_state_dict(torch.load("experiments/spearhead_a/a3_model.pt", weights_only=True))
    model.eval()
    P_est = estimate_P(np.random.default_rng(99))  # probe-task subspace, fixed once

    rows = {f: [] for f in FIELDS}
    with torch.no_grad():
        for _ in range(N_TRIALS):
            k = int(rng.integers(6, KMAX + 1))
            w = sample_task(rng)
            X = sample_context(k, rng)
            y = X @ w + SIGMA * rng.standard_normal(k)
            xq = rng.standard_normal(len(w))
            toks = torch.from_numpy(build_tokens(X, y, xq)[None])
            pred, maps = model(toks.to(DEV), return_attn=True)
            w_att = query_attention_weights(maps, k)[0]
            tgt = float(xq @ w)
            err_tx = (pred.item() - tgt) ** 2
            err_ridge = (ridge_predict(X, y, xq) - tgt) ** 2
            d = trial_diagnostics(X, xq, w_att=w_att, P_est=P_est)
            rec = dict(k=float(k), err_tx=err_tx, err_ridge=err_ridge,
                       err_gap=err_tx - err_ridge, **d)
            for f in FIELDS:
                rows[f].append(rec[f])

    np.savez("experiments/spearhead_a/a3_trials.npz", **{f: np.array(rows[f]) for f in FIELDS})
    print(f"saved {N_TRIALS} trials -> a3_trials.npz")
    print(f"mean err: transformer {np.mean(rows['err_tx']):.4f}  ridge-floor {np.mean(rows['err_ridge']):.4f}")


if __name__ == "__main__":
    main()
