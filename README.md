# PE ML — Persistency of Excitation for Machine Learning

**A research program porting the identifiability mathematics of adaptive control —
persistency of excitation (PE) — into modern ML: when is a learner's data rich enough to
pin down what it's trying to learn, and what breaks when it isn't?**

Classical control has a precise answer to "why did my adaptive system track perfectly yet
learn nothing": the data stopped asking linearly independent questions. This repo asks the
same question of in-context learning, sparse-autoencoder features, and exploration — and
documents, with equal care, where the transplant *works* and where it collapses into
classical statistics.

## Headline results

| Result | Where | Status |
|---|---|---|
| **Label-free ICL failure diagnostic**: prompts whose examples under-excite the task subspace have **~3× the error at fixed shot count**; the subspace is estimable from 8 probe tasks (no oracle) | [`experiments/spearhead_a/A3_VERDICT.md`](experiments/spearhead_a/A3_VERDICT.md) | ✅ built, deconfounded |
| **Aliasing separation**: one *informative* demo (new task direction) beats one *similar* demo (near-duplicate) **16×** — informativeness ≠ similarity, and excitation measures the right one | [`A3_ALIASING.md`](experiments/spearhead_a/A3_ALIASING.md) | ✅ built |
| **Excitation predicts ICL-specific failure**: low excitation predicts the transformer's *excess error over the Bayes floor on the same prompts* — degenerate context geometry is where learned inference breaks from Bayes | [`A3_RESULTS.md`](experiments/spearhead_a/A3_RESULTS.md) | ✅ built |
| **SAE feature identifiability**: does per-feature excitation predict which interp features wander across seeds? (frequency-confound-controlled, pre-registered) | [`experiments/e1_sae/DESIGN.md`](experiments/e1_sae/DESIGN.md) | 🔬 running |
| **Prompt order as uniform observability**: a mechanistic (not heuristic) account of ICL order sensitivity | [`experiments/b4_ordering/PROPOSAL.md`](experiments/b4_ordering/PROPOSAL.md) | 📋 pre-registered, unbuilt |

**Honest demotions, published alongside** (see [`A3_VERDICT.md`](experiments/spearhead_a/A3_VERDICT.md)):
per-query, classical Bayes predictive variance beats the excitation score (as pre-registered —
the instrument's honest scope is the query-agnostic certificate); the attention-budget
mechanism this program originally proposed tested *negative* and is retired with data.

## Why the process is half the point

Every hypothesis here ran a gated adversarial pipeline: prior-art sweep → theory gate
(prosecution/defense/judge, pre-registered kill criteria) → experiment-design gate →
execution → results red-team. The program then **audited its own past verdicts** and
overturned two of them. The paper trail:

- [`process/track-status.md`](process/track-status.md) — every route (A–G) with its verdict; nothing quietly dropped
- [`process/gate-verdicts-2026-07-19.md`](process/gate-verdicts-2026-07-19.md) — full theory-gate case files
- [`process/process-audit-2026-07-19.md`](process/process-audit-2026-07-19.md) — the self-audit that found our own headline was confounded (it was; the honest number is 3×, not 4.88×)

Killed ideas stay visible: the misspecification-excitation probe died against classical
robust experiment design ([`experiments/spearhead_b/RESULTS.md`](experiments/spearhead_b/RESULTS.md)) and is documented as thoroughly as the survivors.

## Map

```
theory/              research notes (incl. superseded ones, marked as such) + idea ledger
process/             status board, audits, gate verdicts — the epistemic paper trail
experiments/
  spearhead_a/       ICL excitation diagnostic (A1→A3; A3 is current)
  spearhead_b/       misspecification probe (negative result, documented)
  e1_sae/            SAE feature identifiability pilot (pre-registered, running)
  b4_ordering/       prompt-order proposal (pre-registered, unbuilt)
article/             essay draft: "the geometry of asking good questions"
```

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install torch numpy scipy matplotlib
# ICL diagnostic (CPU, ~30 min total):
python experiments/spearhead_a/exp_a3_train.py
python experiments/spearhead_a/exp_a3_eval.py
python experiments/spearhead_a/exp_a3_analysis.py
python experiments/spearhead_a/exp_a3_aliasing.py
# SAE pilot (GPU): see experiments/e1_sae/DESIGN.md
```

*Kulbir Singh, 2026. All verdicts trace to pre-registered criteria and saved arrays.*
