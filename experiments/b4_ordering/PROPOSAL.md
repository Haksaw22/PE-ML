# B4 — Demo ordering by windowed uniform observability (pre-registered proposal)

**Status: EXECUTED 2026-07-22 — KILLED at Tier A** (pre-registered criterion (b): sign
prediction at chance despite strong order sensitivity; sensitivity appendix equally null).
See `B4_VERDICT.md`, `B4_TIER_A.md`, `B4_APPENDIX.md`. Original pre-registration below,
unedited.

## The idea in one paragraph

In-context learning is known to be sensitive to demonstration *order* (Lu et al. 2022),
but existing order scores are entropy/calibration heuristics measured post hoc. Control
theory has a mechanistic candidate: a static parameter observed through a *time-varying*
map — which is exactly what a fixed task seen through prefix-dependent attention is — is
identified iff the system is **uniformly observable**: every window of the sequence, not
just the whole, must excite the parameter subspace. The proposal: score an ordering
`pi` by its worst window, `U(pi) = min_j lam_min(Omega_j(pi))`, where `Omega_j` is the
attention-weighted windowed Gram, and test whether U predicts (and improves) ordering
performance where terminal-Gram scores cannot — because the *terminal* Gram is
order-invariant by construction, while ICL performance is not. Whatever explains order
sensitivity must therefore be a *windowed/sequential* quantity; U(pi) is the canonical one.

## Why this is the right kind of object (gate finding)

The prosecution's charge that uniform observability "requires a moving state" failed:
constant parameter + time-varying regressor is the *canonical* setting of classical PE
theory (Anderson's persistence-of-excitation line, adaptive-control identification).
The transplant is licensed; the open question is purely empirical content.

## Pre-registered protocol (repairs closed)

1. **Pinned before any run (forking-paths repair):** window length N = 4 demos; attention
   weights = mean over heads of the final layer only; demo feature = its input embedding
   in task coordinates. No post-hoc sweeps of these — a sensitivity appendix may vary
   them *after* the primary result is recorded.
2. **Label-free scope (P-is-not-free repair):** with raw (unprojected) window Grams, the
   claim is only "beats recency/random ordering at zero label cost." Matching Lu et al.'s
   labeled-probe scores requires a probe-estimated projection and is claimed only in that
   arm, with probe cost reported.
3. **Existence check (control-construction repair):** before the main experiment, verify
   constructively that permutation pairs with matched terminal Gram but different U(pi)
   exist at k = 8–12 (they do trivially in the linear model; verify for real attention
   weights). If they do not exist at realistic k, the design is unrunnable and the line
   stops — recorded as such.
4. **Primary test:** Tier A (Garg-style ICL transformer, as `spearhead_a/a3_*`): over
   matched-terminal-Gram permutation pairs, does sign(U(pi_1) - U(pi_2)) predict
   sign(MSE(pi_2) - MSE(pi_1))? Pre-registered success: accuracy > 0.5 with 95% CI
   excluding 0.5. Tier B (open LLM, classification suite): correlation of U with
   per-permutation accuracy, against GlobalE/LocalE baselines at equal probe budget.
5. **Kill criteria:** (a) existence check fails at realistic k; (b) Tier A pairwise
   prediction no better than chance; (c) Tier B adds nothing over entropy baselines at
   equal label cost. Any of these is published as a negative result.

## Relation to the rest of the program

The windowed Gram `Omega_j` is the same object as the concurrent-learning replay
certificate (C4) and the per-mode scheduling primitive (C3) — see the cross-line finding
in `process/gate-verdicts-2026-07-19.md`: B4-as-per-mode-scheduler is the one well-formed
cross-line combination, contingent on this proposal and the G-narrow certificate both
surviving their tests.
