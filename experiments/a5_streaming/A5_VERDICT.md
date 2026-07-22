# A5 verdict — the "persistent" in persistent excitation, demonstrated

Pre-registered design: `DESIGN.md`. Raw numbers: `A5_RESULTS.md`, `a5_arrays.npz`.

## Both pre-registered claims resolved

**P1 (static task): PASS.** Excitation-greedy eviction (keep the retained set's projected
lam_min high — the Chowdhary–Johnson concurrent-learning transplant) nearly halves query
error vs FIFO and reservoir at equal memory budget (0.215 vs 0.400/0.396; paired
difference +0.185 ± 0.056). The transplant has real content in a *learned* ICL system.

**P2 (drifting task): PASS at high drift, with the failure mode on display.** At
rho = 0.15, *undiscounted* excitation curation collapses to **worse than FIFO**
(0.774 vs 0.548): the policy hoards demos that were maximally informative about a task
that no longer exists — stale excitation beats no curation at being wrong. The
time-discounted variant degrades gracefully and wins (0.525; paired +0.249 ± 0.121).
At mild drift (rho = 0.05) the two are statistically indistinguishable (honest NULL,
as pre-registered).

**Guards:** oracle ceiling far below all policies (0.011 vs best 0.215) — the setting
genuinely discriminates; nothing is saturating. Honest cost noted: in the static
regime the discount is a small tax (0.252 vs 0.215) — you pay for drift-robustness
when there is no drift.

## Why this matters beyond the toy

This is the cleanest empirical separation in the repo between **one-off excitation**
(experiment-design informativeness, the thing our audits kept showing is classical) and
**persistent excitation** (every window must stay informative — the sequential clause
that classical optimal design does not own). It is also the natural PoE statement about
**agent memory**: curate context for *current* identifiability, not historical
informativeness. Scope honesty: toy linear-ICL scale; one drift model (subspace
rotation); the transplant claim (P1) is classical machinery in a new venue — the
PE-specific content is P2's discounted-vs-undiscounted separation and its
worse-than-recency failure mode.
