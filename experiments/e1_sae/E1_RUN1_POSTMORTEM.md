# E1 run 1 — postmortem: INVALID RUN (not a kill)

Run 1 (2026-07-22, A100) returned "K1 KILL: partial correlation CI includes 0". The
results red-team **voids this verdict on construct-validity grounds** before it counts
against the hypothesis:

- **The SAEs were not sparse.** Final L0 ≈ 1,530–1,550 of 6,144 (healthy: ~20–100).
  Root cause: reconstruction loss is a *sum* over 768 dims while the L1 coefficient
  (5e-4) was scaled as if against a *mean* — the sparsity penalty was ~500x too weak,
  yielding dense overcomplete autoencoders, not feature dictionaries.
- **The diagnostic therefore measured nothing feature-like.** Mean per-feature
  excitation alpha = 0.996 (max 1.0 by construction): conditioning on "feature fires"
  left the activation distribution essentially unrestricted — the conditional Grams all
  equal the corpus Gram. The frequency-matched null half-spaces (mean alpha 0.40) were
  *more* selective than the "features."
- **Bimodal population:** ~1,600 dense always-on features + ~4,500 near-dead ones
  (only 1,596/6,144 met the n >= 500 floor on 1M tokens despite L0 = 1,540).

**Status of the pre-registered test:** unresolved. K1 applies to a valid sparse regime
only. Pipeline machinery (dump, training, matching, accumulation, partial-correlation
test, null population) validated end-to-end by this run.

**Fix for run 2:** calibrate l1 by short sweep (target L0 in [30, 100]), retrain all
seeds, rerun the unchanged pre-registered analysis. No changes to the test itself —
only to the SAE training hyperparameters. This document stays in the repo either way.
