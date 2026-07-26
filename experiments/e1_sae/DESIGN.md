# E1 pilot — Per-feature excitation predicts SAE feature instability (pre-registered design)

**Status:** EXECUTED — see `E1_VERDICT.md`. (Per program process:
theory gate passed 2026-07-19 with instrument-bar REVISE; this document closes the
required repairs before execution. See `process/gate-verdicts-2026-07-19.md`.)

## Claim under test (narrowed per gate verdict)

For a sparse autoencoder trained on LLM activations, the **per-feature excitation**
`alpha_j` — the minimum eigenvalue of the (subspace-projected) second moment of the
activation distribution *conditional on feature j firing*, computed on a **fixed exogenous
corpus** — predicts which features are **unstable across training seeds**, *beyond what
firing frequency alone predicts*.

This is an instrument claim (label-free reliability diagnostic), not a novelty claim.
The PoE reading: a feature is a parameter direction; its firing set is the data that
excites it; a feature whose exciting data is rank-deficient in context space is
under-identified and free to land differently across seeds.

## Setup

- **Model:** GPT-2-small (124M), residual stream after layer 8.
- **Corpus:** OpenWebText sample; 10M tokens for SAE training (streamed), 2M held-out
  tokens for all diagnostics. Fixed and identical across seeds.
- **SAEs:** S = 5 seeds. Standard ReLU + L1 SAE, width 6144 (8x), same data order, same
  hyperparameters, different init seeds. (Data order held fixed so *init* is the only
  varying factor — cleaner attribution than the seed-instability literature's default.)
- **Instability measure:** for each feature j of seed-0, `instab_j = 1 - mean_s max_i
  cos(d_j^(0), d_i^(s))` over the other seeds s (decoder-direction matching, standard).
- **Excitation measure:** project held-out activations to the corpus's top r=64 PCs
  (fixed, seed-independent). For feature j with firing set S_j (activations where f_j > 0):
  `alpha_j = lam_min( E[zz^T | z in S_j] )` in the projected space. Secondary statistic:
  effective rank (participation ratio) of the same conditional Gram.

## Controls (the gate's required repairs)

1. **Frequency-matched null (K1 repair):** the trivial confound is that rare features are
   less pinned down. Controls: (a) partial Spearman of alpha_j vs instab_j controlling for
   log firing frequency AND conditional sample count n_j; (b) a matched-null population:
   random half-space "features" (random direction, threshold tuned to match each real
   feature's frequency) — their alpha–instability relation estimates the generic
   geometry effect; the real-feature relation must exceed it.
2. **Geometry-matched null (partial):** compare within frequency-decile bins so the
   alpha-instability relation is not driven by cross-bin composition.
3. **Non-circularity note (owed from gate):** alpha_j conditions on S_j, which is defined
   by the fitted w_j — but the *predicted* quantity (cross-seed instability) is measured
   on independently trained SAEs, and alpha_j is evaluated at the fitted parameter in the
   standard local-Fisher / Hessian-at-MLE sense (M-estimation practice). This is
   estimator-reproducibility, not ground-truth recovery; stated explicitly in the writeup.

## Pre-registered outcomes

- **P1 (primary):** partial-Spearman(alpha_j, instab_j | log f_j, n_j) < 0 with 95%
  bootstrap CI excluding 0, and the effect exceeds the matched-null relation.
- **K1 (kill):** CI includes 0, or the matched-null matches the real effect size →
  the diagnostic is generic geometry, published as a negative result.
- **P2 (secondary, descriptive):** low-alpha features concentrate the known pathologies
  (dead/dense/high-frequency-drift features).
- **Stretch (only if days allow):** concurrent-learning repair leg — retrain one seed
  oversampling contexts that excite the bottom-decile-alpha features; test whether
  targeted features' instability drops vs a matched control set. This is the novel-repair
  wedge; it is OPTIONAL and will not gate the pilot's publication.

## Compute plan

- **Primary:** A100 host (ssh haksaw22@harjot-metadog — currently down, spin-up
  requested). Activation streaming + 5 SAE trainings ≈ 2–4 A100-hours total.
- **Fallback:** local GTX 1650 Ti (4GB): width 3072 (4x), 4M tokens, 3 seeds — same
  design, smaller; ~1 overnight.
- All code hand-rolled (no SAELens dependency): `dump_activations.py`, `train_sae.py`,
  `analyze_stability.py` — small, reviewable files.
