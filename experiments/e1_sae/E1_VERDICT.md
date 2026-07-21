# E1 verdict — the pre-registered prediction FAILED; the sign flip is the finding

Run 2 (healthy SAEs: L0 ≈ 25, 0% dead, 2,693/6,144 features above the sample floor;
run 1 was voided, see `E1_RUN1_POSTMORTEM.md`).

## The pre-registered result

**P1 predicted:** higher per-feature conditional excitation α_j ⇒ *lower* cross-seed
instability, after controlling frequency. **The data says the opposite:**
partial Spearman(α, instability | log freq, n) = **+0.259** [95% CI +0.226, +0.293].
P1 is falsified as stated. Per the pre-registration's "unexpected sign" branch, we
investigated before claiming anything:

- **Not a control artifact.** The raw correlation is −0.07 while the partial is +0.26 —
  a classic suppressor: frequency dominates both variables (ρ(freq, instab) = −0.67;
  ρ(freq, α) = +0.37). Within frequency deciles — no regression adjustment at all — the
  correlation is positive in 10/10 bins (+0.02 to +0.35), and split-half replication
  gives +0.28 / +0.24. The within-frequency effect is real, positive, and robust.

## What the sign flip means (interpretation — post-hoc, flagged as such)

α_j measures how *diffuse* a feature's firing region is in activation space (how
unrestricted the conditional input distribution remains). The finding is therefore:

> **At matched frequency, features that fire diffusely are unstable across seeds;
> features that fire selectively are stable.**

The naive transplant treated an SAE feature like a regression parameter, whose estimate
improves with richer excitation of its design matrix. The data rejects that reading and
supports the other face of this program's own theory: a dictionary feature is a
**decomposition component**, and identifiability of decomposition components comes from
**separation/concentration** (the Kruskal-uniqueness / clustering regime — tight, distinct
firing regions every seed rediscovers), not from within-region isotropy. Diffuse
high-α features look like seed-dependent *tilings* of broad activation regions — the
geometry underlying feature splitting — which different seeds tile differently.

Two program-level consequences:
1. **The instrument survives with inverted sign and honest framing:** a label-free,
   activation-geometry-only score (within-frequency |ρ| ≈ 0.22–0.35) for flagging
   *unreliable* SAE features — no retraining or second seed needed at deployment,
   after one-time calibration.
2. **The theory correction is itself a PoE result:** the static-Gram reading fails for
   dictionary features exactly where the algebraic/decomposition face (catalecticant/
   Kruskal) takes over — matching where this program's theory gates said the real
   PoE content lives.

## Not yet established (named, unrun)

Whether α predicts feature *splitting* under width scaling; whether the sign persists
across layers, sites, widths, and real SAE suites (Gemma Scope); whether a
concentration-targeted training intervention improves stability (the repair leg).
These are the follow-ups; none is claimed.

Figures: `figures/e1_deciles.png`, `figures/e1_scatter_mid.png`. Arrays: `e1_arrays.npz`.
