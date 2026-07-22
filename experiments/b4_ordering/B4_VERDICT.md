# B4 verdict — killed at Tier A, and the corpse is informative

**Pre-registered primary (B4_TIER_A.md): KILL** per PROPOSAL.md criterion (b).
The design was runnable (U-spread everywhere while the terminal additive Gram is
order-invariant by construction) and the precondition was strongly met (the causal
transformer shows median 1.8× MSE range across permutations of the *same* prompt) —
but the pinned windowed-observability score predicted none of it: pairwise sign
accuracy 0.495, Wilson95 [0.488, 0.502] over 18,000 pairs; within-task Spearman −0.002.

**Labeled sensitivity appendix (B4_APPENDIX.md, exploratory): equally null.**
Window lengths 2/4/6, unweighted windows, recency-discounted Grams (γ = 0.5, 0.8 — the
A5-unification hypothesis), attention-weighted terminal Gram, and crucial-demo position
all sit within ±0.05 of zero. No variant earns a new pre-registration.

## What this actually establishes

1. In this setting, **order sensitivity is not excitation geometry.** A learned causal
   ICL model can be massively order-sensitive while every excitation/observability
   functional of the ordered prompt is flat against error. Whatever drives the
   sensitivity here (positional-embedding idiosyncrasy, the learned algorithm's
   recency habits interacting with labels), it is not "which windows were informative."
2. The control-theoretic account of ICL order sensitivity now carries a **negative
   prior from us at toy scale**. Untested: LLM scale, semantic tasks, classification
   (the Lu et al. regime). Anyone (including us) proposing observability-based ordering
   for real LLMs should cite this null first.
3. Process note: the theory gate had judged B4's *mechanism* correctly licensed
   (constant parameter, time-varying regressor is PE's canonical setting) — licensing
   was never the problem. The mechanism is simply not what the phenomenon is made of
   here. Right kind of object; wrong phenomenon. That distinction is exactly what
   pre-registered Tier-A tests are for, and this one cost ~40 minutes of CPU.

Status board: B4 ⚗️→🔬 killed-as-tested; the ordering line closes unless a Tier-B-scale
motivation appears. The A5 pilot's P2 (discounted excitation under drift) is UNAFFECTED —
it concerns eviction under streaming, not ordering — but inherits a caution flag.
