# Workspace excitation: two PE questions asked of Anthropic's J-space

**Status:** discussion note (2026-07-22).
Builds explicitly on the published global-workspace work (J-lens; capacity bounded at
no more than ~25 concepts); our wedge is the excitation/identifiability treatment of
*their* object.

## Why we have standing here

The J-lens is the Jacobian of output likelihood w.r.t. internal state — literally the
nonlinear-PE information carrier (our Section-3 ladder object). And our E1 result is the
closest thing we own to a prior: for SAE features, *separation* of the exciting data —
not its richness — predicts which features are stable. The workspace note asks the same
two questions one level up, with that sign lesson in hand.

## Q1 — Are workspace concepts identifiable? (measurable now, mirrors E1)

A concept's J-vector is extracted from contexts that "excite" it. Pre-register (informed
by E1's flip): **concepts whose exciting contexts form tight, well-separated regions
have stable J-vectors across contexts/checkpoints; diffusely-excited concepts drift.**
If true: a label-free reliability score for workspace readouts — directly useful to
anyone doing J-space safety auditing, since an unstable J-vector steers/reads
unreliably. If the sign flips *again* (richness wins at the workspace level), that
itself would be a sharp structural fact about the difference between dictionary
features and broadcast concepts.

Build path (open models, A100): reimplement a minimal J-lens on GPT-2/Gemma-2-2B
(averaged linearized effect of mid-layer state on token likelihood — the paper
describes enough to attempt a faithful small-scale version; reimplementation risk is
real and gets logged honestly). Then E1's exact analysis pipeline transfers: excitation
geometry of activating contexts vs cross-context/cross-checkpoint J-vector stability,
frequency-matched nulls included.

## Q2 — Is capacity an excitation budget? (flashier, riskier)

The workspace holds no more than ~25 concurrent concepts. PE reading: a bound on
**simultaneous excitation rank** — tasks requiring more concurrently-identified latent factors than
the budget should fail *predictably*, with a measurable occupancy cliff, not merely
gradually. Build path: parametric m-factor composition tasks (answer depends on m
simultaneously-tracked quantities); measure J-space occupancy (number of
simultaneously-active, causally-necessary J-directions) and accuracy as m sweeps
across the claimed capacity. Prediction: accuracy cliff co-located with occupancy
saturation. Honest caveats stapled on: our toy-scale attention-budget probe was
negative (A3 aliasing suite), and small open models may have a different (or no crisp)
workspace — a negative here is informative about small models, not about Claude.
