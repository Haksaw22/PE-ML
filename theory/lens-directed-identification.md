# Lens-directed identification: opinionated models at lower representational cost

**Status:** discussion note (2026-07-22) — promoted from the extension in
`commitment-collapse-note.md`. Not gated; decision points below.

## The claim being sharpened

Full system identification is wasteful when you only care about the system *through a
lens*. If the lens is a functional ℓ(model), excitation should target only the
directions ℓ depends on. Applied to LLMs: a model holding superposed contradictory
stances pays representation for all of them; a **lens-directed collapse** would keep
behavior-through-the-lens intact while shedding the off-lens superposition — an
*opinionated* model that is cheaper yet, under its lens, as capable.

## Classical anchors (the transplant is licensed, and by name)

- **c-optimal experiment design:** design excitation to estimate one functional c᷀θ,
  not all of θ. The lens is the c-vector.
- **Identification for control (Gevers):** identify only the dynamics the controller
  will use, to the accuracy the loop needs. "The use defines the required accuracy" is
  exactly the opinionated-model economics.
- **Estimable functions:** even when θ is not identifiable, specific functionals may
  be — partial identifiability is a developed theory, not a hack.

## The three open problems (in build order)

1. **Parameterizing a semantic lens.** Candidates: (a) a probe direction trained on
   lens-labeled contrast pairs (CCS-style for truth); (b) a Jacobian against a
   lens-reward head (J-lens machinery pointed at a scorer); (c) an implicit lens = a
   contrast task-set. Note the prior-art line to respect: CCS/Burns owns unsupervised
   truth-direction finding; RepE owns steering vectors. Our wedge is not finding the
   direction — it is the **identification/excitation treatment of it** (which data pins
   the lens-relevant subsystem down; certificate of sufficiency; targeted collapse).
2. **The collapse mechanism** (the admitted unknown). Candidates: (i) lens-filtered
   self-distillation — generate with the teacher, keep only lens-consistent outputs,
   distill a smaller student; (ii) off-lens subspace ablation (task-arithmetic-style
   surgery, cheap but blunt); (iii) commitment-gate training — per-question, excite the
   discriminating direction (deduction/retrieval), commit iff it resolves. My ranking:
   (i) first — it is measurable, safe, and the compression claim attaches naturally.
3. **The calibration guard** (the safety content): shed only *resolvable*
   contradictions. An opinionated model that is opinionated about genuinely open
   questions is miscalibrated by construction. Operationally: a held-out set of
   underdetermined questions on which the student's entropy must NOT collapse. This
   guard is what separates the proposal from "bias injection" — it is the part I would
   lead with in any safety-audience framing.

## Cheapest first experiment (concrete, A100-scale)

Lens = **arithmetic/logical consistency** first, not truth-in-the-world (crisp ground
truth, no politics, clean contradiction sets), on a small open model:
1. Build contradiction-prone prompt sets where the model demonstrably holds both
   stances (measure stance-direction superposition mass via probes).
2. Collapse via mechanism (i): self-generate, filter by lens, distill into a student
   at 50% width.
3. Pre-registered outcomes: student matches teacher on lens-aligned tasks (compression
   claim); stance-superposition mass drops vs a same-size vanilla-distilled control
   (collapse claim — the control is essential, plain distillation compresses too);
   entropy retained on the underdetermined holdout (guard).
4. Kill: if vanilla distillation collapses superposition just as much, the lens did no
   work and the idea reduces to ordinary task distillation.

## Decision points

- **D1:** first lens — logical consistency (my recommendation: crisp, defensible) or
  truthfulness (flashier, muddier ground truth)?
- **D2:** primary payoff claim — representational *compression*, or *interpretability*
  (an opinionated model is easier to audit), or *stability* (no stance flip-flop)? The
  experiment's headline metric follows from this choice.
- **D3:** is the calibration guard a constraint (my strong preference) or out of scope
  for v1?
