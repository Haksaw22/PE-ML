# Commitment collapse: excitation-gated belief bifurcation (idea note, ungated)

**Status:** idea-stage, logged 2026-07-21. NOT gated, NOT scheduled. Wave-2 candidate.

## The question

LLMs hold many mutually inconsistent stances in superposition, bloating representation
space. Can a PoE-shaped mechanism make a model *collapse onto one opinion* when
warranted — e.g., commit to one side of a logically counterintuitive claim?

## The PoE translation

- Superposed stances = an unresolved posterior: multiple parameter settings consistent
  with all evidence so far. PE theory: the posterior contracts along a direction iff that
  direction is **excited**. Indecision = the *discriminating direction* (along which the
  stances differ) was never excited.
- Classical spine (be honest): active discrimination = optimal experiment design / dual
  control; commit-at-threshold = Wald's SPRT. The collapse mechanism is 80 years old.
- The ML-novel pieces:
  1. **Self-excitation:** for an LLM the discriminating "experiment" can be internal
     computation — chain-of-thought as active experiment design over one's own beliefs.
  2. **Collapse-iff-excitable (calibrated decisiveness):** forced collapse on genuinely
     underdetermined questions is miscalibration. The right gate: commit exactly when the
     discriminating direction CAN be excited (evidence / retrieval / derivation);
     stay superposed when it can't. PoE supplies the certificate for which regime holds.
  3. **Excitation hygiene:** sycophancy = the stance direction excited by social pressure
     rather than evidence. "Which inputs may excite which belief directions" is an
     alignment property statable (and maybe measurable) in excitation terms.
- Workspace link: a capacity-limited workspace can't hold many live hypotheses;
  commitment = workspace admission. The J-lens directed-modulation protocol is a natural
  experimental handle.

## Cheapest first test (if ever gated)

Probe a model on paired contradictory-stance prompts; find the stance direction (probe
or J-lens); measure whether targeted "discriminating" context (a resolving deduction)
collapses the internal superposition more than matched-length neutral context; check
calibration on genuinely underdetermined controls (collapse there = failure).

## Kill risks (for the future gate)

Sycophancy/belief-probing literature may already cover the measurement half; SPRT +
BED subsume the mechanism half; the wedge, if any, is the excitation-*certificate* gate
and the hygiene framing. Same-territory-vs-same-object discipline applies.

## Extension (2026-07-22): lens-directed identification / "opinionated" models

The follow-up: instead of full system identification, use excitation to identify
only the SUBSYSTEM aligned with a chosen lens (e.g., "as truth-consistent as possible"),
yielding models that are representationally much cheaper while as capable *under that
lens* — dropping the superposed contradictory stances that bloat representation space.

**Classical anchors (the transplant is licensed):** c-optimal experiment design —
design excitation to estimate a specific functional c^T·theta, not all of theta — and
Gevers-line "identification for control": identify only the dynamics the controller
will actually use, to the accuracy the loop actually needs. The lens = the functional.

**The ML-shaped wedge:** (1) lenses here are *semantic* functionals (truth-consistency,
a value system, a domain) with no obvious linear parameterization — defining the
"c-vector" of a worldview is the open problem; (2) the payoff claim is representational
COMPRESSION (an opinionated model as a cheap projection of a superposed one), which is
distillation-with-a-lens rather than estimation; (3) the collapse mechanism (how the
representation actually sheds off-lens mass) is admitted-unknown — candidates: distill
on lens-consistent self-generated data; targeted ablation of off-lens subspaces
(task-arithmetic style); or the commitment gate above applied per-question at training
time. Honest risk: for underdetermined questions an opinionated model is *miscalibrated
by construction* — the safety-interesting version keeps calibration on genuinely open
questions and sheds only *resolvable* contradictions (collapse-iff-excitable, applied
with a lens).
