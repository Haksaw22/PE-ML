# Independent-Mode Excitation: Block-Diagonal PoE as the Bridge Between Few-Shot and RL

**Status:** research note / proposal (v0.1)
**Companion to:** `fixed-point-poe-note.md`
**Scope:** the spine (group **G**) unifying algebraic PoE, the "independent skills" idea, and the few-shot architecture question
**Date:** 2026-07

---

## 0. Abstract

Three ideas that arose separately — (i) decomposing a task into **independent skills that don't trade off**
(the "football" objective: `max Σreward = Σ local optima`), (ii) **algebraic persistency of excitation** as
cyclic-module maximality of a block-Hankel matrix, and (iii) **finding the few-shot structures with the greatest
correspondence to the output** — are three views of a single object: *decompose a task's representation into
independent, individually-identifiable modes.* The organizing result is a certificate:

> **A task separates into independently-optimizable skills exactly when its excitation (Fisher/information)
> matrix block-diagonalizes across the mode-subspaces, with each block persistently exciting.**

Cross-excitation between two subspaces is precisely the coupling that forces joint optimization; its absence is
what lets you learn kicking and tackling independently. This note formalizes the certificate, gives its **static
face** (few-shot: a catalecticant/CP decomposition → output-correspondence modes via reduced-rank regression) and
its **sequential face** (RL: a block-Hankel decomposition → skill modes that subsume the C-wedges as per-mode
operations), and states the discovery problem as **learning a representation that block-diagonalizes excitation.**

---

## 1. Intuition

Learning to play football, you eventually notice that *kicking* and *tackling* barely interact — improving one
doesn't cost the other — while *running* overlaps with both. If you could **find** that decomposition, you would
never need to optimize the whole game jointly; you'd optimize each skill locally and stitch overlaps. The question
"when can I do that?" has a precise answer: **when the directions that excite one skill are information-orthogonal
to the directions that excite another.** Independence of skills = block-diagonal information matrix = a *reducible
representation*. And a reducible representation is exactly what a Hankel matrix detects. So "find independent skills"
and "find the irreducible components the sequence excites" are the same problem.

---

## 2. Two faces of algebraic PoE (they are one object)

- **Static / superposition (few-shot).** A task is a superposition of `R` sub-skills; identifiability = uniqueness
  of a symmetric CP / tensor decomposition, certified by the rank of a **catalecticant** matrix. *The catalecticant
  is a Hankel matrix* (a moment matrix), so this is Hankel structure with a symmetric shift.
- **Sequential / dynamical (RL, sequences).** A sequence `g₀,g₁,…` is **algebraically persistently exciting** iff
  `span{ρ(gₜ)} = 𝒜` (the full representation algebra), equivalently `rank(block-Hankel) = dim 𝒜`, equivalently the
  cyclic module `𝒜v = V` — no invariant subspace stays hidden. This is realization theory: `H = 𝒪·𝓡`
  (observability × reachability), `rank(H) =` minimal system order = number of modes.

They are the **static and sequential shifts of the same Hankel/moment operator**, so a result proved on one transfers
to the other. This is the structural reason few-shot and RL are the same problem here.

---

## 3. Core object and the separability certificate

Let `φ(x)∈ℝ^d` be the task-relevant feature (frozen, in the few-shot case; the value/policy feature in RL), and let
`M = 𝔼[ gg^⊤ ]` be the task **excitation/information matrix** (`g = ∇log p` or the readout gradient — the same Gram
that appears throughout the program). A **mode decomposition** is a set of orthogonal projectors `{Π₁,…,Π_k}`,
`ΣΠᵢ = I`, onto subspaces (the "skills").

**Definition (independent modes).** The modes are *independent at level α* if
$$\Pi_i M \Pi_j = 0 \ (i\ne j)\quad\text{and}\quad \lambda_{\min}(\Pi_i M \Pi_i)\ge \alpha_i>0\ \forall i,$$
i.e. `M` is **block-diagonal** in the `{Πᵢ}` basis and **each block is persistently exciting**.

**Separability theorem (informal).** If the modes are independent at level α, then
1. each skill is **separately identifiable** (block `i` recovers its parameters at rate set by `αᵢ`, unaffected by the others);
2. the estimator/optimizer **decomposes**: `θ* = Σᵢ θᵢ*`, where `θᵢ*` solves the mode-`i` subproblem *independently*;
3. for a reward objective with block-diagonal curvature, **`max Σreward = Σᵢ (per-mode optimum)`** — the football condition holds exactly.

Conversely, a nonzero cross-block `ΠᵢMΠⱼ` is *precisely* the coupling that forces joint optimization and the
skill trade-off. So "skills that don't trade off" ⟺ "info-orthogonal excitation subspaces" ⟺ "reducible
representation `V = V₁⊕…⊕V_k`."

**The discovery problem (what we actually build).** Real `M` is not block-diagonal in the given basis. So the task is
to **find the projectors `{Πᵢ}` (equivalently a representation φ) that make `M` as block-diagonal as possible while
keeping each block PE** — a *joint / simultaneous block-diagonalization* of the excitation operator (static: over the
CP factors; sequential: over the block-Hankel SVD), with a penalty on residual off-block mass `Σ_{i≠j}‖ΠᵢMΠⱼ‖`.

---

## 4. Few-shot face: output-correspondence modes and the architecture

"Structures with the greatest correspondence to the output" has an exact meaning. Given demos `{(φ(xᵢ), yᵢ)}`, the
directions `v` that most "correspond to output" maximize the correlation between `⟨v,φ(x)⟩` and `y` — i.e. the top
**canonical directions of the input–output cross-covariance**, equivalently the **reduced-rank regression** solution,
equivalently the leading left factors of the **input–output cross-Hankel**. These output-correspondent directions are
the modes worth exciting: they combine *high excitation* (large `λ` in `M`) with *high output relevance* (this is
exactly the task-Fisher projector `P` of the program, now given a constructive form).

**Architecture principle (the thing to converge on).** Build a model that (a) **discovers** the output-correspondent
independent modes and (b) **routes context through them as separately-identifiable channels**, under the attention
budget of `fixed-point-poe-note.md`. Concretely: a block-structured read/route where each block corresponds to one
excitation mode; overlaps between skills are **shared components** (soft assignment of a feature to multiple blocks).
Refined iteratively — measure residual off-block excitation mass and split/merge blocks to drive it down.

---

## 5. RL face: block-Hankel modes subsume the C-wedges

Collect features `φ(sₜ,aₜ)` along trajectories, form the **block-Hankel** `H`, take its SVD → the **dynamic excitation
modes** (a DMD/subspace-ID of the value-relevant dynamics); block-diagonalize the value-Fisher into independent skill
subspaces. Then the previously-thin C wedges become **per-mode operations on one object** (this is group **G6**):

- **C2 (multisine mode-excitation)** = the per-mode *excitation primitive*: drive each block toward `λ_min ≥ αᵢ`.
- **C1 (task-weighted excitation reward)** = reward the *marginal per-mode* `λ_min` gain.
- **C3 (self-terminating exploration)** = stop exciting a block once its `αᵢ`-certificate is met.
- **C4 (replay curation)** = keep each block's stored-exciting set (concurrent-learning PoE, per mode).

Plus the **football decomposition** itself (**G4**): dynamically-sized, overlapping skill windows discovered by where
the cross-block excitation `ΠᵢMΠⱼ` is weak; and the **global=local objective** (**G5**): train toward block-diagonality
so trajectory optimization separates.

---

## 6. Block-diagonalizing representation learning (revives A6, un-blocked)

Meta-train `φ` so that the excitation matrix `M(φ)` block-diagonalizes into independently-identifiable skills. The
objective couples two terms: **(i) each block PE** (`Σᵢ −log λ_min(ΠᵢMΠᵢ)`, keep skills identifiable) and **(ii) minimal
cross-excitation** (`Σ_{i≠j}‖ΠᵢMΠⱼ‖²`, keep skills independent). This is *not* the blocked A6 ("condition the support
Gram"); it is "**learn features whose excitation operator is reducible into PE blocks**," which — as far as the
prior-art pass found — nobody has posed.

---

## 7. Tooling inherited for free

The Hankel/realization framing means we do **not** build the machinery from scratch: subspace identification
(Ho-Kalman, N4SID, MOESP), spectral learning of latent-variable models (HMM/PSR/weighted automata), Koopman/DMD for
the dynamic modes, CCA / reduced-rank regression for the output-correspondence modes, and joint-diagonalization
algorithms (JADE-style) for the block-diagonalization step. Each is a mature, off-the-shelf component.

---

## 8. Predictions and minimal experiments

1. **Separability is measurable and predictive.** Compute residual off-block excitation `ρ = Σ_{i≠j}‖ΠᵢMΠⱼ‖ / ‖M‖`
   on a discovered decomposition; **the drop in joint-vs-separate optimization gap is monotone in `ρ`** (small `ρ`
   ⇒ separate optimization ≈ joint optimum). Falsifiable on a synthetic multi-skill task with a *known* block structure.
2. **Few-shot (static).** On a 2–3 task in-context superposition (Garg-style), the output-correspondence / CCA modes
   predict which skills are identifiable and beat similarity/DPP selection at equal shots (ties to B1/B2's `k*` law).
3. **RL (sequential).** Block-Hankel modes of the value features recover the ground-truth skills of a compositional
   control task (e.g. reach+grasp), and per-mode excitation (G6) reaches identifiability with fewer environment steps
   than entropy exploration — *and self-terminates per block*.
4. **Kill criterion.** If the joint-vs-separate gap does **not** track `ρ`, the separability certificate is wrong; if
   discovered modes match a graph-Laplacian/eigenoption decomposition better than the excitation decomposition, the
   "excitation-driven, not diversity-driven" novelty claim is dead.

---

## 9. Failure modes / open questions

1. **Exact block-diagonalization is generically impossible.** Real Fisher matrices don't block-diagonalize exactly;
   everything must be *approximate/soft* (minimize off-block mass), and the separability guarantee degrades to a bound
   in `ρ`. Open: a clean perturbation theorem "optimization gap `≤ f(ρ)`."
2. **Overlaps break clean separability.** Running∩kicking = shared components; soft assignment reintroduces coupling.
   Open: the right object is likely a *block-diagonal-plus-low-rank* excitation matrix (independent skills + a shared
   core), and the theory needs the low-rank correction.
3. **Model selection (k and the partition) is hard.** Choosing the number of modes and the partition is the crux;
   Hankel singular-value gaps / catalecticant rank give a principled `k`, but noise makes the gap soft.
4. **Is the discovered decomposition the *task's* or the *model's*?** Excitation modes come from `φ`; a bad `φ` gives
   spurious skills. This is why G3 (learn `φ` to block-diagonalize) and mode discovery must co-train — a chicken-and-egg
   the experiments must control for.

---

## 10. Prior art (honest)

The **tools are all mature** — CP/Kruskal & tensor method-of-moments (Anandkumar et al.), subspace ID (Ho-Kalman;
Van Overschee–De Moor), spectral learning (Hsu–Kakade–Zhang; Balle; PSRs), Koopman/DMD (Schmid; Williams EDMD),
CCA / reduced-rank regression (Hotelling; Izenman), skill/option discovery (option-critic; DIAYN; Laplacian
eigenoptions, Machado et al.), gradient-conflict methods (PCGrad), block-diagonal Fisher (K-FAC). The defensible
novelty is **not** any single tool but:

1. the **separability ⟺ block-diagonal-excitation certificate** as the organizing principle (with the football
   `max Σr = Σ local optima` as its exact statement);
2. discovering modes by **excitation / identifiability**, not by diversity/entropy (DIAYN) or the transition-graph
   Laplacian (eigenoptions) — different modes, and a certificate the others lack;
3. **unifying static few-shot and sequential RL** under one Hankel/moment object;
4. the **block-diagonalizing representation objective** (G3).

> Net: if a reviewer shows that eigenoptions or PCGrad already recover the excitation decomposition in practice, the
> wedge narrows to (1)+(3) — the certificate and the unification — which is still a real contribution.

---

## References (from memory — verify before citing)

Anandkumar et al. 2014 (tensor decompositions for latent variable models). Ho & Kalman 1966; Van Overschee & De Moor
(subspace ID). Hsu, Kakade & Zhang 2012 (spectral HMM); Balle et al. (weighted automata); Littman/Singh (PSRs).
Schmid 2010 (DMD); Williams et al. 2015 (EDMD/Koopman). Hotelling 1936 (CCA); Izenman 1975 (reduced-rank regression).
Bacon et al. 2017 (option-critic); Eysenbach et al. 2019 (DIAYN); Machado et al. 2017 (eigenoptions). Yu et al. 2020
(PCGrad). Martens & Grosse 2015 (K-FAC). Willems et al. 2005 (fundamental lemma; data-driven control).
