# Fixed-Point Persistency of Excitation: An Excitation Theory of In-Context Learning

**Status:** research note / proposal (v0.1)
**Scope:** a novel, buildable extension of persistency of excitation (PoE) to few-shot / in-context learning
**Date:** 2026-07

---

## 0. One-paragraph abstract

Classical persistency of excitation says a regressor identifies a system iff its accumulated
information (Gram) matrix is uniformly positive-definite. In-context learning (ICL) behaves like
implicit regression whose *regressor is the set of demonstrations, embedded in the model's own
feature space*. So ICL should succeed exactly when the demonstrations are persistently exciting in
that space. The twist that makes this new — and that no classical PoE has — is **softmax
normalization**: adding a demonstration *steals attention mass from the others*, so the excitation
matrix depends on the demonstration set through a fixed point. This yields a **finite excitation
budget**: a hard trade-off between how many task directions ("spectral lines") you excite and the
signal-to-noise on each. The central, falsifiable prediction is a **non-monotone shots curve** —
more demonstrations can *reduce* identifiability — predicted quantitatively by an attention-weighted
minimum eigenvalue and invisible to similarity- or diversity-based demonstration selection.

---

## 1. Intuition (the whole idea in three sentences)

- **Few-shot learning is system identification.** The demonstrations are "inputs" you use to identify
  a hidden task; you succeed only if those inputs *probe every direction the task depends on*. That is
  literally persistency of excitation.
- **The catch is attention is a fixed budget.** A transformer's context has finite attention mass;
  every demonstration you add dilutes the attention paid to the rest. So excitation is not free the way
  a longer time-window is free in classical control — you are *spending a budget*, and past some point a
  new demonstration steals more signal than it adds.
- **Therefore excitation obeys a self-referential (fixed-point) law**, and the sweet spot is not "as many
  diverse examples as possible" but "the set that maximizes the worst-excited task direction *after*
  accounting for attention dilution."

---

## 2. Background objects

**Classical PoE.** A regressor $\phi(t)\in\mathbb{R}^d$ is persistently exciting (level $\alpha$, window
$T$) iff
$$\int_t^{t+T}\phi(\tau)\phi(\tau)^\top\,d\tau \;\succeq\; \alpha I \quad\forall t.$$
This lower bound on the smallest eigenvalue of the information matrix is what converts "the loss went
down" into "the parameters were *identified*," with exponential error decay. Frequency view: PE of order
$n \iff$ the input contains $\ge n/2$ distinct spectral lines.

**ICL as regression.** Trained transformers implement (approximately) ridge regression / a few steps of
gradient descent on the in-context examples in a learned feature space (Garg et al. 2022; von Oswald
et al. 2023; Akyürek et al. 2023). Write $\phi(x)\in\mathbb{R}^d$ for a demonstration's feature (e.g.
last-token residual stream at a mid layer). For a support set $S=\{x_1,\dots,x_k\}$ the *naive* induced
Gram is $M_S=\sum_i \phi(x_i)\phi(x_i)^\top$.

**Task-relevant subspace.** Let $g(x)=\nabla_w f_w(x)$ be the readout's gradient w.r.t. the effective
in-context parameters, $F_{\text{task}}=\mathbb{E}_x[g\,g^\top]$ its Fisher, and $P\in\mathbb{R}^{d\times r}$
the top-$r$ eigenvectors of $F_{\text{task}}$ — the **information-dense directions** ("modes"). Define the
task-projected Gram $M_{\text{task}}(S)=P^\top M_S\,P$.

---

## 3. The new object: attention-normalized fixed-point PoE

In a real softmax-attention model the effective regressor is **not additive**: the model attends to the
demonstrations with weights that depend on the *entire* set. Replace the additive Gram with an
**attention-weighted Gram**
$$M_S \;=\; \sum_{i=1}^{k} w_i(S)\,\phi(x_i)\phi(x_i)^\top,\qquad
w(S)=\operatorname{softmax}\big(s(S)\big),\quad \textstyle\sum_i w_i(S)=1,$$
where $s(S)$ are the (query-conditioned) attention scores over the demonstrations. Because $w$ depends on
$S$, the excitation matrix appears on both sides — **a fixed-point excitation condition with no classical
analog.** Define the **restricted excitation constant**
$$\boxed{\;\alpha(S)\;=\;\lambda_{\min}\!\big(P^\top M_S\,P\big)\;=\;\lambda_{\min}\!\Big(\textstyle\sum_i w_i(S)\,P^\top\phi_i\phi_i^\top P\Big).\;}$$

- **Success predictor (label-free).** Under the ICL-as-ridge surrogate, the worst-direction predictive
  variance obeys $\displaystyle \sup_{\|u\|=1,\,u\in\mathrm{range}(P)}\mathrm{Var}(u^\top\hat\theta)\le
  \sigma^2/(\alpha(S)+\lambda)$, so ICL error is **monotone decreasing in $\alpha(S)$** — computable with
  *zero test labels*.
- **Order / PE view.** $\operatorname{rank}(M_{\text{task}})$ = number of distinct task modes present =
  PE order ⇒ a **shot-scaling law: shots-to-learn $\approx$ effective task rank $r$.**
- **The budget (the novel phenomenon).** Because $\sum_i w_i=1$, raising the count $k$ spreads a fixed
  mass over more terms. Adding a demonstration that duplicates an already-excited mode *lowers* $w$ on the
  others without adding a new mode ⇒ $\alpha(S)$ can **decrease**. There is an optimal $k^\star$ and an
  optimal *composition*, set by a trade-off between PE-order (how many modes) and per-mode SNR (attention
  per mode). This is a genuine **excitation budget**, absent from all additive / time-window PoE.

**Sequential selection (greedy, budget-aware).**
$$x^\star=\arg\max_{x\in\text{pool}}\;\lambda_{\min}\!\Big(P^\top M_{S\cup\{x\}}\,P\Big)\ \text{with $w$ recomputed on }S\cup\{x\},\qquad
\text{stop when } \alpha \ge \alpha_{\text{crit}} \text{ or } \alpha \text{ drops.}$$

---

## 4. Central predictions (all sharply falsifiable)

1. **Monotonicity in $\alpha$.** ICL accuracy is monotone in $\log\alpha(S)$ with high $R^2$, computed
   with no test labels.
2. **Non-monotone shots curve (the signature result).** For a fixed pool, there exists $k^\star$ beyond
   which adding demonstrations *reduces* accuracy, and the turning point is predicted by the
   **attention-weighted** $\alpha$ — *not* by the additive $M_S$, cosine relevance, or DPP volume.
3. **Aliasing.** Injecting a near-duplicate of a relevant demonstration leaves $\alpha\approx$ unchanged
   (or lower) and does not help; an orthogonal-but-less-similar demonstration that raises $\alpha$ does.
4. **Rank law.** Shots-to-threshold-accuracy $\approx r$ (effective task rank), independent of pool size.

**Kill criteria.** If accuracy tracks the *unprojected* / *additive* $\lambda_{\min}(M_S)$ or cosine
relevance better than the attention-weighted $\alpha(S)$, the fixed-point/Fisher claims are dead. If the
aliased duplicate helps as much as the orthogonal demo, the PoE framing is dead.

---

## 5. Theory route: don't prove it from scratch — instantiate LPV-PoE

The scary objection ("attention makes the regressor context-dependent and non-additive") is actually the
door. A context-/order-dependent regressor is a **linear-parameter-varying (LPV) / linear-time-varying
(LTV) regressor**, and PoE for LTV systems is a developed classical theory (Kreisselmeier; Bitmead &
Anderson on directional/conditional excitation). So the make-or-break theorem is *not* "prove a new ICL
bound" but **"instantiate a known LTV-PoE bound for the softmax-weighted regressor,"** which is far more
tractable. Target theorem:

> **(Target.)** For a softmax-attention ICL learner with task-Fisher subspace $P$, the excess in-context
> risk on the query is $O\!\big(\sigma^2/(\alpha(S)+\lambda)\big)$, where $\alpha(S)$ is the
> attention-weighted restricted excitation constant, and the minimizing $S$ has $|S|=k^\star$ set by the
> attention-budget trade-off.

---

## 6. Minimal experiment protocol

**Tier A — controlled (ground truth known).** Transformer trained on in-context linear regression
(Garg-style), where the task subspace and rank $r$ are known, so $P$, $M_{\text{task}}$, and $\alpha$ are
unambiguous. Verify predictions 1–4 exactly; measure the fixed-point effect by ablating softmax → hard/
uniform attention (should *remove* the non-monotonicity).

**Tier B — real model.** Frozen open LLM (7–8B). $\phi$ = mid-layer last-token residual stream; $P$
estimated from ~8 labeled probe demos per task; pool ~200 candidates per task over a suite of
classification tasks.

**Methods at equal shot budget $k$:** (a) EXCITE-ICL greedy budget-aware $\alpha$; (b) additive-Gram
$\lambda_{\min}$ (ablation isolating the fixed-point term); (c) kNN cosine retrieval; (d) DPP diversity;
(e) random.

**Report:** accuracy vs. $\log\alpha$; accuracy vs. $k$ per method (look for $k^\star$); the aliasing arm;
$R^2$ of the label-free predictor. **Must beat** kNN and DPP at equal $k$, and must *predict* where they
break.

---

## 7. Failure modes / open questions

1. **Additivity of $M_S$ is only a surrogate.** The honest object is the attention-weighted Gram; if even
   that doesn't capture cross-demo attention (composition, order), $\alpha$ won't predict accuracy. → the
   LPV framing (§5) is the mitigation and the central theoretical bet.
2. **$P$ is not free.** Estimating $F_{\text{task}}$ needs a few labeled probes, so the method is
   *conditionally* label-free. Open: recover $P$ from unlabeled task structure or transfer it across tasks.
3. **E-optimality is non-smooth / NP-hard; the recovery bound is a linear surrogate.** Restricted-
   invertibility results (Allen-Zhu et al.; Kadison–Singer line) give approximation guarantees, but
   stitching them to the nonlinear ICL bound *and* the greedy approximation simultaneously is unproven.
4. **Is $\lambda_{\min}$ even the right functional?** If $P$ localizes $\theta$ well, average-case
   (G-optimal, trace) design may dominate worst-case (E-optimal, $\lambda_{\min}$). The distinctive PoE
   claim leans on the *fixed-point / budget* effect (§3), not on E- vs. G-optimality per se — keep the
   novelty anchored there.

---

## 8. Relation to prior art (honest)

- **Optimal design for ICL** (Mukherjee 2024): does G/A-optimal (average-case) demonstration design — *not*
  the min-eigenvalue / PoE object, and *not* attention-normalized. Closest neighbor; the wedge is the
  fixed-point budget + worst-direction certificate.
- **E-optimal subset selection** (Joshi & Boyd 2009; sensor/actuator placement): mature — so the *criterion*
  is not new; the *transplant to attention-normalized in-context features* is.
- **Directional/conditional excitation** (Bitmead & Anderson): greedy max-$\lambda_{\min}$ ≈ conditional PE
  — again, new venue, not new object.
- **ICL = GD/ridge** (Garg 2022; von Oswald 2023; Akyürek 2023): the bridge this note stands on.
- **Conditioning meta-learning** (Hiller, Harandi & Drummond 2022): conditions the support Gram for
  adaptation — overlaps the *representation-side* variant, not the fixed-point/attention-budget claim.

> Net: the genuinely unclaimed contribution is the **attention-normalized fixed-point excitation condition
> and its finite excitation budget**, with the non-monotone shots curve as its falsifiable fingerprint —
> not the E-optimal selection rule, which is old.

---

## References (from memory — verify before citing)

Åström & Wittenmark, *Adaptive Control*. Narendra & Annaswamy, *Stable Adaptive Systems*.
Bitmead & Anderson (directional/conditional excitation). Kreisselmeier (LTV excitation).
Chowdhary & Johnson (concurrent learning). Garg, Tsipras, Liang, Valiant 2022 (ICL of function classes).
von Oswald et al. 2023; Akyürek et al. 2023 (ICL as GD/ridge). Mukherjee 2024 (optimal design for ICL).
Joshi & Boyd 2009 (convex sensor selection). Ash et al. 2021 (BAIT). Hiller, Harandi & Drummond 2022
(conditioning meta-learning).
