# Persistency of Excitation: The Geometry of Asking Good Questions

*Sixty years ago control theory proved a theorem about why learning stops - roughly,
that your data ran out of new questions. I first met it as an engineer, before I knew
it had a name. This essay takes that theorem - persistency of excitation - for a walk
through machine learning: what it has to say about prompts, features and memory, the
traps I fell into along the way, and the experiments that came out the other side.*

**Code & experiments:** [github.com/Haksaw22/PE-ML](https://github.com/Haksaw22/PE-ML) -
every number below links to a verdict document in the repo.

---

## How I ran into it

I met persistency of excitation before I knew its name. At Magway I worked on control
for an electrodynamic carriage system - vehicles surfing magnetic fields down a track.
In motion the carriages produced almost pure sinusoidal waveforms, and part of my job
was making sure our sensing was granular enough to actually capture them. So for a
while I assumed that was the whole story of knowing a system: measure often enough,
and the truth arrives.

Then I spent months designing experiments to get our physical rig to show dynamics
that our simulations swore it had, and learnt the harder lesson. You can sample a
system at absurd rates and still learn nothing, because the system only answers what
you ask - and we had been asking one question, with great frequency and enthusiasm. It
turns out this exact failure has been a theorem in the adaptive-control literature for
sixty years, under probably the least glamorous name in applied mathematics:
**persistency of excitation**. Nyquist is about sampling often enough. This is its
less famous cousin, about asking *differently* enough - coverage of directions rather
than coverage of time. (We'll make that precise in a moment.)

The everyday version is something everyone already runs socially. You don't get to
know a person by asking whether they like food - you ask the question whose answer you
can't predict: what they'd defend in an argument, what they'd do with an unscheduled
Tuesday. You're looking, pretty literally, for what excites them, and each answer
tells you which question is worth asking next. Control theory wrote that down, proved
when it converges, and then mostly kept it to itself.

## The whole idea in one equation

Suppose something you care about is governed by unknown parameters
$\theta = (\theta_1, \theta_2)$, and each observation gives you one linear glimpse:

$$y = \phi^\top \theta$$

where $\phi$ is whatever you measured alongside the outcome. One observation is one
equation in two unknowns - a whole line of candidate explanations, not an answer. A
second observation pins it down, unless your second $\phi$ points the same direction
as the first, in which case you've asked the same question twice and got the same
information once. The bit that took me embarrassingly long to properly internalise:
ten thousand further observations along that direction won't help either. The
currency here isn't really sample count, it's directions.

The bookkeeping object is the **information matrix** $\sum_k \phi_k \phi_k^\top$ - a
running record of which directions have been probed and how hard. Its smallest
eigenvalue is your worst-illuminated direction, i.e. the question you've most
neglected. And the "persistency" part is the sharpest clause in the definition: it
demands that **every window** of recent data stay well-conditioned,

$$\sum_{k=t}^{t+N} \phi_k \phi_k^\top \;\succeq\; \alpha I \quad \text{for all } t,$$

because anything that adapts also forgets, and the rich data you supplied last year
doesn't excuse the monotone diet you're feeding it now. The picture I keep in my head
is an object in a dark room, with each measurement a torch beam from one angle. From
any single angle, plenty of different shapes cast the same shadow - so what you
actually want to collect is angles, not light.

![The ellipse is what you still don't know](figures/f_angles.gif)
*Twelve measurements, two interrogation styles. Left: the same question repeated - the
posterior stops shrinking along the direction nobody asked about. Right: varied
angles - the ellipse collapses. Same sample count in both.*

## How general is this, really?

The equation above looks aggressively linear, but the idea climbs.

**Linear.** The regressor $\phi$ carries the information, and everything reduces to
the rank story above.

**Nonlinear.** Let $y = f(\theta, u)$. Each observation now carves a curved surface
through parameter space, and the local information carrier becomes the sensitivity
$J = \partial f / \partial \theta$ - which in the linear case is exactly $\phi$. The
information matrix becomes $\int J^\top J\,dt$, at which point a statistician will
point out, correctly, that this is **Fisher information** and has been since 1925.
Persistency of excitation for nonlinear systems is the demand that the Fisher
information stay uniformly positive definite along your trajectory - no parameter
direction leaves the conversation. And where even local sensitivity fails, you're into
identifiability proper: can two different $\theta$'s produce identical outputs under
the inputs you actually applied? If yes, no estimator can save you - that failure was
locked in when you chose the inputs.

**Function space.** For neural networks, parameter space is honestly the wrong room to
be illuminating. Permute two hidden neurons: identical function. Rescale through a
ReLU: identical function. Enormous tracts of parameter space are pure gauge (which is
partly why networks tolerate savage pruning without complaint). The thing that matters
is the network's function-space geometry - what the model can distinguishably *do* -
and the honest excitation question becomes: has the data excited every functionally
distinct direction the task cares about? This rung is younger and less settled, and
it's where I think the interesting problems live.

## Whose idea is this anyway?

Here's a part I really enjoy. Take $M = \sum_k \phi_k \phi_k^\top$ and go on a tour of
the neighbouring fields:

- **Adaptive control** calls $\lambda_{\min}(M) \geq \alpha$ *persistent excitation*
  and pays out exponential parameter convergence.
- **Statistics** calls $M$ *Fisher information* and its inverse the Cramér-Rao floor:
  worst-excited direction, worst-estimable parameter, exactly.
- **Optimal experiment design** has spent decades shaping $M$ on purpose: maximise its
  determinant (D-optimal), minimise its trace-inverse (A-optimal), or raise its
  smallest eigenvalue (E-optimal) - and that last one simply *is* the excitation
  constant with a budget attached (Kiefer, 1959).
- **Active learning** scores an unlabeled point by how much it would grow $M$:
  information gain, coverage, leverage - it's the same ledger again.
- **System identification** stacks time-shifted data into a Hankel matrix and reads
  its rank as the number of dynamical modes the data actually touched (Ho-Kalman;
  Willems' fundamental lemma is a persistent-excitation statement). In the frequency
  domain, PE of order $n$ means the input carries $n$ distinct spectral lines - a
  model with $n$ parameters needs a song with $n$ notes. That's also the precise form
  of the Nyquist kinship from earlier: both are statements that your observations must
  span the relevant space, they just disagree about which space.
- **Deep learning** meets $M$ as the gradient covariance / empirical Fisher,
  load-bearing in natural gradient, K-FAC, and Laplace approximations, usually without
  control theory getting a mention (it's used to that by now).

Six different fields, and it's the same positive-semidefinite matrix every time - six
vocabularies for "which questions has the data asked". I do mean that literally: not a
metaphor connecting fields, a shared object whose theorems transfer. Which is lovely,
and also a warning label - if your exciting new ML idea lives entirely inside this
table, one of these six fields already owns it, and has since before you were born.
The traps section below is mostly about the day I read that label properly.

![One matrix, six vocabularies](figures/f_sixnames.png)

## Where MDL fits in

There's a seventh face, and it's the one that connects my two research obsessions. The
minimum-description-length principle says: prefer the hypothesis that compresses your
observations best. I've always loved it, but it has a gap you only notice when you try
to act on it - it ranks the explanations you already have, and says nothing at all
about what to go and look at next. Excitation turns out to be exactly the missing
half. In Bayesian terms, the expected information gain of an experiment equals the
expected reduction in posterior code length, so choosing the most informative
experiment is the same thing as choosing the observation that shortens your future
description fastest. Greedy "active MDL" turns out to be classical Bayesian experiment
design - Lindley worked this out in 1956, and every generation since has rediscovered
it with better logos. Twenty questions is the folk algorithm: a good question
compresses the remaining hypothesis space by about a bit, whatever the answer.

What's genuinely not classical is the structured version. MDL is most interesting over
two-part codes - model class plus residual, abstractions plus exceptions - and I know
of no satisfying account of excitation over *abstractions*: which observation most
reorganises the structural part of your code, rather than the parameters within it? My
parallel work on structural approaches to abstract reasoning lives in that
structured-code world, and the synthesis - an agent that acts to shorten its own
structured description of the world - is, as far as I can tell, still unclaimed. I'm
flagging it as a question rather than a result, for reasons the next section makes
fairly obvious.

## Over-enthusiastic traps

Everything so far is the enthusiast's tour, and enthusiasm is cheap. So over recent
months I ran these ideas through a deliberately adversarial process - pass/fail lines
fixed in advance, theory gates argued prosecution-and-defense style, and a scheduled
audit of my own past conclusions ([paper trail](../process/)). Three lessons came out,
and each one cost me an idea I was fond of.

**The circularity trap.** My first "excitation theory of in-context learning" scored a
prompt by $\lambda_{\min}(P^\top M_S P)$ and predicted ICL error from it via a
ridge-regression bound. The theory gate dispatched it in one sentence: that bound *is*
the ridge estimator's own posterior variance, restated. I had built an elaborate
machine for predicting a quantity from itself! The trap generalises, and it's worth
stating plainly: any excitation score derived from your model's own uncertainty
machinery will "predict" that machinery's behaviour perfectly and tell you nothing
about the world. So an excitation claim only counts for anything once the predicted
quantity is measured *independently* of the score. Every experiment in the next
section is built around that rule - and not for theoretical reasons.

**The classical wall.** Take excitation to few-shot example selection: "choose
demonstrations that excite the task subspace". Sounds novel. It's E-optimal experiment
design, from 1959. The audit's verdict on my static few-shot theory was blunt - static
excitation questions collapse into optimal design, where sixty years of literature
have mined the scalar criteria down to bedrock. What the classical field doesn't own
is the sequential structure: windows, ordering, forgetting, closed loops - the
*persistent* part of persistent excitation. That distinction ended up reshaping my
whole research programme, and I pass it on as the cheapest inheritance going: the
static questions are taken, but the sequential ones are still open.

**You can't out-excite Bayes from inside the model class.** I spent a month on a
hypothesis I still think was pretty: that you could design excitation toward
*model-class error* - the part of reality your hypothesis space can't represent, which
the posterior doesn't self-report. I built the closed-form study, ran the controls,
and watched it come apart. The effect exists, but it's exactly classical robust
experiment design,
split along the standard minimax-versus-Bayes axis
([full post-mortem](../experiments/spearhead_b/RESULTS.md)). The moral: within a
linear-Gaussian world with free design, there's no secret excitation channel that
beats the posterior at its own game. If misspecification-seeking excitation exists, it
lives in sequential, capacity-constrained, nonlinear regimes - and the burden of proof
is on whoever claims it. The whole study is still in the repo - to be frank, I'm still
a bit fond of it.

## What survived

Same discipline, pointed at the claims that lived. The routine each time: derive what
the object should do in one line, pre-register it, then look.

**A label-free failure diagnostic for in-context learning.** If ICL behaves like
regression in some feature space, then worst-case error on a task subspace $P$ is
controlled by $1/\lambda_{\min}(P^\top G P)$, with $G$ the prompt's Gram matrix - a
purely input-side score, no labels, no logits, that should bound where ICL cannot
succeed. Pre-registered: the score must survive shot-count deconfounding (adding
examples mechanically inflates any $\lambda_{\min}$), must beat its unprojected
version, and must still work with $P$ *estimated* from a handful of probe tasks rather
than handed over by an oracle. What happened: on a trained ICL transformer, prompts in
the worst excitation quartile carry roughly **3× the error** of the best quartile at
fixed shot count; the projected score wins every slice (the unprojected one is
provably degenerate when shots < dimension); and the probe-estimated subspace matches
the oracle to three decimal places
([results](../experiments/spearhead_a/A3_VERDICT.md)). Two disclosures. Per-query,
classical Bayes predictive variance - which needs the query and the prior - beats the
excitation score, exactly as pre-registered, so the instrument's honest niche is the
query-agnostic certificate. And an earlier version of this experiment reported 4.88× -
my own audit found that number confounded with shot count, and the deconfounded 3×
replaced it. Both the audit and the original 4.88× are still in the repo.

![Excitation predicts ICL failure at fixed shot count](figures/f_quartiles.png)

**Similar examples versus informative ones.** A near-duplicate demonstration adds zero
new rank to $G$; a demonstration along an unprobed task direction adds an entire
eigendirection. Retrieval-by-similarity is structurally blind to that difference.
Pre-registered: with one task direction left dark, an informative demo should cut
error sharply and a maximally similar demo should do roughly nothing, at identical
shot count. What happened: error 1.36 → 0.36 for the informative demo, 1.36 → 1.30 for
the duplicate - a **16× difference in improvement**, called in advance by
$\Delta\lambda_{\min}$ ([results](../experiments/spearhead_a/A3_ALIASING.md)). This is
probably the cleanest three-bar summary of the whole thesis: what your prompt needs
isn't more relevant examples so much as unasked questions. The same run also returned
a tidy negative - an attention-dilution "budget" mechanism I once considered my
central idea did nothing measurable, and is retired.

![Informativeness beats similarity 16x](figures/f_aliasing.png)

**The one that came back backwards.** One more finding from the diagnostic first: low
excitation predicts the transformer's *excess* error over the Bayes-optimal answer
computed on the same prompt - degenerate prompt geometry isn't just hard, it's
specifically where the learned inference algorithm parts company with Bayes. The SAE
pilot took that logic up a level. Treat each sparse-autoencoder feature as a parameter
and the data where it fires as its excitation, and you'd predict that features with
degenerate exciting data are the ones that wander across training seeds -
interpretability's known reliability problem, assigned a control-theoretic cause. I
pre-registered it with a frequency-matched control and trained five seed-varied SAEs
on GPT-2 activations. The prediction failed, and the sign flipped, robustly: at
matched frequency it's the *diffusely* firing features that are unstable (partial
$\rho = +0.26$, positive in ten out of ten frequency deciles, split-half replicated;
[verdict](../experiments/e1_sae/E1_VERDICT.md)). That one took me a while to digest,
and then it reorganised how I think about the whole thing. An SAE feature isn't a
regression parameter, whose estimate sharpens with richer excitation - it behaves like
a *decomposition component*, whose identifiability comes from separation. Tight,
distinctive firing regions get rediscovered by every seed; diffuse regions get tiled
differently every time. That's exactly the boundary between the two faces of
excitation theory - Gram conditioning versus uniqueness of decompositions - and here
it was, located empirically in a real model by a falsified prediction. The instrument
survives with its sign corrected: a label-free score, computed from activation
geometry alone, that flags which interpretability features not to trust. Alongside it
I'd also specified, before any result existed, a test of prompt-*order* sensitivity as
a failure of uniform observability
([proposal](../experiments/b4_ordering/PROPOSAL.md)) - which brings us to the
postscript.

![Within-frequency, diffuse firing predicts instability](figures/f_e1_deciles.png)

**The "persistent" part.** After this essay first went up, two more pre-registered
tests ran. One failed right on schedule: the windowed-observability account of
prompt-order sensitivity was ruled out by its own criterion - the model is massively
order-sensitive and no excitation functional predicts any of it
([verdict](../experiments/b4_ordering/B4_VERDICT.md)). So order sensitivity is not
excitation geometry, at least at this scale, and that idea is done. The other is the
cleanest positive in the repo. Stream demonstrations past a model with a fixed memory
budget and ask what to evict. Curating memory to keep the retained set *exciting*
halves error against recency at equal budget - classical concurrent learning
transplanted into the prompt world, and it works. But let the task drift and the
undiscounted version becomes
**worse than naive recency** - it hoards memories that were maximally informative
about a task that no longer exists. The time-discounted variant, the one that takes
the every-window clause seriously, degrades gracefully and wins
([verdict](../experiments/a5_streaming/A5_VERDICT.md)). That's the difference between
informativeness and persistent excitation, measured - an agent's memory should be
curated for whatever keeps the *present* identifiable, and the window condition from
earlier turns out to be the design principle rather than the fine print.

![Hoarding stale excitation is worse than forgetting](figures/f_a5_policies.png)

## What I want to do next

Everything above treats the model as the subject of excitation. The questions I most
want to work on turn the lens around.

**Capacity as an excitation budget.** Recent interpretability work at Anthropic finds
that language models route a privileged handful of concepts - order tens, not
thousands - through a small, flexible, verbally-reportable workspace, read out via a
*Jacobian* lens. If you've read the nonlinear rung above, you'll recognise the
Jacobian as the nonlinear-PE information carrier. A capacity-limited workspace is, in
excitation language, a bound on simultaneous excitation rank - tasks requiring more
concurrently-identified modes than the budget should fail in a characteristic,
predictable way. An excitation-order ceiling for reasoning, if it holds up. I'm
stating it as a question rather than a result, partly on principle and partly because
my own toy-scale probe of a budget effect came back negative - which is exactly the
sort of detail you're tempted to leave out, so it stays in. The twin question: are the
workspace's concepts *identifiable* - is each one pinned down by the contexts that
excite it? That's the same identifiability worry as the SAE-stability result, just
about concepts instead of features.

**When should a model make up its mind?** A model holding contradictory stances in
superposition is, in estimation terms, an unresolved posterior - and PE theory says
beliefs collapse along a direction exactly when that direction gets excited. For a
language model, the discriminating experiment doesn't even need external data:
chain-of-thought on a counterintuitive claim is *self-excitation* of the
discriminating direction. The design principle would be something like calibrated
decisiveness - commit when the discriminating direction can be excited, by evidence,
retrieval or derivation, and stay honestly uncertain when it can't. There's also a
corollary about what should *never* excite a stance direction: social pressure.
Sycophancy, restated as an excitation-hygiene violation, becomes something you can
actually measure.

**Oversight as dual control.** Dual control - act so as to perform *and* to remain
identifiable - is adaptive control's oldest dilemma. An aligned model under oversight
faces its mirror image: behave so that your overseer's data about you stays
persistently exciting, so that what you are stays identifiable from what you do. I
honestly don't know how far this framing carries - it's the one on this list I'd most
like an excuse to pursue properly.

---

Everything here - code, pre-registrations, audits, the dead ideas alongside the live
ones - is in [the repository](https://github.com/Haksaw22/PE-ML), arranged so you can
check me. The theorem itself is sixty years old and still mostly sits in the
control-theory literature, which I think is a shame - so if any of the open questions
above catches you, do get in touch, I'd genuinely like the excuse!
