# Persistency of Excitation: The Geometry of Asking Good Questions

*Draft v1. Math is KaTeX-ready for Notion. Placeholders for
personal detail are marked 【like this】. Every quantitative claim links to the repo.*

---

## 1. A control engineer's confession

I came to machine learning from magnetic levitation. 【One sentence on Warwick Hyperloop /
Magway in your own words — the room, the rig, the sound of it.】 In that world you meet a
theorem early, usually the hard way: an adaptive controller can track its target
*perfectly* while learning *nothing true* about the system it controls. The tracking
error goes to zero. The parameter estimates go wherever they like. Everything looks like
success until the setpoint changes, and then the controller reveals that it never
understood the plant at all — it had memorized a performance, not identified a system.

Control theory doesn't treat this as a mystery. It has a precise diagnosis, with a
sixty-year-old name: the input was not **persistently exciting**. The data stopped asking
new questions, so learning stopped — silently, invisibly, behind a perfectly flat error
curve.

I now believe this is one of the most quietly transferable ideas that classical
engineering owns, because modern machine learning keeps rediscovering its symptoms
without inheriting its diagnosis. A language model that answers confidently from a prompt
that never pinned the task down; an interpretability feature that dissolves when you
retrain with a new seed; an RL agent whose exploration revisits what it already knows —
each of these is, I'll argue, the tracking-without-identification failure wearing new
clothes. This essay is my attempt to lay out the idea from the ground up, show that it is
secretly the *same object* as half a dozen things you already know, be honest about where
the transplant into ML breaks down — I have the scars, and I'll show them — and end with
the questions I most want to answer next.

## 2. The whole idea, from one equation

Suppose something you care about is governed by unknown parameters
$\theta = (\theta_1, \theta_2)$, and each observation gives you one linear glimpse:

$$y = \phi^\top \theta$$

where $\phi$ is whatever you measured alongside the outcome. One observation is one
equation in two unknowns: a *line* of candidate explanations, not an answer. A second
observation nails it — **unless** your second $\phi$ points the same direction as the
first, in which case you've asked the same question twice and learned nothing you didn't
already know. Ten thousand observations along one direction still leave an entire axis of
ignorance. Diversity of the $\phi$'s — not their number — is what learning runs on.

The bookkeeping object for this is the **information matrix**
$\sum_k \phi_k \phi_k^\top$: the running total of which directions have been probed and
how hard. Its smallest eigenvalue is your *worst-illuminated direction* — the question
you've asked least. And "persistency" is the sharpest part of the definition: it demands
that **every window** of recent data keep the matrix well-conditioned,

$$\sum_{k=t}^{t+N} \phi_k \phi_k^\top \;\succeq\; \alpha I \quad \text{for all } t,$$

because a learner that adapts (as every interesting learner does) forgets; richness you
supplied last year doesn't cover for the monotony you're feeding it now. Persistent
excitation is the condition that your data *keeps asking linearly independent questions*.
The flashlight picture: $\theta$ is an object in a dark room, each measurement a beam
from one angle. From one angle, many shapes cast the same shadow. Learning is not
accumulating light — it's accumulating *angles*.

## 3. The ladder of generality

The equation above looks aggressively linear. The idea is not; it climbs a ladder.

**Rung one: linear.** The regressor $\phi$ *is* the information carrier, and everything
reduces to the rank story above.

**Rung two: nonlinear.** Let $y = f(\theta, u)$. Each observation now carves a curved
surface through parameter space, and the local information carrier is the sensitivity
$J = \partial f / \partial \theta$ — which, in the linear case, is exactly $\phi$. The
information matrix becomes $\int J^\top J\,dt$, and statisticians will recognize their
oldest friend: this is **Fisher information**. Persistency of excitation for nonlinear
systems is the demand that your trajectory keep the Fisher information uniformly positive
definite — that no parameter direction fall out of the conversation between model and
data. (Where even local sensitivity fails, the right notion becomes observability and
identifiability proper: can two different $\theta$'s produce identical outputs under the
inputs you actually applied?)

**Rung three: function space.** For neural networks, parameter space is the wrong
ballroom. Permute two hidden neurons: same function. Millions of parameter directions are
pure gauge — exciting them is meaningless, and modern networks tolerate savage pruning
precisely because so much of parameter space carries no function. The object that
matters is the network's *function-space* geometry (the NTK/GP view of what the model can
distinguishably do), and the honest excitation question becomes: **has the data excited
every functionally distinct direction the task cares about?** This rung is younger and
less settled — which is exactly what makes it interesting.

## 4. One matrix, many names

Here is the part I find genuinely beautiful. Take the object
$M = \sum_k \phi_k \phi_k^\top$ and walk it through the neighboring fields:

- **Adaptive control** calls $\lambda_{\min}(M) \geq \alpha$ *persistent excitation*, and
  buys exponential parameter convergence with it.
- **Statistics** calls $M$ the *Fisher information* and its inverse the Cramér–Rao floor:
  the worst-excited direction is the worst-estimable parameter, exactly.
- **Optimal experiment design** (Kiefer, 1959) spent decades choosing observations to
  shape $M$: maximize its determinant (D-optimal), its trace-inverse (A-optimal), or its
  smallest eigenvalue (E-optimal) — that last one *is* the excitation constant, optimized.
- **Active learning** scores an unlabeled point by how much it would grow $M$ —
  information gain, coverage, leverage.
- **System identification** stacks time-shifted data into a Hankel matrix and reads its
  rank as the number of dynamical modes the data touched (Ho–Kalman; Willems'
  "fundamental lemma" is a persistent-excitation statement). In the frequency domain, PE
  of order $n$ means the signal carries $n$ independent spectral lines — a model with
  $n$ parameters needs a song with $n$ notes.
- **Deep learning** meets $M$ as the gradient covariance / empirical Fisher, the object
  behind natural gradient, K-FAC, and Laplace approximations.

Six fields, one PSD matrix, six dialects for "which questions has the data asked?" I find
this the right kind of beauty — not a metaphor connecting fields, but a *literal shared
object* whose theorems port across. That's also a warning, and Section 6 is about heeding
it: if your "new" ML excitation idea lives entirely inside this table, one of these six
fields already owns it.

## 5. Compression's active twin

There's a seventh face, and it's the one that bridges my two research obsessions. The
minimum-description-length principle says: prefer the hypothesis that *compresses* your
observations best. It is a philosophy of what to believe, given data — and it is
completely **passive**: MDL ranks explanations; it never tells you what to look at next.

Excitation is the missing active half. In Bayesian terms, the expected information gain
of an experiment is the expected reduction in the posterior's code length — so *choosing
the maximally informative experiment is choosing the observation that shortens your
future description fastest*. Greedy "active MDL" turns out to be classical Bayesian
experiment design (this is Lindley's 1956 insight, rediscovered by every generation).
Twenty questions is the folk version: a good question is one whose answer compresses the
remaining hypothesis space by a bit, whatever the answer.

What is *not* classical is the structured version. MDL is at its most interesting over
two-part codes — model class plus residual, abstractions plus exceptions — and nobody has
a satisfying account of *excitation over abstractions*: which observation most
reorganizes the structural part of the code, not just the parameters within it? My
parallel work on program-induction approaches to abstract reasoning (ARC) lives exactly
in that structured-code world, and the open synthesis — an agent that acts to shorten its
own structured description of the world — is, I think, one of the deepest unclaimed
questions in this whole area. I flag it here as a question; I have no results, only the
conviction that the two faces belong together.

## 6. Where the analogy breaks — three scars

Everything so far is the enthusiast's tour. Now the part the enthusiast's tour never
includes. Over the past months I ran these ideas through an adversarial research process —
pre-registered kill criteria, prosecution-and-defense theory gates, and a full audit of
my own past conclusions ([the paper trail is public](../process/)). Three lessons
survived that I have not seen stated plainly anywhere, and each one killed something I
liked.

**Scar one: the circularity trap.** My first "excitation theory of in-context learning"
scored a prompt by $\lambda_{\min}(P^\top M_S P)$ and predicted ICL error from it via a
ridge-regression bound. The theory gate killed it with one sentence: *that bound is the
ridge estimator's own posterior variance, restated*. The "prediction" was an algebraic
identity of the surrogate — true by construction, empty as science. The trap generalizes:
any excitation score you derive from your model's own uncertainty machinery will
"predict" that machinery's behavior perfectly and tell you nothing about the world. An
excitation claim earns its keep only when the predicted quantity is *measured
independently* of the score. Every experiment in Section 7 is built around that rule.

**Scar two: the classical wall.** Take excitation to few-shot example selection: "choose
demonstrations that excite the task subspace." Sounds novel; *is* E-optimal experiment
design, 1959, transplanted. The audit verdict on my static few-shot theory was blunt:
static excitation questions collapse into optimal design, where sixty years of literature
have already mined the scalar criteria to bedrock. What classical design does **not**
own is the *sequential* structure — windows, ordering, forgetting, closed loops: the
distinctly *persistent* part of persistent excitation. That's where the live questions
are, and it reshaped my whole research program toward them.

**Scar three: you can't out-excite Bayes from inside the model class.** I bet months on
a seductive hypothesis — that one could design excitation toward *model-class error*,
the part of reality your hypothesis space can't represent, which the posterior
"doesn't self-report." Built the closed-form study, ran the controls. Result: the effect
exists but is exactly classical robust experiment design in disguise, split along the
standard minimax-vs-Bayes axis ([full negative writeup](../experiments/spearhead_b/RESULTS.md)).
The general lesson: within a linear-Gaussian world with free design, *there is no secret
excitation channel that beats the posterior at its own game*. If misspecification-seeking
excitation is possible at all, it lives in sequential, capacity-constrained, nonlinear
regimes — and the burden of proof is on the claimant. I keep the corpse on display
because the field publishes too few of them.

## 7. What survives: three experiments

The same discipline, pointed at claims that *did* survive. Each follows the same
micro-arc: derive the object in one line, pre-register what it must do, show the result.

**7a. A label-free failure diagnostic for in-context learning.**
*Derive:* if ICL behaves like regression in some feature space, worst-case error on a
task subspace $P$ is controlled by $1/\lambda_{\min}(P^\top G P)$, with $G$ the prompt's
Gram matrix — so a purely *input-side* score (no labels, no logits) should bound where
ICL can't succeed. *Predict:* the score must survive shot-count deconfounding (adding
examples mechanically raises any $\lambda_{\min}$), must beat its unprojected version,
and must work with $P$ *estimated* from a handful of probe tasks, not oracle-given.
*Show:* on a trained ICL transformer, prompts in the worst excitation quartile carry
**~3× the error** of the best quartile *at fixed shot count*; the projected score wins
in every slice (the unprojected one is provably degenerate when shots < dimension); the
probe-estimated subspace matches the oracle exactly
([results](../experiments/spearhead_a/A3_VERDICT.md)). Honesty box: per-query, classical
Bayesian predictive variance — which needs the query and the prior — beats the excitation
score, as pre-registered; the instrument's honest niche is the query-*agnostic*
certificate. And an earlier version of this experiment reported 4.88×; my own audit found
that number confounded and the deconfounded 3× replaced it. The audit is in the repo too.

**7b. Informativeness is not similarity — a 16× demonstration.**
*Derive:* a near-duplicate demonstration adds zero new rank to $G$; a demonstration along
an unprobed task direction adds a full new eigendirection. Retrieval-by-similarity cannot
see this difference; excitation is *exactly* this difference. *Predict:* with one task
direction left dark, adding an informative demo should slash error; adding a
demo that is maximally similar to existing ones should do nothing — at identical shot
count. *Show:* error 1.36 → 0.36 for the informative demo, 1.36 → 1.30 for the duplicate:
a **16× difference in improvement**, called in advance by $\Delta\lambda_{\min}$
([results](../experiments/spearhead_a/A3_ALIASING.md)). This is the cleanest three-bar
statement I know of the whole thesis: *what your prompt needs is not more relevant
examples but unasked questions*. The same run also produced a clean negative: an
attention-dilution "budget" mechanism I once believed in did nothing measurable at this
scale, and is retired with data.

**7c. Two pre-registered bets, running and waiting.**
The most surprising A3 finding deserves its own sentence: low excitation predicts the
transformer's *excess* error over the Bayes-optimal answer computed on the same prompt —
degenerate prompt geometry isn't just hard, it's specifically **where the learned
inference algorithm parts company with Bayes**. That observation is what the SAE pilot
scales up: treat each sparse-autoencoder feature as a parameter and the data where it
fires as that parameter's excitation; then features whose exciting data is degenerate
should be exactly the ones that *wander across training seeds* — interpretability's
known reliability problem, given a control-theoretic cause and a cheap label-free
diagnostic. Pre-registered with a frequency-matched null (rare features are trivially
unstable; that confound is controlled by design), running on an A100 as I write
([design](../experiments/e1_sae/DESIGN.md)). Alongside it, a proposal I deliberately
haven't built: prompt-*order* sensitivity as a failure of **uniform** observability —
every window of the sequence must stay informative, not just the whole — with the test
specified before any result exists ([proposal](../experiments/b4_ordering/PROPOSAL.md)).
Pre-registration cuts both ways; that's the point.

## 8. The frontier: excitation inside the model

Everything above treats the model as the *subject* of excitation. The questions I most
want to work on next turn the lens around.

**Capacity as an excitation budget.** Anthropic's recent global-workspace work finds that
language models route a privileged ~10–25 concepts at a time through a small, flexible,
verbally-reportable subspace — read out via a *Jacobian* lens, which a reader of Section 3
will recognize as the sensitivity object $\partial(\text{output})/\partial(\text{state})$,
the exact nonlinear-PE information carrier. A capacity-limited workspace is, in
excitation language, a bound on *simultaneous excitation rank*: tasks needing more
concurrently-identified modes than the budget should fail in a characteristic,
predictable way — an excitation-order ceiling for reasoning. I state this as a testable
question, not a result (my own toy-scale probe of an attention-budget effect came back
negative, which is exactly why the real-scale question is open). Its twin: are the
concepts in the workspace *identifiable* — is each one pinned down by the contexts that
excite it? That is the SAE-stability question asked one level higher.

**When should a model make up its mind?** A model holding contradictory stances in
superposition is, in estimation terms, an unresolved posterior — and PE theory says
beliefs collapse along a direction *exactly when that direction gets excited*. The
discriminating "experiment" needn't be external data: for an LLM, chain-of-thought on a
counterintuitive claim is *self-excitation* of the discriminating direction. The design
principle this suggests I'd call **calibrated decisiveness**: commit when the
discriminating direction can be excited (by evidence, retrieval, or derivation); stay
honestly uncertain when it can't — and never let the stance direction be excited by
social pressure, which is sycophancy stated as an excitation-hygiene violation. The
classical spine (sequential tests, dual control) is old and strong; the internal-belief
version is, as far as I can tell, open.

**Oversight as dual control.** Dual control — act to perform *and* to stay identifiable —
is adaptive control's oldest dilemma. An aligned model under oversight faces its mirror
image: behave so that your overseer's data about you remains persistently exciting, so
that what you are stays identifiable from what you do. I don't know how far that framing
carries. I'd like to find out.

## 9. Coda

The through-line of this essay is a single matrix and a single discipline. The matrix
counts the questions your data has asked. The discipline is refusing to believe your own
excitement until the object survives an attempt to kill it — because the graveyard
sections above did more to sharpen the surviving claims than any success did. Everything
here — code, pre-registrations, audits, corpses, and results — is in
[the repository](https://github.com/【repo-url】), arranged so you can check me.

*【Closing line in your own voice.】*
