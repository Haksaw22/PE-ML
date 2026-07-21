# A3 verdict — what the repaired experiment actually established

(Interpretation of `A3_RESULTS.md` + `A3_ALIASING.md`; all claims below survive the
2026-07-19 audit's repair list. Raw arrays: `a3_trials.npz`. Pre-registrations were
written into the experiment docstrings before results were seen.)

## Confirmed

1. **The projected-excitation diagnostic is real and survives deconfounding.** The honest
   headline is **~3.0x** MSE between low- and high-excitation quartiles *at fixed shot
   count* (2.42x / 3.17x / 3.31x at k = 8/12/16). The old pooled "4.88x" (here 6.52x on
   the rerun) was, as the audit predicted, roughly double-counted by the shot-count
   confound. 3x-at-fixed-k is the number to cite.
2. **Projection is the value-add.** At fixed k the *unprojected* Gram is useless — for
   k < D it is exactly singular, so its lam_min is rank-noise (this is why the paired
   test at k=8 returns NaN; the +0.090 "correlation" there is noise on numerically-zero
   eigenvalues). The task-projected version stays predictive in every slice.
3. **The oracle-P objection is dead.** A subspace estimated from 8 labeled probe tasks
   performs identically to the oracle subspace (pooled rho -0.282 vs -0.284; paired
   differences ns, slightly favoring estimated). The instrument does not require knowing
   the answer in advance.
4. **The aliasing fingerprint — the note's signature prediction — passed decisively.**
   Adding one demo along the dark task direction cuts MSE ~4x (1.36 -> 0.36); adding a
   near-duplicate (equally "relevant" by any similarity score) does nothing (1.36 -> 1.30).
   A 16x improvement ratio, predicted by delta-lam_min (+2.29 vs +0.00). *Informativeness
   and similarity dissociate, and excitation tracks the one that matters.*
5. **New finding — excitation predicts ICL-SPECIFIC failure.** Against the Bayes-ridge
   floor computed on the same contexts, low excitation predicts the transformer's *excess*
   error over Bayes (rho -0.12..-0.17 at fixed k): under-excited prompts are where the
   learned in-context algorithm deviates from the Bayesian ideal, not merely where the
   problem is hard for everyone. This is the construct-validity control the audit
   demanded, and it is the most interp-relevant fact on the page: *degenerate context
   geometry is where learned inference breaks.*

## Demoted (as pre-registered)

6. **Per-query, Bayes predictive variance dominates** (rho +0.47 vs -0.28; significantly
   better in every paired test). Per the pre-registration in `exp_a3_eval.py`: for
   per-query error prediction, the excitation lam_min is a *derived summary* of Bayes
   uncertainty, and the classical object wins. The honest scope of the instrument is the
   **query-agnostic certificate**: against the matched query-agnostic baseline
   (tr posterior variance on the task subspace), lam_min ties (E-optimal vs A-optimal
   scalarizations of the same object — as the program's own pivot predicted).
7. **The attention-weighted Gram adds nothing measurable** (ns vs the plain Gram in every
   slice), and the **budget probe is negative at this scale**: piling on duplicate demos
   drives attention-weighted alpha down 34% but MSE stays flat. The "fixed-point /
   attention-budget" story — this program's original central claim — remains retired,
   now with direct experimental evidence rather than only the theory-gate argument.

## One-line summary

A label-free, probe-estimable, deconfounded excitation certificate that predicts where
in-context learning breaks from Bayes (3x at fixed k; 16x aliasing separation) — and a
clean negative on the attention-budget mechanism, published alongside.
