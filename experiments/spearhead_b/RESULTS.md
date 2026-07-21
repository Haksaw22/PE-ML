# Spearhead B1 — Misspecification-complement excitation: does it separate, or collapse?

**Verdict (one line): (b) — the effect is REAL but SUBSUMED by classical robust /
minimax-vs-Bayes optimal experiment design. The "misspecification-complement" designer
does not uniquely excite model-error directions; the robust and minimax designers excite
them as much or more, and the only stable difference from robust design is *which*
complement direction (true vs worst-case) — the textbook minimax-vs-Bayes distinction the
hypothesis pre-registered as the likely collapse.**

---

## 1. Setup (linear-Gaussian, closed form, no training)

Feature space `R^D` in a basis whose first `r` coords span the **representable** subspace
and last `D-r` the **unrepresentable complement** (`P_rep = diag(1..1,0..0)`).

- Ground truth `y = phi^T theta* + noise`, `noise ~ N(0,sigma^2)`. We set `theta_r = 0`
  (isolates the misspecification question — the shrinkage bias from `theta_r` is a
  separate classical effect) and `||theta_c|| = rho`. So the **entire** true signal that
  the learner cannot represent lives in the complement.
- The learner regresses `y` on the representable features only (Bayesian ridge, prior
  `beta ~ N(0, tau^2 I_r)`). Its posterior precision `A = N·M_rr + kappa·I_r`
  (`kappa = sigma^2/tau^2`) is a function of the representable design block `M_rr` **alone**
  — hence the posterior variance is **blind** to the complement and to cross-moments. This
  is the crux: *posterior variance does not flag misspecification.*
- A **design** is its second-moment matrix `M = E[phi phi^T]`, `M ⪰ 0`, `trace(M)=1` (unit
  budget). Leaving `M` free over the spectrahedron is the cleanest test of what each
  *objective* wants to excite, uncontaminated by input geometry.
- **Excitation allocation** = `(rep, comp) = (trace M_rr, trace M_cc)`; the complement
  fraction is `trace(M_cc)`.

**Closed-form prediction risk** (test second moment `S`, blocks `S_rr, S_rc, S_cc`), with
`theta_r = 0`:

```
Risk(M, theta_c) = theta_c^T Q(M) theta_c + v(M)
  G   = A^{-1}(N·M_rc)                              (leakage / induced cross-correlation)
  Q(M)= G^T S_rr G - (G^T S_rc + S_rc^T G) + S_cc
  v(M)= trace(S_rr · A^{-1}(sigma^2 N M_rr) A^{-1}) (estimator variance)
```

The cross term `-(G^T S_rc + ·)` is the only channel by which exciting the complement can
help: it lets a *beneficial induced bias* cancel part of the complement error — **but only
when the test metric correlates representable and complement features (`S_rc != 0`).**

Two test regimes: `aligned` (`S_rc = 0`, rep ⟂ comp) and `correlated` (`S_rc != 0`,
generic — the natural case). Defaults `D=6, N=100, sigma^2=1, kappa=1, rho=1`.

The closed-form `Risk` was validated against a Monte-Carlo learn-then-predict simulation
(`exp_b0_mc_check.py`, see §6).

## 2. The four designers (each minimises its loss over `{M ⪰ 0, trace 1}`)

| # | designer | loss | what it acts on |
|---|----------|------|-----------------|
| 1 | **Bayes** (D-opt on rep params) | `-logdet(A)` | posterior precision (blind to complement) |
| 2 | **Minimax / E-optimal** (full Gram) | `-lambda_min(M)` | full-space coverage |
| 3 | **Robust / DR-OED** (worst case over `||theta_c||<=rho`) | `rho^2·lambda_max(Q) + v` | worst-case misspecification |
| 4 | **Misspec-complement** ("oracle", knows `theta_c` via probe) | `theta_c^T Q theta_c + v` | actual misspecification |

Designers 3 and 4 share `Q, v` and differ **only** in scalarisation — worst-case eigenvalue
vs value at the true direction. That is exactly the minimax-vs-Bayes(average-case) axis.

Optimisation: unconstrained factor parameterisation `M = LL^T/tr(LL^T)` + scipy L-BFGS-B
with exact torch gradients, best of many restarts (robust/oracle are non-convex and
multimodal — robust has a shallow spurious local min). **Validation** against known optima
passed exactly: Bayes → `M_rr = I_r/r` (diag 0.333), Minimax → `M = I/D` (all eigenvalues
0.167), Bayes comp = 0.0000, Minimax comp = (D-r)/D = 0.5000.

## 3. Allocation table (r = 3 / D = 6, mean ± std over 6 draws)

**Aligned test (`S_rc = 0`):**

| designer | comp | comp/dim | PR(comp) | cos²(θc) |
|----------|-----:|---------:|---------:|---------:|
| bayes    | 0.0000 ± 0.0000 | 0.0000 | – | – |
| minimax  | 0.5000 ± 0.0000 | 0.1667 | 3.00 (isotropic) | 0.333 (=1/(D-r), no targeting) |
| robust   | **0.0000 ± 0.0000** | 0.0000 | – | – |
| oracle   | **0.0000 ± 0.0000** | 0.0000 | – | – |

→ When rep ⟂ comp in the test metric, robust and oracle **collapse onto Bayes** (zero
complement). Only the coverage-based minimax touches the complement, and it does so
**blindly/uniformly** (isotropic, no alignment with the true error). The whole
"misspec-complement" effect is *contingent on test correlation*.

**Correlated test (`S_rc != 0`):**

| designer | comp | comp/dim | PR(comp) | cos²(θc) |
|----------|-----:|---------:|---------:|---------:|
| bayes    | 0.0000 ± 0.0000 | 0.0000 | – | – |
| minimax  | 0.5000 ± 0.0000 | 0.1667 | 3.00 (isotropic) | 0.333 |
| robust   | 0.1039 ± 0.0616 | 0.0346 | 1.00 (rank-1) | 0.332 (**not** the true dir) |
| oracle   | 0.1028 ± 0.0618 | 0.0343 | 1.00 (rank-1) | **1.000** (targets θc exactly) |

→ Robust and oracle both excite the complement, both **rank-1** (a single direction).
Averaged magnitudes nearly coincide (0.104 vs 0.103; these are 20-restart values — the
better-converged 12-draw paired numbers in §6 give 0.084 vs 0.080, same story). They differ
in **direction**: oracle
puts all complement energy on the true error `theta_c` (cos²=1.0); robust guards a
worst-case direction fixed by the test geometry (cos²(θc)=0.33). Direction cosine between
the two: `cos²(dir_robust, dir_oracle) = 0.332 ± 0.279`.

## 4. Capacity scaling — complement fraction `trace(M_cc)` vs r

**Aligned:**
```
  r    bayes  minimax   robust   oracle
  1    0.000    0.833    1.000    1.000     <- degenerate edge (theta_r=0, 1 rep dim)
  2    0.000    0.667    0.000    0.000
  3    0.000    0.500    0.000    0.000
  4    0.000    0.333    0.000    0.002
  5    0.000    0.167    0.000    0.000
  6    0.000    0.000    0.000    0.000
```
Robust = oracle = 0 for all `r>=2`. Effect absent without test correlation.

**Correlated:**
```
  r    bayes  minimax   robust   oracle   rob-orc
  1    0.000    0.833    1.000    1.000     0.000   <- degenerate edge
  2    0.000    0.667    0.017    0.018    -0.002
  3    0.000    0.500    0.118    0.051     0.067   (robust > oracle)
  4    0.000    0.333    0.033    0.080    -0.046   (oracle > robust)
  5    0.000    0.167    0.024    0.022     0.002
  6    0.000    0.000    0.000    0.000     0.000
```
The robust-vs-oracle magnitude ordering **flips** with `r` (r=3 robust > oracle; r=4 oracle
> robust) → **no consistent scalar-magnitude separation** between designers 3 and 4. Both
vanish as `r → D` (complement disappears). Bayes is 0 throughout; minimax is `(D-r)/D`.

## 5. Scale sweep (correlated, r = 3) — complement fraction

```
  rho      N    bayes  minimax   robust   oracle
  0.5     30    0.000    0.500    0.676    0.359
  0.5    100    0.000    0.500    0.027    0.065
  0.5    500    0.000    0.500    0.014    0.021
  1.0     30    0.000    0.500    0.678    0.366
  1.0    100    0.000    0.500    0.033    0.053
  1.0    500    0.000    0.500    0.014    0.022
  2.0     30    0.000    0.500    0.679    0.044
  2.0    100    0.000    0.500    0.032    0.058
  2.0    500    0.000    0.500    0.014    0.022
  4.0     30    0.000    0.500    0.679    0.045
  4.0    100    0.000    0.500    0.089    0.099
  4.0    500    0.000    0.500    0.014    0.022
```
At small `N=30` the *worst-case* robust designer over-hedges dramatically (comp ≈ 0.68) —
several times the oracle — again showing designer 4 is **not** the one that most excites
model-error directions. At larger `N` both shrink toward zero and stay comparable (oracle
slightly above robust; e.g. N=500 → 0.014 vs 0.022). Bayes and minimax are `N`-independent
(0 and 0.5). (Small-`N` values are the least optimizer-stable; the trend is the point.)

## 6. Paired per-draw test (`exp_b2_paired.py`) + Monte-Carlo validation (`exp_b0_mc_check.py`)

**Monte-Carlo validation of the closed-form risk** (Gaussian design ~ `N(0,M)`, ridge fit
on the representable block, evaluate on the test distribution; `N=100`, 4000 reps):

```
mode         design    analytic   MC mean   rel.err
aligned      oracle      0.7807    0.7843    0.0046
aligned      robust      0.7807    0.7843    0.0046
aligned      I/D         0.9506    0.9889    0.0387
correlated   oracle      1.1915    1.2124    0.0172
correlated   robust      1.1675    1.1906    0.0194
correlated   I/D         1.3995    1.4521    0.0362
```
Analytic and simulated risks agree to 0.5–2% on the optimized designs (the 3–4% on the
worse `I/D` design is the expected finite-sample Gram fluctuation, MC slightly higher). **The
objective driving designers 3 and 4 is correct.**

**Paired robust-vs-oracle, 12 draws, correlated, r=3, N=100, rho=1** (robust 60 restarts,
oracle 40 restarts):

```
  quantity                                          mean +/- std
  comp robust                                       0.0841 +/- 0.0698
  comp oracle                                       0.0802 +/- 0.0270
  |comp_rob - comp_orc|  (paired magnitude gap)     0.0527 +/- 0.0505   (~56% of the value)
  cos2(dir_robust, dir_oracle)   (1 = same dir)     0.3519 +/- 0.2987
  cos2(dir_oracle, theta_c)      (1 = targets truth)0.9999 +/- 0.0004   <-- DEFINITIVE
  cos2(dir_robust, theta_c)      (1 = targets truth)0.3546 +/- 0.3025
```

Read-out:
- **Direction (the clean, convergence-insensitive separator):** the oracle designer puts its
  complement excitation on the true error `theta_c` in *every* draw (`cos2 = 0.9999`); the
  robust designer does not (`0.35 ± 0.30`, guarding a worst-case direction fixed by the test
  geometry). This is the textbook average-case-vs-worst-case difference.
- **Magnitude:** comparable order (means 0.084 vs 0.080), with moderate per-draw gaps (~56%)
  and *no consistent ordering* (draw 1: robust 0.244 ≫ oracle 0.091; draw 5: oracle 0.106 >
  robust 0.042). Not a clean scalar separation.

**Optimizer reliability:** the oracle objective is also mildly multimodal. A single unlucky
restart-20 start under-converged for one instance (loss 1.1915 vs the true 1.1624), but
best-of-several-starts recovers the true optimum, stable across 20/60/120 restarts
(loss 1.1624, comp 0.0595). Crucially the **direction** result (`cos2(oracle, theta_c)≈1`) is
rock-solid regardless. The correlated magnitudes above (B2, ≥40 restarts) supersede the
r=3 row of §3 (B1, 20 restarts; qualitatively identical, oracle 0.10 vs 0.08).

## 7. Honest verdict

The **KEY TEST** was: does designer 4's complement allocation SEPARATE from designers 1, 2,
AND 3 — or coincide with minimax (2) or robust (3)?

- **vs Bayes (1):** separates trivially — Bayes never excites the complement (structural,
  comp = 0). But so do minimax, robust, oracle in the correlated case, so this is no
  evidence of anything special about designer 4.
- **vs Minimax (2):** separates. Minimax excites the complement **most** (0.5 = (D-r)/D),
  isotropically and blindly (no targeting of the error). Designer 4 excites it far less and
  rank-1.
- **vs Robust (3):** no clean scalar-magnitude separation. Complement magnitudes are the
  same order (paired means 0.084 vs 0.080), with only moderate per-draw gaps and **no
  consistent ordering** (robust bigger at r=3 and small N; oracle bigger at r=4; averages
  nearly equal). The one **stable** difference is **direction**: oracle targets the true
  error `theta_c` in every draw (`cos2 = 0.9999`); robust guards a worst-case direction
  (`cos2(θc) = 0.35`). But that is *exactly* how average-case (Bayes/oracle) OED always
  differs from worst-case (minimax) OED over the **same** misspecification set — i.e.
  Wiens-type robust design for approximately-linear models vs local/oracle
  misspecification-optimal design. Designers 3 and 4 are the two faces (worst-case,
  average-case) of one robust-OED family; their difference is the pre-registered
  minimax-vs-Bayes axis, not a new one.

Two further nails:

1. The whole effect is **contingent on test-metric correlation** `S_rc != 0`; in the aligned
   case robust = oracle = Bayes = 0. The mechanism is the well-known design-dependent
   best-linear-approximation / *beneficial bias* under misspecification, not a new
   excitation principle.
2. The hypothesis's core intuition — that the misspec-complement designer **uniquely**
   allocates excitation to model-error directions the others miss — is **refuted**: the
   robust and minimax designers allocate as much or (small `N`, and the whole minimax
   column) far MORE to the complement than designer 4.

**Conclusion: (b).** The effect exists but collapses to a known classical distinction. It is
subsumed by robust / distributionally-robust OED: designer 4 is the average-case (oracle /
local-Bayes) member of the same family whose worst-case member is designer 3, separated only
along the standard minimax-vs-Bayes axis, and only when the test geometry correlates the
representable and unrepresentable subspaces. No genuinely novel "misspecification-complement
excitation" phenomenon survives the controls.

## Files
- `exp_b1_misspec.py` — main experiment (allocation tables, capacity + scale sweeps, validation).
- `exp_b2_paired.py` — paired per-draw robust-vs-oracle magnitude/direction test.
- `exp_b0_mc_check.py` — Monte-Carlo validation of the closed-form risk.
- `b1_stdout.log`, `b2_stdout.log`, `b0_stdout.log` — raw run outputs.
