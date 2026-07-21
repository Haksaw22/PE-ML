"""
Spearhead B / Stage B1 -- "misspecification-complement excitation".

HYPOTHESIS UNDER TEST
---------------------
When a learner's model class has FINITE CAPACITY / is SYSTEMATICALLY MISSPECIFIED
in some directions, an excitation/experiment-design objective allocates PERSISTENT
excitation to directions of MODEL-CLASS ERROR that the learner's OWN posterior
variance does NOT flag -- and this allocation SEPARATES from what standard
designers do. The hypothesis warns this most likely COLLAPSES to a classical
distinction (minimax-vs-Bayes design, or robust / DR-OED). We find out which.

TOY SETUP (linear-Gaussian, closed form, no training)
-----------------------------------------------------
Work directly in feature space R^D with an orthonormal basis whose first r
coordinates span the REPRESENTABLE subspace and last (D-r) the UNREPRESENTABLE
complement (P_rep = diag(1..1,0..0)).

  * Ground truth   y = phi^T theta*  + noise,  phi in R^D,  noise ~ N(0,sigma^2).
    theta* = (theta_r, theta_c).  theta_c lives ONLY in the complement -> it is the
    model-class error.  We set theta_r = 0 (isolates the misspecification question;
    the shrinkage bias from theta_r is an orthogonal, classical effect) and
    ||theta_c|| = rho.
  * Learner regresses y on the representable features phi_r ONLY (Bayesian ridge,
    prior beta ~ N(0, tau^2 I_r)). Its posterior precision is
        A = N*M_rr + kappa*I_r,   kappa = sigma^2/tau^2,
    a function of M_rr (the representable block of the design) ALONE. Hence the
    posterior variance A^{-1} is BLIND to the complement and to cross-moments --
    the crux: posterior variance does not flag misspecification.

  * A DESIGN is its second-moment matrix M = E[phi phi^T], a D x D PSD matrix with
    trace(M)=1 (unit excitation budget). This is the standard "approximate design"
    relaxation; leaving M free (any PSD, trace 1) is the cleanest possible test of
    what each OBJECTIVE *wants* to excite, uncontaminated by input geometry.
  * EXCITATION ALLOCATION = (trace(M_rr), trace(M_cc)); complement fraction =
    trace(M_cc) since trace(M)=1.

CLOSED-FORM PREDICTION RISK  (test second moment Sigma_test = S, blocks S_rr,S_rc,S_cc)
-------------------------------------------------------------------------------------
Under a deterministic (continuous) design with per-sample moment M and N samples,
the ridge estimate satisfies  beta_hat = theta_r + G theta_c + noise  with
    A   = N*M_rr + kappa*I_r,
    G   = A^{-1} (N*M_rc)            (leakage / induced cross-correlation, r x (D-r))
    Sig_b = A^{-1} (sigma^2 N M_rr) A^{-1}   (estimator covariance).
With theta_r = 0 the expected prediction risk over x ~ test is PURELY quadratic in
theta_c plus a design-only variance term:
    Risk(M, theta_c) = theta_c^T Q(M) theta_c + v(M),
    Q(M) = G^T S_rr G - (G^T S_rc + S_rc^T G) + S_cc      ((D-r)x(D-r), symmetric)
    v(M) = trace(S_rr Sig_b).
Q collects (i) leakage variance G^T S_rr G, (ii) the cross term -(G^T S_rc + .)
that lets a *beneficial* induced bias CANCEL part of the complement error when the
test metric correlates rep & comp (S_rc != 0), and (iii) the irreducible S_cc.

THE FOUR DESIGNERS  (each minimises its loss over {M PSD, trace 1})
-------------------------------------------------------------------
 1. BAYES (posterior-precision / D-optimal on representable params):
        loss = -logdet(A).   Depends on M_rr only -> excites representable only.
 2. MINIMAX / E-optimal on the FULL Gram:
        loss = -lambda_min(M).   Optimum M = I/D -> uniform, comp frac (D-r)/D.
 3. ROBUST / DR-OED  (worst case over ||theta_c|| <= rho):
        loss = rho^2 * lambda_max(Q(M)) + v(M).       [minimax over misspec direction]
 4. MISSPEC-COMPLEMENT ("oracle", knows theta_c via a probe of f*):
        loss = theta_c^T Q(M) theta_c + v(M).         [best design AT the true theta_c]

Designers 3 and 4 share Q,v and differ ONLY in scalarisation (worst-case eigenvalue
vs value at the true direction) -- i.e. exactly the minimax-vs-Bayes(oracle) axis.

WHAT WE MEASURE
---------------
 * complement fraction trace(M_cc) for each designer, over a sweep of capacity r
   and of noise/prior/misspec scales, for an ALIGNED test (S_rc = 0) and a
   CORRELATED test (S_rc != 0);
 * whether designer 4 SEPARATES from 1, 2 AND 3 (scalar fraction AND within-
   complement direction);
 * whether the complement excitation VANISHES as r -> D.

The verdict (a: no effect / b: subsumed by minimax or robust / c: separates from
all three) is decided by the printed numbers, not by assertion.
"""
from __future__ import annotations
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Problem construction
# --------------------------------------------------------------------------- #
def make_test_cov(D: int, r: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Test second-moment matrix S (PSD, trace = D).

    mode='aligned'    : block-diagonal in the rep/comp split (S_rc = 0). The special
                        case where rep and comp are orthogonal in the test metric.
    mode='correlated' : generic PSD with nonzero rep/comp cross-block (S_rc != 0),
                        the natural case (the model split need not align with the
                        test geometry).
    """
    if mode == "aligned":
        d = np.exp(rng.uniform(-0.5, 0.5, size=D))
        S = np.diag(d)
    elif mode == "correlated":
        # random rotation mixes rep & comp coordinates -> generic cross-block.
        Z = rng.standard_normal((D, D))
        Qo, _ = np.linalg.qr(Z)
        d = np.exp(rng.uniform(-0.5, 0.5, size=D))
        S = Qo @ np.diag(d) @ Qo.T
    else:
        raise ValueError(mode)
    S = 0.5 * (S + S.T)
    S *= D / np.trace(S)  # normalise trace to D
    return S


def make_theta_c(D: int, r: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """A complement-only true parameter of norm rho (the model-class error)."""
    if r >= D:
        return np.zeros(0)
    v = rng.standard_normal(D - r)
    v /= np.linalg.norm(v)
    return rho * v


# --------------------------------------------------------------------------- #
# Design parameterisation:  M is PSD with trace 1 (projected-gradient variable)
# --------------------------------------------------------------------------- #
def project_to_spectahedron(M: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {M symmetric PSD, trace(M)=1} (the spectrahedron).
    Symmetrise, eigen-decompose, project eigenvalues onto the probability simplex."""
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w_proj = project_simplex(w)
    return (V * w_proj) @ V.T


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto {x >= 0, sum x = 1} (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - css / ind > 0
    rho_idx = ind[cond][-1]
    theta = css[cond][-1] / rho_idx
    return np.maximum(v - theta, 0.0)


def _project_simplex_torch(v: torch.Tensor) -> torch.Tensor:
    u, _ = torch.sort(v, descending=True)
    css = torch.cumsum(u, 0) - 1.0
    ind = torch.arange(1, len(v) + 1, dtype=v.dtype)
    cond = (u - css / ind) > 0
    rho_idx = int(cond.nonzero()[-1]) + 1
    theta = css[rho_idx - 1] / rho_idx
    return torch.clamp(v - theta, min=0.0)


def _project_spectahedron_torch(M: torch.Tensor) -> torch.Tensor:
    M = 0.5 * (M + M.T)
    w, V = torch.linalg.eigh(M)
    w = _project_simplex_torch(w)
    return (V * w) @ V.T


# --------------------------------------------------------------------------- #
# Objectives (torch, differentiable in M)
# --------------------------------------------------------------------------- #
class Problem:
    def __init__(self, D, r, S, theta_c, N, sigma2, kappa, rho):
        self.D, self.r = D, r
        self.N, self.sigma2, self.kappa, self.rho = N, sigma2, kappa, rho
        self.S = torch.tensor(S)
        self.S_rr = self.S[:r, :r]
        self.S_rc = self.S[:r, r:]
        self.S_cc = self.S[r:, r:]
        self.theta_c = torch.tensor(theta_c) if len(theta_c) else None
        self.Ir = torch.eye(r)

    def _AGv(self, M):
        """Return A, G, and variance term v(M) from a design M (torch)."""
        r = self.r
        M_rr = M[:r, :r]
        M_rc = M[:r, r:]
        A = self.N * M_rr + self.kappa * self.Ir
        A_inv = torch.linalg.inv(A)
        G = A_inv @ (self.N * M_rc)                      # r x (D-r)
        Sig_b = A_inv @ (self.sigma2 * self.N * M_rr) @ A_inv
        v = torch.trace(self.S_rr @ Sig_b)
        return A, G, v

    def Q(self, M):
        """The (D-r)x(D-r) misspecification-risk matrix Q(M)."""
        _, G, _ = self._AGv(M)
        cross = G.T @ self.S_rc
        return G.T @ self.S_rr @ G - (cross + cross.T) + self.S_cc

    # ---- the four losses (all minimised) ----
    def loss_bayes(self, M):
        A = self.N * M[: self.r, : self.r] + self.kappa * self.Ir
        return -torch.logdet(A)

    def loss_minimax(self, M):
        return -torch.linalg.eigvalsh(M)[0]

    def loss_robust(self, M):
        if self.r >= self.D:
            _, _, v = self._AGv(M)
            return v
        Q = self.Q(M)
        _, _, v = self._AGv(M)
        return self.rho ** 2 * torch.linalg.eigvalsh(Q)[-1] + v

    def loss_oracle(self, M):
        if self.r >= self.D:
            _, _, v = self._AGv(M)
            return v
        Q = self.Q(M)
        _, _, v = self._AGv(M)
        return self.theta_c @ Q @ self.theta_c + v


# --------------------------------------------------------------------------- #
# Projected-gradient optimiser over the spectrahedron, with restarts
# --------------------------------------------------------------------------- #
from scipy.optimize import minimize


def _tril_to_M(x, D):
    """Map free vector x (D(D+1)/2 entries) -> M = L L^T / trace(L L^T), PSD, trace 1."""
    L = torch.zeros(D, D, dtype=torch.float64)
    idx = torch.tril_indices(D, D)
    L[idx[0], idx[1]] = x
    G = L @ L.T
    return G / torch.trace(G)


def optimize_design(loss_fn, D, restarts=8, iters=None, lr=None, seed=0):
    """Minimise loss_fn(M) over {M PSD, trace 1} via the unconstrained factor
    parameterisation M = L L^T / trace(L L^T), using scipy L-BFGS-B with exact
    (torch autograd) gradients, best of `restarts` random starts.

    The robust/oracle objectives are non-convex with a FLAT valley in the allocation
    direction; L-BFGS from many starts reaches a sharper optimum than projected SGD.
    (`iters`/`lr` kept for signature compatibility; unused.)"""
    rng = np.random.default_rng(seed)
    npar = D * (D + 1) // 2

    def fg(xv):
        x = torch.tensor(xv, dtype=torch.float64, requires_grad=True)
        M = _tril_to_M(x, D)
        loss = loss_fn(M)
        loss.backward()
        return float(loss.detach()), x.grad.numpy().astype(np.float64)

    best_val, best_M = np.inf, None
    for rs in range(restarts):
        if rs == 0:
            x0 = np.zeros(npar)
            x0[np.cumsum(np.arange(1, D + 1)) - 1] = 1.0  # ~ identity factor
        else:
            x0 = rng.standard_normal(npar)
        res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=500, ftol=1e-12, gtol=1e-9))
        if res.fun < best_val:
            x = torch.tensor(res.x, dtype=torch.float64)
            best_val = float(res.fun)
            best_M = _tril_to_M(x, D).detach().numpy().copy()
    return best_M, best_val


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def allocation(M, r):
    """Fraction of excitation energy in representable vs complement subspace."""
    rep = float(np.trace(M[:r, :r]))
    comp = float(np.trace(M[r:, r:])) if r < M.shape[0] else 0.0
    return rep, comp


def complement_profile(M, r, theta_c):
    """Within-complement structure of the design.
    Returns (participation_ratio, cos2_with_theta_c, top_eigvec).
      participation_ratio in [1, D-r]: 1 = rank-1 (concentrated), D-r = isotropic;
      cos2_with_theta_c = fraction of complement energy along the true error direction;
      top_eigvec = leading eigenvector of M_cc (the main excited complement direction)."""
    D = M.shape[0]
    if r >= D:
        return float("nan"), float("nan"), None
    Mcc = M[r:, r:]
    w, V = np.linalg.eigh(Mcc)
    w = np.clip(w, 0, None)
    s = w.sum()
    if s <= 1e-12:
        return float("nan"), float("nan"), None
    pr = (s ** 2) / (np.sum(w ** 2) + 1e-18)
    top = V[:, -1]
    if theta_c is not None and len(theta_c) and np.linalg.norm(theta_c) > 0:
        u = theta_c / np.linalg.norm(theta_c)
        cos2 = float(u @ Mcc @ u) / (s + 1e-18)
    else:
        cos2 = float("nan")
    return float(pr), cos2, top


# --------------------------------------------------------------------------- #
# Experiment driver
# --------------------------------------------------------------------------- #
DESIGNERS = ["bayes", "minimax", "robust", "oracle"]
LOSS_ATTR = {
    "bayes": "loss_bayes",
    "minimax": "loss_minimax",
    "robust": "loss_robust",
    "oracle": "loss_oracle",
}
# per-designer restart budget: bayes/minimax are convex (few starts suffice);
# robust/oracle are non-convex/multimodal and need many starts to find the global opt.
# NOTE: robust has a shallow spurious local min at comp~0.01 and oracle can under-converge
# from a single unlucky start, so we use generous budgets. (An MC cross-check confirmed
# these budgets recover the true optima; single-seed restarts=20 was occasionally short.)
RESTARTS = {"bayes": 3, "minimax": 4, "robust": 30, "oracle": 20}


def solve_all(prob: Problem, seed=0):
    out = {}
    for name in DESIGNERS:
        loss_fn = getattr(prob, LOSS_ATTR[name])
        M, val = optimize_design(loss_fn, prob.D, restarts=RESTARTS[name], seed=seed)
        rep, comp = allocation(M, prob.r)
        tc = prob.theta_c.numpy() if prob.theta_c is not None else None
        pr, cos2, top = complement_profile(M, prob.r, tc)
        out[name] = dict(M=M, loss=val, rep=rep, comp=comp, pr=pr, cos2=cos2, top=top)
    return out


def build(D, r, mode, N, sigma2, kappa, rho, seed):
    rng = np.random.default_rng(seed)
    S = make_test_cov(D, r, mode, rng)
    theta_c = make_theta_c(D, r, rho, rng)
    return Problem(D, r, S, theta_c, N, sigma2, kappa, rho)


def msd(a):
    a = np.asarray(a, float)
    return float(np.nanmean(a)), float(np.nanstd(a))


def main():
    D = 6
    N = 100.0
    sigma2 = 1.0
    kappa = 1.0     # sigma^2/tau^2  (tau=1)
    rho = 1.0       # ||theta_c||

    print("#" * 82)
    print("SPEARHEAD B1 -- misspecification-complement excitation")
    print(f"D={D}  N={N:.0f}  sigma^2={sigma2}  kappa=sigma^2/tau^2={kappa}  rho=||theta_c||={rho}")
    print("Design set: {M PSD, trace(M)=1}.  Allocation: rep=trace(M_rr), comp=trace(M_cc).")
    print("comp/dim = comp/(D-r).  PR(comp) in [1,D-r] (1=rank-1 concentrated, D-r=isotropic).")
    print("cos2(tc) = fraction of complement energy along the TRUE error direction theta_c.")
    print("#" * 82)

    # ---------- Part 1: allocation tables at a representative capacity ---------- #
    r = 3
    NDRAW = 6
    for mode in ["aligned", "correlated"]:
        agg = {name: dict(rep=[], comp=[], pr=[], cos2=[], top=[]) for name in DESIGNERS}
        for s in range(NDRAW):
            prob = build(D, r, mode, N, sigma2, kappa, rho, seed=100 + s)
            res = solve_all(prob, seed=100 + s)
            for name in DESIGNERS:
                for k in ("rep", "comp", "pr", "cos2"):
                    agg[name][k].append(res[name][k])
                agg[name]["top"].append(res[name]["top"])
        print(f"\n=== ALLOCATION  test={mode}  r={r}/{D}  (mean +/- std over {NDRAW} draws) ===")
        print(f"{'designer':9s} {'comp':>14s} {'comp/dim':>9s} {'PR(comp)':>9s} {'cos2(tc)':>9s}")
        for name in DESIGNERS:
            cm, cs = msd(agg[name]["comp"])
            prm, _ = msd(agg[name]["pr"])
            csm, _ = msd(agg[name]["cos2"])
            print(f"{name:9s} {cm:7.4f} +/-{cs:6.4f} {cm/(D-r):9.4f} "
                  f"{prm:9.3f} {csm:9.3f}")
        # direction cosine between robust and oracle complement excitation (per draw)
        dcos = []
        for tr, to in zip(agg["robust"]["top"], agg["oracle"]["top"]):
            if tr is not None and to is not None:
                dcos.append(float(np.dot(tr, to)) ** 2)
        if dcos:
            dm, ds = msd(dcos)
            print(f"cos2(robust top-dir, oracle top-dir) = {dm:.3f} +/- {ds:.3f}"
                  f"   (0 => guard DIFFERENT directions)")

    # ---------- Part 2: capacity-scaling curve (complement fraction vs r) ---------- #
    for mode in ["aligned", "correlated"]:
        print(f"\n=== CAPACITY SCALING  test={mode}   comp fraction trace(M_cc) vs r ===")
        print(f"{'r':>3s} {'bayes':>8s} {'minimax':>8s} {'robust':>8s} {'oracle':>8s} "
              f"{'orc-bay':>8s} {'rob-orc':>8s}")
        for r in range(1, D + 1):
            row = {name: [] for name in DESIGNERS}
            for s in range(4):
                prob = build(D, r, mode, N, sigma2, kappa, rho, seed=200 + s)
                res = solve_all(prob, seed=200 + s)
                for name in DESIGNERS:
                    row[name].append(res[name]["comp"])
            m = {name: float(np.mean(row[name])) for name in DESIGNERS}
            print(f"{r:3d} {m['bayes']:8.3f} {m['minimax']:8.3f} {m['robust']:8.3f} "
                  f"{m['oracle']:8.3f} {m['oracle']-m['bayes']:8.3f} "
                  f"{m['robust']-m['oracle']:8.3f}")

    # ---------- Part 3: scale sweeps (rho, N) at fixed r, correlated test ---------- #
    print(f"\n=== SCALE SWEEP  test=correlated  r=3/{D}  (complement fraction) ===")
    print(f"{'rho':>5s} {'N':>6s} {'bayes':>8s} {'minimax':>8s} {'robust':>8s} {'oracle':>8s}")
    for rho_s in [0.5, 1.0, 2.0, 4.0]:
        for N_s in [30.0, 100.0, 500.0]:
            row = {name: [] for name in DESIGNERS}
            for s in range(3):
                prob = build(D, 3, "correlated", N_s, sigma2, kappa, rho_s, seed=300 + s)
                res = solve_all(prob, seed=300 + s)
                for name in DESIGNERS:
                    row[name].append(res[name]["comp"])
            m = {name: float(np.mean(row[name])) for name in DESIGNERS}
            print(f"{rho_s:5.1f} {N_s:6.0f} {m['bayes']:8.3f} {m['minimax']:8.3f} "
                  f"{m['robust']:8.3f} {m['oracle']:8.3f}")

    # ---------- Part 4: validation of closed-form optima ---------- #
    print("\n=== VALIDATION (optimiser vs known closed-form optima) ===")
    prob = build(D, 3, "correlated", N, sigma2, kappa, rho, seed=1)
    res = solve_all(prob, seed=1)
    print(f"Bayes comp frac       (expect 0.000):              {res['bayes']['comp']:.4f}")
    print(f"Minimax comp frac     (expect (D-r)/D={(D-3)/D:.3f}):        {res['minimax']['comp']:.4f}")
    ev = np.linalg.eigvalsh(res["minimax"]["M"])
    print(f"Minimax eigvals of M  (expect ~1/D={1/D:.3f} each):   "
          f"[{', '.join(f'{e:.3f}' for e in ev)}]")
    print(f"Bayes M_rr diag       (expect ~1/r={1/3:.3f} each):   "
          f"[{', '.join(f'{d:.3f}' for d in np.diag(res['bayes']['M'])[:3])}]")


if __name__ == "__main__":
    main()
