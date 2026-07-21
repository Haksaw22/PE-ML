"""
Spearhead B0 -- Monte-Carlo validation of the closed-form prediction risk.

Sanity check that the analytic Risk(M, theta_c) = theta_c^T Q(M) theta_c + v(M)
used to define designers 3 and 4 actually equals the mean-squared prediction error
of the real learn-then-predict loop (Gaussian design with covariance M, ridge fit on
the representable block, evaluate on the test distribution). If these match, the
objectives driving the whole experiment are correct.
"""
from __future__ import annotations
import numpy as np
import torch
import exp_b1_misspec as e


def analytic_risk(prob: e.Problem, M):
    Mt = torch.tensor(M)
    Q = prob.Q(Mt)
    _, _, v = prob._AGv(Mt)
    tc = prob.theta_c
    return float(tc @ Q @ tc + v)


def mc_risk(prob: e.Problem, M, N, reps=4000, ntest=20000, seed=0):
    """Simulate: Gaussian design phi~N(0,M); ridge fit on representable block with
    kappa; predict on phi~N(0,S). theta_r=0, theta_c given. Return mean SE."""
    rng = np.random.default_rng(seed)
    D, r = prob.D, prob.r
    Mnp = np.asarray(M)
    Snp = prob.S.numpy()
    kappa = prob.kappa
    sigma = np.sqrt(prob.sigma2)
    theta = np.zeros(D)
    theta[r:] = prob.theta_c.numpy()
    # test points (fixed across reps)
    Ls = np.linalg.cholesky(Snp + 1e-12 * np.eye(D))
    Xtest = rng.standard_normal((ntest, D)) @ Ls.T
    ytest_clean = Xtest @ theta
    Lm = np.linalg.cholesky(Mnp + 1e-12 * np.eye(D))
    errs = []
    for _ in range(reps):
        X = rng.standard_normal((int(N), D)) @ Lm.T
        y = X @ theta + sigma * rng.standard_normal(int(N))
        Xr = X[:, :r]
        A = Xr.T @ Xr + kappa * np.eye(r)          # = N*M_rr + kappa I in expectation
        beta = np.linalg.solve(A, Xr.T @ y)
        pred = Xtest[:, :r] @ beta
        errs.append(np.mean((pred - ytest_clean) ** 2))
    return float(np.mean(errs)), float(np.std(errs) / np.sqrt(reps))


if __name__ == "__main__":
    print("Monte-Carlo validation of closed-form risk (theta_r=0)")
    print(f"{'mode':11s} {'N':>5s} {'design':8s} {'analytic':>10s} {'MC mean':>10s} {'MC se':>8s} {'rel.err':>8s}")
    for mode in ["aligned", "correlated"]:
        for N in [100.0]:
            prob = e.build(6, 3, mode, N, 1.0, 1.0, 1.0, seed=7)
            # test on a few designs: oracle optimum, robust optimum, and I/D
            designs = {}
            Mo, _ = e.optimize_design(prob.loss_oracle, 6, restarts=20, seed=7)
            Mr, _ = e.optimize_design(prob.loss_robust, 6, restarts=40, seed=7)
            designs["oracle"] = Mo
            designs["robust"] = Mr
            designs["I/D"] = np.eye(6) / 6
            for dn, M in designs.items():
                a = analytic_risk(prob, M)
                m, se = mc_risk(prob, M, N, seed=11)
                rel = abs(a - m) / (abs(m) + 1e-12)
                print(f"{mode:11s} {N:5.0f} {dn:8s} {a:10.4f} {m:10.4f} {se:8.4f} {rel:8.4f}")
