"""Article figures -> article/figures/. Static PNGs + the angles GIF."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

FIG = "article/figures"
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(3)

# ---- F_ANGLES.GIF: same question repeated vs new angles ---------------------
# Posterior ellipse over theta under sequential measurements phi_k.
def post_cov(phis, s2=0.25, prior=4.0):
    A = np.eye(2) / prior
    for p in phis:
        A = A + np.outer(p, p) / s2
    return np.linalg.inv(A)

def ellipse(C, n=100):
    t = np.linspace(0, 2 * np.pi, n)
    L = np.linalg.cholesky(C)
    return (L @ np.vstack([np.cos(t), np.sin(t)]) * 2).T  # 2-sigma

angles_same = [0.6] * 12
angles_var = list(0.6 + 2.2 * np.sin(np.linspace(0, 4, 12)) + rng.normal(0, .3, 12))
fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
titles = ["the same question, twelve times", "twelve questions, twelve angles"]
els = []
for ax, ttl in zip(axes, titles):
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect(1)
    ax.set_title(ttl, fontsize=11)
    ax.plot(0.8, -0.5, "k*", ms=12)
    ax.set_xticks([]); ax.set_yticks([])
    (l,) = ax.plot([], [], color="#348", lw=2)
    arr = ax.plot([], [], color="#c44", lw=1.6)[0]
    els.append((l, arr))

def frame(k):
    for (l, arr), angs in zip(els, (angles_same, angles_var)):
        phis = [np.array([np.cos(a), np.sin(a)]) for a in angs[: k + 1]]
        e = ellipse(post_cov(phis)) + np.array([0.8, -0.5])
        l.set_data(e[:, 0], e[:, 1])
        a = angs[k]
        arr.set_data([0.8 - 2.6 * np.cos(a), 0.8], [-0.5 - 2.6 * np.sin(a), -0.5])
    fig.suptitle(f"measurement {k+1}/12 — the ellipse is what you still don't know", fontsize=11)
    return sum(([l, a] for l, a in els), [])

anim = FuncAnimation(fig, frame, frames=12, interval=600)
anim.save(f"{FIG}/f_angles.gif", writer=PillowWriter(fps=1.8), dpi=100)
plt.close(fig)

# ---- F_SIXNAMES.PNG: one matrix, six vocabularies ---------------------------
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.axis("off")
ax.add_patch(plt.Circle((0.5, 0.5), 0.13, color="#348", alpha=0.15))
ax.text(0.5, 0.5, r"$M=\sum_k \phi_k\phi_k^{\top}$", ha="center", va="center", fontsize=15)
labels = [
    ("Adaptive control", r"$\lambda_{\min}(M)\geq\alpha$: persistent excitation"),
    ("Statistics", "Fisher information / Cramér–Rao"),
    ("Experiment design", "D-, A-, E-optimality"),
    ("Active learning", "information gain, coverage, leverage"),
    ("System ID", "Hankel rank = modes touched"),
    ("Deep learning", "gradient covariance, K-FAC, Laplace"),
]
for i, (name, sub) in enumerate(labels):
    a = np.pi / 2 - i * np.pi / 3
    x, y = 0.5 + 0.36 * np.cos(a), 0.5 + 0.36 * np.sin(a)
    ax.annotate("", xy=(0.5 + 0.15 * np.cos(a), 0.5 + 0.15 * np.sin(a)),
                xytext=(x - 0.05 * np.cos(a), y - 0.05 * np.sin(a)),
                arrowprops=dict(arrowstyle="-", color="#888", lw=1))
    ax.text(x, y + 0.022, name, ha="center", fontsize=11, fontweight="bold")
    ax.text(x, y - 0.022, sub, ha="center", fontsize=8.5, color="#444")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.suptitle("One matrix, six vocabularies", fontsize=13)
fig.tight_layout()
fig.savefig(f"{FIG}/f_sixnames.png", dpi=150)
plt.close(fig)

# ---- copy headline experiment figures into article/figures ------------------
import shutil
for src, dst in [
    ("experiments/spearhead_a/figures/f1_quartiles_fixed_k.png", "f_quartiles.png"),
    ("experiments/spearhead_a/figures/f2_aliasing.png", "f_aliasing.png"),
    ("experiments/e1_sae/figures/e1_deciles.png", "f_e1_deciles.png"),
]:
    shutil.copy(src, f"{FIG}/{dst}")
print("figures done:", sorted(os.listdir(FIG)))
