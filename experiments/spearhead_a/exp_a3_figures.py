"""A3 headline figures -> experiments/spearhead_a/figures/*.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z = np.load("experiments/spearhead_a/a3_trials.npz")
k, err, lmp = Z["k"], Z["err_tx"], Z["lmp"]
FIG = "experiments/spearhead_a/figures"
import os; os.makedirs(FIG, exist_ok=True)

# F1: quartile MSE at fixed k (deconfounded headline)
ks, lo_m, hi_m = (8, 12, 16), [], []
for k0 in ks:
    m = k == k0
    x = lmp[m]
    lo_m.append(err[m][x < np.quantile(x, 0.25)].mean())
    hi_m.append(err[m][x > np.quantile(x, 0.75)].mean())
xpos = np.arange(len(ks))
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(xpos - 0.18, lo_m, 0.36, label="low-excitation quartile", color="#c44")
ax.bar(xpos + 0.18, hi_m, 0.36, label="high-excitation quartile", color="#286")
for i, (a, b) in enumerate(zip(lo_m, hi_m)):
    ax.text(i, a + 0.01, f"{a/b:.1f}x", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(xpos, [f"k={v}" for v in ks])
ax.set_ylabel("transformer query MSE")
ax.set_title("Excitation predicts ICL failure — at fixed shot count")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/f1_quartiles_fixed_k.png", dpi=150)

# F2: aliasing arm
fig, ax = plt.subplots(figsize=(5.4, 4))
vals = [1.3607, 0.3593, 1.2990]  # from A3_ALIASING.md
ax.bar(["base (dark\ndirection)", "+ orthogonal\ndemo", "+ duplicate\ndemo"], vals,
       color=["#888", "#286", "#c44"], width=0.6)
ax.set_ylabel("transformer query MSE")
ax.set_title("Informativeness beats similarity 16x (aliasing arm)")
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
fig.tight_layout(); fig.savefig(f"{FIG}/f2_aliasing.png", dpi=150)

# F3: scatter at k=12
m = k == 12
fig, ax = plt.subplots(figsize=(5.4, 4))
ax.scatter(lmp[m], err[m], s=8, alpha=0.35, color="#348")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"projected excitation  $\lambda_{\min}(P^\top G P)$")
ax.set_ylabel("transformer query MSE")
ax.set_title("k = 12 slice")
fig.tight_layout(); fig.savefig(f"{FIG}/f3_scatter_k12.png", dpi=150)
print("figures written")
