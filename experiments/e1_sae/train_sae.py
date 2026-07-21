"""
E1 stage 1 — train ONE ReLU+L1 SAE on the dumped activations.

Per DESIGN.md: data order is FIXED across seeds (permutation seed 0, shared);
only the INIT seed varies, so init is the sole source of cross-SAE variation.

  python train_sae.py --seed 3            # full (A100)
  python train_sae.py --seed 3 --smoke    # tiny local validation
"""
import argparse
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--smoke", action="store_true")
p.add_argument("--width", type=int, default=None)
p.add_argument("--l1", type=float, default=5e-4)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--batch", type=int, default=4096)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--outdir", default=".")
a = p.parse_args()
WIDTH = a.width or (512 if a.smoke else 6144)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D = 768

X = np.load(f"{a.outdir}/acts_train.npy", mmap_mode="r")
N = len(X)
# global scalar normalization so E||x||^2 = D (computed on a fixed prefix, seed-free)
scale = float(np.sqrt(D) / np.linalg.norm(np.asarray(X[:200_000], np.float32), axis=1).mean())
b_dec0 = torch.tensor(np.asarray(X[:200_000], np.float32).mean(0) * scale, device=DEV)

g = torch.Generator().manual_seed(a.seed)
W_dec = torch.nn.functional.normalize(torch.randn(WIDTH, D, generator=g), dim=1).to(DEV)
W_dec.requires_grad_(True)
W_enc = W_dec.detach().clone().T.contiguous().requires_grad_(True)
b_enc = torch.zeros(WIDTH, device=DEV, requires_grad=True)
b_dec = b_dec0.clone().requires_grad_(True)
opt = torch.optim.Adam([W_enc, b_enc, W_dec, b_dec], lr=a.lr)

perm_rng = np.random.default_rng(0)                      # SHARED across seeds
steps_per_epoch = N // a.batch
last_fire = torch.zeros(WIDTH, dtype=torch.long, device=DEV)
step_t = 0
for ep in range(a.epochs):
    order = perm_rng.permutation(steps_per_epoch)        # same batch order for all seeds
    for si, b0 in enumerate(order):
        xb = torch.tensor(np.asarray(X[b0 * a.batch:(b0 + 1) * a.batch], np.float32),
                          device=DEV) * scale
        f = torch.relu((xb - b_dec) @ W_enc + b_enc)
        xhat = f @ W_dec + b_dec
        loss = ((xhat - xb) ** 2).sum(-1).mean() + a.l1 * f.abs().sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            W_dec.data = torch.nn.functional.normalize(W_dec.data, dim=1)
            last_fire[(f > 0).any(0)] = step_t
        if si % 500 == 0:
            l0 = float((f > 0).float().sum(-1).mean())
            print(f"seed {a.seed} ep {ep} step {si}/{steps_per_epoch} "
                  f"loss {loss.item():.3f} L0 {l0:.1f}", flush=True)
        step_t += 1

dead = float((last_fire < step_t - 1000).float().mean())
torch.save({"W_enc": W_enc.detach().cpu(), "b_enc": b_enc.detach().cpu(),
            "W_dec": W_dec.detach().cpu(), "b_dec": b_dec.detach().cpu(),
            "scale": scale, "width": WIDTH, "dead_frac": dead},
           f"{a.outdir}/sae_seed{a.seed}.pt")
print(f"seed {a.seed} done; dead fraction {dead:.3f}")
