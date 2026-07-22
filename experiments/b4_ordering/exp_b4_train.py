"""B4 stage 0 — train the CAUSAL ICL transformer (order sensitivity structural).
Same data/hyperparameters as A3; only the attention mask differs.
Run from repo root: python experiments/b4_ordering/exp_b4_train.py"""
import sys
import numpy as np
import torch

sys.path.insert(0, "experiments/spearhead_a")
from a3_common import ICLModel, gen_batch, KMAX, DEV

STEPS, B, VAL_EVERY, VAL_N = 8000, 64, 250, 512
torch.manual_seed(0)
torch.set_num_threads(8)


def make_val_set(seed=12345):
    rng = np.random.default_rng(seed)
    return [gen_batch(32, int(rng.integers(6, KMAX + 1)), rng) for _ in range(VAL_N // 32)]


@torch.no_grad()
def val_loss(model, val):
    model.eval()
    tot, n = 0.0, 0
    for toks, tgt in val:
        tot += ((model(toks.to(DEV)) - tgt.to(DEV)) ** 2).sum().item()
        n += len(tgt)
    model.train()
    return tot / n


def main():
    rng = np.random.default_rng(0)
    model = ICLModel(causal=True).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    val = make_val_set()
    best, log = float("inf"), []
    model.train()
    for step in range(STEPS + 1):
        if step % VAL_EVERY == 0:
            vl = val_loss(model, val)
            log.append(f"step {step:5d}  val {vl:.4f}")
            print(log[-1], flush=True)
            if vl < best:
                best = vl
                torch.save(model.state_dict(), "experiments/b4_ordering/b4_model_causal.pt")
        if step == STEPS:
            break
        k = int(rng.integers(6, KMAX + 1))
        toks, tgt = gen_batch(B, k, rng)
        loss = ((model(toks.to(DEV)) - tgt.to(DEV)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    log.append(f"best val {best:.4f}")
    with open("experiments/b4_ordering/b4_train_log.txt", "w") as f:
        f.write("\n".join(log) + "\n")
    print(log[-1])


if __name__ == "__main__":
    main()
