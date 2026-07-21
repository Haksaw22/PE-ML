"""
A3 stage 1 — train the ICL transformer WITH validation tracking + best-checkpoint selection
(repairs Audit 4 finding: A2 evaluated an unverified late-training snapshot).

Writes: a3_model.pt (best val checkpoint), a3_train_log.txt
Run:    python experiments/spearhead_a/exp_a3_train.py
"""
import numpy as np
import torch
from a3_common import ICLModel, gen_batch, KMAX, DEV

STEPS, B, VAL_EVERY, VAL_N = 8000, 64, 250, 512
torch.manual_seed(0)
torch.set_num_threads(8)


def make_val_set(seed=12345):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(VAL_N // 32):
        k = int(rng.integers(6, KMAX + 1))
        batches.append(gen_batch(32, k, rng))
    return batches


@torch.no_grad()
def val_loss(model, val):
    model.eval()
    tot, n = 0.0, 0
    for toks, tgt in val:
        pred = model(toks.to(DEV))
        tot += ((pred - tgt.to(DEV)) ** 2).sum().item()
        n += len(tgt)
    model.train()
    return tot / n


def main():
    rng = np.random.default_rng(0)
    model = ICLModel().to(DEV)
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
                torch.save(model.state_dict(), "experiments/spearhead_a/a3_model.pt")
        if step == STEPS:
            break
        k = int(rng.integers(6, KMAX + 1))
        toks, tgt = gen_batch(B, k, rng)
        loss = ((model(toks.to(DEV)) - tgt.to(DEV)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    log.append(f"best val {best:.4f}  (checkpoint saved at best, not last)")
    with open("experiments/spearhead_a/a3_train_log.txt", "w") as f:
        f.write("\n".join(log) + "\n")
    print(log[-1])


if __name__ == "__main__":
    main()
