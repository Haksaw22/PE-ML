"""
E1 stage 0 — stream OpenWebText through GPT-2-small; save layer-8 residual-stream
activations as fp16 memmaps (train + held-out). Position-0 tokens are dropped (norm
outliers). Sizes per DESIGN.md; --smoke for a tiny local end-to-end test.

  python dump_activations.py            # full (A100)
  python dump_activations.py --smoke    # tiny local validation
"""
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--smoke", action="store_true")
p.add_argument("--outdir", default=".")
a = p.parse_args()

N_TRAIN, N_HELD = (60_000, 20_000) if a.smoke else (10_000_000, 2_000_000)
SEQ, BATCH, LAYER, D = 256, (4 if a.smoke else 32), 8, 768
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEV == "cuda" else torch.float32

tok = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", dtype=DTYPE).to(DEV).eval()
cache = {}
model.transformer.h[LAYER].register_forward_hook(
    lambda m, i, o: cache.update(h=(o[0] if isinstance(o, tuple) else o)))

out_tr = np.lib.format.open_memmap(f"{a.outdir}/acts_train.npy", mode="w+",
                                   dtype=np.float16, shape=(N_TRAIN, D))
out_he = np.lib.format.open_memmap(f"{a.outdir}/acts_held.npy", mode="w+",
                                   dtype=np.float16, shape=(N_HELD, D))
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

buf, n_tr, n_he = [], 0, 0
with torch.no_grad():
    for ex in ds:
        buf.extend(tok(ex["text"])["input_ids"])
        while len(buf) >= SEQ * BATCH:
            ids = torch.tensor([buf[i * SEQ:(i + 1) * SEQ] for i in range(BATCH)], device=DEV)
            buf = buf[SEQ * BATCH:]
            model(ids)
            h = cache["h"][:, 1:, :].reshape(-1, D).to(torch.float16).cpu().numpy()
            take = min(len(h), N_TRAIN - n_tr)
            if take > 0:
                out_tr[n_tr:n_tr + take] = h[:take]
                n_tr += take
                h = h[take:]
            if len(h) and n_he < N_HELD:
                t2 = min(len(h), N_HELD - n_he)
                out_he[n_he:n_he + t2] = h[:t2]
                n_he += t2
            if n_tr % 500_000 < SEQ * BATCH:
                print(f"train {n_tr}/{N_TRAIN}  held {n_he}/{N_HELD}", flush=True)
        if n_tr >= N_TRAIN and n_he >= N_HELD:
            break
out_tr.flush(); out_he.flush()
print(f"DONE train={n_tr} held={n_he}")
