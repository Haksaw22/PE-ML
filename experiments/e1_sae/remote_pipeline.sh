#!/bin/bash
# E1 full pipeline on the A100 host. Launched via nohup; touches PIPELINE_DONE on success.
set -e
cd ~/e1_sae
source .venv/bin/activate
echo "=== dump $(date) ==="
python dump_activations.py
echo "=== train x5 $(date) ==="
for s in 0 1 2 3 4; do
  python train_sae.py --seed $s
done
echo "=== analyze $(date) ==="
python analyze_stability.py
touch PIPELINE_DONE
echo "=== DONE $(date) ==="
