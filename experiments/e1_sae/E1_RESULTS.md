# E1 pilot results

seeds=3 width=512 eval_tokens=20000 usable_features=512/512 (min_n=100) dead_frac_seed0=0.000

raw     Spearman(alpha, instability) = -0.079
PARTIAL Spearman(alpha, instability | log freq, n) = -0.079  [-0.163, +0.002]  <-- P1/K1 pre-registered test

## Frequency-decile bins (within-bin raw Spearman)
  decile 0: n=   52  rho=-0.178
  decile 1: n=   50  rho=-0.106
  decile 2: n=   52  rho=+0.233
  decile 3: n=   51  rho=-0.037
  decile 4: n=   51  rho=-0.182
  decile 5: n=   50  rho=-0.238
  decile 6: n=   52  rho=+0.027
  decile 7: n=   51  rho=-0.064
  decile 8: n=   51  rho=-0.005
  decile 9: n=   52  rho=-0.265

## Matched-null (generic geometry) comparison
  real features:  mean alpha 0.7083
  null halfspaces: mean alpha 0.7195  (n=200)
  (null features have no instability — they calibrate how alpha varies with
   region geometry/frequency alone; a real effect must survive the partial
   correlation AND not be reproducible by frequency-matched geometry)

## Pre-registered verdict: K1 KILL: CI includes 0 — diagnostic does not beat the frequency confound
