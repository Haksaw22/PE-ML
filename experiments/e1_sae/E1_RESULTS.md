# E1 pilot results

seeds=5 width=6144 eval_tokens=1000000 usable_features=1596/6144 (min_n=500) dead_frac_seed0=0.000

raw     Spearman(alpha, instability) = +0.029
PARTIAL Spearman(alpha, instability | log freq, n) = +0.027  [-0.017, +0.074]  <-- P1/K1 pre-registered test

## Frequency-decile bins (within-bin raw Spearman)
  decile 0: n=  160  rho=+0.054
  decile 1: n=  159  rho=-0.040
  decile 2: n=  159  rho=+0.034
  decile 3: n=  160  rho=+0.054
  decile 4: n=  160  rho=-0.083
  decile 5: n=  159  rho=-0.111
  decile 6: n=  160  rho=+0.015
  decile 7: n=  159  rho=+0.098
  decile 8: n=  160  rho=+0.053
  decile 9: n=  160  rho=+0.130

## Matched-null (generic geometry) comparison
  real features:  mean alpha 0.9959
  null halfspaces: mean alpha 0.4006  (n=1000)
  (null features have no instability — they calibrate how alpha varies with
   region geometry/frequency alone; a real effect must survive the partial
   correlation AND not be reproducible by frequency-matched geometry)

## Pre-registered verdict: K1 KILL: CI includes 0 — diagnostic does not beat the frequency confound
