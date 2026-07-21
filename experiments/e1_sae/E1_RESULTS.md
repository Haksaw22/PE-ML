# E1 pilot results

seeds=5 width=6144 eval_tokens=1000000 usable_features=2693/6144 (min_n=500) dead_frac_seed0=0.000

raw     Spearman(alpha, instability) = -0.068
PARTIAL Spearman(alpha, instability | log freq, n) = +0.259  [+0.226, +0.293]  <-- P1/K1 pre-registered test

## Frequency-decile bins (within-bin raw Spearman)
  decile 0: n=  270  rho=+0.217
  decile 1: n=  269  rho=+0.303
  decile 2: n=  269  rho=+0.265
  decile 3: n=  268  rho=+0.345
  decile 4: n=  270  rho=+0.327
  decile 5: n=  270  rho=+0.264
  decile 6: n=  269  rho=+0.309
  decile 7: n=  269  rho=+0.271
  decile 8: n=  269  rho=+0.231
  decile 9: n=  270  rho=+0.024

## Matched-null (generic geometry) comparison
  real features:  mean alpha 0.2411
  null halfspaces: mean alpha 0.1950  (n=1000)
  (null features have no instability — they calibrate how alpha varies with
   region geometry/frequency alone; a real effect must survive the partial
   correlation AND not be reproducible by frequency-matched geometry)

## Pre-registered verdict: UNEXPECTED SIGN: partial correlation positive — investigate before claiming
