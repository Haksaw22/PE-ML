# Spearhead A / Stage A3 — repaired construct-validity results

n = 4000 trials | mean transformer MSE 0.3095 | mean Bayes-ridge floor 0.0125

## Pooled Spearman vs transformer query MSE (95% bootstrap CI)
(excitation diagnostics: negative = predicts error; variance baselines: positive)
  lmp        : rho = -0.284  [-0.312, -0.257]
  lmf        : rho = -0.221  [-0.250, -0.192]
  lmp_att    : rho = -0.264  [-0.294, -0.234]
  lmp_est    : rho = -0.282  [-0.310, -0.252]
  trace_proj : rho = -0.204  [-0.233, -0.174]
  predvar_q  : rho = +0.471  [+0.447, +0.496]
  predvar_tr : rho = +0.283  [+0.255, +0.312]
  k          : rho = -0.217  [-0.247, -0.188]

## Fixed-k slices (the deconfounded comparison)
  k= 8 (n=281): lmp=-0.177  lmf=+0.090  lmp_att=-0.186  lmp_est=-0.182  trace_proj=-0.108  predvar_q=+0.478  predvar_tr=+0.166
  k=12 (n=239): lmp=-0.202  lmf=-0.149  lmp_att=-0.197  lmp_est=-0.225  trace_proj=-0.112  predvar_q=+0.488  predvar_tr=+0.222
  k=16 (n=256): lmp=-0.152  lmf=-0.052  lmp_att=-0.179  lmp_est=-0.171  trace_proj=-0.082  predvar_q=+0.442  predvar_tr=+0.174

## Paired bootstrap: |rho| differences at fixed k (positive = first wins)
  lmp      vs lmf       @k= 8: +nan [+nan,+nan] ns 
  lmp      vs lmf       @k=12: +0.054 [-0.108,+0.217] ns 
  lmp      vs lmf       @k=16: +0.084 [-0.044,+0.216] ns 
  lmp_att  vs lmp       @k= 8: +0.009 [-0.013,+0.030] ns 
  lmp_att  vs lmp       @k=12: -0.006 [-0.031,+0.019] ns 
  lmp_att  vs lmp       @k=16: +0.026 [-0.004,+0.057] ns 
  lmp_est  vs lmp       @k= 8: +0.005 [-0.030,+0.040] ns 
  lmp_est  vs lmp       @k=12: +0.023 [-0.008,+0.054] ns 
  lmp_est  vs lmp       @k=16: +0.019 [-0.009,+0.046] ns 
  lmp      vs predvar_q @k= 8: -0.299 [-0.417,-0.184] SIG
  lmp      vs predvar_q @k=12: -0.284 [-0.421,-0.148] SIG
  lmp      vs predvar_q @k=16: -0.290 [-0.415,-0.168] SIG

## Quartile MSE ratio (low-excitation / high-excitation, by lmp)
  POOLED (k-confounded, upper bound): 6.52x
  k= 8 (deconfounded): 2.42x
  k=12 (deconfounded): 3.17x
  k=16 (deconfounded): 3.31x
  fixed-k mean: 2.96x  <-- honest headline

## Excess-error analysis (ICL-specific failure vs 'hard for anyone')
  rho(err_ridge, err_tx)      = +0.233  (how much of tx error is just problem hardness)
  rho(lmp      , err_gap)    = -0.223  (gap = tx - ridge floor)
  rho(lmp_att  , err_gap)    = -0.208  (gap = tx - ridge floor)
  rho(predvar_q, err_gap)    = +0.380  (gap = tx - ridge floor)
    k= 8: rho(lmp, gap) = -0.170
    k=12: rho(lmp, gap) = -0.160
    k=16: rho(lmp, gap) = -0.119
