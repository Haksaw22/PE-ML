# A3 aliasing arm

base k=8 (dark direction 3), n_tasks=500
  MSE base: 1.3607
  MSE +orthogonal demo: 0.3593   (delta-lmp +2.294)
  MSE +duplicate demo:  1.2990   (delta-lmp +0.000)
  orth improvement / dup improvement: 16.3x

## Budget probe (append duplicates of demo 0)
  +0 dup: MSE 0.6311  attention-weighted alpha 0.1197
  +1 dup: MSE 0.6658  attention-weighted alpha 0.1205
  +2 dup: MSE 0.5659  attention-weighted alpha 0.1095
  +3 dup: MSE 0.6812  attention-weighted alpha 0.1024
  +4 dup: MSE 0.7059  attention-weighted alpha 0.1017
  +5 dup: MSE 0.7204  attention-weighted alpha 0.0989
  +6 dup: MSE 0.7498  attention-weighted alpha 0.0918
  +7 dup: MSE 0.7197  attention-weighted alpha 0.0905
  +8 dup: MSE 0.6429  attention-weighted alpha 0.0787
  verdict: alpha_att falls; MSE flat (budget effect decorative at this scale)
