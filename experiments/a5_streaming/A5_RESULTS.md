# A5 results — excitation-preserving memory under streaming

T=40 C=8 tasks=120 gamma=0.85 (per DESIGN.md)

## drift rho = 0.0
  fifo        : MSE 0.4005 +/- 0.0739
  reservoir   : MSE 0.3958 +/- 0.0719
  excite      : MSE 0.2154 +/- 0.0487
  excite_disc : MSE 0.2524 +/- 0.0640
  oracle      : MSE 0.0105 +/- 0.0033

## drift rho = 0.05
  fifo        : MSE 0.4205 +/- 0.0802
  reservoir   : MSE 0.4245 +/- 0.0776
  excite      : MSE 0.2981 +/- 0.0740
  excite_disc : MSE 0.2787 +/- 0.0654
  oracle      : MSE 0.0239 +/- 0.0069

## drift rho = 0.15
  fifo        : MSE 0.5476 +/- 0.1029
  reservoir   : MSE 0.6704 +/- 0.1218
  excite      : MSE 0.7741 +/- 0.2015
  excite_disc : MSE 0.5248 +/- 0.1064
  oracle      : MSE 0.0959 +/- 0.0272

## Pre-registered readouts (paired per-task differences)
  P1 static, fifo - excite      : +0.1851 +/- 0.0560  (excite wins)
  P1 static, reservoir - excite : +0.1803 +/- 0.0582
  P2 rho=0.05, excite - excite_disc: +0.0194 +/- 0.0413  (NULL)
  P2 rho=0.15, excite - excite_disc: +0.2493 +/- 0.1207  (disc wins)
  oracle gap (rho=0): best-policy 0.2154 vs oracle 0.0105
