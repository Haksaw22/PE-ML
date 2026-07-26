# A5 pilot — excitation-preserving memory for streaming ICL (design-lite, pre-registered)

**Status:** EXECUTED — see `A5_VERDICT.md`. (A5 was reopened by the
2026-07-19 pivot audit: the streaming/online demo-arrival axis is the genuinely
sequential PoE reading the static-face kill never touched.)

## The question

An agent with a bounded context watches demonstrations arrive over time and must evict.
Classical concurrent learning (Chowdhary–Johnson) says: curate the retained set to keep
the information matrix persistently exciting, not merely recent. Does that beat recency
in a learned ICL system — and does the *windowed* PE condition (stay exciting at all
times) beat terminal-only curation when the task drifts?

## Setup (existing infra, CPU)

Stream of demos (x_t, y_t) from a task; context budget C = 8 slots; the A3 (or causal
B4) transformer consumes the retained set; query error measured continuously.
Two regimes: (i) static task; (ii) drifting task (w rotates slowly within the task
subspace — the regime where PERSISTENT excitation, not one-off excitation, should matter).

## Eviction policies (equal budget, label-free unless stated)

1. FIFO/recency (the default agents use)
2. Random reservoir
3. **Excitation-greedy (the PoE policy):** evict the demo whose removal least reduces
   lam_min of the projected retained Gram (Chowdhary–Johnson transplant)
4. Excitation-windowed: same, but on an exponentially time-discounted Gram (the
   drift-aware variant — the genuinely *persistent* condition)
5. Oracle upper bound: evict to minimize true query error (label-using; ceiling only)

## Pre-registered predictions & kills

- P1 (static): (3) beats (1) and (2) at equal budget on query MSE.
- P2 (drift — the PoE-specific claim): (4) beats (3) under drift, because undiscounted
  curation hoards stale excitation; the gap grows with drift rate.
- K1: if (3) fails to beat recency in the static regime, the transplant is empty here.
- K2: if (4) does not beat (3) under drift, the "persistent vs one-off" distinction has
  no measurable content at this scale — published as such.
- Honesty guard: report the oracle gap; if all policies sit within noise of the oracle,
  the problem is too easy to discriminate and the result is a null, not a win.

Cost: ~half a day CPU. No new theory claimed: this is a transplant test with a named
classical anchor, and P2 is the only part that is distinctively "persistency."
