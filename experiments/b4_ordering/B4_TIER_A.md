# B4 Tier A — pre-registered result

tasks=300 k=10 perms=16 queries=6 window=4 (params pinned in PROPOSAL.md before any run)

## Existence check (repair 3)
  terminal additive Gram: permutation-invariant by construction (order signal is windowed-only)
  fraction of tasks with U-spread max/min > 1.2: 1.00
  median U-spread: 1764.64

## Order-sensitivity precondition
  within-task MSE relative range across permutations: median 1.791 (p10 0.897, p90 2.886)

## PRIMARY: pairwise sign prediction (large-gap pairs)
  n_pairs=18000  accuracy=0.495  Wilson95=[0.488, 0.502]
  pre-registered success: CI excludes 0.5 -> NULL (CI includes 0.5)

## Secondary: within-task Spearman(U, MSE)
  mean rho = -0.002  (n=300 tasks; negative = high-U orderings do better)
