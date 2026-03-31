# Example Data (Synthetic, Public-safe)

Rebuilt with strict path constraints.

## Path Constraint
For each core TF -> each TG:
- total paths: 20
- 15 paths with 2 steps
- 5 paths with 3 steps
- all paths <= 5 steps

## Simulation
Expression is simulated from synthetic VCF genotypes using linear logic with:
- h_cis = 0.1
- h_trans = 0.3
- h_tf = 0.6
