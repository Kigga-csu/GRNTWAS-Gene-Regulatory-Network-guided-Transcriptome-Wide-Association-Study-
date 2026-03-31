# Example Data (Synthetic)

This folder contains synthetic, public-safe demo data for GRNTWAS.

## Contents
- `gene_example_55.bed`: 55 genes (5 TG + 50 TF)
- `Combination_trrust_HTF_Brain_example.tsv`: synthetic GRN edges (TF->Target)
- `Combination_trrust_HTF_Brain_example.gexf`: synthetic GRN graph
- `sample_ids.txt`: sample IDs (1..20)
- `tpm_normalized_gene_peer_vcf.csv`: synthetic expression matrix
- `vcf/`: synthetic VCF files (`example_chr*.vcf.gz` + `.tbi`)

## Path Constraints
For each core TF to each TG:
- total paths: 20
- 15 paths of length 2
- 5 paths of length 3
- all paths <= 5 steps

## Simulation Parameters
Expression is simulated from synthetic genotypes with:
- `h_cis = 0.1`
- `h_trans = 0.3`
- `h_tf = 0.6`
