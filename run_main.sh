#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/wangshixian/Documents/project/paper_manuscript/GRNTWAS"
PY="$ROOT/.conda_env/bin/python"

# Data paths
BED="$ROOT/data/gene.bed"
GRN_GEXF="$ROOT/data/Combination_trrust_HTF_Brain.gexf"
GRN_TSV="$ROOT/data/Combination_trrust_HTF_Brain.tsv"
EQTL="$ROOT/data/TF_eQTL_all.txt"
EXP="$ROOT/data/tpm_normalized_gene_peer_vcf.csv"
SAMPLE="$ROOT/data/sample_ids.txt"
GENO="/Volumes/WSX19819083255_data/data_genomic/"

# Output paths
OUT_WEIGHT="$ROOT/result_eqtl_100k_TF_GRN_brainscope/weight_all_exp_gene_156_peer/"
OUT_INFO="$ROOT/result_eqtl_100k_TF_GRN_brainscope/info_all_exp_gene_156_peer/"

mkdir -p "$OUT_WEIGHT" "$OUT_INFO"

exec "$PY" "$ROOT/code/main.py" \
  --bed "$BED" \
  --grn "$GRN_GEXF" \
  --grn-tsv "$GRN_TSV" \
  --geno "$GENO" \
  --exp "$EXP" \
  --sample "$SAMPLE" \
  --eqtl "$EQTL" \
  --out-weight "$OUT_WEIGHT" \
  --out-info "$OUT_INFO" \
  --threads 1 \
  --method LTM \
  --mode eqtl \
  --windows 100000 \
  --tf-numbers 10 \
  --maf 0.01 \
  --missing-rate 0.2 \
  --hwe 0.001 \
  --cv-r2 \
  --cvr2-threshold 0.001
