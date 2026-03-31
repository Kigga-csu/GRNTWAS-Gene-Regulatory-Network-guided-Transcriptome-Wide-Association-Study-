#!/usr/bin/env bash
# 
source activate GRN_TLP

# -------  -------
trait_list=(
  #"AD_GCST90027158" "AD_2021" "AD_2019"
  #"ADHD2022" "BIP" "AN" "ASD" "PTSD" "SCZ"
  "AD_2019"
  #"CDG" "BV" "NTSM" "OCD" "PANIC"
)

# -------  -------
for trait in "${trait_list[@]}"; do
    echo "=== : $trait ==="

    # 
    gwasfile="/data/lab/wangshixian/data/GWAS_summary/${trait}/${trait}_tigar_GWAS.hg38.sorted.indexed.tsv.gz"
    outdir="../result_eqtl_100k_TF_filter/Association/${trait}"
    result_file="${outdir}/GRNTWAS_results_${trait}_protein_e5.tsv"

    # ---------- Step 1: Association ----------
    python Association_GRNTWAS.py \
        --TIGAR_dir 1 \
        --gene_anno /data/lab/wangshixian/GRNTWAS_STAR/GRNTWAS2mayo-ad/vcf_project/result_eqtl_100k_TF_filter/info_all_exp_gene_156_peer/can_predicted_gene_protein.txt \
        --weight /data/lab/wangshixian/GRNTWAS_STAR/GRNTWAS2mayo-ad/vcf_project/result_eqtl_100k_TF_filter/weight_all_exp_gene_156_peer/weight_GRN.csv \
        --Zscore "${gwasfile}" \
        --LD_pattern /data/lab/wangshixian/data/LD/1000G_phase3_EUR/no_chr_prefix/CHR{chrom}_reference_cov.nochr.txt.gz \
        --window 100000 \
        --weight_threshold 0.0001 \
        --test_stat both \
        --thread 100 \
        --out_dir "${outdir}" \
        --out_twas_file "GRNTWAS_results_${trait}_protein_e5.tsv" \
        --gtf /data/lab/wangshixian/data/gtf/gene.bed

    # ---------- Step 2: FDR ----------
    #   -  --input
    #   -  outdir  FDR 
    fdr_outdir="${outdir}/FDR_protein"
    mkdir -p "${fdr_outdir}"

    python FDR_calculate.py \
        --input  "${result_file}" \
        --outdir "${fdr_outdir}"

    echo "=== $trait  ==="
done
