# GRNTWAS Association GWAS Pipeline

This folder runs TWAS-style association tests using pre-trained GRNTWAS weights and GWAS summary statistics, then applies FDR correction.

The workflow is driven by `run_all_step_Association.sh`.

## What the script does

For each trait in `trait_list`:
1. **Association step** (`Association_GRNTWAS.py`)
   - Loads GWAS summary statistics (Z-scores)
   - Loads GRNTWAS weight file
   - Uses LD reference panels
   - Runs association tests for genes
2. **FDR step** (`FDR_calculate.py`)
   - Applies FDR correction to the association results

Outputs are written to a per-trait directory under `../result_eqtl_100k_TF_filter/Association/<trait>/`.

## Prerequisites

- Conda environment with required packages
- GWAS summary stats in TIGAR-compatible format
- GRNTWAS weight file
- Gene annotation file
- LD reference panel files

The script currently assumes a Conda env named `GRN_TLP`.

## Data locations

The script uses hard-coded paths. If your data are in the local `data/` folder, update the paths in `run_all_step_Association.sh` to point to your local locations.

Paths in the script:

- GWAS summary statistics:
  - `gwasfile="/data/lab/wangshixian/data/GWAS_summary/${trait}/${trait}_tigar_GWAS.hg38.sorted.indexed.tsv.gz"`
- GRNTWAS gene list:
  - `--gene_anno /data/lab/wangshixian/GRNTWAS_STAR/GRNTWAS2mayo-ad/vcf_project/result_eqtl_100k_TF_filter/info_all_exp_gene_156_peer/can_predicted_gene_protein.txt`
- GRNTWAS weights:
  - `--weight /data/lab/wangshixian/GRNTWAS_STAR/GRNTWAS2mayo-ad/vcf_project/result_eqtl_100k_TF_filter/weight_all_exp_gene_156_peer/weight_GRN.csv`
- LD reference panels:
  - `--LD_pattern /data/lab/wangshixian/data/LD/1000G_phase3_EUR/no_chr_prefix/CHR{chrom}_reference_cov.nochr.txt.gz`
- Gene annotation BED:
  - `--gtf /data/lab/wangshixian/data/gtf/gene.bed`

## Usage

Edit the trait list and paths in `run_all_step_Association.sh`, then run:

```bash
bash run_all_step_Association.sh
```

## Script parameters (Association step)

Key arguments passed to `Association_GRNTWAS.py`:

- `--TIGAR_dir`: TIGAR model directory flag (set to `1` in the script)
- `--gene_anno`: gene list to analyze
- `--weight`: GRNTWAS weight file
- `--Zscore`: GWAS summary statistics file
- `--LD_pattern`: LD reference panel pattern, with `{chrom}` placeholder
- `--window`: cis window size (bp)
- `--weight_threshold`: minimum absolute weight
- `--test_stat`: `both` (runs both statistics)
- `--thread`: number of threads
- `--out_dir`: output directory
- `--out_twas_file`: output filename
- `--gtf`: gene annotation BED

## Output structure

For each trait:

- Association results:
  - `../result_eqtl_100k_TF_filter/Association/<trait>/GRNTWAS_results_<trait>_protein_e5.tsv`
- FDR results:
  - `../result_eqtl_100k_TF_filter/Association/<trait>/FDR_protein/`

## Notes

- The script assumes the GWAS summary statistics are **sorted, indexed, and hg38**.
- LD files must exist for each chromosome referenced by the GWAS data.
- If you are using a different environment name, change `source activate GRN_TLP`.
