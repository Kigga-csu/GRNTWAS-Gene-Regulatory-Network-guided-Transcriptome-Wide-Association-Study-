# GRNTWAS: GRN-guided TWAS Training

GRNTWAS integrates gene regulatory networks (GRNs) into TWAS model training. It uses graph-guided TF selection and eQTL filtering to improve trans-SNP selection.

## Features
- Graph-guided TF selection
- eQTL-based SNP filtering
- Lasso/ElasticNet/DPR model comparison
- Parallel processing
- 5-fold cross-validation

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python >= 3.8
- `tabix` for VCF queries

## Usage

Edit paths in `config.py` or pass them by CLI.

```bash
python main.py
```

Example with explicit paths:

```bash
python main.py \
  --bed /path/to/gene.bed \
  --grn /path/to/network.gexf \
  --grn-tsv /path/to/network.tsv \
  --geno /path/to/vcf/ \
  --exp /path/to/expression.tsv \
  --sample /path/to/sample_ids.txt \
  --eqtl /path/to/eqtl.txt
```

## Key CLI Arguments
- `--bed`: gene annotation BED
- `--grn`: GRN GEXF
- `--grn-tsv`: GRN TSV
- `--geno`: VCF directory
- `--exp`: expression matrix
- `--sample`: sample IDs
- `--eqtl`: eQTL file
- `--threads`: parallel workers
- `--windows`: gene window size (bp)
- `--method`: influence method (RRW/LTM/KATZ/PATH)
- `--mode`: `lasso` or `eqtl`
- `--cv-r2`: enable CVR2
- `--cvr2-threshold`: CVR2 threshold

## Input Formats
- BED: `chrom start end strand gene_id gene_name gene_type`
- Expression: `CHROM GeneStart GeneEnd TargetID GeneName Sample1 Sample2 ...`
- GRN TSV: two columns (TF, Target)

## Outputs
- `weight_GRN.csv`: SNP weights
- `info_GRN.csv`: training summary

## Project Structure
```
GRNTWAS/
├── main.py
├── config.py
├── GRN_guided_adaptive_selection.py
├── GRNutils.py
├── Regular_subgraph_build.py
├── model/
│   └── Group_spares_lasso.py
└── README.md
```
