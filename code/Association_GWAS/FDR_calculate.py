import pandas as pd
import numpy as np
import os
import argparse
from statsmodels.stats.multitest import multipletests # For FDR calculation

# --- Configuration ---
# Input file containing the combined TWAS results
parser = argparse.ArgumentParser(
    description=" GRNTWAS / TWAS  FDR "
)
parser.add_argument(
    "-i", "--input", required=True, help=" TWAS （TSV/CSV ）"
)
parser.add_argument(
    "-o", "--outdir", default="twas_FDR_processed_results",
    help="（: twas_FDR_processed_results）"
)
args = parser.parse_args()

input_file = args.input
output_dir = args.outdir

# Output filenames
fusion_output_file = os.path.join(output_dir, 'fusion_results_sorted_fdr.tsv')
spred_output_file = os.path.join(output_dir, 'spred_results_sorted_fdr.tsv')
summary_output_file = os.path.join(output_dir, 'significance_counts.txt')

# Significance thresholds
fdr_threshold = 0.05
pval_threshold = 0.05

# --- Ensure output directory exists ---
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# --- Load the data ---
try:
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file, sep='\t')
    print(f"Loaded {df.shape[0]} results.")

    # Convert P-value columns to numeric, coercing errors to NaN
    if 'FUSION_PVAL' in df.columns:
        df['FUSION_PVAL'] = pd.to_numeric(df['FUSION_PVAL'], errors='coerce')
    if 'SPred_PVAL' in df.columns:
        df['SPred_PVAL'] = pd.to_numeric(df['SPred_PVAL'], errors='coerce')

except FileNotFoundError:
    print(f"Error: Input file not found at '{input_file}'")
    exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# --- Calculate FDR (Benjamini-Hochberg) ---

# Function to safely calculate FDR
from statsmodels.stats.multitest import fdrcorrection_twostage
def calculate_fdr_bky(p_values):
    """Calculates FDR using BKY two-stage method, handling NaNs."""
    pvals_nonan = p_values.dropna()
    if pvals_nonan.empty:
        return pd.Series(np.nan, index=p_values.index)

    reject, q_values, _, _ = fdrcorrection_twostage(pvals_nonan, method='bky', alpha=0.05, maxiter=1)

    fdr_series = pd.Series(np.nan, index=p_values.index)
    fdr_series.loc[pvals_nonan.index] = q_values
    return fdr_series

def calculate_fdr_by(p_values):
    """Calculates FDR using Benjamini-Yekutieli method, handling NaNs."""
    pvals_nonan = p_values.dropna()
    if pvals_nonan.empty:
        return pd.Series(np.nan, index=p_values.index)

    reject, q_values, _, _ = multipletests(pvals_nonan, method='bonferroni')

    fdr_series = pd.Series(np.nan, index=p_values.index)
    fdr_series.loc[pvals_nonan.index] = q_values
    return fdr_series

def calculate_fdr(p_values):
    """Calculates FDR handling NaNs."""
    pvals_nonan = p_values.dropna()
    if pvals_nonan.empty:
        # Return NaNs of the same shape if no valid p-values
        return pd.Series(np.nan, index=p_values.index)

    # Calculate FDR on non-NaN values
    reject, q_values, _, _ = multipletests(pvals_nonan, method='fdr_bh')

    # Create a Series with NaNs, then fill in the calculated q-values
    fdr_series = pd.Series(np.nan, index=p_values.index)
    fdr_series.loc[pvals_nonan.index] = q_values
    return fdr_series

# Calculate FUSION FDR if column exists
if 'FUSION_PVAL' in df.columns:
    print("Calculating FDR for FUSION...")
    df['FUSION_FDR'] = calculate_fdr_by(df['SPred_PVAL'])
    print("FUSION FDR calculation complete.")
else:
    print("FUSION_PVAL column not found, skipping FUSION FDR calculation.")
    df['FUSION_FDR'] = np.nan # Add column with NaNs if PVAL missing

# Calculate SPrediXcan FDR if column exists
if 'SPred_PVAL' in df.columns:
    print("Calculating FDR for SPrediXcan...")
    df['SPred_FDR'] = calculate_fdr_by(df['SPred_PVAL'])
    print("SPrediXcan FDR calculation complete.")
else:
    print("SPred_PVAL column not found, skipping SPrediXcan FDR calculation.")
    df['SPred_FDR'] = np.nan # Add column with NaNs if PVAL missing

# --- Split into FUSION and SPrediXcan DataFrames ---

# Define common columns
common_cols = ['CHROM', 'GeneStart', 'GeneEnd', 'TargetID', 'GeneName', 'n_snps', 'used_regions']
# Ensure common columns actually exist in the DataFrame
common_cols = [col for col in common_cols if col in df.columns]

# Create FUSION DataFrame
if 'FUSION_PVAL' in df.columns:
    fusion_cols = common_cols + ['FUSION_Z', 'FUSION_PVAL', 'FUSION_FDR']
    # Filter out columns that might not exist (e.g., if FUSION_Z wasn't calculated)
    fusion_cols = [col for col in fusion_cols if col in df.columns]
    df_fusion = df[fusion_cols].copy()
    # Sort by FUSION FDR
    print("Sorting FUSION results by FDR...")
    df_fusion.sort_values(by='FUSION_FDR', ascending=True, inplace=True, na_position='last')
    # Save FUSION results
    print(f"Saving sorted FUSION results to: {fusion_output_file}")
    df_fusion.to_csv(fusion_output_file, sep='\t', index=False, na_rep='NA')
else:
    df_fusion = pd.DataFrame() # Create empty DataFrame if no FUSION results
    print("No FUSION results found to save.")

# Create SPrediXcan DataFrame
if 'SPred_PVAL' in df.columns:
    spred_cols = common_cols + ['SPred_Z', 'SPred_PVAL', 'SPred_FDR']
    # Filter out columns that might not exist
    spred_cols = [col for col in spred_cols if col in df.columns]
    df_spred = df[spred_cols].copy()
    # Sort by SPred FDR
    print("Sorting SPrediXcan results by FDR...")
    df_spred.sort_values(by='SPred_FDR', ascending=True, inplace=True, na_position='last')
    # Save SPrediXcan results
    print(f"Saving sorted SPrediXcan results to: {spred_output_file}")
    df_spred.to_csv(spred_output_file, sep='\t', index=False, na_rep='NA')
else:
    df_spred = pd.DataFrame() # Create empty DataFrame if no SPred results
    print("No SPrediXcan results found to save.")

# --- Count Significant Results ---
print("Counting significant results...")
results_summary = []

if not df_fusion.empty:
    fusion_sig_fdr = df_fusion[df_fusion['FUSION_FDR'] < fdr_threshold].shape[0]
    fusion_sig_pval = df_fusion[df_fusion['FUSION_PVAL'] < pval_threshold].shape[0]
    results_summary.append(f"FUSION Results:")
    results_summary.append(f"  - Significant at FDR < {fdr_threshold}: {fusion_sig_fdr}")
    results_summary.append(f"  - Significant at P-value < {pval_threshold}: {fusion_sig_pval}")
    results_summary.append(f"  - Total FUSION results written: {df_fusion.shape[0]}")
else:
    results_summary.append(f"FUSION Results: Not available.")

if not df_spred.empty:
    spred_sig_fdr = df_spred[df_spred['SPred_FDR'] < fdr_threshold].shape[0]
    spred_sig_pval = df_spred[df_spred['SPred_PVAL'] < pval_threshold].shape[0]
    results_summary.append(f"\nSPrediXcan Results:")
    results_summary.append(f"  - Significant at FDR < {fdr_threshold}: {spred_sig_fdr}")
    results_summary.append(f"  - Significant at P-value < {pval_threshold}: {spred_sig_pval}")
    results_summary.append(f"  - Total SPrediXcan results written: {df_spred.shape[0]}")
else:
     results_summary.append(f"\nSPrediXcan Results: Not available.")


# --- Print and Save Summary ---
summary_text = "\n".join(results_summary)
print("\n--- Significance Summary ---")
print(summary_text)

try:
    with open(summary_output_file, 'w') as f:
        f.write(summary_text)
    print(f"\nSummary saved to: {summary_output_file}")
except Exception as e:
    print(f"Error saving summary file: {e}")

print("\nProcessing complete.")
