#!/usr/bin/env python

###################################################################
import argparse
import multiprocessing
import sys
import os # 
from time import time

import numpy as np
import pandas as pd

from scipy.stats import chi2
from natsort import natsorted # 

###############################################################
start_time = time()

###############################################################
parser = argparse.ArgumentParser(description=' 02 - Trans  TWAS')

parser.add_argument('--TIGAR_dir', type=str, required=True) # 

parser.add_argument('--gene_anno', type=str, dest='annot_path', required=True) # 

# parser.add_argument('--chr', type=str, dest='chrm')

parser.add_argument('--weight', type=str, dest='w_path', required=True) # 

parser.add_argument('--Zscore', type=str, dest='z_path', required=True) # 

parser.add_argument('--LD_pattern', type=str, dest='ld_pattern', required=True,
                    help="LD ， /path/to/LD/chr{chrom}.MCOV.gz。'{chrom}' 。")
parser.add_argument('--window',type=int, default=1000000, help="GeneID  (+/-) (: 1Mb)")

parser.add_argument('--weight_threshold',type=float, default=0.0) #  0

parser.add_argument('--test_stat', type=str, choices=['FUSION', 'SPrediXcan', 'both'], required=True) # 

parser.add_argument('--thread',type=int, default=1) #  1

parser.add_argument('--out_dir', type=str, required=True) # 

parser.add_argument('--out_twas_file', type=str, required=True) # 

parser.add_argument('--gtf', type=str, required=True) # 

args = parser.parse_args()



#sys.path.append(args.TIGAR_dir)
import utils as tg

def get_pval(z):
    return np.format_float_scientific(chi2.sf(z**2, 1), precision=15, exp_digits=0)



def get_z_denom(V, w):
    return np.sqrt(np.linalg.multi_dot([w, V, w]))

def get_spred_zscore(V_cov, w, Z_gwas, snp_sd):
    denom = np.sqrt(np.linalg.multi_dot([w, V_cov, w]))
    if denom == 0: return 0.0, 1.0 # 
    Z_twas = np.dot(snp_sd * w, Z_gwas) / denom
    Z_twas = np.sum(snp_sd * w * Z_gwas) / denom
    return Z_twas, get_pval(Z_twas)

def get_fusion_zscore(V_cor, w, Z_gwas, snp_sd=None): # FUSION  snp_sd
    denom = np.sqrt(np.linalg.multi_dot([w, V_cor, w]))
    if denom == 0: return 0.0, 1.0 # 
    Z_twas = np.dot(w, Z_gwas) / denom
    return Z_twas, get_pval(Z_twas)

def get_burden_zscore(test_stat, get_zscore_args):
    if test_stat =='FUSION':
        return get_fusion_zscore(*get_zscore_args)
    if test_stat == 'SPrediXcan':
        return get_spred_zscore(*get_zscore_args)

#############################################################

print('###############################################################')
print(' TWAS  (Trans-)')
print('###############################################################')

out_twas_path = os.path.join(args.out_dir, args.out_twas_file)

os.makedirs(args.out_dir, exist_ok=True)

print(f" TWAS : {out_twas_path}")

print({
    **args.__dict__,
    'test_stat_str': 'FUSION and SPrediXcan' if args.test_stat == 'both' else args.test_stat,
    'out_twas_path': out_twas_path,
})


print(' GeneID ...')
try:
    gene_annot_all = pd.read_csv(
        args.annot_path,
        sep='\t',
        usecols=['CHROM', 'GeneStart', 'GeneEnd', 'TargetID', 'GeneName'],
        dtype={'CHROM': object, 'GeneStart': np.int64, 'GeneEnd': np.int64, 'TargetID': object, 'GeneName': object}
    )
    gene_annot_all = tg.optimize_cols(gene_annot_all) # 


    TargetID = natsorted(gene_annot_all['TargetID'].unique())
    n_targets = len(TargetID)

    print(f' {gene_annot_all.shape[0]} 。')
    print(f' {n_targets}  TargetID 。')
    if n_targets == 0:
        raise ValueError(" TargetID。")

    gene_gtf_all = pd.read_csv(
        args.gtf,
        sep='\t',
        header=None,  # 
        names=['CHROM', 'GeneStart', 'GeneEnd', '_,', 'TargetID', 'GeneName', 'type'],  # 
        usecols=[0, 1, 2, 4, 5],  # （：CHROM, GeneStart, GeneEnd, TargetID, GeneName）
        dtype={'CHROM': object, 'GeneStart': np.int64, 'GeneEnd': np.int64, 'TargetID': object, 'GeneName': object}
    )
    gene_loc_dict = gene_gtf_all.set_index('TargetID')[['CHROM', 'GeneStart', 'GeneEnd']].to_dict('index')
    gene_info_dict = gene_gtf_all.set_index('TargetID')[['CHROM', 'GeneStart', 'GeneEnd', 'GeneName']].to_dict(
        'index')


except Exception as e:
    print(f": {args.annot_path}")
    print(e)
    sys.exit(1)


print(f': {args.w_path}...')
try:
    weight_cols = ['CHROM', 'POS', 'snpID', 'TargetID', 'GeneID', 'MAF', 'p_HWE', 'ES']
    weight_dtype = {'CHROM': object, 'POS': np.int64, 'snpID': object, 'TargetID': object, 'GeneID': object, 'MAF': np.float64, 'p_HWE': np.float64, 'ES': np.float64}

    weight_df_all = pd.read_csv(
        args.w_path,
        sep='\t', # 
        usecols=weight_cols,
        dtype=weight_dtype,
        low_memory=False # ，
    )
    weight_df_all = tg.optimize_cols(weight_df_all) # 

    if args.weight_threshold > 0:
         print(f" ES <= {args.weight_threshold} ...")
         initial_count = weight_df_all.shape[0]
         weight_df_all = weight_df_all[np.abs(weight_df_all['ES']) > args.weight_threshold].copy()
         print(f"   {initial_count - weight_df_all.shape[0]} 。: {weight_df_all.shape[0]}")


    print(" TargetID ...")
    weights_by_targetid = {
        target: group_df.reset_index(drop=True)
        for target, group_df in weight_df_all.groupby('TargetID')
        if not group_df.empty # 
    }
    print(f" {len(weights_by_targetid)}  TargetID  ()。")

    target_ids_in_weights = set(weights_by_targetid.keys())
    original_target_count = len(TargetID)
    TargetID = natsorted([tid for tid in TargetID if tid in target_ids_in_weights])
    n_targets = len(TargetID)
    print(f" TargetID :  {n_targets} 。")
    if n_targets == 0:
        print(":  TargetID。")

except Exception as e:
    print(f": {args.w_path}")
    print(e)
    sys.exit(1)


print(' Z-score ...')
try:
    zscore_header = tg.get_header(args.z_path, zipped = True) #  tabix 
    zscore_info = tg.get_cols_dtype(
        zscore_header,
		cols=['CHROM','POS','REF','ALT','Zscore'], #  zscore 
		ind_namekey=True) #  dtype 
    zscore_info['path'] = args.z_path #  read_tabix 


except Exception as e:
    print(f" Z-score : {args.z_path}")
    print(e)
    sys.exit(1)

print(f': {out_twas_path}' )
out_cols = ['CHROM','GeneStart','GeneEnd','TargetID','GeneName','n_snps', 'used_regions'] #  used_regions 
if args.test_stat == 'both':
    out_cols += ['FUSION_Z','FUSION_PVAL','SPred_Z','SPred_PVAL']
else:
    test_stat_prefix = 'FUSION' if args.test_stat == 'FUSION' else 'SPred' # 
    out_cols += [f'{test_stat_prefix}_Z', f'{test_stat_prefix}_PVAL'] # 

pd.DataFrame(columns=out_cols).to_csv(
	out_twas_path,
	sep='\t',
	index=None,
	header=True,
	mode='w') # 'w' 

@tg.error_handler #  TIGARutils 
def thread_process(num):
    target = TargetID[num]
    print(f'\n---  {num+1}/{n_targets}: {target} ---')

    if target not in weights_by_targetid:
        print(f" TargetID {target} 。。")
        return None
    target_weights_df = weights_by_targetid[target]

    if target_weights_df.empty:
        print(f"{target}  DataFrame  ()。。")
        return None

    print(f"   {target}  {target_weights_df.shape[0]} 。")

    print("   GeneID  SNP ...")
    unique_geneids = target_weights_df['GeneID'].unique()
    regions_unmerged = []
    missing_geneids = []
    for gene_id in unique_geneids:
        if gene_id in gene_loc_dict:
            loc = gene_loc_dict[gene_id]
            chrom = loc['CHROM']
            start = max(0, loc['GeneStart'] - args.window) # 0
            end = loc['GeneEnd'] + args.window
            regions_unmerged.append((str(chrom), int(start), int(end)))
        else:
            missing_geneids.append(gene_id)

    if missing_geneids:
        print(f"  :  {len(missing_geneids)}  GeneID : {', '.join(map(str, missing_geneids[:5]))}{'...' if len(missing_geneids)>5 else ''}")

    if not regions_unmerged:
        print(f"   {target}  SNP 。。")
        return None

    merged_regions = tg.merge_regions(regions_unmerged) # 
    merged_regions_str_list = [f"{r[0]}:{r[1]}-{r[2]}" for r in merged_regions] # 
    print(f"   {len(merged_regions)} : {', '.join(merged_regions_str_list)}")

    print("   GWAS Z-scores...")
    all_zscore_dfs = []
    for region_chrom, region_start, region_end in merged_regions:
        try:
            z_df_region = tg.read_tabix(
                start=str(region_start),
                end=str(region_end),
                chrm=str(region_chrom),
                sampleID=[],
                return_empty_df=True, # ，
                raise_error=False, # ， df
                **zscore_info #  zscore_info['path'] 
            )
            if not z_df_region.empty:
                all_zscore_dfs.append(z_df_region)
        except tg.NoTargetDataError: # （ raise_error=False ）
             print(f"   {region_chrom}:{region_start}-{region_end}  Z-score ")
             continue # 
        except Exception as e:
            print(f"   {region_chrom}:{region_start}-{region_end}  Z-scores : {e}")
            continue # 

    if not all_zscore_dfs:
        print(f"   {target}  GWAS Z-score 。。")
        return None

    Zscore_df = pd.concat(all_zscore_dfs, ignore_index=True)
    Zscore_df = Zscore_df.drop_duplicates(subset=['snpID'], keep='first').reset_index(drop=True)
    print(f"   {Zscore_df.shape[0]}  Z-scores。")

    print("   Z-scores...")
    Zscore_df['snpIDflip'] = tg.flip_snpIDs(Zscore_df['snpID'].values) # 

    print("   Z-scores (/)...")
    Zscore_df['snpIDflip'] = tg.flip_snpIDs(Zscore_df['snpID'].values)  # 
    Zscore_df['snpIDcomp'] = Zscore_df['snpID'].apply(tg.get_complement_snpID_2)
    Zscore_df['snpIDcompflip'] = tg.flip_snpIDs(Zscore_df['snpIDcomp'].values)

    target_weights_snpids = set(target_weights_df['snpID'])  #  SNP ID 

    z_indices_potential_match = Zscore_df[
        Zscore_df['snpID'].isin(target_weights_snpids) |
        Zscore_df['snpIDflip'].isin(target_weights_snpids) |
        Zscore_df['snpIDcomp'].isin(target_weights_snpids) |
        Zscore_df['snpIDcompflip'].isin(target_weights_snpids)
        ].index

    if z_indices_potential_match.empty:
        print(f'   {target}  GWAS Z-scores  SNP ()。。')
        return None

    Zscore_filtered = Zscore_df.loc[z_indices_potential_match].copy()
    print(f"   {Zscore_filtered.shape[0]}  Z-score  SNP 。")

    flip_multipliers = {}  # : Z-score  ->  (1  -1)
    zscore_id_to_use = {}  # : Z-score  ->  ** snpID
    final_weight_snps_matched = set()  #  SNP 

    for idx, row in Zscore_filtered.iterrows():
        matched_in_this_row = False  #  Z-score 

        if row['snpID'] in target_weights_snpids:
            flip_multipliers[idx] = 1
            zscore_id_to_use[idx] = row['snpID']
            final_weight_snps_matched.add(row['snpID'])
            matched_in_this_row = True
            # print(f"  Match Direct: Z-idx {idx} ({row['snpID']}) -> Wgt {row['snpID']} (mult=1)") # Debug

        if not matched_in_this_row and row['snpIDcompflip'] in target_weights_snpids:

            flip_multipliers[idx] = 1  # Same effect direction
            zscore_id_to_use[idx] = row['snpIDcompflip']  # Use the matched weight SNP ID
            final_weight_snps_matched.add(row['snpIDcompflip'])
            matched_in_this_row = True
            # print(f"  Match CompFlip: Z-idx {idx} ({row['snpIDcompflip']}) -> Wgt {row['snpIDcompflip']} (mult=1)") # Debug

        if not matched_in_this_row and row['snpIDflip'] in target_weights_snpids:
            flip_multipliers[idx] = -1  # Opposite effect direction
            zscore_id_to_use[idx] = row['snpIDflip']  # Use the matched weight SNP ID
            final_weight_snps_matched.add(row['snpIDflip'])
            matched_in_this_row = True
            # print(f"  Match Flip: Z-idx {idx} ({row['snpIDflip']}) -> Wgt {row['snpIDflip']} (mult=-1)") # Debug

        if not matched_in_this_row and row['snpIDcomp'] in target_weights_snpids:
            flip_multipliers[idx] = -1  # Opposite effect direction
            zscore_id_to_use[idx] = row['snpIDcomp']  # Use the matched weight SNP ID
            final_weight_snps_matched.add(row['snpIDcomp'])
            matched_in_this_row = True
            print(f"  Match Comp: Z-idx {idx} ({row['snpIDcomp']}) -> Wgt {row['snpIDcomp']} (mult=-1)") # Debug

        if not matched_in_this_row: # ， Zscore_filtered 
            print(f"  : Z-score  {idx} (SNP: {row['snpID']})  SNP (?)")


    if not zscore_id_to_use:  # 
        print(f"   {target} ， Z-score  SNP。。")
        return None

    ZW_filtered_weights = target_weights_df[target_weights_df['snpID'].isin(final_weight_snps_matched)].copy()
    if ZW_filtered_weights.empty:
        print(f"   SNP  Z-score 。 {target}.")
        return None

    indices_to_keep = list(zscore_id_to_use.keys())
    Zscore_matched = Zscore_df.loc[indices_to_keep].copy()

    Zscore_matched['snpID_final'] = Zscore_matched.index.map(zscore_id_to_use.get)
    Zscore_matched['Zscore_aligned'] = Zscore_matched['Zscore'] * Zscore_matched.index.map(flip_multipliers.get)

    Zscore_aligned = Zscore_matched[['snpID_final', 'Zscore_aligned']].rename(
        columns={'snpID_final': 'snpID', 'Zscore_aligned': 'Zscore'})

    Zscore_aligned = Zscore_aligned.drop_duplicates(subset=['snpID'], keep='first')

    ZW = ZW_filtered_weights.merge(Zscore_aligned, on='snpID', how='inner')  # 

    if ZW.empty:
        print(f"   {target}  Z-scores  SNP。。")
        return None

    print(f"   {ZW.shape[0]}  Z-scores  SNP。")

    snp_search_ids = natsorted(ZW['snpID'].unique())

    ZW = ZW[ZW['snpID'].isin(snp_search_ids)].drop_duplicates(subset=['snpID'], keep='first')
    ZW = ZW.set_index('snpID').loc[snp_search_ids].reset_index()

    if ZW.empty:
         print(f"   {target}  Z-scores  SNP。。")
         return None

    print(f"   {ZW.shape[0]}  Z-scores  SNP。")

    print(f"   {len(snp_search_ids)}  SNP  LD  ( strand/allele )...")

    print("     SNP ...")
    query_snps_expanded = set(snp_search_ids) #  SNP 
    temp_pos_map = {} # 
    snps_to_generate_forms = list(snp_search_ids) # 

    for snp_id in snps_to_generate_forms:
        try:
            flipped = tg.flip_snpIDs([snp_id])[0]
            comp = tg.get_complement_snpID(snp_id)
            if comp is None:
                print(snp_id)
            compflip = tg.flip_snpIDs([comp])[0] if comp else None

            if flipped: query_snps_expanded.add(flipped)
            if comp: query_snps_expanded.add(comp)
            if compflip: query_snps_expanded.add(compflip)

            parts = snp_id.split(':')
            if len(parts) >= 2: temp_pos_map[snp_id] = int(parts[1])

        except Exception as e:
            print(f"      :  SNP ID '{snp_id}' : {e}")

    query_snps_expanded_list = natsorted(list(query_snps_expanded))
    print(f"     {len(query_snps_expanded_list)}  SNP ID 。")

    print("     SNP ...")
    from collections import defaultdict
    region_to_expanded_snps_map = defaultdict(list)
    unassigned_expanded_snps = []
    expanded_pos_map = {}
    for snp_id in query_snps_expanded_list:
         try:
             parts = snp_id.split(':')
             if len(parts) >= 2: expanded_pos_map[snp_id] = int(parts[1])
             else: raise ValueError("")
         except (ValueError, IndexError):
              unassigned_expanded_snps.append(snp_id) # 
              continue

         assigned = False
         for r_chrom, r_start, r_end in merged_regions:
             snp_chrom = snp_id.split(':')[0]
             snp_pos = expanded_pos_map[snp_id]
             if snp_chrom == str(r_chrom) and int(r_start) <= snp_pos <= int(r_end):
                 region_key = f"{r_chrom}:{r_start}-{r_end}"
                 region_to_expanded_snps_map[region_key].append(snp_id)
                 assigned = True
                 break
         if not assigned:
             unassigned_expanded_snps.append(snp_id)

    if unassigned_expanded_snps:
         pass # print(f"      : {len(unassigned_expanded_snps)}  SNP 。")
    print(f"       SNP  {len(region_to_expanded_snps_map)}  LD 。")


    all_mcov_returned_dfs = [] #  get_ld_data  MCOV 
    found_ld_files = {}
    processed_regions_count = 0

    print("     LD  ( SNP )...")
    if not region_to_expanded_snps_map:
        print(f"       SNP 。 LD 。")
        return None

    for region_key, snps_in_region_expanded in region_to_expanded_snps_map.items():
        processed_regions_count += 1
        print(f"       {processed_regions_count}/{len(region_to_expanded_snps_map)}: {region_key} ( {len(snps_in_region_expanded)}  SNP)")

        if not snps_in_region_expanded: continue

        try: chrom = region_key.split(':')[0]
        except IndexError: print(f"         '{region_key}' 。。"); continue

        ld_file_path = args.ld_pattern.format(chrom=chrom)
        if chrom not in found_ld_files:
            if not os.path.exists(ld_file_path): print(f"        :  {chrom}  LD 。。"); continue
            if not os.path.exists(ld_file_path + ".tbi"): print(f"        :  {chrom}  LD 。。"); continue
            print(f"         LD : {ld_file_path}")
            found_ld_files[chrom] = ld_file_path
        else: ld_file_path = found_ld_files[chrom]

        try:
            snps_in_region_expanded_sorted = sorted(snps_in_region_expanded, key=lambda snp: expanded_pos_map.get(snp, float('inf')))
        except Exception as e: print(f"         SNP : {e}。。"); continue

        try:
            mcov_df_region_returned = tg.get_ld_data(ld_file_path, snps_in_region_expanded_sorted)
            if not mcov_df_region_returned.empty:
                if not all(col in mcov_df_region_returned.columns for col in ['snpID', 'COV']) or mcov_df_region_returned.index.name != 'row':
                     print(f"        :  {region_key}  LD 。")
                     continue
                all_mcov_returned_dfs.append(mcov_df_region_returned)
                print(f"         {region_key}  {mcov_df_region_returned.shape[0]}  LD 。")

        except tg.NoTargetDataError: pass # 
        except Exception as e: print(f"         {region_key} LD : {e}")


    if not all_mcov_returned_dfs:
         print(f"   LD  SNP 。 {target}.")
         return None

    MCOV_returned_all = pd.concat(all_mcov_returned_dfs)
    MCOV_returned_all = MCOV_returned_all.loc[~MCOV_returned_all.index.duplicated(keep='first')]
    MCOV_returned_all = MCOV_returned_all.drop_duplicates(subset=['snpID'], keep='first') # 
    print(f"  ， LD  {MCOV_returned_all.shape[0]}  LD 。")

    if MCOV_returned_all.empty:
         print(f"   LD 。 {target}.")
         return None

    print("  :  SNP  LD ...")

    MCOV_returned_all['snpIDflip'] = tg.flip_snpIDs(MCOV_returned_all['snpID'].values)
    MCOV_returned_all['snpIDcomp'] = MCOV_returned_all['snpID'].apply(tg.get_complement_snpID)
    MCOV_returned_all['snpIDcompflip'] = tg.flip_snpIDs(MCOV_returned_all['snpIDcomp'].values)

    target_snp_to_mcov_idx = {}
    target_snp_to_mcov_snp = {}
    mcov_indices_to_keep = set()

    mcov_lookup = {}
    mcov_lookup['direct'] = pd.Series(MCOV_returned_all.index, index=MCOV_returned_all['snpID']).to_dict()
    mcov_lookup['flip'] = pd.Series(MCOV_returned_all.index, index=MCOV_returned_all['snpIDflip']).to_dict()
    mcov_lookup['comp'] = pd.Series(MCOV_returned_all.index, index=MCOV_returned_all['snpIDcomp']).to_dict()
    mcov_lookup['compflip'] = pd.Series(MCOV_returned_all.index, index=MCOV_returned_all['snpIDcompflip']).to_dict()

    target_snp_to_mcov_idx={}
    for target_snp in snp_search_ids:
        matched_mcov_idx = -1 # 
        matched_mcov_snp = None

        if target_snp in mcov_lookup['direct']:
            matched_mcov_idx = mcov_lookup['direct'][target_snp]
            matched_mcov_snp = target_snp # MCOV snpID is the same

        elif target_snp in mcov_lookup['compflip']:
             matched_mcov_idx = mcov_lookup['compflip'][target_snp]
             matched_mcov_snp = MCOV_returned_all.loc[matched_mcov_idx, 'snpID']

        elif target_snp in mcov_lookup['flip']:
             matched_mcov_idx = mcov_lookup['flip'][target_snp]
             matched_mcov_snp = MCOV_returned_all.loc[matched_mcov_idx, 'snpID']


        elif target_snp in mcov_lookup['comp']:
             matched_mcov_idx = mcov_lookup['comp'][target_snp]
             matched_mcov_snp = MCOV_returned_all.loc[matched_mcov_idx, 'snpID']

        if matched_mcov_idx != -1:
             if matched_mcov_idx not in mcov_indices_to_keep:
                 target_snp_to_mcov_idx[target_snp] = matched_mcov_idx
                 target_snp_to_mcov_snp[target_snp] = matched_mcov_snp
                 mcov_indices_to_keep.add(matched_mcov_idx)
             else: #  MCOV ， target_snp  MCOV 
                  print(f"    : MCOV  {matched_mcov_idx} ，target SNP {target_snp} 。")

    snps_with_ld = natsorted(list(target_snp_to_mcov_idx.keys()))
    n_snps_final = len(snps_with_ld)

    if n_snps_final == 0:
        print(f"   SNP  LD 。 {target}.")
        return None

    original_zw_count = ZW.shape[0]
    ZW = ZW[ZW['snpID'].isin(snps_with_ld)].copy()
    if ZW.empty:
         print(f"   LD  ZW 。 {target}.")
         return None
    ZW = ZW.set_index('snpID').loc[snps_with_ld].reset_index() # ZW is final and sorted by target SNPs
    print(f"   {n_snps_final}  SNP ( {original_zw_count}  SNP  LD)  TWAS。")

    final_mcov_indices = list(target_snp_to_mcov_idx.values())
    MCOV_final = MCOV_returned_all.loc[final_mcov_indices].copy()


    MCOV_final = MCOV_final.sort_index()

    mcov_snps_order = MCOV_final['snpID'].tolist()

    print("   TWAS  ( LD)...")

    w_final_by_target_snp = ZW.set_index('snpID')['ES'] #  target_snp  Series
    z_gwas_final = ZW['Zscore'].values #  snps_with_ld

    results_dict = {}
    region_ld_matrices = {} # : {'region_key': {'V_cov': V, 'snp_sd': sd, 'snps_order': order}}

    print("     LD ...")
    try:
        MCOV_final['CHROM'] = MCOV_final['snpID'].map(lambda x: x.split(':')[0])
        mcov_blocks = {} #  MCOV 

        processed_snps_in_matrices = set()
        block_ld_results = {} # 

        for chrom, mcov_block_grp in MCOV_final.groupby('CHROM'):
             print(f"       LD  ( {chrom})...")
             mcov_block_sorted = mcov_block_grp.sort_index() #  'row' 
             if mcov_block_sorted.empty: continue

             try:
                 snp_sd_block, V_cov_block, _ = tg.get_ld_matrix(mcov_block_sorted, return_diag=True)
                 snps_order_in_block = mcov_block_sorted.drop_duplicates(subset=['snpID'])['snpID'].tolist() # LD  SNP ID 

                 block_ld_results[chrom] = {
                     'V_cov': V_cov_block,
                     'snp_sd': snp_sd_block,
                     'snps_order': snps_order_in_block #  LD  snpID
                 }
                 processed_snps_in_matrices.update(snps_order_in_block)
                 print(f" {chrom}  LD  ({len(snps_order_in_block)} SNPs)。")

             except Exception as e:
                 print(f" {chrom}  LD : {e}")
                 continue

        mcov_snps_in_target_map = set(target_snp_to_mcov_snp.values())
        missing_snps = mcov_snps_in_target_map - processed_snps_in_matrices
        if missing_snps:
             print(f"    : {len(missing_snps)}  MCOV SNP  LD 。TWAS 。")

    except Exception as e:
         print(f"     MCOV_final  LD : {e}。 TWAS 。")
         return None

    if args.test_stat == 'SPrediXcan' or args.test_stat == 'both':
        print("     SPrediXcan  ( LD )...")
        denominator_sq_spred = 0.0
        snp_sd_map = {} #  SD， target_snpID ( ZW)
        processed_target_snps_for_sd = set()

        for chrom, block_data in block_ld_results.items():
            snps_order_ld = block_data['snps_order'] # LD  snpID 
            V_cov_block = block_data['V_cov']
            snp_sd_block = block_data['snp_sd']

            target_snps_in_block = []
            mcov_snp_to_target_snp = {v: k for k, v in target_snp_to_mcov_snp.items()} # 

            for ld_snp in snps_order_ld:
                if ld_snp in mcov_snp_to_target_snp:
                    target_snps_in_block.append(mcov_snp_to_target_snp[ld_snp])
                else:
                    print(f"      : SPred  {chrom}  LD SNP {ld_snp}  SNP。")
                    target_snps_in_block = None # 
                    break

            if target_snps_in_block is None or len(target_snps_in_block) != len(snps_order_ld):
                 print(f"      SPred:  {chrom}  LD SNP  SNP。。")
                 continue

            try:
                 w_block_by_target = w_final_by_target_snp[target_snps_in_block]
                 w_block = w_block_by_target.values
            except KeyError as e:
                 print(f"      SPred:  {chrom}  (Target SNP: {e})。。")
                 continue

            try:
                denom_contrib_sq = np.linalg.multi_dot([w_block, V_cov_block, w_block])
                if denom_contrib_sq < 0: denom_contrib_sq = 0.0
                denominator_sq_spred += denom_contrib_sq
            except Exception as e:
                 print(f"      SPred:  {chrom} : {e}。。")
                 continue

            for i, target_snp in enumerate(target_snps_in_block):
                if target_snp not in processed_target_snps_for_sd:
                     if i < len(snp_sd_block):
                         snp_sd_map[target_snp] = snp_sd_block[i]
                         processed_target_snps_for_sd.add(target_snp)
                     else:
                          print(f"        : SPred  {chrom} target SNP {target_snp}  SD 。")

        if len(processed_target_snps_for_sd) != n_snps_final:
            print(f"    : SPred  SD  SNP  ({len(processed_target_snps_for_sd)})  SNP  ({n_snps_final}) 。。")
            results_dict['SPred_Z'], results_dict['SPred_PVAL'] = np.nan, np.nan
        else:
            snp_sd_final = np.array([snp_sd_map[snp_id] for snp_id in snps_with_ld])

            numerator_spred = np.sum(ZW['ES'].values * snp_sd_final * ZW['Zscore'].values)

            twas_denominator_spred = np.sqrt(denominator_sq_spred) if denominator_sq_spred > 1e-10 else 0
            if twas_denominator_spred > 0:
                Z_spred = numerator_spred / twas_denominator_spred
                P_spred = get_pval(Z_spred)
            else:
                Z_spred, P_spred = np.nan, np.nan
            results_dict['SPred_Z'] = Z_spred; results_dict['SPred_PVAL'] = P_spred
            print(f"    SPrediXcan (): Z={Z_spred:.4f}, P={P_spred}")


    if args.test_stat == 'FUSION' or args.test_stat == 'both':
        print("     FUSION  ( LD )...")
        denominator_sq_fusion = 0.0

        for chrom, block_data in block_ld_results.items():
            snps_order_ld = block_data['snps_order']
            V_cov_block = block_data['V_cov']
            snp_sd_block = block_data['snp_sd']

            target_snps_in_block = []
            mcov_snp_to_target_snp = {v: k for k, v in target_snp_to_mcov_snp.items()}
            for ld_snp in snps_order_ld:
                if ld_snp in mcov_snp_to_target_snp: target_snps_in_block.append(mcov_snp_to_target_snp[ld_snp])
                else: target_snps_in_block = None; break # 
            if target_snps_in_block is None or len(target_snps_in_block) != len(snps_order_ld):
                print(f"      FUSION:  {chrom}  LD SNP  SNP。。")
                continue

            try:
                 w_block = w_final_by_target_snp[target_snps_in_block].values
            except KeyError as e:
                 print(f"      FUSION:  {chrom}  (Target SNP: {e})。。")
                 continue

            try:
                snp_sd_inv = np.zeros_like(snp_sd_block)
                valid_sd = snp_sd_block > 1e-6
                snp_sd_inv[valid_sd] = 1.0 / snp_sd_block[valid_sd]
                D_inv = np.diag(snp_sd_inv)
                V_cor_block = np.linalg.multi_dot([D_inv, V_cov_block, D_inv])
                np.fill_diagonal(V_cor_block, 1.0)
                V_cor_block = np.clip(V_cor_block, -1.0, 1.0)
            except Exception as e:
                print(f"      FUSION:  {chrom}  V_cor : {e}。。")
                continue

            try:
                denom_contrib_sq = np.linalg.multi_dot([w_block, V_cor_block, w_block])
                if denom_contrib_sq < 0: denom_contrib_sq = 0.0
                denominator_sq_fusion += denom_contrib_sq
            except Exception as e:
                print(f"      FUSION:  {chrom} : {e}。。")
                continue

        numerator_fusion = np.dot(ZW['ES'].values, ZW['Zscore'].values)

        twas_denominator_fusion = np.sqrt(denominator_sq_fusion) if denominator_sq_fusion > 1e-10 else 0
        if twas_denominator_fusion > 0:
            Z_fusion = numerator_fusion / twas_denominator_fusion
            P_fusion = get_pval(Z_fusion)
        else:
            Z_fusion, P_fusion = np.nan, np.nan
        results_dict['FUSION_Z'] = Z_fusion; results_dict['FUSION_PVAL'] = P_fusion
        print(f"    FUSION (): Z={Z_fusion:.4f}, P={P_fusion}")


    base_info = gene_info_dict.get(target, {'CHROM': 'NA', 'GeneStart': 0, 'GeneEnd': 0, 'GeneName': 'NA'}) # 
    output_df = pd.DataFrame({
        'CHROM': [base_info['CHROM']],
        'GeneStart': [base_info['GeneStart']],
        'GeneEnd': [base_info['GeneEnd']],
        'TargetID': [target],
        'GeneName': [base_info['GeneName']],
        'n_snps': [n_snps_final], #  SNP 
        'used_regions': [";".join(merged_regions_str_list)] # 
    })

    if args.test_stat == 'both':
        output_df['FUSION_Z'] = results_dict.get('FUSION_Z', np.nan) #  .get ， NaN
        output_df['FUSION_PVAL'] = results_dict.get('FUSION_PVAL', np.nan)
        output_df['SPred_Z'] = results_dict.get('SPred_Z', np.nan)
        output_df['SPred_PVAL'] = results_dict.get('SPred_PVAL', np.nan)
    elif args.test_stat == 'FUSION':
        output_df['FUSION_Z'] = results_dict.get('FUSION_Z', np.nan)
        output_df['FUSION_PVAL'] = results_dict.get('FUSION_PVAL', np.nan)
    elif args.test_stat == 'SPrediXcan':
        output_df['SPred_Z'] = results_dict.get('SPred_Z', np.nan)
        output_df['SPred_PVAL'] = results_dict.get('SPred_PVAL', np.nan)

    output_df.to_csv(out_twas_path, sep='\t', index=None, header=None, mode='a', na_rep='NA')


    print(f'---  {target} TWAS 。 ---')
    return # 

###############################################################
if __name__ == '__main__':
    if n_targets > 0:
        print(f'\n {n_targets}  TWAS ， {args.thread} 。')
        pool = multiprocessing.Pool(args.thread)
        pool.imap_unordered(thread_process, range(n_targets))
        pool.close() #  Pool，
        pool.join() # 
        print('\n。')
    else:
        print('\n。')

###############################################################
elapsed_sec = time()-start_time
elapsed_time = tg.format_elapsed_time(elapsed_sec) # 
print('\n (:::): ' + elapsed_time)
print('###############################################################')
