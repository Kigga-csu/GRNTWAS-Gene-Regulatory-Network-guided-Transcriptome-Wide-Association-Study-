import numpy as np
import pandas as pd
import GRNutils as tg
import operator
import os
from sklearn.preprocessing import StandardScaler
import subprocess
from sklearn.model_selection import KFold
import warnings
import model
from scipy.stats import pearsonr
from sklearn.linear_model import LassoCV
import seaborn as sns
import matplotlib.pyplot as plt

param_combinations = [
    {'n_TG': 300, 'h_cis': 0.0375, 'h_trans': 0.0125, 'h_tf': 0.1},
    {'n_TG': 300, 'h_cis': 0.0375, 'h_trans': 0.075, 'h_tf': 0.2},
    {'n_TG': 300, 'h_cis': 0.0375, 'h_trans': 0.0375, 'h_tf': 0.1},
    {'n_TG': 300, 'h_cis': 0.05, 'h_trans': 0.01, 'h_tf': 0.1},
    {'n_TG': 300, 'h_cis': 0.05, 'h_trans': 0.05, 'h_tf': 0.1},
    {'n_TG': 300, 'h_cis': 0.05, 'h_trans': 0.2, 'h_tf': 0.3},
    {'n_TG': 300, 'h_cis': 0.075, 'h_trans': 0.025, 'h_tf': 0.1},
    {'n_TG': 300, 'h_cis': 0.075, 'h_trans': 0.075, 'h_tf': 0.2},
    {'n_TG': 300, 'h_cis': 0.075, 'h_trans': 0.3, 'h_tf': 0.5},
    {'n_TG': 300, 'h_cis': 0.1, 'h_trans': 0.03, 'h_tf': 0.2},
    {'n_TG': 300, 'h_cis': 0.1, 'h_trans': 0.1, 'h_tf': 0.3},
    {'n_TG': 300, 'h_cis': 0.1, 'h_trans': 0.3, 'h_tf': 0.6},
    {'n_TG': 300, 'h_cis': 0.2, 'h_trans': 0.05, 'h_tf': 0.2},
    {'n_TG': 300, 'h_cis': 0.2, 'h_trans': 0.1, 'h_tf': 0.3},
    {'n_TG': 300, 'h_cis': 0.2, 'h_trans': 0.2, 'h_tf': 0.3},
    {'n_TG': 300, 'h_cis': 0.2, 'h_trans': 0.4, 'h_tf': 0.8},
    {'n_TG': 300, 'h_cis': 0.2, 'h_trans': 0.6, 'h_tf': 0.3},
    {'n_TG': 300, 'h_cis': 0.3, 'h_trans': 0.1, 'h_tf': 0.5},
    {'n_TG': 300, 'h_cis': 0.3, 'h_trans': 0.3, 'h_tf': 0.5}
    ]


# 参数设置（可调整）
class SimulationParams:
    def __init__(self, n_TG=500, n_TF_class1=10, n_TF_class2=5, real_TF_class1=3, real_TF_class2=2,
                 p_eqtl_cis=0.1, h_cis=0.2, h_trans=0.1, h_tf=0.1, h_t=0.3, windows=5000):
        self.n_TG = n_TG
        self.n_TF_class1 = n_TF_class1
        self.n_TF_class2 = n_TF_class2
        self.real_TF_class1 = real_TF_class1
        self.real_TF_class2 = real_TF_class2
        self.p_eqtl_cis = p_eqtl_cis
        self.h_cis = h_cis
        self.h_trans = h_trans
        self.h_tf = h_tf
        self.h2 = h_cis + h_trans
        self.h_U_tf = max(1 - self.h_tf, 0) * 0.8
        self.h_U_tg = max(1 - self.h_cis - (self.h_trans / self.h_tf), 0) * 0.8
        # Ensure h_E_tf and h_E_tg are non-negative
        self.h_E_tf = max(1 - self.h_tf, 0) * 0.2
        self.h_E_tg = max(1 - self.h_cis - (self.h_trans / self.h_tf), 0) * 0.2
        self.windows = windows
        self.h_t = h_t


# 文件路径设置
GENO_PATH = '/data/lab/wangshixian/data/genotype/1000G/1000G_Phase3_30X/'
BED_PATH = "/data/lab/wangshixian/GRNTWAS_STAR/data/anno_info/gene.bed"
SAMPLE_PATH = "/data/lab/wangshixian/GRNTWAS_STAR/simulation/GRNTWAS_Simulation/sample_names/EUR.txt"
TF_INFO_PATH = "/data/lab/wangshixian/data/GRN/TFlist_human/Homo_sapiens_TF.txt"
BASE_OUTPUT_DIR = "/data/lab/wangshixian/GRNTWAS_STAR/simulation/GRNTWAS_Simulation/simulation_expression_1.2_compare_other_TWAS/p_casual_0.1"
DPR_path = "/data/lab/wangshixian/TIGAR-master/Model_Train_Pred/DPR"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# 数据预处理参数
MAF_THRESHOLD = 0.01
MISSING_RATE_THRESHOLD = 0.15
HWE_THRESHOLD = 0.05


def gene_info_4_id(bed_file_path, gene_ids, windows):
    df = pd.read_csv(bed_file_path, sep='\t', header=None,
                     names=['chrom', 'start', 'end', 'strand', 'gene_id', 'gene_name', 'gene_type'])
    df_filtered = df[df['gene_id'].isin(gene_ids)]
    chrom = df_filtered['chrom'].values
    gene_names = df_filtered['gene_name'].values
    GeneStart = df_filtered['start'].values
    GeneEnd = df_filtered['end'].values
    start = max(int(GeneStart[0]) - windows, 0)  # 假设单基因，避免多基因问题
    end = int(GeneEnd[0]) + windows
    return gene_names, start, end, chrom


def extract_genotype_vcf(genofile_type="vcf", data_format="GT", bed_file=None, geno_path=None,
                         gene_ID=None, sampleid_path=None, windows=None):
    target_name, start, end, chrom = gene_info_4_id(bed_file, [gene_ID], windows)
    try:
        chrom_int = int(chrom[0].replace('chr', ''))  # 处理可能的 'chr' 前缀
        if chrom_int not in range(1, 23):
            print(f"{target_name} not have genotype")
            return None, None
    except ValueError:
        print(f"{target_name} not have genotype")
        return None, None
    genofile_path = f'{geno_path}CCDG_14151_B01_GRM_WGS_2020-08-05_chr{chrom[0]}.filtered.shapeit2-duohmm-phased.vcf.gz'
    sampleID, sample_size, geno_info = tg.sampleid_startup_simulation(
        chrm=chrom[0], genofile_type=genofile_type, data_format=data_format,
        geno_path=genofile_path, sampleid_path=sampleid_path
    )
    if not isinstance(sampleID, np.ndarray):
        print(f"Failed to get sample IDs for {target_name}")
        return None, None
    df_Geno = tg.read_tabix_revise(str(start), str(end), sampleID, **geno_info)
    if df_Geno is False or df_Geno.empty:
        print(f"No genotype data for {target_name}")
        return None, None
    df_Geno.drop_duplicates(subset='snpID', keep='first', inplace=True)
    df_Geno = tg.handle_missing_wsx(df_Geno, sampleID, MISSING_RATE_THRESHOLD)
    if df_Geno.empty:
        return None, None
    df_Geno = tg.calc_maf(df_Geno, sampleID, MAF_THRESHOLD, op=operator.ge)
    if df_Geno.empty:
        return None, None
    df_Geno = tg.calc_p_hwe(df_Geno, sampleID, HWE_THRESHOLD, op=operator.ge)
    
    # 如果 SNP 数量超过 100，随机采样 100 个 SNP
    if len(df_Geno) > 100:
        df_Geno = df_Geno.sample(n=100, random_state=42)  # 可设定 random_state 保证可重复性

    if df_Geno.empty:
        return None, None
    print("预处理结束")
    return df_Geno, sampleID


def simulate_gene_expression(snpsMat, n_qtl, p_causal, h_component):
    print("构建表达")
    n_qtls = max(1, int(np.floor(p_causal * n_qtl)))
    c_qtls = np.random.choice(n_qtl, size=n_qtls, replace=False)
    b_qtls = np.zeros(n_qtl)
    b_qtls[c_qtls] = np.random.normal(loc=0, scale=np.sqrt(h_component / n_qtls), size=n_qtls)
    gexpr_genetics = np.dot(snpsMat, b_qtls)
    return gexpr_genetics, b_qtls


def simulate_corr(dimension, low=-0.1, high=0.1):
    # 生成随机矩阵
    mat = np.random.uniform(low, high, size=(dimension, dimension))
    # 对称化
    mat = (mat + mat.T) / 2
    # 对角线设为 1
    np.fill_diagonal(mat, 1)
    # 归一化到最大值
    mat = mat / np.max(np.abs(mat))
    return mat


def simulate_TF_expression(tf_gene_id, params):
    print(f"模拟TF: {tf_gene_id}")
    geno, sampleID = extract_genotype_vcf(
        bed_file=BED_PATH, geno_path=GENO_PATH, gene_ID=tf_gene_id,
        sampleid_path=SAMPLE_PATH, windows=params.windows
    )
    if geno is None:
        print(f"No SNP data for TF {tf_gene_id}")
        with open(SAMPLE_PATH, 'r') as f:
            sampleID = np.array([line.strip() for line in f])
        n_samples = len(sampleID)
        gexpr_tot = np.random.normal(loc=0, scale=np.sqrt(params.h_U_tf + params.h_E_tf), size=n_samples)
        weights_df = pd.DataFrame(columns=['CHROM', 'POS', 'snpID', 'REF', 'ALT', 'beta'])
        return gexpr_tot, weights_df, sampleID, None
    print("SNP extract successful", geno.shape)
    snpsMat = geno.drop(columns=['CHROM', 'POS', 'snpID', 'REF', 'ALT', 'p_HWE', 'MAF', 'missing_rate']).values
    snpsMat = snpsMat.T
    snpsMat = StandardScaler().fit_transform(snpsMat)
    n_qtl = geno['snpID'].size
    gexpr_cis, b_cis = simulate_gene_expression(snpsMat, n_qtl, params.p_eqtl_cis,
                                                params.h_tf)  # 使用 params.h_tf  控制TF的遗传力
    # 结构化噪声 U
    W = simulate_corr(dimension=len(sampleID), low=-0.05, high=0.05)
    U = np.random.multivariate_normal(mean=np.zeros(len(sampleID)),
                                      cov=params.h_U_tf * W, size=1).flatten()
    E = np.random.normal(loc=0, scale=np.sqrt(params.h_E_tf), size=len(sampleID))
    gexpr_tot = gexpr_cis + U + E
    weights_df = pd.DataFrame({
        'CHROM': geno['CHROM'], 'POS': geno['POS'], 'snpID': geno['snpID'],
        'REF': geno['REF'], 'ALT': geno['ALT'], 'beta': b_cis
    })
    weights_df = weights_df[weights_df['beta'] != 0]
    return gexpr_tot, weights_df, sampleID, geno


def simulate_TG_expression(tg_gene_id, real_tf_ids, tf_expressions, tf_weights, tf_genos, params):
    tg_geno, sampleID = extract_genotype_vcf(
        bed_file=BED_PATH, geno_path=GENO_PATH, gene_ID=tg_gene_id,
        sampleid_path=SAMPLE_PATH, windows=params.windows
    )
    if tg_geno is None:
        return None, None, None, None

    tg_snpsMat = tg_geno.drop(columns=['CHROM', 'POS', 'snpID', 'REF', 'ALT', 'p_HWE', 'MAF', 'missing_rate']).values
    tg_snpsMat = tg_snpsMat.T
    tg_snpsMat = StandardScaler().fit_transform(tg_snpsMat)
    n_qtl_cis = tg_geno['snpID'].size

    gexpr_cis, b_cis = simulate_gene_expression(tg_snpsMat, n_qtl_cis, params.p_eqtl_cis, params.h_cis)
    cis_weights = pd.DataFrame({
        'CHROM': tg_geno['CHROM'], 'POS': tg_geno['POS'], 'snpID': tg_geno['snpID'],
        'REF': tg_geno['REF'], 'ALT': tg_geno['ALT'], 'beta': b_cis
    })
    cis_weights = cis_weights[cis_weights['beta'] != 0]
    ############################################################################################################
    # 使用 TF 表达量构建 trans 部分
    real_tf_expression_values = []
    for tf_id in real_tf_ids:
        # 找到对应的 TF 表达数据 DataFrame
        tf_expr_df = next((tf_expr for tf_expr in tf_expressions if tf_expr['TargetID'].iloc[0] == tf_id), None)
        if tf_expr_df is not None:
            # 提取样本的表达值并转置为行向量
            tf_sample_expr_df = tf_expr_df[sampleID]
            tf_expression_for_tg = tg.center(tf_sample_expr_df, sampleID).values.flatten()
            real_tf_expression_values.append(tf_expression_for_tg)

    if real_tf_expression_values:
        tf_expression_matrix = np.vstack(real_tf_expression_values).T  # 样本 x TF
        n_tf = tf_expression_matrix.shape[1]
        b_trans_expr = np.random.normal(loc=0, scale=np.sqrt(params.h_trans / params.h_tf / n_tf),
                                        size=n_tf)  #   基于 TF 数量和 TF 遗传力调整 scale
        gexpr_trans = np.dot(tf_expression_matrix, b_trans_expr)

        # Scale gexpr_trans to have variance proportional to h_trans (这里不需要再次scale, 因为beta已经基于 params.h_trans调整)
        # trans_var = np.var(gexpr_trans)
        # if trans_var > 0:
        #     gexpr_trans = gexpr_trans * np.sqrt(params.h_trans / trans_var)
    else:
        gexpr_trans = np.zeros(len(sampleID))
        b_trans_expr = np.zeros(0)

    ##############################################################################################################
    W = simulate_corr(dimension=len(sampleID), low=-0.02, high=0.02)
    U = np.random.multivariate_normal(mean=np.zeros(len(sampleID)),
                                      cov=params.h_U_tg * W, size=1).flatten()
    # U = np.random.normal(loc=0, scale=np.sqrt(params.h_U), size=len(sampleID))
    E = np.random.normal(loc=0, scale=np.sqrt(params.h_E_tg), size=len(sampleID))
    gexpr_tot = gexpr_cis + gexpr_trans + U + E

    # trans 效应的 weights_df  现在基于 TF 表达量的系数
    selected_trans_weights = pd.DataFrame({
        'TF_GeneID': real_tf_ids,
        'beta_expr': b_trans_expr  # TF 表达量的beta
    })

    weights_df = pd.concat([cis_weights, selected_trans_weights]).reset_index(
        drop=True)  # concat cis weights 和 trans weights (基于TF 表达量)
    return gexpr_tot, weights_df, sampleID, tg_geno


def run_simulation(params, OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    anno_data = pd.read_csv(BED_PATH, sep="\t", header=None,
                            names=['chrom', 'start', 'end', 'strand', 'gene_id', 'gene_name', 'gene_type'])
    tf_data = pd.read_csv(TF_INFO_PATH, sep="\t")
    all_genes = anno_data['gene_id'].tolist()
    tf_candidates_all = tf_data['Ensembl'].tolist()
    tf_candidates = [gene for gene in tf_candidates_all if gene in all_genes]
    tg_list = [gene for gene in all_genes if gene not in tf_candidates]
    tg_ids = np.random.choice(tg_list, size=params.n_TG, replace=False)

    tf_class1_ids_selected = []
    tf_class1_candidates_pool = list(tf_candidates)  # 使用 list 以允许移除元素
    while len(tf_class1_ids_selected) < params.n_TF_class1 and tf_class1_candidates_pool:
        tf_id = np.random.choice(tf_class1_candidates_pool)
        print(f"Trying TF class 1: {tf_id}")
        expr, weights, sampleID, geno = simulate_TF_expression(tf_id, params)
        if geno is not None and not geno.empty:
            tf_class1_ids_selected.append(tf_id)
            print(f"成功为 tf_class1_ids {tf_id} 提取到SNP数据")
        else:
            print(f"为 tf_class1_ids {tf_id} 未提取到SNP数据, 重新选择")
        tf_class1_candidates_pool.remove(tf_id)  # 确保不再重复尝试相同的tf_id

    if len(tf_class1_ids_selected) < params.n_TF_class1:
        print(
            f"Warning: Could not find SNP data for {params.n_TF_class1} TF class 1 TFs. Found only {len(tf_class1_ids_selected)}.")

    tf_class2_ids_selected = []
    tf_class2_candidates_pool = list(t for t in tf_candidates if t not in tf_class1_ids_selected)  # 避免重复使用class1已选的
    while len(tf_class2_ids_selected) < params.n_TF_class2 and tf_class2_candidates_pool:
        tf_id = np.random.choice(tf_class2_candidates_pool)
        print(f"Trying TF class 2: {tf_id}")
        expr, weights, sampleID, geno = simulate_TF_expression(tf_id, params)
        if geno is not None and not geno.empty:
            tf_class2_ids_selected.append(tf_id)
            print(f"成功为 tf_class2_ids {tf_id} 提取到SNP数据")
        else:
            print(f"为 tf_class2_ids {tf_id} 未提取到SNP数据, 重新选择")
        tf_class2_candidates_pool.remove(tf_id)  # 确保不再重复尝试相同的tf_id

    if len(tf_class2_ids_selected) < params.n_TF_class2:
        print(
            f"Warning: Could not find SNP data for {params.n_TF_class2} TF class 2 TFs. Found only {len(tf_class2_ids_selected)}.")

    tf_class1_ids = np.array(tf_class1_ids_selected)
    tf_class2_ids = np.array(tf_class2_ids_selected)

    print(f"tf_class1_ids: {tf_class1_ids}")
    print(f"tf_class2_ids: {tf_class2_ids}")
    tf_expressions = []
    tf_weights = {}  # tf_weights 存储 TF 的 cis-eQTL weights
    tf_genos = {}
    valid_tf_class1_ids = []
    valid_tf_class2_ids = []

    for tf_id in np.concatenate([tf_class1_ids, tf_class2_ids]):
        print(f"Processing TF {tf_id}")
        expr, weights, sampleID, geno = simulate_TF_expression(tf_id, params)
        if expr is not None:
            if geno is not None and len(geno) > 0:  # 检查geno是否为空或者None
                gene_info = anno_data[anno_data['gene_id'] == tf_id].iloc[0]
                expr_df = pd.DataFrame({
                    'CHROM': [gene_info['chrom']],
                    'GeneStart': [gene_info['start']],
                    'GeneEnd': [gene_info['end']],
                    'TargetID': [tf_id],
                    'GeneName': [gene_info['gene_name']],
                    **{sampleID[i]: [expr[i]] for i in range(len(sampleID))}  # 使用真实样本名
                })
                tf_expressions.append(expr_df)
                tf_weights[tf_id] = weights  # 存储 TF 的 cis-eQTL weights
                tf_genos[tf_id] = geno
                weights.to_csv(f"{OUTPUT_DIR}/TF_eQTL_beta_{tf_id}.csv", sep="\t", index=False)
                if tf_id in tf_class1_ids:
                    valid_tf_class1_ids.append(tf_id)
                elif tf_id in tf_class2_ids:
                    valid_tf_class2_ids.append(tf_id)
            else:
                print(f"Warning: geno for TF {tf_id} is empty or None, skipping.")
        else:
            print(f"Warning: expr for TF {tf_id} is None, skipping.")
    tf_class1_ids = np.array(valid_tf_class1_ids)
    tf_class2_ids = np.array(valid_tf_class2_ids)

    tg_expressions = []
    tg_genos = {}
    tg_4_real_tf = {}
    all_tf_ids = [tf['TargetID'].iloc[0] for tf in tf_expressions]
    for tg_id in tg_ids:
        real_tf1 = np.random.choice(tf_class1_ids, size=min(params.real_TF_class1, len(tf_class1_ids)), replace=False)
        real_tf2 = np.random.choice(tf_class2_ids, size=min(params.real_TF_class2, len(tf_class2_ids)), replace=False)
        real_tf_ids = np.concatenate([real_tf1, real_tf2])
        tg_4_real_tf[tg_id] = real_tf_ids
        expr, weights, sampleID, tg_geno = simulate_TG_expression(
            tg_id, real_tf_ids, tf_expressions, tf_weights, tf_genos, params
        )
        tg_genos[tg_id] = tg_geno

        if expr is not None:
            gene_info = anno_data[anno_data['gene_id'] == tg_id].iloc[0]
            expr_df = pd.DataFrame({
                'CHROM': [gene_info['chrom']],
                'GeneStart': [gene_info['start']],
                'GeneEnd': [gene_info['end']],
                'TargetID': [tg_id],
                'GeneName': [gene_info['gene_name']],
                **{sampleID[i]: [expr[i]] for i in range(len(sampleID))}  # 使用真实样本名
            })
            tg_expressions.append(expr_df)
            weights.to_csv(f"{OUTPUT_DIR}/TG_eQTL_beta_{tg_id}_TFexpr.csv", sep="\t",
                           index=False)  # TG trans eQTL beta 现在基于 TF 表达量

    # 文件路径设置 (modified)
    TG_CHR_DIR = os.path.join(OUTPUT_DIR,
                              f"TG_expression_h_cis_{params.h_cis}_h_trans_{params.h_trans}_hU_{params.h_U_tg}_by_chr")
    TF_CHR_DIR = os.path.join(OUTPUT_DIR,
                              f"TF_expression_h_cis_{params.h_cis}_h_trans_{params.h_trans}_hU_{params.h_U_tf}_by_chr")  # TF 文件夹名字也统一
    os.makedirs(TG_CHR_DIR, exist_ok=True)
    os.makedirs(TF_CHR_DIR, exist_ok=True)

    # Inside run_simulation function:
    # 保存 TF 表达量（整体和按染色体切分）
    if tf_expressions:
        final_tf_expr = pd.concat(tf_expressions, ignore_index=True)
        final_tf_expr.to_csv(f"{OUTPUT_DIR}/TF_expression_h_tf_{params.h_tf}_hU_{params.h_U_tf}.csv",  # TF 文件名使用 h_tf
                             sep="\t", index=False)  # Keeping TF overall file with h2 for consistency
        # 按染色体切分并保存
        for chrom, group in final_tf_expr.groupby('CHROM'):
            chrom_file = os.path.join(TF_CHR_DIR, f"TF_expression_{chrom}.csv")
            group.to_csv(chrom_file, sep="\t", index=False)
            print(f"Saved TF expression for {chrom} to {chrom_file}")

    # 保存 TG 表达量（整体和按染色体切分）
    if tg_expressions:
        final_tg_expr = pd.concat(tg_expressions, ignore_index=True)
        final_tg_expr.to_csv(
            f"{OUTPUT_DIR}/TG_expression_h_cis_{params.h_cis}_h_trans_{params.h_trans}_hU_{params.h_U_tg}.csv",
            sep="\t", index=False)  # Modified TG overall file name
        # 按染色体切分并保存
        for chrom, group in final_tg_expr.groupby('CHROM'):
            chrom_file = os.path.join(TG_CHR_DIR, f"TG_expression_{chrom}.csv")
            group.to_csv(chrom_file, sep="\t", index=False)
            print(f"Saved TG expression for {chrom} to {chrom_file}")

    return tg_expressions, tf_expressions, all_tf_ids, tf_class1_ids, tf_class2_ids, tf_genos, tg_genos, tg_4_real_tf


def read_result(base_path, method):
    if method not in ['TIGAR', 'PRDX']:
        raise ValueError("method 参数必须是 'TIGAR' 或 'PRDX'")
    # 根据 method 设置文件夹和文件前缀
    prefix = 'DPR' if method == 'TIGAR' else 'EN'
    method_column = method  # 列名直接使用 method
    dfs = []
    # 循环所有染色体（1到22）
    for i in range(1, 23):
        folder_path = os.path.join(base_path, f'{prefix}_CHR{i}')
        file_path = os.path.join(folder_path, f'CHR{i}_{prefix}_train_GeneInfo.txt')
        # 检查文件是否存在
        if os.path.exists(file_path):
            # 读取文件为 DataFrame，仅读取第5列（GeneName）和第9列（结果列）
            df = pd.read_csv(file_path, sep='\t', usecols=[3, 8]) \
                .drop_duplicates(subset=['TargetID'], keep='first')
            df.columns = ['TargetID', method_column]
            dfs.append(df)
        else:
            print(f"文件 {file_path} 不存在")
    # 竖向组合所有 DataFrame
    if dfs:  # 检查是否为空
        return pd.concat(dfs, ignore_index=True)
    else:
        print("没有找到任何有效文件，返回空 DataFrame")
        return pd.DataFrame(columns=['TargetID', method_column])


def run_Tigar(params, OUTPUT_DIR):
    h_cis = params.h_cis
    h_trans = params.h_trans
    h_U = params.h_U_tg
    TIGAR_outpath = OUTPUT_DIR + "model_result/TIGAR/"
    command_list = ["bash", "train_TIGAR_all_chr.sh", h_cis, h_trans, h_U, GENO_PATH, SAMPLE_PATH, TIGAR_outpath, 'DPR',
                    params.windows, OUTPUT_DIR]
    # 将列表中的元素转换为字符串，并用空格连接
    command_string = " ".join(map(str, command_list))
    print(command_string)

    subprocess.run(command_string, shell=True, check=True)
    result = read_result(TIGAR_outpath, "TIGAR")
    return result


def run_PrediXcan(params, OUTPUT_DIR):
    h_cis = params.h_cis
    h_trans = params.h_trans
    h_U = params.h_U_tg
    TIGAR_outpath = OUTPUT_DIR + "model_result/PrediXcan/"
    command_list = ["bash", "train_TIGAR_all_chr.sh", h_cis, h_trans, h_U, GENO_PATH, SAMPLE_PATH, TIGAR_outpath,
                    'elastic_net',
                    params.windows, OUTPUT_DIR]
    # 将列表中的元素转换为字符串，并用空格连接
    command_string = " ".join(map(str, command_list))
    print(command_string)
    subprocess.run(command_string, shell=True, check=True)
    result = read_result(TIGAR_outpath, "PRDX")
    return result


def BGW_TWAS():
    return


def select_snps_with_lasso(tf_geno, tf_expr, sampleID):
    # 提取基因型矩阵并转置为样本 x SNP 的形式
    X = tf_geno.set_index('snpID')[sampleID].T
    y = tf_expr[sampleID].values.flatten()

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用 LassoCV 进行特征选择
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_scaled, y)

    # 返回系数非零的 SNP ID
    selected_snps = X.columns[lasso.coef_ != 0].tolist()
    return selected_snps


def filter_tf_genos(all_tf_expressions, tf_genos, sampleID):
    filtered_tf_genos = {}
    for tf_id in tf_genos.keys():
        # 提取该 TF 的表达量数据
        tf_expr_row = all_tf_expressions[all_tf_expressions['TargetID'] == tf_id]
        if not tf_expr_row.empty:
            tf_expr = tf_expr_row.iloc[0]
            # 使用 LassoCV 选择 SNP
            selected_snps = select_snps_with_lasso(tf_genos[tf_id], tf_expr, sampleID)
            # 过滤基因型数据
            filtered_tf_genos[tf_id] = tf_genos[tf_id][tf_genos[tf_id]['snpID'].isin(selected_snps)]
        else:
            print(f"警告: TF {tf_id} 没有表达量数据，跳过过滤")
            filtered_tf_genos[tf_id] = pd.DataFrame()  # 或保留原始数据，根据需求调整
    return filtered_tf_genos


def do_cv_transTFTWAS(i, groups, Geno_Exp_df, cv_trainID, cv_testID):
    Geno_Exp_df = Geno_Exp_df.copy()
    train_geno_exp = Geno_Exp_df.loc[cv_trainID[i]].dropna()
    test_geno_exp = Geno_Exp_df.loc[cv_testID[i]].dropna()
    cv_rsquared = model.transTF_TWAS(groups, train_geno_exp, test_geno_exp)
    print(f"completed 1 {cv_rsquared}")
    return cv_rsquared

def do_cv_GRNTWAS(i, groups, Geno_Exp_df, cv_trainID, cv_testID, Geno_meta, dpr_path, DPR_tmp_path):
    train_geno_exp = Geno_Exp_df.loc[cv_trainID[i]].dropna()
    test_geno_exp = Geno_Exp_df.loc[cv_testID[i]].dropna()
    cv_rsquared = model.compare_lasso_enet_cv(groups, train_geno_exp, test_geno_exp, Geno_meta=Geno_meta, dpr_path=dpr_path, tmp_DPR=DPR_tmp_path)
    return cv_rsquared

def run_transTFTWAS(all_tg_expression, tf_class1_ids, tf_genos, tg_genos, CV_trainID, CV_testID, sampleID):
    '''
    从 all_tg_expression（list）中遍历每一个元素（df）
    '''
    filtered_tf_genos = filter_tf_genos(all_tf_expressions, tf_genos, sampleID)
    tf_genos = filtered_tf_genos
    tf_class1_geno_list = [tf_genos[i] for i in tf_class1_ids]
    # 创建一个列表来收集所有的结果字典
    results_list = []

    for index in all_tg_expression.index:
        row = all_tg_expression.loc[[index]]
        TargetID = row.iloc[0, 3]
        target_Expr = tg.center(row, sampleID)

        # 5. 从 tg_genos 中提取并拼接 Geno_list
        Geno_list = [tg_genos[TargetID]]
        Geno_list.extend(tf_class1_geno_list)

        # 过滤掉 None 值，只保留有效的 DataFrame
        for i, df in enumerate(Geno_list):
            if df is not None:
                df['group'] = i  # 只对非 None 的 DataFrame 添加 group 列

        # 过滤掉 None 并进行拼接
        Geno_list_filtered = [df for df in Geno_list if df is not None]
        if Geno_list_filtered:  # 确保列表非空
            Geno = pd.concat(Geno_list_filtered, axis=0, ignore_index=True)
        else:
            # 如果过滤后为空，处理空情况（根据需求调整）
            print("警告: Geno_list 中没有有效的 DataFrame")
            Geno = pd.DataFrame()  # 或者抛出异常，取决于你的需求
        groups = Geno['group'].tolist()

        Geno = tg.center(Geno, sampleID)
        Geno_Exp = pd.concat([
            Geno.set_index(['snpID'])[sampleID],
            target_Expr.set_index(['TargetID'])[sampleID]
        ]).T

        do_cv_args = [Geno_Exp, CV_trainID, CV_testID]
        warnings.filterwarnings('ignore')
        k_fold_R2 = [do_cv_transTFTWAS(i, groups, *do_cv_args) for i in range(5)]
        avg_r2_cv = sum(k_fold_R2) / 5

        # 将结果添加到列表中
        results_list.append({'TargetID': TargetID, 'transTFTWAS': avg_r2_cv})

        warnings.filterwarnings('default')

    # 在循环结束后一次性将所有结果转换为 DataFrame
    results_df = pd.DataFrame(results_list)

    return results_df


def run_GRNTWAS(all_tg_expressions, all_tf_expressions, tf_genos, tg_genos, sampleID,
                CV_trainID, CV_testID,
                tg_4_real_tf, dpr_path=None, OUTPUT_DIR=None):
    """
    运行 GRNTWAS，调用 DPR、LassoCV 和 ElasticNetCV 进行模型对比。

    参数:
    - all_tg_expressions: 目标基因表达量 DataFrame
    - all_tf_expressions: 转录因子表达量 DataFrame
    - tf_genos: 转录因子基因型字典
    - tg_genos: 目标基因基因型字典
    - sampleID: 样本 ID 列表
    - CV_trainID: 交叉验证训练集 ID 列表
    - CV_testID: 交叉验证测试集 ID 列表
    - tg_4_real_tf: 目标基因与真实转录因子的映射
    - dpr_path: DPR 可执行文件路径（可选）

    返回:
    - results_df: 包含每个目标基因的 R² 结果的 DataFrame
    """
    filtered_tf_genos = filter_tf_genos(all_tf_expressions, tf_genos, sampleID)
    tf_genos = filtered_tf_genos
    results_list = []
    DPR_tmp_path = OUTPUT_DIR + 'DPR/'
    for index in all_tg_expressions.index:
        row = all_tg_expressions.loc[[index]]
        TargetID = row['TargetID'].values[0]
        real_tf = tg_4_real_tf[TargetID]

        # 计算转录因子与目标基因的相关性
        filter_results = []
        for i in all_tf_expressions.index:
            tf_expression = all_tf_expressions.loc[[i]]
            corr, p_value = pearsonr(row[sampleID].values.flatten(), tf_expression[sampleID].values.flatten())
            filter_results.append((tf_expression['TargetID'].values[0], corr, p_value))
        filter_results = pd.DataFrame(filter_results, columns=['TargetID', 'Pearson_Correlation', 'P_Value'])

        # 选择显著相关的 TF（p < 0.05），并取相关性绝对值前 3
        significant_tf_genes = filter_results[filter_results['P_Value'] < 0.05].copy()
        significant_tf_genes.sort_values(by='Pearson_Correlation', key=abs, ascending=False, inplace=True)
        significant_tf_genes = significant_tf_genes.head(3)
        coexpression_TF_ids = significant_tf_genes['TargetID'].tolist()

        # 合并目标基因和显著 TF 的基因型数据
        tf_geno_list = [tf_genos[i] for i in coexpression_TF_ids]
        Geno_list = [tg_genos[TargetID]]
        Geno_list.extend(tf_geno_list)
        Geno_list_filtered = [df for df in Geno_list if df is not None and not df.empty]

        if Geno_list_filtered:
            Geno = pd.concat(Geno_list_filtered, axis=0, ignore_index=True)
        else:
            print(f"警告: 目标基因 {TargetID} 的 Geno_list 中没有有效的 DataFrame")
            continue

        # 保存 Geno 的元信息（snpID, REF, ALT）
        Geno_meta = Geno[['snpID', 'REF', 'ALT']].copy()

        # 构建 Geno_Exp 数据框
        Geno_Exp = pd.concat([
            Geno.set_index(['snpID'])[sampleID],
            row.set_index(['TargetID'])[sampleID]
        ]).T

        # 执行 5 折交叉验证
        do_cv_args = [None, Geno_Exp, CV_trainID, CV_testID, Geno_meta, dpr_path, DPR_tmp_path]
        k_fold_R2 = [do_cv_GRNTWAS(i, *do_cv_args) for i in range(5)]
        avg_r2_cv = sum(k_fold_R2) / 5

        results_list.append({'TargetID': TargetID, 'GRNTWAS': avg_r2_cv})

    results_df = pd.DataFrame(results_list)
    return results_df


'''
def compare_model(tg_expressions, tf_expressions, all_tf_ids, tf_class1_ids, tf_class2_ids, tf_genos, tg_genos):
    # 对于 PrediXcan 和 Tigar、BGW，直接利用 tg_expressions 建立的本地文件，通过 bash 脚本计算最后结果
    # 对于 GRNTWAS 和 transTF-TWAS 和 GRN-TI 直接传递
    return
'''

import os  # 导入 os 模块，用于文件路径操作
import datetime  # 导入 datetime 模块，用于生成唯一的文件名


def print_mean_r2(results_df, column_name, model_name, output_dir="./method/"):
    """
    检查 DataFrame 是否为空，计算指定列的均值，并打印结果，以及保存 DataFrame 到文件。

    参数:
        results_df (pd.DataFrame): 包含模型结果的 DataFrame.
        column_name (str): 包含 R2 值的列名.
        model_name (str): 模型名称，用于打印信息和文件名.
        output_dir (str, optional): 输出文件保存的目录路径。默认为当前目录 "."。
    """
    print(f"\n{model_name} results:")
    print(results_df)

    if not results_df.empty:
        if column_name in results_df.columns:
            mean_r2 = results_df[column_name].mean()
            print(f"Mean R2 for {model_name}: {mean_r2}")

            # 获取当前日期时间，避免文件覆盖
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{model_name}_CVR2_{timestamp}.tsv"  # 文件名加上时间戳
            output_path = os.path.join(output_dir, output_filename)  # 组合完整路径

            # 检查输出目录是否存在，不存在则创建
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Output directory {output_dir} created.")

            try:
                # 保存 DataFrame 到文件，使用 Tab 分割
                results_df.to_csv(output_path, sep='\t', index=False)
                print(f"Results DataFrame for {model_name} saved to: {output_path}")  # 打印保存路径
            except Exception as e:
                print(f"Error saving results for {model_name}: {e}")
        else:
            print(f"Error: '{column_name}' column not found in {model_name} results.")
    else:
        print(f"{model_name} results DataFrame is empty, cannot calculate mean R2.")


# Main execution with multiple parameter combinations
if __name__ == "__main__":
    all_results = []  # List to collect results from all parameter sets

    # Loop over each parameter combination
    for idx, param_set in enumerate(param_combinations):
        print(f"\nRunning simulation for parameter set {idx}: {param_set}")

        # Create a unique output directory for this parameter set
        OUTPUT_DIR = f"{BASE_OUTPUT_DIR}param_set_{idx}/"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Initialize parameters
        params = SimulationParams(
            n_TG=param_set['n_TG'],
            h_cis=param_set['h_cis'],
            h_trans=param_set['h_trans'],
            h_tf=param_set['h_tf'],
            n_TF_class1=6,  # Default values, adjust if needed
            n_TF_class2=3,
            real_TF_class1=2,
            real_TF_class2=2,
            p_eqtl_cis=0.1,
            windows=5000
        )

        # Run simulation
        tg_expressions, tf_expressions, all_tf_ids, tf_class1_ids, tf_class2_ids, tf_genos, tg_genos, tg_4_real_tf = run_simulation(
            params, OUTPUT_DIR)
        print(f"Simulation completed for param_set_{idx}. Results saved in {OUTPUT_DIR}")

        # Concatenate expressions
        all_tg_expressions = pd.concat(tg_expressions, axis=0, ignore_index=True)
        all_tf_expressions = pd.concat(tf_expressions, axis=0, ignore_index=True)

        # Set up 5-fold cross-validation
        sampleID = all_tg_expressions.columns[5:].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        kf_splits = [(sampleID[x], sampleID[y]) for x, y in kf.split(sampleID)]
        CV_trainID, CV_testID = zip(*kf_splits)

        # Run all five models
        result_tigar = run_Tigar(params, OUTPUT_DIR)
        result_GRNTWAS = run_GRNTWAS(all_tg_expressions, all_tf_expressions, tf_genos, tg_genos, sampleID, CV_trainID,
                                     CV_testID, tg_4_real_tf, DPR_path, OUTPUT_DIR)

        result_PRDX = run_PrediXcan(params, OUTPUT_DIR)
        result_transTFTWAS = run_transTFTWAS(all_tg_expressions, tf_class1_ids, tf_genos, tg_genos, CV_trainID,
                                             CV_testID, sampleID)


        # Merge results into a single DataFrame
        results = pd.merge(result_tigar, result_PRDX, on='TargetID', how='outer')
        results = pd.merge(results, result_transTFTWAS, on='TargetID', how='outer')
        results = pd.merge(results, result_GRNTWAS, on='TargetID', how='outer')

        # Add parameter set identifier
        results[
            'param_set'] = f"Set {idx}: h_cis={param_set['h_cis']}, h_trans={param_set['h_trans']}, h_tf={param_set['h_tf']}"

        # Save individual results
        results.to_csv(f"{OUTPUT_DIR}/model_results_param_set_{idx}.csv", index=False)
        all_results.append(results)

    # Combine all results into a single DataFrame
    all_results_df = pd.concat(all_results, ignore_index=True)

    # Reshape data to long format for plotting
    melted_df = pd.melt(
        all_results_df,
        id_vars=['TargetID', 'param_set'],
        value_vars=['TIGAR', 'PRDX', 'transTFTWAS', 'GRNTWAS'],
        var_name='Model',
        value_name='CV_R2'
    )

    # Generate boxplots
    plt.figure(figsize=(15, 10))
    g = sns.FacetGrid(melted_df, col='param_set', col_wrap=4, sharey=True, height=4, aspect=1.5)
    g.map(sns.boxplot, 'Model', 'CV_R2', order=['TIGAR', 'PRDX', 'transTFTWAS', 'GRNTWAS'], palette='Set2')
    g.set_titles("{col_name}")
    g.set_axis_labels("Model", "CV R²")
    g.fig.suptitle("Comparison of CV R² Across Models and Parameter Sets", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{BASE_OUTPUT_DIR}/boxplot_comparison.png", dpi=300)
    plt.show()
