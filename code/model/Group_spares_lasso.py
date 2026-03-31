import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from group_lasso import GroupLasso
import statsmodels.api as sm
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import uniform, pearsonr
import os
import tempfile
import tempfile
import sys
import subprocess

# Add parent directory to path for importing GRNutils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import GRNutils as tg

def mse_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return mean_squared_error(y, y_pred)


# Compatibility wrapper used by downstream wrappers
def sparse_group_lasso_Qualified(groups, train_geno_exp, test_geno_exp):
    return compare_lasso_enet_cv(groups, train_geno_exp, test_geno_exp)


# 调用 DPR 的函数
def run_dpr(Bimbam_df, Pheno_df, output_dir, target_id, dpr_path, method, es_type):
    """
    调用 DPR 模型并读取结果。

    参数:
    - Bimbam_df: BIMBAM 格式的基因型数据
    - Pheno_df: 表型数据
    - output_dir: 输出目录
    - target_id: 目标 ID
    - dpr_path: DPR 可执行文件路径
    - method: DPR 方法参数
    - es_type: 效应大小类型

    返回:
    - dpr_result: DPR 输出结果 DataFrame
    """
    if not dpr_path or (not os.path.exists(dpr_path)):
        print(f"警告：未找到 DPR 可执行文件：{dpr_path}")
        return None

    bimbam_file = f"{output_dir}{target_id}.bimbam"
    pheno_file = f"{output_dir}{target_id}.pheno"
    out_args = {'header': False, 'index': None, 'sep': '\t', 'mode': 'w', 'float_format': '%f'}
    Bimbam_df.to_csv(bimbam_file, **out_args)
    Pheno_df.to_csv(pheno_file, **out_args)
    output_prefix = f"{output_dir}output/{target_id}.param.txt"
    cmd = [dpr_path, '-g', f"{target_id}.bimbam", '-p', f"{target_id}.pheno", '-dpr', method,
           '-notsnp',
           '-o', target_id]
    print(cmd)
    try:
        subprocess.check_call(cmd, cwd=output_dir)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"警告：DPR 执行失败，将跳过 DPR 模型。原因: {e}")
        return None
    finally:
        os.remove(bimbam_file)
        os.remove(pheno_file)

    print("esx")  # 确保这个语句会执行

    if os.path.exists(output_prefix):
        dpr_result = pd.read_csv(output_prefix,
                                 sep='\t',
                                 header=0,
                                 names=['CHROM', 'snpID', 'POS', 'n_miss', 'b', 'beta', 'gamma'],
                                 usecols=['snpID', 'b', 'beta'],
                                 dtype={'snpID': object, 'b': np.float64, 'beta': np.float64})
        os.remove(output_prefix)
        os.remove(f"{output_dir}output/{target_id}.log.txt")
        DPR_Out = tg.optimize_cols(dpr_result)

        # GET EFFECT SIZE
        DPR_Out['ES'] = DPR_Out['beta']

        return DPR_Out

    else:
        print(f"错误：DPR 输出文件 {output_prefix} 不存在")
        return None


def compare_lasso_enet_cv(groups, train, test=None, k=5, random_state=42, Geno_meta=None, dpr_path=None, dpr_method='1',
                          es_type='fixed', tmp_DPR=None, used_model=None):

    # 数据准备
    trainX = train.iloc[:, :-1]
    trainY = train.iloc[:, -1].values
    if test is not None:
        testX = test.iloc[:, :-1]
        testY = test.iloc[:, -1].values
    else:
        testX = trainX
        testY = trainY

    # 如果提供分组信息，使用 GroupLasso 筛选特征
    if groups is not None:
        gl = GroupLasso(
            groups=groups, l1_reg=0, group_reg=0.1, scale_reg="inverse_group_size",
            supress_warning=True, frobenius_lipschitz=True, fit_intercept=False
        )
        gl.fit(trainX.values, trainY)
        nonzero_groups = [i for i, group in enumerate(gl.groups_) if np.any(gl.coef_[group] != 0)]
        selected_features = [trainX.columns[gl.groups_[group_index]] for group_index in nonzero_groups]
        selected_features = [item for sublist in selected_features for item in sublist]
    else:
        selected_features = trainX.columns.tolist()
    if not selected_features:
        print("警告：Group Lasso 没有选择任何特征。使用所有特征进行后续训练。")
        selected_features = trainX.columns.tolist()

    # LassoCV
    lasso_cv_model = LassoCV(
        cv=k, n_jobs=max(1, int(0.7 * os.cpu_count())), random_state=random_state,
        fit_intercept=True, alphas=np.array([0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.3, 0.5])
    )
    print("开始训练 LassoCV 模型...")
    lasso_cv_model.fit(trainX[selected_features].values, trainY)
    lasso_predY = lasso_cv_model.predict(testX[selected_features].values)
    lasso_lm = sm.OLS(testY, sm.add_constant(lasso_predY)).fit()
    lasso_Rsquared = lasso_lm.rsquared
    print("LassoCV 模型训练完成。")

    # ElasticNetCV
    enet_cv_model = ElasticNetCV(
        max_iter=100, l1_ratio=[0.05, 0.1, 0.2, 0.5], fit_intercept=True,
        alphas=np.array([0.0001, 0.001, 0.01, 0.05, 0.15]), selection='random',
        cv=k, n_jobs=max(1, int(0.7 * os.cpu_count())), random_state=random_state
    )
    print("开始训练 ElasticNetCV 模型...")
    enet_cv_model.fit(trainX[selected_features].values, trainY)
    enet_predY = enet_cv_model.predict(testX[selected_features].values)
    enet_lm = sm.OLS(testY, sm.add_constant(enet_predY)).fit()
    enet_Rsquared = enet_lm.rsquared
    print("ElasticNetCV 模型训练完成。")

    # DPR
    Rsquared_dpr = -np.inf
    predY_dpr = None
    lm_dpr = None
    weights_dpr = None
    if dpr_path is not None and Geno_meta is not None:
        print("开始训练 DPR 模型...")
        selected_snps = selected_features
        # 提取 Geno_meta 中与 selected_snps 匹配的元信息
        meta_info = Geno_meta[Geno_meta['snpID'].isin(selected_snps)][['snpID', 'REF', 'ALT']].copy()

        # 转置 trainX[selected_snps]，使 SNP ID 成为行索引
        geno_data = trainX[selected_snps].T
        geno_data.index.name = 'snpID'

        # 按 'snpID' 合并元信息和基因型数据
        Bimbam_df = meta_info.merge(geno_data, left_on='snpID', right_index=True)

        # 调整列顺序：'snpID', 'REF', 'ALT' 后跟样本列
        sample_columns = [col for col in Bimbam_df.columns if col not in ['snpID', 'REF', 'ALT']]
        Bimbam_df = Bimbam_df[['snpID', 'REF', 'ALT'] + sample_columns]

        Pheno_df = pd.DataFrame(trainY, columns=['expression'])

        dpr_output_dir = tmp_DPR if tmp_DPR else tempfile.mkdtemp(prefix='grntwas_dpr_')
        os.makedirs(dpr_output_dir, exist_ok=True)
        os.makedirs(f"{dpr_output_dir}output", exist_ok=True)
        target_id = train.columns[-1]
        dpr_result = run_dpr(Bimbam_df, Pheno_df, dpr_output_dir, target_id, dpr_path, dpr_method, es_type)
        if dpr_result is not None:
            weights_dpr = dpr_result.set_index('snpID')['ES']
            common_snps = [snp for snp in weights_dpr.index if snp in testX.columns]
            if common_snps:
                testX_selected = testX[common_snps]
                predY_dpr = np.dot(testX_selected.values, weights_dpr[common_snps].values)
                lm_dpr = sm.OLS(testY, sm.add_constant(predY_dpr)).fit()
                Rsquared_dpr = lm_dpr.rsquared
            else:
                print("警告：DPR 输出与测试集 SNP 不匹配")
        else:
            print("警告：DPR 调用失败")
        print("DPR 模型训练完成。")
    # 记录每个模型的信息
    models = {
        "LassoCV": {
            "Rsquared": lasso_Rsquared,
            "predY": lasso_predY,
            "model": lasso_cv_model,
            "lm": lasso_lm,
            "Alpha": None,  # LassoCV 无 l1_ratio
            "Lambda": lasso_cv_model.alpha_,
            "cvm": np.min(lasso_cv_model.mse_path_),
            "beta": lasso_cv_model.coef_
        },
        "ElasticNetCV": {
            "Rsquared": enet_Rsquared,
            "predY": enet_predY,
            "model": enet_cv_model,
            "lm": enet_lm,
            "Alpha": enet_cv_model.l1_ratio_,
            "Lambda": enet_cv_model.alpha_,
            "cvm": np.min(enet_cv_model.mse_path_),
            "beta": enet_cv_model.coef_
        },
        "DPR": {
            "Rsquared": Rsquared_dpr,
            "predY": predY_dpr,
            "model": None,  # DPR 无模型对象
            "lm": lm_dpr,
            "Alpha": None,
            "Lambda": None,
            "cvm": None,
            "beta": weights_dpr  # DPR 的权重
        }
    }
    # 选择 R² 最高的模型
    if used_model:
        best_model_name = used_model
    else:
        best_model_name = max(models, key=lambda x: models[x]["Rsquared"])

    best_info = models[best_model_name]
    print(f"最佳模型: {best_model_name}, R² = {best_info['Rsquared']}")

    # 根据 test 是否为 None 返回结果
    if test is not None:
        return best_info["Rsquared"], best_model_name
    else:
        beta_model = best_info["beta"]
        Rsquared = best_info["Rsquared"]
        Pvalue = best_info["lm"].f_pvalue if best_info["lm"] is not None else None
        Alpha = best_info["Alpha"]
        Lambda = best_info["Lambda"]
        cvm = best_info["cvm"]

        # 初始化 beta，长度与 trainX 特征数一致
        beta = np.zeros(trainX.shape[1])
        if isinstance(beta_model, pd.Series):  # DPR 的情况
            # 将 weights_dpr 的值映射到 beta 中
            for snp in beta_model.index:
                if snp in selected_features:
                    feature_index = trainX.columns.get_loc(snp)
                    beta[feature_index] = beta_model[snp]
        else:  # LassoCV 或 ElasticNetCV 的情况
            # 将 beta_model 的值映射到 selected_features 对应的位置
            for i, feature_name in enumerate(selected_features):

                feature_index = trainX.columns.get_loc(feature_name)
                beta[feature_index] = beta_model[i]

        return beta, Rsquared, Pvalue, Alpha, Lambda, cvm


def compare_lasso_enet_cv_revise(groups, train, test=None, k=5, random_state=42, Geno_meta=None, dpr_path=None, dpr_method='1',
                          es_type='fixed', tmp_DPR=None, used_model=None):
    # 数据准备
    trainX = train.iloc[:, :-1]
    trainY = train.iloc[:, -1].values
    if test is not None:
        testX = test.iloc[:, :-1]
        testY = test.iloc[:, -1].values
    else:
        testX = trainX
        testY = trainY

    # 如果提供分组信息，使用 GroupLasso 筛选特征
    if groups is not None:
        gl = GroupLasso(
            groups=groups, l1_reg=0, group_reg=0.1, scale_reg="inverse_group_size",
            supress_warning=True, frobenius_lipschitz=True, fit_intercept=False
        )
        gl.fit(trainX.values, trainY)
        nonzero_groups = [i for i, group in enumerate(gl.groups_) if np.any(gl.coef_[group] != 0)]
        selected_features = [trainX.columns[gl.groups_[group_index]] for group_index in nonzero_groups]
        selected_features = [item for sublist in selected_features for item in sublist]
    else:
        selected_features = trainX.columns.tolist()
    if not selected_features:
        print("警告：Group Lasso 没有选择任何特征。使用所有特征进行后续训练。")
        selected_features = trainX.columns.tolist()

    # LassoCV
    lasso_cv_model = LassoCV(
        cv=k, n_jobs=max(1, int(0.7 * os.cpu_count())), random_state=random_state,
        fit_intercept=True, alphas=np.array([0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.3, 0.5])
    )
    print("开始训练 LassoCV 模型...")
    lasso_cv_model.fit(trainX[selected_features].values, trainY)
    lasso_predY = lasso_cv_model.predict(testX[selected_features].values)
    lasso_lm = sm.OLS(testY, sm.add_constant(lasso_predY)).fit()
    lasso_Rsquared = lasso_lm.rsquared
    print("LassoCV 模型训练完成。")

    # ElasticNetCV
    enet_cv_model = ElasticNetCV(
        max_iter=100, l1_ratio=[0.05, 0.1, 0.2, 0.5], fit_intercept=True,
        alphas=np.array([0.0001, 0.001, 0.01, 0.05, 0.15]), selection='random',
        cv=k, n_jobs=max(1, int(0.7 * os.cpu_count())), random_state=random_state
    )
    print("开始训练 ElasticNetCV 模型...")
    enet_cv_model.fit(trainX[selected_features].values, trainY)
    enet_predY = enet_cv_model.predict(testX[selected_features].values)
    enet_lm = sm.OLS(testY, sm.add_constant(enet_predY)).fit()
    enet_Rsquared = enet_lm.rsquared
    print("ElasticNetCV 模型训练完成。")

    # DPR
    Rsquared_dpr = -np.inf
    predY_dpr = None
    lm_dpr = None
    weights_dpr = None
    if dpr_path is not None and Geno_meta is not None:
        print("开始训练 DPR 模型...")
        selected_snps = selected_features
        meta_info = Geno_meta[Geno_meta['snpID'].isin(selected_snps)][['snpID', 'REF', 'ALT']].copy()
        geno_data = trainX[selected_snps].T
        geno_data.index.name = 'snpID'
        Bimbam_df = meta_info.merge(geno_data, left_on='snpID', right_index=True)
        sample_columns = [col for col in Bimbam_df.columns if col not in ['snpID', 'REF', 'ALT']]
        Bimbam_df = Bimbam_df[['snpID', 'REF', 'ALT'] + sample_columns]
        Pheno_df = pd.DataFrame(trainY, columns=['expression'])
        dpr_output_dir = tmp_DPR if tmp_DPR else tempfile.mkdtemp(prefix='grntwas_dpr_')
        os.makedirs(dpr_output_dir, exist_ok=True)
        os.makedirs(f"{dpr_output_dir}output", exist_ok=True)
        target_id = train.columns[-1]
        dpr_result = run_dpr(Bimbam_df, Pheno_df, dpr_output_dir, target_id, dpr_path, dpr_method, es_type)
        if dpr_result is not None:
            weights_dpr = dpr_result.set_index('snpID')['ES']
            common_snps = [snp for snp in weights_dpr.index if snp in testX.columns]
            if common_snps:
                testX_selected = testX[common_snps]
                predY_dpr = np.dot(testX_selected.values, weights_dpr[common_snps].values)
                lm_dpr = sm.OLS(testY, sm.add_constant(predY_dpr)).fit()
                Rsquared_dpr = lm_dpr.rsquared
            else:
                print("警告：DPR 输出与测试集 SNP 不匹配")
        else:
            print("警告：DPR 调用失败")
        print("DPR 模型训练完成。")

    # 记录每个模型的信息
    models = {
        "LassoCV": {
            "Rsquared": lasso_Rsquared,
            "predY": lasso_predY,
            "model": lasso_cv_model,
            "lm": lasso_lm,
            "Alpha": None,
            "Lambda": lasso_cv_model.alpha_,
            "cvm": np.min(lasso_cv_model.mse_path_),
            "beta": lasso_cv_model.coef_
        },
        "ElasticNetCV": {
            "Rsquared": enet_Rsquared,
            "predY": enet_predY,
            "model": enet_cv_model,
            "lm": enet_lm,
            "Alpha": enet_cv_model.l1_ratio_,
            "Lambda": enet_cv_model.alpha_,
            "cvm": np.min(enet_cv_model.mse_path_),
            "beta": enet_cv_model.coef_
        },
        "DPR": {
            "Rsquared": Rsquared_dpr,
            "predY": predY_dpr,
            "model": None,
            "lm": lm_dpr,
            "Alpha": None,
            "Lambda": None,
            "cvm": None,
            "beta": weights_dpr
        }
    }

    # 选择最佳模型
    if used_model:
        best_model_name = used_model
    else:
        best_model_name = max(models, key=lambda x: models[x]["Rsquared"])

    best_info = models[best_model_name]
    print(f"最佳模型: {best_model_name}, R² = {best_info['Rsquared']}")

    # 检查 beta_model 是否为 None
    beta_model = best_info["beta"]
    if beta_model is None:
        # 写入 error.txt
        error_file = "/data/lab/wangshixian/GRNTWAS_STAR/GRNTWAS2mayo-ad/vcf_project/result_eqtl_5k_TF_filter/error.txt"
        with open(error_file, "a") as f:
            f.write(train.columns[-1] + "\n")

        # 先尝试使用 DPR
        best_info = models["DPR"]
        beta_model = best_info["beta"]
        if beta_model is None:
            # 如果 DPR 的 beta 也为 None，则使用 ElasticNetCV
            best_info = models["ElasticNetCV"]
            beta_model = best_info["beta"]

    # 根据 test 是否为 None 返回结果
    if test is not None:
        return best_info["Rsquared"], best_model_name
    else:
        Rsquared = best_info["Rsquared"]
        Pvalue = best_info["lm"].f_pvalue if best_info["lm"] is not None else None
        Alpha = best_info["Alpha"]
        Lambda = best_info["Lambda"]
        cvm = best_info["cvm"]

        # 初始化 beta，长度与 trainX 特征数一致
        beta = np.zeros(trainX.shape[1])
        if isinstance(beta_model, pd.Series):  # DPR 的情况
            for snp in beta_model.index:
                if snp in trainX.columns:
                    feature_index = trainX.columns.get_loc(snp)
                    beta[feature_index] = beta_model[snp]
        else:  # LassoCV 或 ElasticNetCV 的情况
            for i, feature_name in enumerate(selected_features):
                feature_index = trainX.columns.get_loc(feature_name)
                beta[feature_index] = beta_model[i]

        return beta, Rsquared, Pvalue, Alpha, Lambda, cvm
