# GRNTWAS: GRN-guided Transcriptome-Wide Association Study

基因调控网络引导的转录组关联分析 (Gene Regulatory Network-guided TWAS) 训练工具。

## 简介

GRNTWAS 是一个将基因调控网络 (GRN) 信息整合到 TWAS 模型训练中的工具。通过利用网络传播算法识别关键调控因子，并结合 eQTL 信息筛选 trans-SNP，构建更准确的基因表达预测模型。

### 主要特点

- 支持多种网络影响力传播算法 (LTM, KATZ, RRW, PATH)
- 整合 eQTL 数据进行 SNP 过滤
- 多种正则化模型选择 (ElasticNet, Lasso, DPR)
- 支持并行处理加速训练
- 5 折交叉验证评估模型性能

## 安装

### 依赖项

```bash
pip install -r requirements.txt
```

### 系统要求

- Python >= 3.8
- tabix (用于处理 VCF 文件)

## 使用方法

### 1. 配置文件

在运行前，请修改 `config.py` 中的路径配置：

```python
# 输入数据路径配置
BED_PATH = '/path/to/gene.bed'
GRN_NETWORK_PATH = '/path/to/network.gexf'
GENO_PATH = '/path/to/genotype/'
GENE_EXPRESSION_PATH = '/path/to/expression.csv'
SAMPLE_PATH = '/path/to/sample_ids.txt'
EQTL_PATH = '/path/to/eQTL.txt'

# 输出路径配置
OUT_WEIGHT_PATH = 'result/weight/'
OUT_INFO_PATH = 'result/info/'
```

### 2. 运行训练

基本用法：

```bash
python main.py
```

使用命令行参数：

```bash
# 指定线程数和影响力算法
python main.py --threads 8 --method LTM

# 指定自定义路径
python main.py --bed /path/to/gene.bed --geno /path/to/vcf/ --exp /path/to/exp.csv
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--bed` | 基因注释 BED 文件路径 | config.BED_PATH |
| `--grn` | GRN 网络 GEXF 文件路径 | config.GRN_NETWORK_PATH |
| `--grn-tsv` | GRN 网络 TSV 文件路径 | config.NET_RAW_FILE_PATH |
| `--geno` | 基因型 VCF 目录路径 | config.GENO_PATH |
| `--exp` | 基因表达数据文件路径 | config.GENE_EXPRESSION_PATH |
| `--sample` | 样本 ID 文件路径 | config.SAMPLE_PATH |
| `--eqtl` | eQTL 数据文件路径 | config.EQTL_PATH |
| `--out-weight` | 权重输出目录 | config.OUT_WEIGHT_PATH |
| `--out-info` | 信息输出目录 | config.OUT_INFO_PATH |
| `--threads` | 并行线程数 | 5 |
| `--windows` | 基因窗口大小 (bp) | 100000 |
| `--method` | 影响力算法 (RRW/LTM/KATZ/PATH) | LTM |
| `--tf-numbers` | 选择的 TF 数量 | 10 |
| `--cv-r2` | 进行交叉验证 R² 计算 | True |
| `--no-cv-r2` | 跳过交叉验证 | - |

## 输入文件格式

### 基因注释文件 (BED)

制表符分隔，包含以下列：
```
chrom  start  end  strand  gene_id  gene_name  gene_type
```

### 基因表达文件

制表符分隔，格式如下：
```
CHROM  GeneStart  GeneEnd  TargetID  GeneName  Sample1  Sample2  ...
```

### GRN 网络文件

支持两种格式：
- **GEXF**: NetworkX 可读的图格式
- **TSV**: 两列（TF, Target）的调控关系表

### eQTL 文件

制表符分隔，至少包含 `Gene` 和 `SNPPos` 列。

## 输出文件

### weight_GRN.csv

模型权重文件，包含以下列：
```
CHROM  POS  snpID  REF  ALT  TargetID  MAF  p_HWE  ES
```

### info_GRN.csv

训练信息文件，包含以下列：
```
CHROM  GeneStart  GeneEnd  TargetID  GeneName  sample_size  n_snp  
n_effect_snp  CVR2  TrainPVALUE  TrainR2  k-fold  alpha  Lambda  cvm  CVR2_threshold
```

## 项目结构

```
GRNTWAS/
├── main.py                    # 主程序入口
├── config.py                  # 配置文件
├── GRN_guided_adaptive_selection.py  # graph-guided adaptive selection（Lasso）
├── GRNutils.py               # 基因型处理工具函数
├── Regular_subgraph_build.py  # 图算法模块
├── model/                     # 正则化模型
│   └── Group_spares_lasso.py
├── requirements.txt           # Python 依赖
└── README.md                 # 项目文档
```

## 算法说明

### 影响力传播算法

- **LTM (Linear Threshold Model)**: 线性阈值模型，考虑直接和间接调控关系
- **KATZ**: Katz 中心性，基于路径长度计算节点重要性
- **RRW (Restarted Random Walk)**: 重启随机游走，模拟网络信息传播
- **PATH**: 基于前驱节点的直接连接关系

### 模型选择

系统通过 5 折交叉验证自动选择最佳模型：
- ElasticNet
- Lasso
- DPR (Dirichlet Process Regression)

## 引用

如果您使用了 GRNTWAS，请引用：

```bibtex
@article{grntwas2024,
  title={GRNTWAS: Gene Regulatory Network-guided Transcriptome-Wide Association Study},
  author={Wang, Shixian},
  year={2024}
}
```

## 许可证

MIT License

## 联系方式

如有问题，请提交 Issue 或联系作者。
