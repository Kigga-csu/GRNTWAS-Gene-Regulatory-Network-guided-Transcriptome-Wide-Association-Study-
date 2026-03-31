import pandas as pd
import argparse
import networkx as nx




def extract_filename_from_path(address):
    # 使用str.rsplit来分割路径，限制分割次数为1，从右边开始
    filename_with_extension = address.split('/')[-1]
    
    # 使用str.split来分割文件名和扩展名
    filename_without_extension = filename_with_extension.split('.')[0]
    
    return filename_without_extension

def csv_information(file_path):
    # 读取TSV文件
    try:
        df = pd.read_csv(file_path, sep='\t')  # 使用sep参数指定分隔符为制表符
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()

    # 打印行列数
    print(f"行数: {df.shape[0]}")
    print(f"列数: {df.shape[1]}")

    # 打印每一列的不同数据数量
    for column in df.columns:
        unique_count = df[column].nunique()
        print(f"列 '{column}' 有 {unique_count} 种不同的数据")

    # 打印列名
    print("列名:", df.columns.tolist())
    return df

def GRN_extract_1(df, gene_name):
    filtered_data = df[df['gene'] == gene_name]
    print(filtered_data[['TF', 'bestMotif', 'NES', 'Genie3Weight', 'Confidence']])
    return filtered_data

def GRN_build_no_relationship(df):
    df = df.drop_duplicates()
    G = nx.DiGraph()
    # 遍历DataFrame中的每一行来添加边
    for index, row in df.iterrows():
        # 获取出发节点和接受节点
        source = row[0]  # 出发节点
        target = row[1]  # 接受节点

        # 添加有向边，不再包含关系属性
        G.add_edge(source, target)

    # 计算节点和边的数量
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # 输出节点和边的数量
    print(f"图中的节点数量: {num_nodes}")
    print(f"图中的边的数量: {num_edges}")
    return G

def GRN_DLG_build(df, filename):
    # 创建一个有向图
    G = nx.DiGraph()

    # 遍历DataFrame中的每一行来添加边
    for index, row in df.iterrows():
        G.add_edge(row['TF'], row['gene'], weight=row['NES'])

    # 计算节点和边的数量
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # 输出节点和边的数量
    print(f"图中的节点数量: {num_nodes}")
    print(f"图中的边的数量: {num_edges}")

    # 保存图到文件
    nx.write_gexf(G, '../data/GRN_data/'+filename+'.gexf')
    return G

# 主程序
if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="分析TSV文件的数据信息")
    parser.add_argument('file_path', type=str, help="指定TSV文件的路径")

    # 解析命令行参数
    args = parser.parse_args()
    df = csv_information(args.file_path)  # 调用函数并传入文件路径
    # 假设我们要提取名为"特定基因"的数据
    filename = extract_filename_from_path(args.file_path)
    #GRN_extract_1(df, "ACSM2A")  # 调用函数并传入DataFrame和基因名称
    G = GRN_DLG_build(df, filename)  # 调用函数并传入DataFrame
