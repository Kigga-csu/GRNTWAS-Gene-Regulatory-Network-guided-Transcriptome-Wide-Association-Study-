import networkx as nx
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
#############
# 对graph中的有向cycle进行统计
def analyze_gexf_graph(gexf_file_path, output_file):
    """
    Analyzes a GEXF graph and saves cycle information to a TSV file.

    Args:
        gexf_file_path (str): Path to the GEXF file.
        output_file (str): Path to save the TSV file.

    Returns:
        None
    """

    def read_gexf_file(file_path):
        return nx.read_gexf(file_path)

    def find_cycles(graph):
        return nx.simple_cycles(graph)

    def get_cycle_info(cycles):
        cycle_info = []
        for cycle in cycles:
            num_nodes = len(cycle)
            num_edges = sum(graph[cycle[i]][cycle[i + 1]]['weight'] for i in range(num_nodes - 1))
            cycle_info.append((num_nodes, num_edges))
        return cycle_info

    def save_to_tsv(cycle_info, output_file):
        df = pd.DataFrame(cycle_info, columns=['Nodes', 'Edges'])
        df.to_csv(output_file, sep='\t', index=False)

    graph = read_gexf_file(gexf_file_path)
    cycles = find_cycles(graph)
    cycle_info = get_cycle_info(cycles)
    save_to_tsv(cycle_info, output_file)

    print(f"找到了 {len(cycles)} 个环路，已保存到 {output_file} 文件中。")


# 读取.gexf格式的图数据
def read_gexf_file(file_path):
    return nx.read_gexf(file_path)
def is_weighted_graph(graph):
    # 检查图是否包含权重属性
    for u, v, data in graph.edges(data=True):
        if 'weight' in data:
            return True  # 如果边包含 'weight' 属性，说明图是加权的
    return False  # 否则，图是无权重的

# 使用广度优先搜索构建子图
def build_subgraph(graph, target_node):
    visited = set()
    subgraph = nx.DiGraph()

    queue = [target_node]
    visited.add(target_node)

    while queue:
        current_node = queue.pop(0)
        for neighbor in graph.predecessors(current_node):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                subgraph.add_edge(neighbor, current_node, weight=graph[neighbor][current_node]['weight'])

    return subgraph


def build_subgraph_cycle(graph, target_node):
    visited = set()
    subgraph = nx.DiGraph()

    queue = [target_node]
    visited.add(target_node)

    while queue:
        current_node = queue.pop(0)
        for neighbor in graph.predecessors(current_node):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                subgraph.add_edge(neighbor, current_node, weight=graph[neighbor][current_node]['weight'])

    # 添加前驱节点之间的边
    for node in subgraph.nodes():
        for neighbor in graph.predecessors(node):
            if neighbor in subgraph.nodes():
                subgraph.add_edge(neighbor, node, weight=graph[neighbor][node]['weight'])

    return subgraph

def build_subgraph_cycle_no_weight(graph, target_node):
    """
    基于输入的有向图构建以目标节点为根的子图，子图中只保留有向边，无权重。

    参数:
    - graph: 有向图（NetworkX DiGraph）
    - target_node: 目标节点

    返回:
    - subgraph: 以目标节点为根的有向无权子图
    """
    visited = set()
    subgraph = nx.DiGraph()

    queue = [target_node]
    visited.add(target_node)

    while queue:
        current_node = queue.pop(0)
        for neighbor in graph.predecessors(current_node):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                subgraph.add_edge(neighbor, current_node)

    # 添加前驱节点之间的边，避免重复添加
    for node in subgraph.nodes():
        for neighbor in graph.predecessors(node):
            if neighbor in subgraph.nodes() and not subgraph.has_edge(neighbor, node):
                subgraph.add_edge(neighbor, node)

    return subgraph

# 示例用法
def build_subgraph_from_gexf_plt(graph, target_node):
    target_node1 = target_node

    if target_node1 not in graph:
        print(f"The node {target_node1} is not in the digraph.")
        return None  # 或者抛出异常: raise nx.NetworkXError(f"The node {target_node} is not in the digraph.")
    subgraph = build_subgraph_cycle(graph, target_node1)

    # 输出子图的边和节点
    #for edge in subgraph.edges(data=True):
        #print(f"Edge from {edge[0]} to {edge[1]}, weight: {edge[2]['weight']}")

    # 计算节点和边的数量
    num_nodes = subgraph.number_of_nodes()
    num_edges = subgraph.number_of_edges()

    # 输出节点和边的数量
    print(f"图中的节点数量: {num_nodes}")
    print(f"图中的边的数量: {num_edges}")

    return subgraph

def build_subgraph_from_gexf_noweight(graph, target_node):
    target_node1 = target_node
    if target_node1 not in graph:
        print(f"The node {target_node1} is not in the digraph.")
        return None
    subgraph = build_subgraph_cycle_no_weight(graph, target_node1)
    if not subgraph.nodes():
        print(f"No predecessors found for {target_node1}, returning empty subgraph.")
        return None

    num_nodes = subgraph.number_of_nodes()
    num_edges = subgraph.number_of_edges()
    print(f"Subgraph nodes: {num_nodes}, edges: {num_edges}")
    return subgraph

def build_subgraph_from_gexf(gexf_path, target_node):
    graph = read_gexf_file(gexf_path)
    subgraph = build_subgraph(graph, target_node)

    # 计算节点和边的数量
    num_nodes = subgraph.number_of_nodes()
    num_edges = subgraph.number_of_edges()

    # 输出节点和边的数量
    print(f"图中的节点数量: {num_nodes}")
    print(f"图中的边的数量: {num_edges}")

    # 可以根据需要绘制子图
    return subgraph


def normalize_graph(graph0):
    """
    对图的边权重进行标准化。
    每个节点的出边权重之和应该为1。（实际情况下 需要出边权重之和为1吗？）
    """
    for node in graph0.nodes():
        total_weight = sum(graph0[node][neighbor]['weight'] for neighbor in graph0.successors(node))
        for neighbor in graph0.successors(node):
            if total_weight > 0:
                graph0[node][neighbor]['weight'] = graph0[node][neighbor]['weight'] / total_weight
#######################################################################################
'''
以下部分函数均是计算节点重要性算法 通过比较来得到最后结果
'''
#######################################################################################
def katz_influence_on_target(graph, target, alpha=0.1, beta=1.0):
    target = target[0]
    """
    使用 Katz 中心性计算每个节点对目标节点的影响力。

    参数:
    - graph: 有向加权图（NetworkX图）。
    - target: 目标节点。
    - alpha: Katz 中心性的衰减因子 (0 < alpha < 1)，表示每次迭代的权重衰减。
    - beta: Katz 中心性的初始值，通常设置为 1。

    返回:
    - influence: 以节点为键，对目标节点的影响力为值的字典。
    """
    # 计算所有节点的 Katz 中心性
    katz_centrality = nx.katz_centrality(graph, alpha=alpha, beta=beta, weight='weight')

    # 初始化影响力字典，目标节点自身的影响力为0
    influence = {node: 0 for node in graph.nodes() if node != target}

    # 逐个检查节点对目标节点的路径
    for node in graph.nodes():
        if node != target:
            # 计算从该节点到目标节点的路径
            paths = nx.all_simple_paths(graph, source=node, target=target)

            # 对于每一条路径，根据长度和Katz公式计算贡献
            for path in paths:
                path_length = len(path) - 1  # 路径长度
                influence[node] += (alpha ** path_length) * katz_centrality[node]

    # 按照影响力从大到小排序


    return influence

def betweenness_centrality_influence(graph, target):
    target = target[0]
    """
    使用介数中心性模型计算所有节点对指定目标节点的影响力。

    参数:
    - graph: 有向加权图（NetworkX图）。
    - target: 目标节点。

    返回:
    - influence: 以节点为键，对目标节点的影响力为值的字典。
    """
    # 计算所有节点的介数中心性 (忽略目标节点自身)
    betweenness_centrality = nx.betweenness_centrality(graph, weight='weight', normalized=True)

    # 初始化影响力字典，并去除目标节点自身
    influence = {node: betweenness_centrality[node] for node in graph.nodes() if node != target}

    # 按照影响力从大到小排序
    #sorted_influence = sorted(influence.items(), key=lambda item: item[1], reverse=True)

    return influence

def betweenness_centrality_influence4target(graph, target):
    target = target[0]
    """
    计算每个节点对目标节点的影响力，基于从所有节点到目标节点的最短路径的介数中心性。

    参数:
    - graph: 有向加权图（NetworkX图）。
    - target: 目标节点。

    返回:
    - influence: 以节点为键，对目标节点的影响力为值的字典。
    """
    # 初始化影响力字典，目标节点的影响力为0
    influence = {node: 0 for node in graph.nodes() if node != target}

    # 遍历每个节点，计算其到目标节点的最短路径
    for source in graph.nodes():
        if source == target:
            continue  # 跳过目标节点自身

        # 找出从 source 到 target 的所有最短路径
        try:
            paths = list(nx.all_shortest_paths(graph, source=source, target=target, weight='weight'))
        except nx.NetworkXNoPath:
            continue  # 如果没有路径，则跳过该节点

        # 计算这些路径上每个中间节点的贡献
        for path in paths:
            # 路径上的中间节点（排除 source 和 target）
            for node in path[1:-1]:
                influence[node] += 1

    return influence


def linear_threshold_influence(graph, target):
    """
    使用线性阈值模型计算所有节点对指定目标节点的影响力。

    参数:
    - graph: 有向图（NetworkX图），可以是加权或无权。
    - target: 目标节点。

    返回:
    - influence: 以节点为键，对目标节点的影响力为值的字典。
    """
    influence = {node: 0 for node in graph.nodes() if node != target}
    direct_neighbors = [] #存储直接相连的节点

    # 计算直接影响力，并标记直接相连的节点
    for node in graph.nodes():
        if graph.has_edge(node, target):
            direct_neighbors.append(node)
            if 'weight' in graph[node][target]:
                influence[node] = graph[node][target]['weight']
            else:
                influence[node] = 1  # 无权图，设置直接影响力为1

    # 计算间接影响力 (累加)
    for node in graph.nodes():
        if node != target:
            for neighbor in graph.successors(node):
                if graph.has_edge(node, neighbor) and neighbor != target:
                    if 'weight' in graph[node][neighbor]:
                        influence[node] += (graph[node][neighbor]['weight'] * influence[neighbor])
                    else:
                        influence[node] += influence[neighbor] #无权图，默认weight为1

    # 给直接相连的节点增加影响力
    for node in direct_neighbors:
        influence[node] += 1000000

    return influence

def restarted_random_walk_influence(G, target_node, num_walks=10000, walk_length=5):
    """
    计算有向无权图上节点的影响力，基于重启随机游走。

    参数:
    - G: 有向无权图 (NetworkX DiGraph)。
    - target_node: 目标节点 (列表，仅使用第一个元素)。
    - num_walks: 随机游走的次数。
    - walk_length: 每次游走的长度。

    返回:
    - influence_scores: 以节点为键，影响力得分为值的字典。
    """
    target_node = target_node[0]
    if not G.nodes():  # 检查图是否为空
        print(f"Subgraph for {target_node} is empty, skipping random walk.")
        return {node: 0 for node in G.nodes()}  # 返回空字典
    influence_scores = {node: 0 for node in G.nodes()}

    for walk in range(num_walks):
        current_node = np.random.choice(list(G.nodes()))
        path = [current_node]

        for step in range(walk_length):
            if current_node == target_node:
                influence_scores[path[0]] += 1
                break
            successors = list(G.successors(current_node))
            if successors:
                current_node = np.random.choice(successors)  # 均匀选择后继节点
                path.append(current_node)
            else:
                break

        if current_node != target_node:
            continue

        for node in path:
            influence_scores[node] += 1

    return influence_scores

def weighted_restarted_random_walk_influence(G, target_node, num_walks=10000, walk_length=5):
    """
    使用重启随机游走模型计算所有节点对指定目标节点的影响力。

    参数:
    - graph: 有向加权图（NetworkX图）。
    - target: 目标节点。

    返回:
    - influence: 以节点为键，对目标节点的影响力为值的字典。

    不对权重进行标准化版本
    """

    target_node = target_node[0]
    influence_scores = {node: 0 for node in G.nodes()}

    for walk in range(num_walks):
        # 随机选择一个起始节点
        current_node = np.random.choice(list(G.nodes()))
        path = [current_node]  # 记录游走路径

        for step in range(walk_length):
            # 如果当前节点是目标节点，则增加起始节点的得分，并重新开始游走
            if current_node == target_node:
                influence_scores[path[0]] += 1
                break
            successors = list(G.successors(current_node))
            if successors:
                # 根据边的权重归一化，获取概率分布
                weights = [G[current_node][succ]['weight'] for succ in successors]
                total_weight = sum(weights)
                if total_weight > 0:  # 避免除以零
                    probabilities = [w / total_weight for w in weights]
                    current_node = np.random.choice(successors, p=probabilities)
                    path.append(current_node)  # 将当前节点添加到路径中
                else:
                    # 如果权重总和为0，表示没有有效的边可以走，跳出循环重新开始游走
                    break
            else:
                # 如果当前节点没有后继节点，跳出循环重新开始游走
                break

        # 如果在游走长度内没有到达目标节点，则不增加得分
        if current_node != target_node:
            continue

        # 游走到达目标节点后，增加路径上所有节点的得分
        for node in path:
            influence_scores[node] += 1

    return influence_scores

def get_directly_connected_nodes(graph, target_node):
    #target_node = target_node[0]
    """
    从有向无权图中提取与目标节点直接相连的节点（前驱和后继）。

    参数:
    - graph: 有向无权图（NetworkX DiGraph）
    - target_node: 目标节点

    返回:
    - predecessors: 目标节点的前驱节点列表
    - successors: 目标节点的后继节点列表
    """
    # 获取目标节点的前驱节点（指向目标节点的节点）
    predecessors = list(graph.predecessors(target_node))


    return predecessors

def Independent_Cascade_Model_influence(G, target_node, num_walks=100000, walk_length=20):
    """
    执行独立级联模型并评估图中节点对目标节点的影响力。

    参数:
    - G: 有向加权图
    - target_node: 目标节点，游走的结束点
    - num_walks: 游走次数
    - walk_length: 每次游走的步数

    返回:
    - 一个字典，包含每个节点的影响力得分（到达目标节点的次数）
    """
    # 影响力得分字典
    influence_scores = {node: 0 for node in G.nodes()}

    # 对每个游走进行循环
    for _ in range(num_walks):
        current_node = np.random.choice(list(G.nodes()))
        for _ in range(walk_length):
            if current_node == target_node:
                influence_scores[current_node] += 1
                break
            # 获取当前节点的后继节点和权重，并进行归一化处理
            successors = list(G.successors(current_node))
            if successors:
                weights = [G[current_node][succ].get('weight', 1) for succ in successors]
                # 归一化权重
                if weights:
                    probabilities = weights / np.sum(weights)
                    current_node = np.random.choice(successors, p=probabilities)
                else:
                    break
            else:
                break

    return influence_scores

def calculate_influence_by_paths(graph, target):
    target = target[0]
    """
    计算所有节点对目标节点的影响力，基于从所有节点到目标节点的路径数量。

    参数:
    - graph: 有向无权图（NetworkX DiGraph）。
    - target: 目标节点。

    返回:
    - influence: 以节点为键，对目标节点的影响力为值的字典。
    """
    influence = {node: 0 for node in graph.nodes() if node != target}

    # 计算从其他节点到目标节点的最短路径数量
    for node in graph.nodes():
        if node == target:
            continue
        # 使用 BFS 计算从 node 到 target 的最短路径数量
        try:
            # 计算所有到目标节点的最短路径
            shortest_paths = nx.all_shortest_paths(graph, source=node, target=target)
            influence[node] = len(list(shortest_paths))  # 记录最短路径的数量作为影响力
        except nx.NetworkXNoPath:
            influence[node] = 0  # 如果没有路径，影响力为 0

    return influence

#######################################################################################
'''
以下部分函数均是计算节点重要性算法 通过比较来得到最后结果
'''
#######################################################################################


def gene_name_2_ID(bed_file_path, gene_names):
    """
    专家组修改：确保返回的 gene_ids 顺序与输入的 gene_names 严格一致。
    """
    # 读取.bed文件
    df = pd.read_csv(bed_file_path, sep='\t', header=None,
                     names=['chrom', 'start', 'end', 'strand', 'gene_id', 'gene_name', 'gene_type'])

    # 创建 name 到 id 的映射字典
    name_2_id_map = dict(zip(df['gene_name'], df['gene_id']))

    # 按照输入 gene_names 的顺序提取 ID
    # 如果某个 name 不在 bed 文件中，则不包含在结果中（或者填 None，这里根据您原逻辑是剔除）
    valid_ids = []
    valid_names = []  # 用于调试或验证

    for name in gene_names:
        if name in name_2_id_map:
            valid_ids.append(name_2_id_map[name])
            valid_names.append(name)

    return np.array(valid_ids), np.array(valid_names)



def get_influence_tfgeneID_LTM(graph, target_node=None, numbers=None, expression_TF_gene=None, bed_path=None):
    """
    基于线性阈值模型（LTM）计算目标节点在网络中的影响力，并返回影响力最大的若干个节点的基因ID。
    可以使用有向无权图 和 有向加权图
    参数：
        gexf_path (str): GEXF文件的路径，默认为脑转录因子网络文件。
        target_node (str): 目标节点的名称。
        numbers (int): 要筛选出的影响力最大的节点数量。

    返回：
        numpy.ndarray: 影响力最大的节点的基因ID数组。
    """

    # 1. 构建子图

    # 判断图是否是加权图
    if is_weighted_graph(graph):
        print("The graph is weighted. Using build_subgraph_from_gexf_plt.")
        # 使用加权图构建子图
        sub_graph = build_subgraph_from_gexf_plt(graph, target_node)
    else:
        print("The graph is unweighted. Using build_subgraph_from_gexf_noweight.")
        # 使用无权图构建子图
        sub_graph = build_subgraph_from_gexf_noweight(graph, target_node)

    # 2. 检查子图是否构建成功
    if sub_graph is None:
        return None
    # 3. 计算影响力（使用线性阈值模型 LTM）
    pankrang = linear_threshold_influence(sub_graph, target_node)
    # 调用 linear_threshold_influence 函数，使用 LTM 算法计算子图中每个节点的影响力。
    # 4. 排序影响力得分
    sorted_scores = sorted(pankrang.items(), key=lambda item: item[1], reverse=True)
    # 5. 筛选影响力最大的节点
    top_influencers = sorted_scores[:numbers]
    # 6. 提取基因名称
    gene_names = [gene[0] for gene in top_influencers]
    # 7. 转换为 NumPy 数组
    geneNAME = np.array(gene_names, dtype=object).ravel()
    geneNAME = np.intersect1d(geneNAME, expression_TF_gene)
    # 将基因名称列表转换为 NumPy 数组，并展平为一维数组。
    # 8. 获取基因 ID
    if bed_path is None:
        bed_path = '/data/lab/wangshixian/GRNTWAS_STAR/data/anno_info/gene.bed'
    tf_gene_ID, _ = gene_name_2_ID(bed_path, geneNAME)
    # 9. 返回基因 ID 列表
    return tf_gene_ID


def get_influence_tfgeneID_KATZ(gexf_path='/data/lab/wangshixian/GRNTWAS_STAR/data/GRN_data/brain_tf_network.gexf',
                               target_node=None, numbers=None, expression_TF_gene=None):
    """
    使用 Katz 中心性算法计算目标节点在网络中的影响力，并返回影响力最大的若干个节点的基因ID。
    参数：
        gexf_path (str): GEXF文件的路径，默认为脑转录因子网络文件。
        target_node (str): 目标节点的名称。
        numbers (int): 要筛选出的影响力最大的节点数量。
    返回：
        numpy.ndarray: 影响力最大的节点的基因ID数组，如果子图构建失败则返回 None。
    """

    # 1. 构建子图并检查是否成功
    sub_graph = build_subgraph_from_gexf_plt(gexf_path, target_node)
    if sub_graph is None:
        return None  # 如果子图构建失败，直接返回 None

    # 2. 计算 Katz 中心性
    influence_scores = katz_influence_on_target(sub_graph, target_node)

    # 3. 排序并筛选影响力最大的节点
    if influence_scores: # 检查influence_scores是否为空
        sorted_influencers = sorted(influence_scores.items(), key=lambda item: item[1], reverse=True)
        top_influencers = [item[0] for item in sorted_influencers[:numbers]] # 直接提取基因名
    else:
        return np.array([]) # 如果influence_scores为空，返回空数组

    # 6. 提取基因名称
    gene_names = [gene[0] for gene in top_influencers]
    # 7. 转换为 NumPy 数组
    geneNAME = np.array(gene_names, dtype=object).ravel()
    geneNAME = np.intersect1d(geneNAME, expression_TF_gene)
    tf_gene_ids = gene_name_2_ID('/data/lab/wangshixian/GRNTWAS_STAR/data/anno_info/gene.bed', geneNAME)

    return tf_gene_ids

#######################################有向无权网络#############################

def get_influence_tfgeneID_paths(sub_graph, target_node=None, numbers=None):
    """基于路径计算影响力并返回 TF 基因 ID"""
    pankrang = calculate_influence_by_paths(sub_graph, target_node)
    sorted_scores = sorted(pankrang.items(), key=lambda item: item[1], reverse=True)
    top_influencers = sorted_scores[:numbers]
    gene_names = [gene[0] for gene in top_influencers]
    geneNAME = np.array(gene_names, dtype=object).ravel()
    tf_gene_ID = gene_name_2_ID('/data/lab/wangshixian/GRNTWAS_STAR/data/anno_info/gene.bed', geneNAME)
    return tf_gene_ID

def get_influence_tfgeneID_predecessors(sub_graph, target_node=None, numbers=None, expression_TF_gene=None):
    """基于前驱节点计算影响力并返回 TF 基因 ID"""
    pankrang = get_directly_connected_nodes(sub_graph, target_node)
    gene_names = [gene for gene in pankrang]
    geneNAME = np.array(gene_names, dtype=object).ravel()
    geneNAME = np.intersect1d(geneNAME, expression_TF_gene)
    tf_gene_ID = gene_name_2_ID('/data/lab/wangshixian/GRNTWAS_STAR/data/anno_info/gene.bed', geneNAME)
    return tf_gene_ID

def get_influence_tfgeneID_RRW(gexf_path = '/data/lab/wangshixian/GRNTWAS_STAR/data/GRN_data/brain_tf_network.gexf', target_node=None,numbers=None, expression_TF_gene=None):
    sub_graph = build_subgraph_from_gexf_noweight(gexf_path, target_node)
    if sub_graph is None:
        print(f"No valid subgraph for {target_node}, skipping.")
        return None
    pankrang = restarted_random_walk_influence(sub_graph, target_node)
    sorted_scores = sorted(pankrang.items(), key=lambda item: item[1], reverse=True)
    top_influencers = sorted_scores[:numbers]
    gene_names = [gene[0] for gene in top_influencers]
    print(gene_names)
    geneNAME = np.array(gene_names, dtype=object).ravel()
    geneNAME = np.intersect1d(geneNAME, expression_TF_gene)
    tf_gene_ID = gene_name_2_ID('/data/lab/wangshixian/GRNTWAS_STAR/data/anno_info/gene.bed', geneNAME)
    return tf_gene_ID


def select_TFs_via_graph_guided_bayesian(gexf_path, target_node_name, target_expr_values,
                                         gene_exp_df, sampleID, bed_path, expression_TF_gene_names):
    """
    功能：实现基于图拓扑先验的贝叶斯筛选（通过加权 L1 正则化求解）。

    参数:
    - gexf_path: GRN 文件路径
    - target_node_name: 目标基因名称 (e.g., 'TP53')
    - target_expr_values: 目标基因的表达量向量 (numpy array, shape=[n_samples])
    - gene_exp_df: 全量基因表达矩阵 (DataFrame, 包含 'TargetID' 列和样本列)
    - sampleID: 样本 ID 列表 (用于从 gene_exp_df 提取正确列)
    - bed_path: BED 文件路径 (用于 ID 转换)
    - expression_TF_gene_names: 在表达矩阵中存在的 TF 名称列表
    返回:
    - selected_tf_ids: 筛选出的关键 TF ID 列表
    """

    # -------------------------------------------------------
    # 1. 图计算 (RWR 捕捉网络拓扑先验)
    # -------------------------------------------------------
    # 构建子图
    sub_graph = build_subgraph_from_gexf_noweight(gexf_path, target_node_name)
    if sub_graph is None or len(sub_graph.nodes) == 0:
        return []

    # 计算 RWR 分数 (使用 NetworkX PageRank 作为解析解)
    # 针对 TF->TG 有向图，我们在反向图上计算以寻找上游 TF
    try:
        rwr_scores_dict = nx.pagerank(sub_graph.reverse(), alpha=0.85, personalization={target_node_name: 1})
    except Exception as e:
        print(f"RWR Error: {e}")
        return []

    # -------------------------------------------------------
    # 2. 数据对齐与 ID 映射
    # -------------------------------------------------------
    # 找出既在 RWR 结果中(有拓扑关系)，又在表达矩阵中(有数据)的 TF
    # 过滤掉 Target 自己（防止自回归）
    candidate_names = [
        name for name in rwr_scores_dict.keys()
        if name in expression_TF_gene_names and name != target_node_name
    ]

    # 如果候选 TF 太少，无法进行有效回归筛选
    if len(candidate_names) < 2:
        return []

    # 提取分数并排序 (取 Top 200 进行广义回归，避免计算量过大)
    # 这是一个"软截断"，主要为了工程性能，Top 200 足够包含所有潜在调节子
    candidate_names = sorted(candidate_names, key=lambda x: rwr_scores_dict[x], reverse=True)[:100]

    # ID 转换 (使用修正后的 gene_name_2_ID，确保返回两个对齐的数组)
    tf_ids, valid_tf_names = gene_name_2_ID(bed_path, candidate_names)

    if len(tf_ids) == 0:
        return []

    # 重新提取对应的分数 (确保顺序与 ID 一致)
    final_scores = np.array([rwr_scores_dict[name] for name in valid_tf_names])

    # -------------------------------------------------------
    # 3. 构建回归矩阵 X 和 Y (无占位符，真实 pandas 操作)
    # -------------------------------------------------------
    # Y: 目标基因表达量 (已中心化)
    y = target_expr_values - np.mean(target_expr_values)

    tf_data = gene_exp_df[gene_exp_df['TargetID'].isin(tf_ids)].copy()
    tf_data = tf_data.drop_duplicates(subset=['TargetID'])  # 防止重复 ID
    tf_data = tf_data.set_index('TargetID')
    tf_data = tf_data.reindex(tf_ids)  # 强制对齐顺序

    # 如果有 NaN (可能某些 ID 没匹配上)，需要清洗
    if tf_data.isnull().values.any():
        valid_mask = ~tf_data.isnull().any(axis=1)
        tf_data = tf_data[valid_mask]
        tf_ids = tf_ids[valid_mask]  # 更新 ID 列表
        final_scores = final_scores[valid_mask]  # 更新分数列表

    if tf_data.empty:
        return []

    # 提取数值矩阵 [n_features, n_samples] -> 转置为 [n_samples, n_features]
    X = tf_data[sampleID].values.T

    # 数据标准化 (Standardization) 对贝叶斯回归至关重要
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # -------------------------------------------------------
    # 4. 构造贝叶斯先验权重 (Bayesian Prior Weights)
    # -------------------------------------------------------
    epsilon = 1e-4
    # 归一化分数，使权重在一个合理范围内
    max_s = np.max(final_scores)
    normalized_scores = final_scores / (max_s + epsilon)

    # 惩罚权重向量
    penalty_weights = 1.0 / (normalized_scores + 0.05)  # 0.05 是超参数，控制先验强度
    # -------------------------------------------------------
    # 5. 求解模型 (使用特征缩放实现加权贝叶斯筛选)
    # -------------------------------------------------------
    X_weighted = X / penalty_weights[np.newaxis, :]

    try:
        # LassoCV 自动通过交叉验证选择最佳的 Lambda (贝叶斯后验的超参数)
        model = LassoCV(cv=5, n_jobs=1, random_state=42, max_iter=2000).fit(X_weighted, y)
        # 获取非零系数的索引
        beta = model.coef_
        support_indices = np.where(model.coef_ != 0)[0]

        if len(support_indices) == 0:
            return []

        # 如果选出的太多（例如超过样本量的一半），为了保证 OLS 的稳定性，可以只取绝对值最大的 Top N
        # 但通常 Lasso 已经够稀疏了，这里直接进行下一步

    except Exception as e:
        print(f"Lasso step failed: {e}")
        return []

    # -------------------------------------------------------
    # 6. 第二阶段：OLS 显著性检验 (Significance Testing)
    # -------------------------------------------------------
    X_subset = X[:, support_indices]
    selected_ids_subset = tf_ids[support_indices]

    try:
        # 添加截距项 (OLS 需要手动添加截距)
        X_subset_with_const = sm.add_constant(X_subset)
        ols_model = sm.OLS(y, X_subset_with_const).fit()
        r2 = ols_model.rsquared_adj
        if ols_model.rsquared_adj < 0.05:
            # print(f"  Target {target_node_name}: Low R2 ({ols_model.rsquared_adj:.4f}), skipped.")
            return []
        p_values = ols_model.pvalues[1:]
        significant_mask = p_values < 0.05

        # 最终选定的 TF ID
        final_selected_tf_ids = selected_ids_subset[significant_mask].tolist()

        # print(f"  Step 1 Lasso: {len(support_indices)} -> Step 2 OLS: {len(final_selected_tf_ids)}")
        return final_selected_tf_ids

    except Exception as e:
        print(f"OLS step failed: {e}")
        # 如果 OLS 失败（例如矩阵奇异），回退到仅 Lasso 的结果
        return tf_ids[support_indices].tolist()
