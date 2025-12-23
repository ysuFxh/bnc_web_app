
import numpy as np
def dag_to_edges(adj_mat: np.ndarray, node_names: list)->list:
    """
    将DAG邻接矩阵转换为pgmpy的DiscreteBayesianNetwork对象
    :param adj_mat: DAG邻接矩阵（n×n），adj_mat[i][j]=1 表示 节点i → 节点j 有边；0表示无边
    :param node_names: 节点名列表（顺序与邻接矩阵的行/列严格对应）
    :return: pgmpy的离散贝叶斯网络对象
    """
    edges = []
    n_nodes = len(node_names)
    for i in range(n_nodes):
        parent = node_names[i]
        for j in range(n_nodes):
            if adj_mat[i][j] == 1:
                child = node_names[j]
                edges.append((parent, child))
    
    #bn = DiscreteBayesianNetwork(edges)
    #bn.check_model()
    return edges
#TODO 
# def check_dag_acyclic(model) -> None:
#     """校验DAG无环性"""
#     try:
#         model.check_model()
#     except Exception as e:
#         raise ValueError(f"DAG不合法（含环/重复边）：{e}")