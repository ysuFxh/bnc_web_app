import random

from backend.app.core.bnc_core.base import get_node_names
from backend.app.core.bnc_core.base.base_bnc import BaseBNC
import numpy as np
from backend.app.core.bnc_core.utils import it_calc_tool
import networkx as nx


class TANClassifier(BaseBNC):
    """TAN算法"""
    def __init__(self, param_estimator: str = "MLE", root_node:int=-1):
        super().__init__(param_estimator)
        self.root_node = root_node


    def _structure_learning(self, train_data):
        """重写了基类中的结构学习方法，其它方法复用基类"""
        # self.train_data=kwargs['train_data']
        all_names, class_name, feature_names = get_node_names(train_data)
        attr_matrix = train_data.drop(columns=[class_name])
        class_label = train_data[class_name]  # 类标签
        Nnodes = len(all_names)# 节点数量

        self.learn_struct_mst2tan(attr_matrix,class_label,Nnodes)
        return self.dag

    def learn_struct_mst2tan(self,attr_matrix,class_label,Nnodes):
        """使用最大生成树方法构建无向图，再使用dfs方法实现边的定向"""
        CMI = np.zeros((Nnodes, Nnodes))

        CMI = it_calc_tool.calc_cmi(CMI, attr_matrix, class_label)
        # 1. 从CMI矩阵构建无向图
        G = nx.Graph()
        n = CMI.shape[0]
        # 遍历邻接矩阵，添加非0权值的边（避免无连接的边）
        for i in range(n):
            for j in range(i + 1, n):  # 无向图，仅遍历上三角（避免重复）
                weight = CMI[i][j]
                if weight > 0:  # 过滤无连接的边（权值为0）
                    G.add_edge(i, j, weight=weight)

        # 2. 求解最大生成树
        mst_undirected = nx.maximum_spanning_tree(G, weight="weight")
        if self.root_node == -1: #如果用户没有指定root_node
            self.root_node = random.randint(0, Nnodes - 2)  # 随机指定根节点

        mst_directed = nx.dfs_tree(mst_undirected, source=self.root_node)  # DFS生成有向树
        # 转化成dag
        # 2. 初始化全0邻接矩阵
        self.dag = np.zeros((Nnodes, Nnodes))
        # 3. 遍历有向边，仅标记“有边=1”（完全不加权值）
        for u, v in mst_directed.edges():  # 只遍历边的拓扑
            self.dag[u][v] = 1  # 有向边u→v的位置设为1
        self.dag[Nnodes - 1, :-1] = 1  #class->features
