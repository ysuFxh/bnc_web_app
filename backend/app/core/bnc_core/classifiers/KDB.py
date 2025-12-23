from backend.app.core.bnc_core.base import get_node_names
from backend.app.core.bnc_core.base.base_bnc import BaseBNC
import numpy as np
from typing import Tuple
from backend.app.core.bnc_core.utils import it_calc_tool


class KDBClassifier(BaseBNC):
    """KDB，k值默认为2，可以在实例化的时候传入k"""

    def __init__(self, param_estimator: str = "MLE", k: int = 2):
        super().__init__(param_estimator)
        self.k = k
        # 初始化成员变量（避免后续方法变量未定义）
        self.MI = None  # 特征与类标签的互信息
        self.CMI = None  # 特征间条件互信息（给定类标签）
        self.sort_attribute = None  # 按MI排序后的特征列表

    # 继承可以使用父类的成员变量
    def _structure_learning(self, train_data):
        """重写了基类中的结构学习方法，其它方法复用基类"""

        all_names, class_name, feature_names = get_node_names(train_data)
        self.attr = train_data.drop(columns=[class_name])
        self.class_label = train_data[class_name]  # 类标签
        self.Nnodes = train_data.shape[1]  # 特征数量

        self.MI, self.CMI, self.sort_attribute = self.attrbute_sort(feature_names)
        self.dag = np.zeros((self.Nnodes, self.Nnodes))
        return self.learn_struct(feature_names,self.dag)

    def attrbute_sort(self, feature_names) -> Tuple[np.ndarray, np.ndarray, list]:
        """按照MI属性排序"""
        CMI = np.zeros((self.Nnodes, self.Nnodes))
        MI = np.zeros(self.Nnodes)
        #计算MI
        MI = it_calc_tool.calc_mi(MI, self.attr, self.class_label)
        # 计算CMI
        CMI = it_calc_tool.calc_cmi(CMI, self.attr, self.class_label)

        feat_mi_pairs = list(zip(feature_names, MI))  # 将特征与MI值绑定
        feat_mi_sorted = sorted(feat_mi_pairs, key=lambda x: x[1], reverse=True)  # lamda表达式声明排序依据为按MI排序
        sort_attribute = [feat for feat, mi in feat_mi_sorted]  # 排序后的特征名列表
        return MI, CMI, sort_attribute

    def learn_struct(self,feature_names, dag: np.ndarray):
        """学习KDB结构"""
        # 步骤1：建立「特征名→索引」的映射（方便通过特征名找CMI矩阵的索引）
        feat2idx = {feat: idx for idx, feat in enumerate(feature_names)}
        # 步骤2：遍历排序后的每个特征（除第一个，第一个无前置特征）
        for i, child_feat in enumerate(self.sort_attribute):
            child_idx = feat2idx[child_feat]
            # 第一个特征无前置特征，跳过
            if i == 0:
                continue

            # 候选父节点：排序中当前特征之前的所有特征
            candidate_feats = self.sort_attribute[:i]

            # 计算当前特征与所有候选父节点的CMI
            cmi_scores = []
            for parent_feat in candidate_feats:
                parent_idx = feat2idx[parent_feat]
                # CMI[child_idx, parent_idx] = I(child; parent | C)
                cmi = self.CMI[child_idx, parent_idx]
                cmi_scores.append((parent_feat, parent_idx, cmi))

            # 按CMI降序排序，选择前k个作为父节点
            cmi_scores_sorted = sorted(cmi_scores, key=lambda x: x[2], reverse=True)
            selected_parents = cmi_scores_sorted[:self.k]

            # 更新DAG邻接矩阵：父→子
            for _, parent_idx, _ in selected_parents:
                dag[parent_idx, child_idx] = 1

        # 构建类节点到其他所有节点的边
        for j in range(self.Nnodes - 1):
            dag[self.Nnodes - 1, j] = 1

        return dag
