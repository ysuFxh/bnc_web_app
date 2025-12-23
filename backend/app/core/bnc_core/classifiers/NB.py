from backend.app.core.bnc_core.base.base_bnc import BaseBNC
import numpy as np
from backend.app.core.bnc_core.base.data_utils import get_node_names

class NBClassifier(BaseBNC):
    """朴素贝叶斯分类器（仅实现结构学习）"""

    def _structure_learning(self,train_data):
        """NB结构学习：class→所有特征（无特征间边）"""
        all_names, _, _=get_node_names(train_data)

        Nnodes = len(all_names)

        self.dag = np.zeros((Nnodes, Nnodes))
        for i in range(Nnodes-1):
            self.dag[Nnodes-1, i] = 1
        return self.dag
