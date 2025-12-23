from pgmpy.models import DiscreteBayesianNetwork

from backend.app.core.bnc_core.base.base_bnc import BaseBNC
import numpy as np
from backend.app.core.bnc_core.utils import dag2bn_util
from backend.app.core.bnc_core.base import param_learning, inference, evaluation


class AODEClassifier(BaseBNC):
    """AODE分类器，继承了基类，但因为是集成分类器，所以重写了基类中方法，来适用于集成方法"""

    def __init__(self, param_estimator: str = "MLE"):
        super().__init__(param_estimator=param_estimator)
        self.metric_results = None
        self.pre_results = None
        self.dags = None
        self.models = None
        self.acc = None

    def structure_learning(self, train_data):
        """AODE结构学习"""
        node_names = train_data.columns.tolist()
        DAGs = []  # 保存所有dag
        Nnodes = len(node_names)
        class_idx = Nnodes - 1
        not_class = np.setdiff1d(range(Nnodes), class_idx)
        for i in range(Nnodes - 1):
            self.dag = np.zeros((Nnodes, Nnodes))

            self.dag[class_idx, not_class] = 1
            curr_feat = not_class[i]
            other_feats = np.setdiff1d(not_class, curr_feat)
            self.dag[curr_feat, other_feats] = 1
            DAGs.append(self.dag)
        self.dags = DAGs

    def param_learning(self, train_data, **kwargs):
        node_names = train_data.columns.tolist()
        Models = []  # 训练所有ODE模型
        for dag in self.dags:
            edges = dag2bn_util.dag_to_edges(dag, node_names)
            self.model = DiscreteBayesianNetwork(edges)
            self.model = param_learning.mle_estimate(
                model=self.model,
                train_data=train_data
            )
            Models.append(self.model)
        self.models = Models

    def predict(self, test_data, class_name):
        pre_results = []  # 保存所有ODE的预测结果
        for model in self.models:
            test_feat = test_data.drop(columns=[class_name])
            self.preds = inference.bn_predict(
                model=model,
                test_feat=test_feat,
            )
            pre_results.append(self.preds)
        self.pre_results = pre_results

    def evaluate(self, test_data, class_name):
        metric_results = []  # 所有ODE的评估结果
        for pred_results in self.pre_results:
            # print("开始评估...")
            test_true = test_data[class_name]
            self.metrics = evaluation.evaluate(y_true=test_true, y_pred=pred_results)
            metric_results.append(self.metrics)
        self.metric_results = metric_results
        acc_list = []
        for metrics in self.metric_results:  # 循环获得每个ode的acc结果
            acc = metrics.acc
            acc_list.append(acc)
        self.acc = np.mean(acc_list)  # 取平均值
        # return self.acc
        print(f"AODE准确率：{self.acc:.4f}")

    # def result_process(self):
    #     #TODO 可以添加其他评价指标
    #     acc_list = []
    #     for metrics in self.metric_results:#循环获得每个ode的acc结果
    #         acc=metrics.acc
    #         acc_list.append(acc)
    #     self.acc=np.mean(acc_list)    #取平均值
    #     print(f"AODE准确率：{self.acc:.4f}")
