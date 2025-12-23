"""
bnc_core.base_bnc.base_bnc 的 Docstring
BNC抽象基类
"""

import abc
from pgmpy.models import DiscreteBayesianNetwork
from backend.app.core.bnc_core.utils import dag2bn_util
from backend.app.core.bnc_core.base import data_utils, param_learning, inference, evaluation


class BaseBNC(abc.ABC):
    """
    贝叶斯网络分类器抽象基类（所有BNC分类器的父类）
    通用逻辑：数据加载/预处理/划分、参数学习、推理、评估、可视化
    扩展点：子类仅需实现 _structure_learning 方法（核心结构学习）

    继承这个基类BNC的子类大部分可以只实现一个结构学习算法即可实现完整的分类流程
    实例化分类器不仅可以自己划分数据集实现单次数据的测试，还可以通过调用外部的5轮5折交叉验证
    来实现多轮实验，
    """

    def __init__(self, param_estimator: str = "MLE"):
        """
        初始化
        :param param_estimator: 参数学习方法（"MLE"/"Bayesian"）
        """
        # 核心属性（通用）
        self.metrics = None

        self.param_estimator = param_estimator  # 参数学习类型

        #self.model = None  # pgmpy的BN对象（DiscreteBayesianNetwork/BayesianModel）

        self.preds = None  # 测试集预测结果

        self.dag=None #有向无环图，领接矩阵形式
        self.BNC_struct = None

        # 日志

    # -------------------------- 抽象方法（子类必须实现） --------------------------
    #@abc.abstractmethod
    def _structure_learning(self,train_data):
        """
        结构学习：返回dag领接矩阵
        子类只需实现该方法，其余逻辑可以复用基类
        :param train_data:
        """
        raise NotImplementedError("子类必须实现结构学习方法！")

    # -------------------------- 通用方法（所有子类复用） --------------------------
    def load_data(self, data_path: str, **kwargs):
        """加载数据（复用data_utils），可以加载excel和CSV
        :param data_path
        :return raw_data:加载的数据"""
        raw_data = data_utils.load_data(data_path, **kwargs) # self.raw_data读入的数据
        return raw_data
        #print(f"数据加载完成，形状：{self.raw_data.shape}")


    def get_node_names(self,train_data):
        """获取训练数据的所有节点名，类节点名，和特征节点名
        :return all_names, class_name, feature_names所有节点名，类节点名，和特征节点名"""
        all_names, class_name, feature_names= data_utils.get_node_names(train_data)
        return all_names, class_name, feature_names
    # TODO 这里可以实现把混乱的原始数据进行离散化等操作
    # def preprocess(self, **kwargs) -> None:
    #     """数据预处理（离散化、缺失值填充，复用data_utils）"""
    #     print("开始数据预处理...")
    #     self.processed_data = data_utils.preprocess(
    #         self.raw_data, 
    #         class_name=self.class_name,
    #         **kwargs
    #     )
    #     self.node_names = list(self.processed_data.columns)
    #     print(f"预处理完成，节点列表：{self.node_names}")

    def split_data(self,data, test_size: float = 0.2, random_state: int = 42) :
        """单次测试时需要使用此方法对数据进行分割来获取train_data, test_data
        :param data:数据
        :param test_size:测试数据的比例
        :return train_data, test_data：返回分割好的数据"""
        #print(f"划分训练/测试集（测试集比例：{test_size}）")
        train_data, test_data = data_utils.my_train_test_split(
            data,
            test_size=test_size,
            random_state=random_state
        )
        print(f"训练集形状：{train_data.shape}，测试集形状：{test_data.shape}")
        return train_data, test_data

    def structure_learning(self,train_data):
        """封装结构学习：调用子类实现，转为BN对象（复用dag_utils）
        子类实现上边的结构学习后使用此方法对其进行封装，并将dag转换成BN对象，然后才能进行参数学习
        :param train_data:训练数据

        :return self.dag,self.BNC_struct:返回dag矩阵形式，和DiscreteBayesianNetwork对象
        """
        #print("开始结构学习...")
        # 调用子类的核心结构学习，得到领接矩阵
        node_names = train_data.columns.tolist()
        self.dag = self._structure_learning(train_data)
        # 领接矩阵转边列表、
        edges = dag2bn_util.dag_to_edges(self.dag, node_names)
        # 边列表转BN对象
        self.BNC_struct = DiscreteBayesianNetwork(edges)

        # 校验DAG合法性 TODO 这里可以实现方法来检查是否存在环
        # dag_util.check_dag_acyclic(self.model)
        print(f"结构学习完成，领接矩阵：{self.dag}")
        return self.dag,self.BNC_struct
    #def structure_learning(self,X_train, y_train):

    def param_learning(self, train_data, **kwargs) :
        """参数学习（复用param_learning）
        param_estimator:支持“MLE”和Bayesian
        :param train_data:训练数据
        :param kwargs:
        :return model:返回训练好（完成参数学习）的模型
        """
        #print(f"开始参数学习（方法：{self.param_estimator}）")
        if self.param_estimator == "MLE":
            model = param_learning.mle_estimate(
                model=self.BNC_struct,
                train_data=train_data,
                **kwargs
            )
        elif self.param_estimator == "Bayesian":
            model = param_learning.bayesian_estimate(
                model=self.BNC_struct,
                train_data=train_data,
                **kwargs
            )
        print("参数学习完成")
        return model

    def predict(self,model,test_data,class_name) :
        """推理预测（复用inference）返回概率和预测标签
        :param model:训练好的模型
        :param test_data:测试数据
        :param class_name:类标签名字
        :return pred_prob_values:预测概率矩阵
        :return self.preds:预测标签
        """
        #print("开始预测...")
        test_feat = test_data.drop(columns=[class_name])
        pred_prob_values,self.preds = inference.bn_predict(
            model=model,
            test_feat=test_feat,
        )
        #print("预测完成")
        return pred_prob_values,self.preds


    def evaluate(self,test_data,class_name):
        """模型评估（复用evaluation）
        :param test_data:测试数据
        :param class_name:类标签名字
        :return metrics:目前包括acc,conf_matricx,he report

        """
        # TODO
        # 后续可以添加其他评价指标
        print("开始评估...")
        test_true = test_data[class_name]
        self.metrics= evaluation.evaluate(y_true=test_true, y_pred=self.preds)
        return self.metrics
        # print(f"准确率：{self.metrics.acc:.4f}")
        # print(f"\n混淆矩阵：\n{self.metrics.cm}")  # 加\n让格式更清晰
        # print(f"\n报告：\n{self.metrics.report}")  # 加\n让格式更清晰

    # TODO 后续可以实现dag的可视化
    # def visualize(self, save_path: str = None, interactive: bool = False) -> None:
    #     """结构可视化（复用visualization）"""
    #     print("开始可视化...")
    #     if interactive:
    #         visualization.interactive_plot(
    #             self.model,
    #             save_path=save_path or "bnc_interactive.html"
    #         )
    #     else:
    #         visualization.static_plot(self.model)
    #     print(f"可视化完成（交互式：{interactive}）")

    def save_model(self, save_path: str) -> None:
        """保存模型（pgmpy原生序列化）
        :param save_path保存路径"""
        print(f"保存模型到：{save_path}")
        self.model.save(save_path)

    def load_model(self, load_path: str) -> None:
        """加载模型
        :param load_path加载路径"""
        print(f"加载模型：{load_path}")
        self.model = DiscreteBayesianNetwork.load(load_path)
