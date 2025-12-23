from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd
import numpy as np

from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

def mle_estimate(model:DiscreteBayesianNetwork, train_data: pd.DataFrame, **kwargs) -> DiscreteBayesianNetwork:
    """最大似然估计（MLE）
    :param model DiscreteBayesianNetwork对象，只有结构
    :param train_data:
    :param kwargs:
    :return model 训练（参数学习）完成的模型
    """
    mle = MaximumLikelihoodEstimator(model=model, data=train_data)
    # 为每个节点估计CPD
    for node in model.nodes():
        cpd = mle.estimate_cpd(node=node)
        model.add_cpds(cpd)
    return model
def bayesian_estimate(model:DiscreteBayesianNetwork, train_data: pd.DataFrame,
                       pseudo_counts: int = 1, **kwargs) -> DiscreteBayesianNetwork:
    """贝叶斯估计（拉普拉斯平滑）"""
    bayes = BayesianEstimator(model=model, data=train_data)
    for node in model.nodes():
        cpd = bayes.estimate_cpd(
            node=node,
            prior_type="dirichlet",
            pseudo_counts=pseudo_counts,
            # **kwargs
        )
        model.add_cpds(cpd)
    return model