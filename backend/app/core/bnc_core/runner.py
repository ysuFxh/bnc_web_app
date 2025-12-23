# bnc/runner.py
from typing import Dict, Any
from backend.app.core.bnc_core.base.base_bnc import BaseBNC
from backend.app.core.bnc_core.classifiers import NBClassifier, KDBClassifier, TANClassifier, AODEClassifier


class ClassifierFactory:
    """分类器工厂：根据算法名创建对应的BNC分类器实例（核心：映射算法名→分类器类）"""
    # 算法名 → 分类器类的映射（新增算法只需在这里加一行）
    _CLASSIFIER_MAP = {
        "nb": NBClassifier,  # 朴素贝叶斯
        "kdb": KDBClassifier,  # KDB分类器
        "tan": TANClassifier,  # TAN分类器
        "aode": AODEClassifier
    }

    @classmethod
    def create_classifier(cls, algorithm: str, **kwargs) -> BaseBNC:
        """
        创建分类器实例
        :param algorithm: 算法名（小写，如"nb"/"kdb"/"tan"）
        :param kwargs: 分类器初始化参数（如class_name、param_estimator、k等）
        :return: 对应的BNC分类器实例
        """
        algorithm = algorithm.lower()  # 统一小写，避免大小写问题
        if algorithm not in cls._CLASSIFIER_MAP:
            raise ValueError(
                f"不支持的算法：{algorithm}，支持的算法：{list(cls._CLASSIFIER_MAP.keys())}"
            )
        # 创建分类器实例（传递初始化参数）
        classifier_cls = cls._CLASSIFIER_MAP[algorithm]
        return classifier_cls(**kwargs)


def run_bnc(
        algorithm: str,
        data_path: str,
        param_estimator: str = "MLE",
        test_size: float = 0.2,
        random_state: int = 42,
        # visualize: bool = True,
        # interactive: bool = False,
        # save_model_path: Optional[str] = None,
        **kwargs
) -> Dict[str, Any]:
    """
    统一的BNC运行接口：仅指定算法名，一键完成全流程
    :param algorithm: 算法名（"nb"/"kdb"/"tan"）
    :param data_path: 数据路径（Excel/CSV）
    :param param_estimator: 参数学习方法（"MLE"/"Bayesian"）
    :param test_size: 测试集比例
    :param random_state: 随机种子
    :param TODO visualize: 是否可视化网络结构
    :param TODO interactive: 可视化是否为交互式（True→HTML，False→静态图）
    :param TODO save_model_path: 模型保存路径（None→不保存）
    :param kwargs: 传递给分类器/预处理的扩展参数（如KDB的k值、预处理的n_bins等）
    :return: 运行结果字典（包含分类器实例、预测结果、评估指标）
    """

    print(f"========== 开始运行BNC分类器：{algorithm.upper()} ==========")
    #
    # # 步骤1：创建分类器实例（工厂模式）
    print(f"创建{algorithm.upper()}分类器实例...")
    classifier = ClassifierFactory.create_classifier(
        algorithm=algorithm,
        param_estimator=param_estimator,
        **kwargs  # 传递分类器专属参数（如KDB的k值）
    )

    # 步骤2：加载数据
    print(f"加载数据：{data_path}")
    classifier.load_data(data_path)

    # 步骤3：预处理（传递扩展参数，如n_bins）
    print("数据预处理...")
    classifier.data_process()

    # 步骤4：划分训练/测试集
    print(f"划分训练/测试集（测试集比例：{test_size}）")
    classifier.split_data(test_size=test_size, random_state=random_state)

    # 步骤5：结构学习
    print("结构学习...")
    classifier.structure_learning()

    # 步骤6：参数学习
    print(f"参数学习（方法：{param_estimator}）")
    classifier.param_learning(**kwargs)

    # 步骤7：预测
    print("模型预测...")
    classifier.predict()

    # 步骤8：评估
    print("模型评估...")
    result = classifier.evaluate()

    # # 步骤9：可视化 TODO
    # if visualize:
    #     logger.info(f"可视化网络结构（交互式：{interactive}）")
    #     classifier.visualize(save_path=f"{algorithm}_bnc.html" if interactive else None, interactive=interactive)
    #
    # # 步骤10：保存模型  TODO
    # if save_model_path:
    #     logger.info(f"保存模型到：{save_model_path}")
    #     classifier.save_model(save_model_path)

    # 整理运行结果
    # result = {
    #     "classifier": classifier,       # 分类器实例
    #     "preds": classifier.preds,      # 预测结果
    #     "metrics": classifier.metrics,  # 评估指标
    #     "model": classifier.model       # BN模型对象
    # }

    print(f"========== {algorithm.upper()}分类器运行完成 ==========\n")
    return result
