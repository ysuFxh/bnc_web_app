
from bnc_core.base.base_bnc import BaseBNC
from bnc_core.classifiers import NBClassifier, KDBClassifier, TANClassifier, AODEClassifier

class ClassifierFactory:
    """分类器工厂：根据算法名创建对应的BNC分类器实例（核心：映射算法名→分类器类）"""
    # 算法名 → 分类器类的映射（新增算法只需在这里加一行）
    _CLASSIFIER_MAP = {
        "nb": NBClassifier,  # 朴素贝叶斯
        "kdb": KDBClassifier,  # KDB分类器
        "tan": TANClassifier,  # TAN分类器
        "aode": AODEClassifier
    }

    # 定义各算法的特有参数（可选，用于过滤）
    _ALGORITHM_PARAMS = {
        "kdb": ["k"],  # KDB仅接收k参数
        "nb": [],  # NB无特有参数
        "tan": ["root_node"],  # TAN的特有参数（示例）
        "aode": []  # AODE的特有参数（示例）
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
        # 过滤参数：仅保留当前算法的特有参数 + 通用参数（如param_estimator）
        common_params = ["param_estimator"]  # 所有分类器都有的通用参数
        allowed_params = common_params + cls._ALGORITHM_PARAMS.get(algorithm, [])
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}

        classifier_cls = cls._CLASSIFIER_MAP[algorithm]

        return classifier_cls(**filtered_kwargs)
