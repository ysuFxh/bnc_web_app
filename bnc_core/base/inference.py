import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
import pandas as pd


def bn_predict(model: DiscreteBayesianNetwork, test_feat: pd.DataFrame):
    """
    模型预测：基于变量消去法计算后验概率，取最大概率类
    :param model:模型
    :param test_feat：测试数据
    :return prediction, pred_prob_values ：预测标签，预测概率
    """
    infer = VariableElimination(model)
    predictions = []
    pred_prob_values=[]

    # 逐样本预测
    for idx, row in test_feat.iterrows():
        # 构造证据（特征值）
        evidence = row.to_dict()
        # 校验并过滤未知状态
        valid_evidence = {}
        for var, val in evidence.items():
            # 获取变量的有效状态
            valid_states = model.get_cpds(var).state_names[var]
            # 转换为np.int64类型 + 校验状态
            val_int64 = np.int64(val)
            if val_int64 in valid_states:
                valid_evidence[var] = val_int64
            else:
                # 未知状态映射为第一个有效状态
                valid_evidence[var] = valid_states[0]
        # 执行推理

        # try:
        # 计算后验概率 P(class|evidence)
        posterior = infer.query(variables=["class"], evidence=valid_evidence)

        # ========== 关键修复：旧版pgmpy无argmax，手动取概率最大的类别 ==========
        # 步骤1：提取后验概率的取值（如array([0.2, 0.8])）
        prob_values = posterior.values
        # 步骤2：提取class的所有状态（如[0, 1]）
        class_states = list(posterior.state_names["class"])
        # 步骤3：找到概率最大的状态索引，匹配对应的类别
        max_prob_idx = np.argmax(prob_values)
        pred_class = class_states[max_prob_idx]

        pred_prob_values.append(prob_values)
        predictions.append(pred_class)


        # except Exception as e:
        #     # 异常时打印详细信息（便于排查）+ 用众数兜底
        #     print(f"样本{idx}推理异常：{str(e)[:100]}，使用训练集众数类 {default_class}")
        #     predictions.append(default_class)

    return pd.DataFrame(pred_prob_values),pd.Series(predictions, index=test_feat.index, name="pred_class")
