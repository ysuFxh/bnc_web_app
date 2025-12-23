import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 1. 导入工厂类和Metrics类（假设已定义）
from bnc_core.utils.classifier_factory import ClassifierFactory
from bnc_core.base import data_utils

def n_round_n_fold_cv(
        algorithm: str,
        data_path: str,
        param_estimator: str = "MLE",
        random_state_base: int = 42,  # 基础随机种子，每轮+1保证划分不同
        round:int=5,
        fold:int=5,
        **kwargs  # 分类器专属参数（如k、min_samples等）
):
    """
    执行5轮5折交叉验证（工厂模式创建分类器）
    :param fold: 折数
    :param round: 轮数
    :param algorithm: 算法名（如"kdb"/"aode"/"tan"）
    :param data_path: 数据文件路径
    :param param_estimator: 参数估计方法（如"MLE"）
    :param random_state_base: 基础随机种子
    :param kwargs: 分类器初始化参数（如k=3）
    :return: 最终平均指标（acc/f1）、详细指标记录
    """
    # ========== 1. 加载数据 ==========
    data = data_utils.load_data(data_path)
    all_names, class_name, feature_names= data_utils.get_node_names(data)
    X = data.drop(columns=[class_name]).values
    y = data[class_name].values
    n_samples = X.shape[0]
    print(f"✅ 加载数据完成，样本数：{n_samples}，特征数：{X.shape[1]}")

    # ========== 2. 初始化指标记录 ==========
    # 记录每轮、每折的指标：round -> fold -> metric
    cv_metrics = {
        "round": [],  # 轮次（1-5）
        "fold": [],  # 折数（1-5）
        "acc": [],  # 该折准确率
        # "f1": []  # 该折F1分数（加权平均）
    }
    # ========== 3. 5轮外层循环 ==========
    for round_idx in range(round):
        print(f"\n===== 开始第 {round_idx + 1} 轮5折交叉验证 =====")
        # 每轮用不同的随机种子（保证划分不同）
        random_state = random_state_base + round_idx
        # 分层KFold（分类任务优先，保证每折类别分布一致）
        skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)

        # ========== 4. 5折内层循环 ==========
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"--- 第 {round_idx + 1} 轮 - 第 {fold_idx + 1} 折 ---")
            # 划分训练/测试集
            # X_train, X_test = X[train_idx], X[test_idx]
            # y_train, y_test = y[train_idx], y[test_idx]
            train_data=data.iloc[train_idx].reset_index(drop=True)
            test_data=data.iloc[test_idx].reset_index(drop=True)

            # ========== 5. 工厂模式创建分类器 ==========
            classifier_kwargs = {
                "param_estimator": param_estimator,
                **kwargs
            }
            # 每次折都创建新分类器，避免上一折的训练状态污染
            classifier = ClassifierFactory.create_classifier(
                algorithm=algorithm,
                **classifier_kwargs
            )

            # ========== 6. 训练 + 预测 + 指标计算 ==========

            #结构学习        可以查看dag
            dag=classifier.structure_learning(train_data)
            # 参数学习
            model=classifier.param_learning(train_data)
            # 预测         可以查看pred_result
            pred_prob_values,pred_result=classifier.predict(model,test_data,class_name)
            # 评价
            fold_metrics=classifier.evaluate(test_data,class_name)



            print(f"准确率：{fold_metrics.acc:.4f}")
            print(f"\n混淆矩阵：\n{fold_metrics.cm}")  # 加\n让格式更清晰
            print(f"\n报告：\n{fold_metrics.report}")  # 加\n让格式更清晰
            # classifier.fit(X_train, y_train)  # 假设分类器有fit方法（输入X_train/y_train）
            # y_pred = classifier.predict(X_test)  # 假设分类器有predict方法
            #
            # # 计算指标（用自定义Metrics类）
            # metrics = Metrics(y_test, y_pred)
            # fold_acc = metrics.acc
            # fold_f1 = metrics.report_dict["weighted avg"]["f1-score"]

            # 记录指标
            cv_metrics["round"].append(round_idx + 1)
            cv_metrics["fold"].append(fold_idx + 1)
            cv_metrics["acc"].append(fold_metrics.acc)
            # cv_metrics["f1"].append(fold_f1)

            print(f"第 {round_idx + 1} 轮 - 第 {fold_idx + 1} 折：准确率={fold_metrics.acc:.4f}，")

    # ========== 7. 结果聚合：计算平均 ==========
    # 转换为DataFrame，方便统计
    cv_df = pd.DataFrame(cv_metrics)

    # 步骤1：计算每轮的平均指标（5折平均）
    round_avg = cv_df.groupby("round")[["acc",]].mean().reset_index()
    print("\n===== 每轮5折平均指标 =====")
    print(round_avg)

    # 步骤2：计算5轮的总平均（最终结果）
    final_acc = round_avg["acc"].mean()
    #final_f1 = round_avg["f1"].mean()
    # 可选：计算标准差（评估稳定性）
    acc_std = round_avg["acc"].std()
    #f1_std = round_avg["f1"].std()

    # ========== 8. 输出最终结果 ==========
    print("\n===== 5轮5折交叉验证最终结果（平均） =====")
    print(f"最终平均准确率：{final_acc:.4f} ± {acc_std:.4f}")
    #print(f"最终平均F1分数：{final_f1:.4f} ± {f1_std:.4f}")

    # TODO 返回值未确定
    # return {
    #     "final_acc": final_acc,
    #     #"final_f1": final_f1,
    #     "acc_std": acc_std,
    #     #"f1_std": f1_std,
    #     "detailed_metrics": cv_df  # 详细的每轮每折指标
    # }


# ========== 调用示例 ==========
if __name__ == "__main__":
    # 运行KDB算法的5轮5折CV，传入k=3
    cv_result = n_round_n_fold_cv(
        algorithm="tan",
        data_path="anneal.xlsx",
        param_estimator="MLE",
        random_state_base=1,
        round=5,
        fold=5,
        root_node=3
        #k=3  # KDB的专属参数（通过kwargs传递给工厂）
    )

    # 可选：保存详细指标到CSV
    #cv_result["detailed_metrics"].to_csv("5round_5fold_cv_metrics.csv", index=False)