import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from joblib import Parallel, delayed  # 核心并行库
import warnings

warnings.filterwarnings("ignore")  # 屏蔽无关警告

# 1. 导入你的工厂类和工具类
from bnc_core.utils.classifier_factory import ClassifierFactory
from bnc_core.base import data_utils

# TODO 后续改造成并行计算的版本
# ========== 核心：封装单折执行函数（独立可并行） ==========
def _run_single_fold(
        fold_idx: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        data: pd.DataFrame,
        class_name: str,
        algorithm: str,
        classifier_kwargs: dict,
        all_names: list,
        feature_names: list
):
    """
    执行单折的训练/预测/评估（独立函数，供并行调用）
    :return: 折索引、准确率、F1分数、是否成功
    """
    fold_full_idx = f"折{fold_idx + 1}"
    try:
        # 划分该折数据（仅传索引，减少数据拷贝）
        train_data = data.iloc[train_idx].reset_index(drop=True)
        test_data = data.iloc[test_idx].reset_index(drop=True)

        # 工厂模式创建全新分类器实例（避免状态污染）
        classifier = ClassifierFactory.create_classifier(
            algorithm=algorithm,
            **classifier_kwargs
        )

        # 赋值数据和元信息
        classifier.train_data = train_data
        classifier.test_data = test_data
        classifier.class_name = class_name
        classifier.node_names = all_names
        classifier.feature_names = feature_names

        # 核心流程：结构学习→参数学习→预测→评估
        classifier.structure_learning()
        classifier.param_learning()
        classifier.predict()
        fold_metrics = classifier.evaluate()

        # 提取指标
        fold_acc = fold_metrics.acc
        # 兼容不同Metrics类的F1提取方式
        if hasattr(fold_metrics, 'report_dict'):
            fold_f1 = fold_metrics.report_dict["weighted avg"]["f1-score"]
        else:
            test_true = test_data[class_name]
            fold_f1 = f1_score(test_true, classifier.preds, average='weighted')

        print(f"✅ {fold_full_idx} 完成 | 准确率：{fold_acc:.4f} | F1：{fold_f1:.4f}")
        return {
            "fold_idx": fold_idx + 1,
            "acc": fold_acc,
            "f1": fold_f1,
            "success": True
        }
    except Exception as e:
        print(f"❌ {fold_full_idx} 失败 | 错误：{str(e)[:100]}")
        return {
            "fold_idx": fold_idx + 1,
            "acc": np.nan,
            "f1": np.nan,
            "success": False
        }


# ========== 并行版CV主函数 ==========
def five_round_five_fold_cv_parallel(
        algorithm: str,
        data_path: str,
        param_estimator: str = "MLE",
        random_state_base: int = 42,
        n_rounds: int = 5,  # 轮数
        n_folds: int = 5,  # 每轮折数
        n_jobs: int = -1,  # 并行核心数：-1=全部CPU核心，1=串行，2=2核心
        verbose: int = 1,  # 并行日志级别：1=显示进度，0=静默
        **kwargs  # 分类器专属参数（如k=3）
):
    """
    并行版多轮多折交叉验证（默认5轮5折）
    :param n_jobs: 并行进程数（核心优化参数）
    :param verbose: 并行进度显示级别
    """
    # ========== 1. 全局数据加载（仅执行一次，避免重复IO） ==========
    try:
        data = data_utils.load_data(data_path)
        all_names, class_name, feature_names = data_utils.get_node_names(data)
        X = data.drop(columns=[class_name]).values
        y = data[class_name].values
        n_samples = X.shape[0]
        print(f"✅ 数据加载完成 | 样本数：{n_samples} | 特征数：{X.shape[1]} | 类节点：{class_name}")
    except Exception as e:
        raise RuntimeError(f"数据加载失败：{str(e)}") from e

    # ========== 2. 初始化指标记录 ==========
    cv_metrics = {
        "round": [],  # 轮次
        "fold": [],  # 折数
        "acc": [],  # 准确率
        "f1": [],  # 加权F1
        "success": []  # 是否执行成功
    }

    # ========== 3. 多轮外层循环（轮次串行，折并行） ==========
    for round_idx in range(n_rounds):
        round_name = f"第{round_idx + 1}轮"
        print(f"\n{'=' * 60}\n{round_name} 开始（并行执行{n_folds}折）\n{'=' * 60}")

        # 每轮使用不同随机种子，保证划分独立性
        random_state = random_state_base + round_idx
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        # 构造分类器通用参数（传递给单折函数）
        classifier_kwargs = {
            "param_estimator": param_estimator,
            **kwargs
        }

        # ========== 4. 生成并行任务列表 ==========
        fold_tasks = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            # 封装单折任务（delayed标记为并行任务）
            task = delayed(_run_single_fold)(
                fold_idx=fold_idx,
                train_idx=train_idx,
                test_idx=test_idx,
                data=data,
                class_name=class_name,
                algorithm=algorithm,
                classifier_kwargs=classifier_kwargs,
                all_names=all_names,
                feature_names=feature_names
            )
            fold_tasks.append(task)

        # ========== 5. 并行执行该轮所有折 ==========
        fold_results = Parallel(
            n_jobs=n_jobs,  # 并行核心数
            verbose=verbose,  # 进度显示
            backend="loky",  # 多进程后端（稳定）
            timeout=3600  # 单折超时时间（1小时）
        )(fold_tasks)

        # ========== 6. 整理该轮结果 ==========
        for res in fold_results:
            cv_metrics["round"].append(round_idx + 1)
            cv_metrics["fold"].append(res["fold_idx"])
            cv_metrics["acc"].append(res["acc"])
            cv_metrics["f1"].append(res["f1"])
            cv_metrics["success"].append(res["success"])

    # ========== 7. 结果聚合与统计 ==========
    cv_df = pd.DataFrame(cv_metrics)
    cv_df_clean = cv_df[cv_df["success"]].dropna()  # 过滤失败折

    if cv_df_clean.empty:
        raise RuntimeError("所有折均执行失败，请检查数据或分类器逻辑！")

    # 每轮平均指标
    round_avg = cv_df_clean.groupby("round")[["acc", "f1"]].mean().reset_index()
    # 最终全局指标
    final_acc = round_avg["acc"].mean()
    final_f1 = round_avg["f1"].mean()
    acc_std = round_avg["acc"].std()
    f1_std = round_avg["f1"].std()
    success_rate = cv_df["success"].mean() * 100

    # ========== 8. 输出最终结果 ==========
    print(f"\n{'=' * 80}")
    print(f"🎯 {algorithm.upper()} - {n_rounds}轮{n_folds}折CV（并行加速）最终结果")
    print(f"{'=' * 80}")
    print(f"最终平均准确率：{final_acc:.4f} ± {acc_std:.4f}")
    print(f"最终平均加权F1：{final_f1:.4f} ± {f1_std:.4f}")
    print(f"任务成功率：{success_rate:.1f}%（{cv_df['success'].sum()}/{len(cv_df)}）")
    print(f"并行核心数：{n_jobs if n_jobs != -1 else '全部CPU核心'}")
    print(f"{'=' * 80}")

    # ========== 9. 返回结构化结果 ==========
    return {
        "algorithm": algorithm,
        "n_rounds": n_rounds,
        "n_folds": n_folds,
        "final_acc": final_acc,
        "final_f1": final_f1,
        "acc_std": acc_std,
        "f1_std": f1_std,
        "success_rate": success_rate,
        "round_avg": round_avg,  # 每轮平均指标
        "detailed_metrics": cv_df,  # 所有折详细指标（含失败）
        "clean_metrics": cv_df_clean  # 仅成功折指标
    }


# ========== 调用示例 ==========
if __name__ == "__main__":
    # 1. 基础调用（使用全部CPU核心并行）
    cv_result = five_round_five_fold_cv_parallel(
        algorithm="tan",
        data_path="../anneal.xlsx",
        param_estimator="MLE",
        random_state_base=1,
        n_rounds=5,  # 5轮
        n_folds=5,  # 每轮5折
        n_jobs=-1,  # 并行核心数：-1=全部，2=2核心，1=串行
        verbose=1  # 显示并行进度
        # k=3               # 如需传分类器专属参数（如KDB的k），取消注释
    )

    # 2. 保存结果到CSV（可选）
    cv_result["detailed_metrics"].to_csv("tan_cv_parallel_detailed.csv", index=False)
    cv_result["round_avg"].to_csv("tan_cv_parallel_round_avg.csv", index=False)