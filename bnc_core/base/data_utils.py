import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer


def load_data(data_path: str, **kwargs) -> pd.DataFrame:
    """加载数据（支持Excel/CSV）
    :param data_path:"""
    if data_path.endswith(".xlsx"):
        return pd.read_excel(data_path, **kwargs)
    elif data_path.endswith(".csv"):
        return pd.read_csv(data_path, **kwargs)
    else:
        raise ValueError("仅支持Excel/CSV格式")

def save_data(data: pd.DataFrame, save_path: str) -> None:
    """
    :param data
    :param save_path:
    """
    if save_path.endswith(".xlsx"):
        data.to_excel(save_path,header=False, index=False)
    elif save_path.endswith(".csv"):
        data.to_csv(save_path, header=False, index=False)
    else:
        raise ValueError("仅支持保存为Excel/CSV格式，请确保带有文件后缀名！")


def get_node_names(data: pd.DataFrame)->tuple:
    """
    获取数据集的表头名字
    :param data
    :return all_names, class_name, feature_names
    """
    all_names = data.columns.tolist()
    class_name=all_names[-1]
    feature_names = all_names[0:-1]
    return all_names, class_name, feature_names


def my_train_test_split(data: pd.DataFrame, test_size: float, random_state: int) -> tuple:
    """在进行单次实验测试时，基类调用此方法进行数据分割
    :param data:数据
        :param test_size:测试数据的比例
        :return train_data, test_data：返回分割好的数据"""
    return train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )
#TODO 可以实现数据预处理，这里还没实现

# def preprocess(data: pd.DataFrame, class_name: str, n_bins: int = 5, **kwargs) -> pd.DataFrame:
#     """
#     通用预处理：
#     1. 缺失值填充（众数）
#     2. 数值特征离散化（等频分箱）
#     3. 转为整数类型（适配pgmpy）
#     """
#     data = data.copy()
#     # 填充缺失值
#     data = data.fillna(data.mode().iloc[0])
#     # 离散化数值特征（排除类标签）
#     num_cols = data.select_dtypes(include=[np.number]).columns
#     num_cols = [col for col in num_cols if col != class_name]
#     if num_cols:
#         discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
#         data[num_cols] = discretizer.fit_transform(data[num_cols])
#     # 转为整数
#     for col in data.columns:
#         data[col] = data[col].astype(int)
#     # 对齐特征值（避免后续推理KeyError）
#     return data



# def align_test_features(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
#     """对齐测试集特征值（替换训练集未见过的值为众数）"""
#     test_aligned = test_data.copy()
#     for col in test_aligned.columns:
#         train_vals = set(train_data[col].unique())
#         train_mode = train_data[col].mode()[0]
#         test_aligned[col] = test_aligned[col].apply(lambda x: x if x in train_vals else train_mode)
#     return test_aligned
