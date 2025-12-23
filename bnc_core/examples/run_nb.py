from bnc_core.classifiers.NB import NBClassifier
import pandas as pd
from bnc_core.base import data_utils
if __name__ == "__main__":
    # 1. 初始化分类器
    nb = NBClassifier(param_estimator="MLE")
    
    # 2. 加载数据D:\\文章整理\\bnc_project\\anneal.xlsx
    train_data=nb.load_data("C:\\Users\\27914\\Desktop\\工区\\蠡县工区\\蠡县离散数据\\蠡县探井离散数据.xlsx")
    train_data=train_data.iloc[:,2:]

    test_data=nb.load_data("C:\\Users\\27914\\Desktop\\工区\\蠡县工区\\蠡县离散数据\\蠡县网格离散数据.xlsx")
    x_y = test_data.iloc[:, :2]
    test_data = test_data.iloc[:, 2:]

    all_names, class_name, feature_names=nb.get_node_names(train_data)

    # # 5. 结构学习
    nb.structure_learning(train_data)
    # # 6. 参数学习
    model=nb.param_learning(train_data)

    # # 7. 预测
    pred_prob_values,preds=nb.predict(model,test_data,class_name)

    result=pd.concat([x_y,pred_prob_values.iloc[:,-1]],axis=1)

    print(result)
    data_utils.save_data(result, "C:\\Users\\27914\\Desktop\\工区\\蠡县工区\\蠡县离散数据\\预测结果测试.xlsx")
