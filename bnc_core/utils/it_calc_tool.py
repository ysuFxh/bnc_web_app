from pyitlib import discrete_random_variable as drv


def calc_mi(MI, attr, class_label):
    """:MI ,空的MI列表,attr,特征矩阵，class_label,标签矩阵"""
    feature_names = attr.columns.tolist()
    class_np = class_label.values.astype(int)  # 类标签转数组
    # 计算MI，这个遍历方法是在遍历可遍历对象时同时获取索引和值的遍历方式
    for i, feat_name in enumerate(feature_names):
        feat_np = attr[feat_name].values.astype(int)
        mi = drv.information_mutual(feat_np, class_np, base=2)
        MI[i] = mi
    return MI


def calc_cmi(CMI, attr, class_label):
    """:CMI ,空的CMI列表,attr,特征矩阵，class_label,标签矩阵"""
    feature_names = attr.columns.tolist()
    class_np = class_label.values.astype(int)
    for i, feat_i_name in enumerate(feature_names):
        feat_i_np = attr[feat_i_name].values.astype(int)
        for j, feat_j_name in enumerate(feature_names):
            # 排除自身（i==j），仅计算不同特征间的CMI
            if i != j:
                feat_j_np = attr[feat_j_name].values.astype(int)
                # 计算条件互信息 I(feat_i;feat_j | 类标签)，强制非负
                cmi = drv.information_mutual_conditional(
                    feat_i_np, feat_j_np, class_np, base=2
                )
                CMI[i, j] = max(cmi, 0.0)

    return CMI
