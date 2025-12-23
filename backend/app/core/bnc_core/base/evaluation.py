from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class Metrics:
    """自定义指标类，（.acc/.cm等）

    """
    def __init__(self, y_true, y_pred):
        # 核心指标计算
        self.acc = accuracy_score(y_true, y_pred)  # 准确率（acc）
        self.cm = confusion_matrix(y_true, y_pred)  # 混淆矩阵（cm）
        self.report = classification_report(y_true, y_pred)  # 完整报告, output_dict=True
        # 扩展指标（可选）
        # self.precision = self.report["weighted avg"]["precision"]
        # self.recall = self.report["weighted avg"]["recall"]
        # self.f1 = self.report["weighted avg"]["f1-score"]
def evaluate(y_true, y_pred):
    """
    :param y_true
    :param y_pred
    :return 封装后的评价指标Metrics，acc,confusion_matrix, classification_report"""
    return Metrics(y_true, y_pred)
