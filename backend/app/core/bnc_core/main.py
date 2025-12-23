# 这是一个示例 Python 脚本。
from backend.app.core.bnc_core import *

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 运行KDB，仅需修改algorithm，并传递k值
    result_kdb = bnc_runner.n_round_n_fold_cv("tan",
                                             "anneal.xlsx",
                                              root_node=3
                                              )
    # print(f"准确率：{result_kdb.acc:.4f}")
    # print(f"\n混淆矩阵：\n{result_kdb.cm}")  # 加\n让格式更清晰
    # print(f"\n报告：\n{result_kdb.report}")  # 加\n让格式更清晰
    # print("KDB准确率：", result_kdb["metrics"]["accuracy"])


