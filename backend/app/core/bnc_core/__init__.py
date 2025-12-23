"""贝叶斯网络分类器（bnc_core）项目主包，包含分类器、工具、示例等模块"""
# 导出子包（让外部可以直接导入子包）
from . import classifiers
from . import utils
from . import examples

# 导出常用模块/函数（简化外部导入）
from .runner import run_bnc

# 指定`from bnc import *`时导入的内容（可选，避免命名污染）
__all__ = ["classifiers", "utils", "examples", "base", "bnc_runner"]