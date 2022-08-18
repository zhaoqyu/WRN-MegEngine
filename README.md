# Wide-Residual-Networks-MegEngine
Wide Residual Networks (MegEngine implementation）

论文介绍

论文题目：Wide Residual Networks

论文链接：https://arxiv.org/abs/1605.07146

对标实现 - official 复现链接：https://github.com/szagoruyko/wide-residual-networks

Wide Residual Networks（WRN）模型的 MegEngine 版 inference 函数，提供了模型的 weight，以及可证明等价对象的脚本（compare.py）。

基于旷视天元 MegEngine 框架（限 v1.9.1 及以上版本）；

推理中所有计算使用 megengine 完成，预处理和权重转换使用 numpy 和其他深度学习框架

在 megengine 缺少算子的情况下，使用 numpy 代替实现

requirements.txt声明了全部所需的 python 依赖项

compare.py 证明了与对标实现之间的等价性：

对于 10 个或以上合理的构造输入，megengine 实现 与 对标实现 在 inference 时的结果相对误差均在 1e-3 以内

pip3 install -r requirements.txt

python3 compare.py 输出两者的误差

