import numpy as np
import matplotlib.pyplot as plt


def Fuzzy_Entropy(x, m, r=0.25, n=2):
    """
    模糊熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    n 计算模糊隶属度时的维度
    """
    # 将x转化为数组
    x = np.array(x)
    # 检查x是否为一维数据
    if x.ndim != 1:
        raise ValueError("x的维度不是一维")
    # 计算x的行数是否小于m+1
    if len(x) < m+1:
        raise ValueError("len(x)小于m+1")
    # 将x以m为窗口进行划分
    entropy = 0  # 近似熵
    for temp in range(2):
        X = []
        for i in range(len(x)-m+1-temp):
            X.append(x[i:i+m+temp])
        X = np.array(X)
        # 计算X任意一行数据与其他行数据对应索引数据的差值绝对值的最大值
        D_value = []  # 存储差值
        for index1, i in enumerate(X):
            sub = []
            for index2, j in enumerate(X):
                if index1 != index2:
                    sub.append(max(np.abs(i-j)))
            D_value.append(sub)
        # 计算模糊隶属度
        D = np.exp(-np.power(D_value, n)/r)
        # 计算所有隶属度的平均值
        Lm = np.average(D.ravel())
        entropy = abs(entropy) - Lm

    return entropy

U = np.array([2, 4, 6, 8, 10] * 17)
print(Fuzzy_Entropy(U,3))