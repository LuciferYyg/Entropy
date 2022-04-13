import numpy as np
def func(n):
    """求阶乘"""
    if n == 0 or n == 1:
        return 1
    else:
        return (n * func(n - 1))


def compute_p(S):
    """计算每一种 m 维符号序列的概率"""
    _map = {}
    for item in S:
        a = str(item)
        if a in _map.keys():
            _map[a] = _map[a] + 1
        else:
            _map[a] = 1

    freq_list = []
    for freq in _map.values():
        freq_list.append(freq / len(S))
    return freq_list


def Permutation_Entropy(x, m, t):
    """计算排列熵值"""
    length = len(x) - (m - 1) * t
    # 重构 k*m 矩阵
    y = [x[i:i + m * t:t] for i in range(length)]

    # 将各个分量升序排序
    S = [np.argsort(y[i]) for i in range(length)]

    # 计算每一种 m 维符号序列的概率
    freq_list = compute_p(S)

    # 计算排列熵
    pe = 0
    for freq in freq_list:
        pe += (- freq * np.log(freq))

    return pe / np.log(func(m))
