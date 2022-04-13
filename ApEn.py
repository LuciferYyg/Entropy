import numpy as np
class BaseApEn(object):
    """
    近似熵基础类
    """

    def __init__(self, m, r):
        """
        初始化
        :param U:一个矩阵列表，for example:
            U = np.array([85, 80, 89] * 17)
        :param m: 子集的大小，int
        :param r: 阀值基数，0.1---0.2
        """
        self.m = m
        self.r = r

    @staticmethod
    def _maxdist(x_i, x_j):
        """计算矢量之间的距离"""
        return np.max([np.abs(np.array(x_i) - np.array(x_j))])

    @staticmethod
    def _biaozhuncha(U):
        """
        计算标准差的函数
        :param U:
        :return:
        """
        if not isinstance(U, np.ndarray):
            U = np.array(U)
        return np.std(U, ddof=1)


class ApEn(BaseApEn):
    """
    Pincus提出的算法，计算近似熵的类
    """

    def _biaozhunhua(self, U):
        """
        将数据标准化，
        获取平均值
        所有值减去平均值除以标准差
        """
        self.me = np.mean(U)
        self.biao = self._biaozhuncha(U)
        return np.array([(x - self.me) / self.biao for x in U])

    def _dazhi(self, U):
        """
        获取阀值
        :param U:
        :return:
        """
        if not hasattr(self, "f"):
            self.f = self._biaozhuncha(U) * self.r
        return self.f

    def _phi(self, m, U):
        """
        计算熵值
        :param U:
        :param m:
        :return:
        """
        # 获取矢量列表
        x = [U[i:i + m] for i in range(len(U) - m + 1)]
        # 获取所有的比值列表
        C = [len([1 for x_j in x if self._maxdist(x_i, x_j) <= self._dazhi(U)]) / (len(U) - m + 1.0) for x_i in x]
        # 计算熵
        return np.sum(np.log(list(filter(lambda a: a, C)))) / (len(U) - m + 1.0)

    def _phi_b(self, m, U):
        """
        标准化数据计算熵值
        :param m:
        :param U:
        :return:
        """
        # 获取矢量列表
        x = [U[i:i + m] for i in range(len(U) - m + 1)]
        # 获取所有的比值列表
        C = [len([1 for x_j in x if self._maxdist(x_i, x_j) <= self.r]) / (len(U) - m + 1.0) for x_i in x]
        # 计算熵
        return np.sum(np.log(list(filter(lambda x: x, C)))) / (len(U) - m + 1.0)

    def jinshishang(self, U):
        """
        计算近似熵
        :return:
        """
        return np.abs(self._phi(self.m + 1, U) - self._phi(self.m, U))

    def jinshishangbiao(self, U):
        """
        将原始数据标准化后的近似熵
        :param U:
        :return:
        """
        eeg = self._biaozhunhua(U)
        return np.abs(self._phi_b(self.m + 1, eeg) - self._phi_b(self.m, eeg))

if __name__ == "__main__":
    U = np.array([2, 4, 6, 8, 10] * 17)
    G = np.array([3, 4, 5, 6, 7] * 17)
    ap = ApEn(2, 0.2)
    print(ap.jinshishang(U)) # 计算近似熵
