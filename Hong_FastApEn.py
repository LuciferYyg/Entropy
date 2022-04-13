import numpy as np
from ApEn import *
class NewBaseApen(object):
    """新算法基类"""

    @staticmethod
    def _get_array_zeros(x):
        """
        创建N*N的0矩阵
        :param U:
        :return:
        """
        N = np.size(x, 0)
        return np.zeros((N, N), dtype=int)

    @staticmethod
    def _get_c(z, m):
        """
        计算熵值的算法
        :param z:
        :param m:
        :return:
        """
        N = len(z[0])
        # 概率矩阵C计算
        c = np.zeros((1, N - m + 1))
        if m == 2:
            for j in range(N - m + 1):
                for i in range(N - m + 1):
                    c[0, j] += z[j, i] & z[j + 1, i + 1]
        if m == 3:
            for j in range(N - m + 1):
                for i in range(N - m + 1):
                    c[0, j] += z[j, i] & z[j + 1, i + 1] & z[j + 2, i + 2]
        if m != 2 and m != 3:
            raise AttributeError('m的取值不正确！')
        data = list(filter(lambda x:x, c[0]/(N - m + 1.0)))
        if not all(data):
            return 0
        return np.sum(np.log(data)) / (N - m + 1.0)

class NewApEn(ApEn, NewBaseApen):
    """
    洪波等人提出的快速实用算法计算近似熵
    """

    def _get_distance_array(self, U):
        """
        获取距离矩阵
        :param U:
        :return:
        """
        z = self._get_array_zeros(U)
        fa = self._dazhi(U)
        for i in range(len(z[0])):
            z[i, :] = (np.abs(U - U[i]) <= fa) + 0
        return z

    def _get_shang(self, m, U):
        """
        计算熵值
        :param U:
        :return:
        """
        # 获取距离矩阵
        Z = self._get_distance_array(U)
        return self._get_c(Z, m)

    def hongbo_jinshishang(self, U):
        """
        计算近似熵
        :param U:
        :return:
        """
        return np.abs(self._get_shang(self.m + 1, U) - self._get_shang(self.m, U))
