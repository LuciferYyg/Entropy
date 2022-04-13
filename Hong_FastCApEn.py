import numpy as np
from Hong_FastApEn import *
from CApEn import *
class NewHuApEn(HuApEn, NewBaseApen):
    """
    洪波等人提出的快速实用算法计算互近似熵
    """
    def _get_distance_array(self, U, G):
        """
        获取距离矩阵
        :param U:模板数据
        :return:比较数据
        """
        z = self._get_array_zeros(U)
        fa = self._dazhi(U, G)
        for i in range(len(z[0])):
            z[i, :] = (np.abs(G - U[i]) <= fa) + 0
        return z

    def _get_shang(self, m, U, G):
        """
        计算熵值
        :param U:
        :return:
        """
        # 获取距离矩阵
        Z = self._get_distance_array(U, G)
        return self._get_c(Z, m)

    def hongbo_hujinshishang(self, U, G):
        """
        对外的计算互近似熵的接口
        :param U:
        :param G:
        :return:
        """
        return np.abs(self._get_shang(self.m + 1, U, G) - self._get_shang(self.m, U, G))
