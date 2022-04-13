import numpy as np
from ApEn import *
class HuApEn(BaseApEn):

    def _xiefangcha(self, U, G):
        """
        计算协方差的函数
        :param U: 序列1，矩阵
        :param G: 序列2，矩阵
        :return: 协方差，float
        """
        if not isinstance(U, np.ndarray):
            U = np.array(U)
        if not isinstance(G, np.ndarray):
            G = np.array(G)
        if len(U) != len(G):
            raise AttributeError('参数错误！')
        return np.cov(U, G, ddof=1)[0, 1]

    def _biaozhunhua(self, U, G):
        """
        对数据进行标准化
        """
        self.me_u = np.mean(U)
        self.me_g = np.mean(G)
        self.biao_u = self._biaozhuncha(U)
        self.biao_g = self._biaozhuncha(G)
        # self.biao_u = self._xiefangcha(U, G)
        # self.biao_g = self._xiefangcha(U, G)
        return np.array([(x - self.me_u) / self.biao_u for x in U]), np.array(
            [(x - self.me_g) / self.biao_g for x in U])

    def _dazhi(self, U, G):
        """
        获取阀值
        :param r:
        :return:
        """
        if not hasattr(self, "f"):
            self.f = self._xiefangcha(U, G) * self.r
        return self.f

    def _phi(self, m, U, G):
        """
        计算熵值
        :param m:
        :return:
        """
        # 获取X矢量列表
        x = [U[i:i + m] for i in range(len(U) - m + 1)]
        # 获取y矢量列表
        y = [G[g:g + m] for g in range(len(G) - m + 1)]
        # 获取所有的条件概率列表
        C = [len([1 for y_k in y if self._maxdist(x_i, y_k) <= self._dazhi(U, G)]) / (len(U) - m + 1.0) for x_i in x]
        # 计算熵
        return np.sum(np.log(list(filter(lambda x_1: x_1, C)))) / (len(U) - m + 1.0)

    def _phi_b(self, m, U, G):
        """
        标准化数据计算熵值
        :param m:
        :param U:
        :return:
        """
        # 获取X矢量列表
        x = [U[i:i + m] for i in range(len(U) - m + 1)]
        # 获取y矢量列表
        y = [G[g:g + m] for g in range(len(G) - m + 1)]
        # 获取所有的条件概率列表
        C = [len([1 for y_k in y if self._maxdist(x_i, y_k) <= self.r]) / (len(U) - m + 1.0) for x_i in x]
        # 计算熵
        return np.sum(np.log(list(filter(lambda x: x, C)))) / (len(U) - m + 1.0)

    def hujinshishang(self, U, G):
        """
        计算互近似熵
        :return:
        """
        return np.abs(self._phi(self.m + 1, U, G) - self._phi(self.m, U, G))

    def hujinshishangbiao(self, U, G):
        """
        将原始数据标准化后的互近似熵
        :param U:
        :param G:
        :return:
        """
        u, g = self._biaozhunhua(U, G)
        return np.abs(self._phi_b(self.m + 1, u, g) - self._phi_b(self.m, u, g))
