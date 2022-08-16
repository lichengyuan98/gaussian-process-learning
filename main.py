# %%
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import time


# %% 定义RBF核函数

@jit
def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    根据核函数计算两个样本序列的协方差
    :param X1: [m, dim]包含m个样本点
    :param X2: [n, dim]包含n个样本点
    :param l: 平滑度
    :param sigma_f: 波动性
    :return: 形状为[m, n]的协方差矩阵
    """
    conv_logits = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(conv_logits.shape[0]):
        for j in range(conv_logits.shape[1]):
            dif = X1[i, :] - X2[j, :]
            conv_logits[i, j] = np.dot(dif, dif)

    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * conv_logits)



def plot_gp(X, mu, cov, X_train=None, Y_train=None, samples=[]):
    """
    绘制位于位置X处的,在mu、cov约束下，每个位置对应值的概率分布。
    若输入X_train和Y_train序列，绘制出来。
    sample中每个元素都是位置在X处的一个采样序列。
    之所以(, dim=1)，是因为只考虑自变量的坐标为一维，每个坐标对应的采样值也是一维
    :param X: [n(, dim=1)]，总共n个位置
    :param mu: [n(, dim=1)]
    :param cov: [n, n]
    :param X_train: [m(, dim=1)]
    :param Y_train: [m(, dim=1)]
    :param samples: [k, n(, dim=1)], 总共k个序列采样值
    :return: None
    """
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)

    plt.plot(X, mu, label="mu")
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'sample {i + 1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
    plt.show()
    return None


# %%
