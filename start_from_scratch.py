# %%
"""
修改自参考文章
https://www.heywhale.com/mw/project/5d8da105037db3002d3a4c4a
"""
from numba import jit
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize
import time


# %% 定义RBF核函数


def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    根据核函数计算两个样本序列的协方差
    :param X1: [m, dim]包含m个样本点
    :param X2: [n, dim]包含n个样本点
    :param l: 平滑度。越大，一个点影响范围越广，越平滑
    :param sigma_f: 波动性。越大，则回归预测点方差越大，至少得是噪声标准差两倍以上
    :return: 形状为[m, n]的协方差矩阵
    """
    cov_logits = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)

    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * cov_logits)


def plot_gp(X, mu, cov, X_train=None, Y_train=None, samples=np.array([]), legend=True):
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
    if legend:
        plt.legend()
    plt.show()
    return None


def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    """
    给定训练数据X_train, Y_train后，计算输入数据点X_s对应多元随机向量的后验均值mu和协方差cov。
    :param X_s: [n(, dim=1)]，n个新数据的位置
    :param X_train: [m(, dim=1)]，m个训练数据的位置
    :param Y_train: [m(, dim=1)]，m个训练数据对应的值（即随机变量的值）
    :param l: RBF核函数中的l
    :param sigma_f: RBF核函数中的sigma_f
    :param sigma_y: 包含噪声的标准差，即方差的算数平方根
    :return: 后验均值mu: [n(, dim=1)]；后验协方差cov: [n, n]
    """
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(X_train.shape[0])  # [m, m]
    K_s = kernel(X_train, X_s, l, sigma_f)  # [m, n]
    K_ss = kernel(X_s, X_s, l, sigma_f) + sigma_y * np.eye(X_s.shape[0])  # [n, n]
    K_inv = inv(K)

    # 公式 (4)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # 公式 (5)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


def nll_fn(X_train, Y_train, square_y, naive=True):
    """
    基于给定数据X_train和Y_train以及噪声水平，返回一个可以利用待优化的超参数计算负对数似然的函数
    :param X_train: [m(, dim=1)]
    :param Y_train: [m(, dim=1)]
    :param square_y: 噪声的标准差
    :param naive: 是否使用数值稳定的方法进行求解
    :return: 以超参数列表为形参的负对数似然函数
    """

    def nll_naive(theta):
        # 使用公式(7)来实现
        # 与下面的nll_stable的实现相比在数值上不稳定
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            square_y ** 2 * np.eye(len(X_train))
        return 0.5 * np.log(det(K)) + \
               0.5 * Y_train.T.dot(inv(K).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2 * np.pi)

    def nll_stable(theta):
        # 数值上更稳定，相比于nll_naive
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            square_y ** 2 * np.eye(len(X_train))
        L = cholesky(K)
        return np.sum(np.log(np.diagonal(L))) + \
               0.5 * Y_train.T.dot(lstsq(L.T, lstsq(L, Y_train)[0])[0]) + \
               0.5 * len(X_train) * np.log(2 * np.pi)

    if naive:
        return nll_naive
    else:
        return nll_stable


# %% 先验分布
# 构造样本点位置
X = np.arange(-10, 10, 0.2).reshape(-1, 1)

# 先验的均值与方差
mu = np.zeros(X.shape)
cov = kernel(X, X)
# 从先验分布（多元高斯分布）中抽取样本点
samples = np.random.multivariate_normal(mu.ravel(), cov, 5)

# 画出GP的均值, 置信区间
plot_gp(X, mu, cov, samples=samples)
# %% 超参数给定的情况下，对训练数据学习
# 无噪音的5个输入数据
X_train = np.linspace(-10, 0, 20).reshape(-1, 1)
Y_train = np.sin(X_train)

# 计算后验预测分布的均值向量与协方差矩阵
mu_s, cov_s = posterior_predictive(X, X_train, Y_train, l=1, sigma_f=0.1, sigma_y=0.01)

# 从后验预测分布中抽取3个样本
samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
plot_gp(X, mu_s, cov_s, X_train=X_train, Y_train=Y_train, samples=samples, legend=False)

# %% 利用极大似然估计超参数

# 首先必需假定知道样本点的不确定度
noise = 0.01

# 求解可满足最小化目标函数的参数 l 及 sigma_f
# 此处最好经过多次优化取最小的值
res = minimize(nll_fn(X_train, Y_train, noise, naive=False),  # 待优化函数
               [1, 1],  # 迭代初始点
               bounds=((1e-5, None), (1e-5, None)),
               method='L-BFGS-B')  # 数值迭代方法

# 提取超参数的优化结果
l_opt, sigma_f_opt = res.x

# 使用优化的核函数参数计算后验预测分布的参数，并绘制结果图
mu_s, cov_s = posterior_predictive(X, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise)
samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 20)
plot_gp(X, mu_s, cov_s, X_train=X_train, Y_train=Y_train, samples=samples, legend=False)
