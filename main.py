import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 定义参数
K = 10  # D2D 用户对数
L = 128  # RIS 反射器单元数
sigma_gi = 1  # TAi 到 RIS 的信道标准差
sigma_hi = 1  # TBi 到 RIS 的信道标准差
sigma_ni = 1  # TBi 的噪声标准差
lb = 1  # 传输速率系数
N = 1000  # 蒙特卡罗模拟次数

# 定义发射功率范围
P_min = 0.001  # 最小发射功率，单位为 W
P_max = 100  # 最大发射功率，单位为 W
P_num = 11  # 发射功率采样点数
P_range = np.logspace(np.log10(P_min), np.log10(P_max), P_num)  # 发射功率范围，单位为 W

# 初始化系统总速率和简化近似的系统总速率的平均值和标准差
C_mean = np.zeros(P_num)  # 系统总速率的平均值
C_std = np.zeros(P_num)  # 系统总速率的标准差
C_prime_mean = np.zeros(P_num)  # 简化近似的系统总速率的平均值
C_prime_std = np.zeros(P_num)  # 简化近似的系统总速率的标准差

# 对每个发射功率进行蒙特卡罗模拟
for p in range(P_num):
    P = P_range[p] * np.ones(K)  # 发射功率向量，单位为 W
    print(str(p*10) + '%')

    # 初始化系统总速率和简化近似的系统总速率的样本向量
    C_sample = np.zeros(N)  # 系统总速率的样本向量
    C_prime_sample = np.zeros(N)  # 简化近似的系统总速率的样本向量

    # 对每次模拟进行计算
    for n in range(N):
        # 生成信道矩阵
        G = np.zeros((L, K), dtype=np.complex128)  # TAi 到 RIS 的信道矩阵
        H = np.zeros((L, K), dtype=np.complex128)  # TBi 到 RIS 的信道矩阵
        for i in range(K):
            G[:, i] = np.random.normal(0, sigma_gi, L) + 1j * np.random.normal(
                0, sigma_gi, L
            )  # 复高斯分布
            H[:, i] = np.random.normal(0, sigma_hi, L) + 1j * np.random.normal(
                0, sigma_hi, L
            )  # 复高斯分布

        # 生成相移矩阵
        theta = np.random.uniform(0, 2 * np.pi, L)  # 随机相移角度
        D = np.diag(np.exp(1j * theta))  # 相移矩阵

        # 计算转发信号
        X = np.zeros((K, K), dtype=np.complex128)  # 转发信号矩阵
        for i in range(K):
            for j in range(K):
                X[i, j] = H[:, j].T @ D @ G[:, i]  # 转发信号公式

        # 计算接收信号的SINR和速率
        SINR = np.zeros(K)  # 接收信号的SINR向量
        R = np.zeros(K)  # 接收信号的速率向量
        for i in range(K):
            SINR[i] = (
                P[i]
                * abs(X[i, i]) ** 2
                / (
                    np.sum(P * abs(X[i, :]) ** 2)
                    - P[i] * abs(X[i, i]) ** 2
                    + sigma_ni**2
                )
            )  # SINR公式（3）
            R[i] = lb * np.log2(1 + SINR[i])  # 速率公式（4）

        # 计算系统总速率
        C = np.sum(R)  # 系统总速率公式（5）
        C_sample[n] = C  # 保存系统总速率的样本

        # 计算简化近似的接收信号的SINR和速率
        SINR_prime = np.zeros(K)  # 简化近似的接收信号的SINR向量
        R_prime = np.zeros(K)  # 简化近似的接收信号的速率向量
        for i in range(K):
            SINR_prime[i] = (
                P[i]
                * L
                * sigma_gi**2
                * sigma_hi**2
                / (
                    np.sum(P * L * sigma_gi**2 * sigma_hi**2)
                    - P[i] * L * sigma_gi**2 * sigma_hi**2
                    + sigma_ni**2
                )
            )  # 简化近似的SINR公式（13）
            R_prime[i] = lb * np.log2(1 + SINR_prime[i])  # 简化近似的速率公式（13）

        # 计算简化近似的系统总速率
        C_prime = np.sum(R_prime)  # 简化近似的系统总速率公式（14）
        C_prime_sample[n] = C_prime  # 保存简化近似的系统总速率的样本

    # 计算系统总速率和简化近似的系统总速率的平均值和标准差
    C_mean[p] = np.mean(C_sample)  # 系统总速率的平均值
    C_std[p] = np.std(C_sample)  # 系统总速率的标准差
    C_prime_mean[p] = np.mean(C_prime_sample)  # 简化近似的系统总速率的平均值
    C_prime_std[p] = np.std(C_prime_sample)  # 简化近似的系统总速率的标准差

# 绘制系统总速率和简化近似的系统总速率随发射功率变化的曲线图
plt.figure()
plt.plot(10 * np.log10(P_range / 0.001), C_mean, label="系统总速率")
plt.plot(10 * np.log10(P_range / 0.001), C_prime_mean, label="简化近似的系统总速率")
plt.fill_between(
    10 * np.log10(P_range / 0.001), C_mean - C_std, C_mean + C_std, alpha=0.3
)
plt.fill_between(
    10 * np.log10(P_range / 0.001),
    C_prime_mean - C_prime_std,
    C_prime_mean + C_prime_std,
    alpha=0.3,
)
plt.xlabel("用户发射功率 (dBm)")
plt.ylabel("系统总速率 (bps/Hz)")
plt.legend()
plt.grid()
plt.show()
