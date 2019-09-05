import numpy as np
import matplotlib.pyplot as plt


density = np.array(
    [0.697, 0.774, 0.634, 0.608, 0.556, 0.430, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593,
     0.719]).reshape(-1, 1)
sugar_rate = np.array(
    [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042,
     0.103]).reshape(-1, 1)
x = np.hstack((density, sugar_rate))
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
x_positive = x[: 8]
x_negative = x[8:]
miu_positive = np.mean(x_positive, axis=0).reshape(-1, 1)
miu_negative = np.mean(x_negative, axis=0).reshape(-1, 1)
cov_positive = np.cov(x_positive, rowvar=False)
cov_negative = np.cov(x_negative, rowvar=False)
S_w = cov_positive + cov_negative
weights = np.mat(S_w).I * (miu_negative - miu_positive)
# print(weights)


plt.plot([0, 1], [0, -weights[0] / weights[1]], label='y', color='green')
plt.scatter(x[:8, 0], x[:8, 1], label='1')
plt.scatter(x[8:, 0], x[8:, 1], label='0')
plt.legend()
plt.show()