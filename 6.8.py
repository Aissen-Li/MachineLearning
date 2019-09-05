import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np


data = [[0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.430, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        [0.666, 0.091, 0],
        [0.243, 0.267, 0],
        [0.245, 0.057, 0],
        [0.343, 0.099, 0],
        [0.639, 0.161, 0],
        [0.657, 0.198, 0],
        [0.360, 0.370, 0],
        [0.593, 0.042, 0],
        [0.719, 0.103, 0]]
column = ['density', 'sugar_rate', 'label']
dataSet = pd.DataFrame(data, columns=column)
X = dataSet[['density']].values
y = dataSet['sugar_rate'].values

for kernel in ['linear', 'rbf']:
    clf = svm.SVR(C=1000, kernel=kernel, gamma='scale')
    clf.fit(X, y)
    sv = clf.support_vectors_
    sv_id = clf.support_
    plt.scatter(X, y, color='black', label='data')
    plt.scatter(X[sv_id], y[sv_id], color='red', label='supporting vectors')
    if kernel == 'linear':
        x_points = np.linspace(0, 1, 10)
        y_ = (clf.coef_ * x_points + clf.intercept_)
        plt.plot(x_points, y_[0], color='green')
    plt.xlabel('density')
    plt.ylabel('sugar_rate')
    plt.legend()
    plt.title('{} kernel'.format(kernel))
    plt.show()
    print('{}核函数的支持向量为{}，支持向量的下标为{}'.format(kernel, sv, sv_id))
