import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


class KNN(object):
    def __init__(self, x, y, k):
        self.x = x
        self.y = y
        self.k = k
        self.n = len(x)  # 样本个数

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def knn(self, x):
        distance = []
        for i in range(self.n):
            dist = self.distance(x, self.x[i])
            distance.append([self.x[i], self.y[i], dist])
        distance.sort(key=lambda x: x[2])
        neighbors = []
        neighbors_labels = []
        for k in range(self.k):
            neighbors.append(distance[k][0])  # 近邻具体数据
            neighbors_labels.append(distance[k][1])  # 近邻标记
        return neighbors, neighbors_labels

    def vote(self, x):
        neighbors, neighbors_labels = self.knn(x)
        vote = {}  # 投票法
        for label in neighbors_labels:
            vote[label] = vote.get(label, 0) + 1
        sort_vote = sorted(vote.items(), key=lambda x:x[1], reverse=True)
        return sort_vote[0][0]  # 返回投票数最多的标记

    def fit(self):
        labels = []
        for sample in self.x:
            label = self.vote(sample)
            labels.append(label)
        return labels  # 返回所有样本的标记

    def accuracy(self):
        predict_labels = self.fit()
        real_labels = self.y
        correct = 0
        for predict, real in zip(predict_labels, real_labels):
            if int(predict) == int(real):
                correct += 1
        return correct / self.n


X = dataSet[['density', 'sugar_rate']].values
y = dataSet['label']
for k in range(1, 4):
    print('本次knn的k值选取为{}'.format(k))
    knn = KNN(X, y, k)
    predict = knn.fit()
    print('本次knn的正确率为{}'.format(knn.accuracy()))

    x_positive = []
    y_positive = []
    x_negative = []
    y_negative = []
    for i in range(len(X)):
        if int(predict[i]) == 1:
            x_positive.append(X[i][0])
            y_positive.append(X[i][1])
        else:
            x_negative.append(X[i][0])
            y_negative.append(X[i][1])

    plt.scatter(x_positive, y_positive, marker='o', color='red', label='1')
    plt.scatter(x_negative, y_negative, marker='o', color='blue', label='0')
    plt.xlabel('密度')
    plt.ylabel('含糖率')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.legend(loc='upper left')
    plt.show()

