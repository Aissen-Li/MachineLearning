from math import exp
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # print(data)
    return data[:, :2], data[:, -1]


def tenfolddata(positive_data, negative_data):
    fold_data = []
    for i in range(10):
        pos_temp = positive_data[i * 5: (i+1) * 5].tolist()
        neg_temp = negative_data[i * 5: (i+1) * 5].tolist()
        temp = pos_temp + neg_temp
        fold_data += temp
    return np.array(fold_data)


x, y = create_data()
x_negative = x[:50]
x_positive = x[50:]
x_tenfold_test = tenfolddata(x_positive, x_negative)
y_tenfold_test = np.array([[1]] * 5 + [[0]] * 5)


class LogisticReressionClassifier:
    def __init__(self, max_iter=50, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid(self, x):  # sigmoid函数
        return 1 / (1 + exp(-x))

    def data_matrix(self, X):  # 构建数据矩阵
        data_mat = []
        for d in X:
            data_mat.append([1.0, *d])  # 数据之前加一个1,为的是拟合不需要另外加偏置b
        return data_mat

    def fit(self, X, y):
        # label = np.mat(y)
        data_mat = self.data_matrix(X)  # m*n
        self.weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32)  # 权重w，生成与数据长度相同的0元素矩阵

        for iter_ in range(self.max_iter):
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights))
                error = y[i] - result
                self.weights += self.learning_rate * error * np.transpose([data_mat[i]])

        print('本次训练完成，权重变为{}'.format(self.weights))

    # def f(self, x):
    #     return -(self.weights[0] + self.weights[1] * x) / self.weights[2]

    def score(self, X_test, y_test):
        right = 0
        X_test = self.data_matrix(X_test)
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weights)
            if (result < 0.5 and y == 0) or (result > 0.5 and y == 1):
                right += 1
        return right / len(X_test)


#  10折验证
lr_clf_tenfold = LogisticReressionClassifier()
y_tenfold_train = np.array([[0]] * 45 + [[1]] * 45)
for i in range(10):
    x_tenfold_train = np.array(x[0: i * 5].tolist() + x[(i + 1) * 5: 50].tolist() +
                               x[50: 50 + i * 5].tolist() + x[50 + (i + 1) * 5: 100].tolist())
    # print('测试数据{}'.format(x_tenfold_test[i * 10: (i + 1) * 10]))
    # print('测试标记{}'.format(y_tenfold_test))
    # print('训练数据个数为{}'.format(len(x_tenfold_train)))
    # print('训练标记个数为{}'.format(len(y_tenfold_train)))
    lr_clf_tenfold.fit(x_tenfold_train, y_tenfold_train)
    correct_rate_tenfold = lr_clf_tenfold.score(x_tenfold_test[i * 10: (i + 1) * 10], y_tenfold_test)
    print('这是第{}次10折验证，正确率为{}'.format(i + 1, correct_rate_tenfold))


# 留一法验证
lr_clf_leftone = LogisticReressionClassifier()
correct = 0
for j in range(len(x)):
    x_leftone_train = np.array(x[0: j].tolist() + x[j + 1:].tolist())
    y_leftone_train = np.array(y[0: j].tolist() + y[j + 1:].tolist())
    x_leftone_test = [x[j].tolist()]
    y_leftone_test = [y[j]]
    lr_clf_leftone.fit(x_leftone_train, y_leftone_train)
    correct_rate_leftone = lr_clf_leftone.score(x_leftone_test, y_leftone_test)
    print('第{}次留一验证完成'.format(j + 1))
    correct += correct_rate_leftone
print('留一验证共100次完成，正确率{}'.format(correct / 100))

