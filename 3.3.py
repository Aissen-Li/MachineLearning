# logistic regression
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


density = np.array(
    [0.697, 0.774, 0.634, 0.608, 0.556, 0.430, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593,
     0.719]).reshape(-1, 1)
sugar_rate = np.array(
    [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042,
     0.103]).reshape(-1, 1)
X = np.hstack((density, sugar_rate))
# xtrain = np.hstack((np.ones([density.shape[0], 1]), xtrain))
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
# xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=0.25, random_state=33)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


class LogisticReressionClassifier:
    def __init__(self, max_iter=200, learning_rate=0.5):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid(self, x):  # sigmoid函数
        return 1 / (1 + exp(-x))

    def data_matrix(self, X):  # 构建数据矩阵
        data_mat = []
        for d in X:
            data_mat.append([1.0, *d])  # 数据之前加一个1,为的是拟合不需要另外加偏置b
        # print(data_mat)
        return data_mat

    def fit(self, X, y):
        # label = np.mat(y)
        data_mat = self.data_matrix(X)  # m*n
        self.weights = np.ones((len(data_mat[0]), 1), dtype=np.float32)  # 权重w，生成与数据长度相同的1元素矩阵

        for iter_ in range(self.max_iter):
            FD = 0
            SD = 0  # 一阶二阶导数
            for i in range(len(X)):
                p = self.sigmoid(np.dot(data_mat[i], self.weights))
                FD += data_mat[i] * (y[i] - p)
                SD += np.matmul(data_mat[i], np.transpose(data_mat[i])) * p * (1 - p)
                # result = self.sigmoid(np.dot(data_mat[i], self.weights))
                # error = y[i] - result
                # self.weights += self.learning_rate * error * np.transpose([data_mat[i]])
                self.weights -= self.learning_rate * -np.transpose([FD]) * (1 / SD)
        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(self.learning_rate, self.max_iter))

    # def f(self, x):
    #     return -(self.weights[0] + self.weights[1] * x) / self.weights[2]

    def score(self, X_test, y_test):
        right = 0
        X_test = self.data_matrix(X_test)
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weights)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(X_test)


lr_clf = LogisticReressionClassifier()
lr_clf.fit(X_train, y_train)
print(lr_clf.weights)
print(lr_clf.score(X_test, y_test))

x_ponits = np.arange(0, 2)
y_ = -(lr_clf.weights[1]*x_ponits + lr_clf.weights[0])/lr_clf.weights[2]
plt.plot(x_ponits, y_)

#lr_clf.show_graph()
plt.scatter(X[:9, 0], X[:9, 1], label='0')
plt.scatter(X[9:, 0], X[9:, 1], label='1')
plt.legend()
plt.show()




# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
#
#
# # print(sigmoid(density))
# def logit_regression(theta, x, y, iteration=100, learning_rate=0.1, lbd=0.01):
#     for i in range(iteration):
#         theta = theta - learning_rate / y.shape[0] * (
#                     np.dot(x.transpose(), (sigmoid(np.dot(x, theta)) - y)) + lbd * theta)
#         cost = -1 / y.shape[0] * (np.dot(y.transpose(), np.log(sigmoid(np.dot(x, theta)))) + np.dot((1 - y).transpose(),
#                                                                                                     np.log(1 - sigmoid(
#                                                                                                         np.dot(x,
#                                                                                                                theta))))) + lbd / (
#                            2 * y.shape[0]) * np.dot(theta.transpose(), theta)
#         print('---------Iteration %d,cost is %f-------------' % (i, cost))
#     return theta
#
#
# def predict(theta, x):
#     pre = np.zeros([x.shape[0], 1])
#     for idx, valu in enumerate(np.dot(x, theta)):
#         if sigmoid(valu) >= 0.5:
#             pre[idx] = 1
#         else:
#             pre[idx] = 0
#     return pre
#
#
# theta_init = np.random.rand(3, 1)
# pre = predict(theta_init, xtest)
# theta = logit_regression(theta_init, xtrain, ytrain, learning_rate=1)
# print('predictions are', pre)
# print('ground truth is', ytest)
# print('theta is ', theta)
# print('the accuracy is', np.mean(pre == ytest))
# print(classification_report(ytest, pre, target_names=['Bad', 'Good']))