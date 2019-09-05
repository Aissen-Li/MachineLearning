import numpy as np
import time


class BP:
    def __init__(self, data):
        self.data = np.array(data)
        self.input_neurons = len(self.data[0]) - 1
        self.output_neurons = len(np.unique(self.data[:, -1]))
        self.lr = 0.1
        self.rows = len(self.data)
        self.cols = len(self.data[0])

    def set_eta(self, lr):  # 设置学习率，默认为0.1
        self.lr = lr

    def get_eta(self):
        return self.lr

    def get_input_neurons(self):
        return self.input_neurons

    def get_output_neurons(self):
        return self.output_neurons

    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def create_data(self, q=10):
        x = self.data[:, :-1]  # 取得data的数据部分
        x = np.insert(x, [0], -1, axis=1)  # 在所有数据前插入一列-1作为哑结点
        y = np.array([self.data[:, -1], 1 - self.data[:, -1]]).transpose()
        # 类别1*l，输出神经元个数跟类别一致，所以取原来的真实输出与真实输出的相反值作为y
        d = self.input_neurons
        l = self.output_neurons
        v = np.mat(np.random.random((d + 1, q)))  # 初始化输入层到隐层的权值
        w = np.mat(np.random.random((q + 1, l)))  # 初始化隐层到输出层的权值
        return x, y, d, l, v, w

    def BP(self, err=0.01):
        time_start = time.perf_counter()
        x, y, d, l, v, w = self.create_data()
        print('随机初始化得到的输入层到隐层的连接权{}'.format(v))
        print('随机初始化得到的隐层到输出层的连接权{}'.format(w))

        lr = self.lr
        e_k = 1
        counter = 0
        while e_k > err:
            counter += 1
            for i in range(self.rows):
                alpha = np.mat(x[i, :]) * v  # 1*q
                b_init = self.sigmod(alpha)  # 1*q
                b = np.insert(b_init, [0], -1, axis=1)  # 1*(q+1)
                beta = b * w  # 1*l
                out_y = np.array(self.sigmod(beta))  # 1*l

                g = out_y * (1 - out_y) * (y[i, :] - out_y)  # 1*l
                # print(g)
                w_g = w[1:, :] * np.mat(g).T  # q*1  哑结点的连接权不需要更新，切片
                # print(w_g.T)
                e = np.array(b_init) * (1 - np.array(b_init)) * np.array(w_g.T)  # 1*q
                # print(e)
                d_w = lr * np.mat(b).T * np.mat(g)  # (q+1)*1 * (1*l) = (q+1)*l
                # print(d_w)
                d_v = lr * np.mat(x[i, :]).T * np.mat(e)  # (d+1)*1 * 1*q = (d+1)*q
                # print(d_v)
                w += d_w
                v += d_v
                e_k = 0.5 * np.sum((y[i, :] - out_y) ** 2)
        time_finish = time.perf_counter()
        print('共经过{}轮BP训练'.format(counter))
        print('得到的输入层到隐层的连接权{}'.format(v))
        print('得到的隐层到输出层的连接权{}'.format(w))
        print('BP模型运行时间{}s'.format(time_finish - time_start))
        return

    def ABP(self, err=0.01):
        time_start = time.perf_counter()
        x, y, d, l, v, w = self.create_data()
        print('随机初始化得到的输入层到隐层的连接权{}'.format(v))
        print('随机初始化得到的隐层到输出层的连接权{}'.format(w))
        lr = self.lr
        e_k = 1
        counter = 0
        while e_k > err:
            d_v = 0
            d_w = 0
            e_k = 0  # 每次遍历完整个数据集更新一次连接权，所以每次大循环需要重置
            counter += 1
            for i in range(self.rows):
                alpha = np.mat(x[i, :]) * v  # 1*q
                b_init = self.sigmod(alpha)  # 1*q
                b = np.insert(b_init, [0], -1, axis=1)  # 1*(q+1)
                beta = b * w  # 1*l
                out_y = np.array(self.sigmod(beta))  # 1*l

                g = out_y * (1 - out_y) * (y[i, :] - out_y)  # 1*l
                # print(g)
                w_g = w[1:, :] * np.mat(g).T  # q*1  哑结点的连接权不需要更新，切片
                # print(w_g.T)
                e = np.array(b_init) * (1 - np.array(b_init)) * np.array(w_g.T)  # 1*q
                # print(e)
                d_w += lr * np.mat(b).T * np.mat(g)  # (q+1)*1 * (1*l) = (q+1)*l
                # print(d_w)
                d_v += lr * np.mat(x[i, :]).T * np.mat(e)  # (d+1)*1 * 1*q = (d+1)*q
                # print(d_v)
                e_k += 0.5 * np.sum((y[i, :] - out_y) ** 2)
            w += d_w / self.rows
            v += d_v / self.rows
            e_k = e_k / self.rows
        time_finish = time.perf_counter()
        print('共经过{}轮ABP训练'.format(counter))
        print('得到的输入层到隐层的连接权{}'.format(v))
        print('得到的隐层到输出层的连接权{}'.format(w))
        print('ABP模型运行时间{}s'.format(time_finish - time_start))
        return


D = np.array([
    [1, 1, 1, 1, 1, 1, 0.697, 0.460, 1],
    [2, 1, 2, 1, 1, 1, 0.774, 0.376, 1],
    [2, 1, 1, 1, 1, 1, 0.634, 0.264, 1],
    [1, 1, 2, 1, 1, 1, 0.608, 0.318, 1],
    [3, 1, 1, 1, 1, 1, 0.556, 0.215, 1],
    [1, 2, 1, 1, 2, 2, 0.403, 0.237, 1],
    [2, 2, 1, 2, 2, 2, 0.481, 0.149, 1],
    [2, 2, 1, 1, 2, 1, 0.437, 0.211, 1],
    [2, 2, 2, 2, 2, 1, 0.666, 0.091, 0],
    [1, 3, 3, 1, 3, 2, 0.243, 0.267, 0],
    [3, 3, 3, 3, 3, 1, 0.245, 0.057, 0],
    [3, 1, 1, 3, 3, 2, 0.343, 0.099, 0],
    [1, 2, 1, 2, 1, 1, 0.639, 0.161, 0],
    [3, 2, 2, 2, 1, 1, 0.657, 0.198, 0],
    [2, 2, 1, 1, 2, 2, 0.360, 0.370, 0],
    [3, 1, 1, 3, 3, 1, 0.593, 0.042, 0],
    [1, 1, 2, 2, 2, 1, 0.719, 0.103, 0]])

test = BP(D)
test.BP()
test.ABP()


