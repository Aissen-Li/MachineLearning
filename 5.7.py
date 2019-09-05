import numpy as np
import time

x = [[0, 0], [0, 1], [1, 0], [1, 1]]  # 异或训练集
y = [[0], [1], [1], [0]]  # 异或输出


class RBF:
    def __init__(self, data, label, q, lr, err):
        self.data = np.array(data)
        self.label = np.array(label)
        self.input_neurons = len(self.data[0]) - 1
        self.output_neurons = len(np.unique(self.data[:, -1]))
        self.hidden_neurons = q
        self.lr = lr
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.center = np.random.rand(self.hidden_neurons, 2)  # 服从0到1均匀分布的隐层神经元对应的中心，q个2维，范围[0,1)
        self.err = err

    def create_data(self):
        w = np.mat(np.random.random((self.hidden_neurons, 1)))  # 初始化隐层到输出层的权值 q*1
        beta = np.mat(np.random.random((self.hidden_neurons, 1)))  # 初始化隐层径向基函数的尺度系数 q*1
        print('初始化隐层到输出层权值为{}'.format(w))
        print('初始化隐层径向基函数的尺度系数为{}'.format(beta))
        return w, beta

    def BP(self):
        t1 = time.perf_counter()
        x = self.data
        y = self.label
        center = self.center
        w, beta = self.create_data()
        lr = self.lr
        e_k = 1
        counter = 0
        while e_k > self.err:
            counter += 1
            d_w = 0
            d_beta = 0
            for i in range(self.rows):
                dis = []
                gauss = []
                for j in range(self.hidden_neurons):
                    dis_j = np.linalg.norm(x[i] - center[j])  # 2范数，样本到第j个隐层中心的模长
                    dis.append([dis_j])  # q*1
                    gauss_j = np.array(np.exp(-beta[j] * dis_j))
                    gauss.append(gauss_j[0])  # q*1
                out_y = np.mat(w).T * np.mat(gauss)  # 1*1
                d_y = out_y - y[i]
                d_w += -lr * np.array(d_y) * gauss  # q*1
                d_beta += lr * np.array(w) * np.array(d_y) * np.array(dis) * gauss # q*1
                e_k += 0.5 * np.sum((y[i] - out_y) ** 2)
            w += d_w / self.rows
            beta += d_beta / self.rows
            e_k = e_k / self.rows
        t2 = time.perf_counter()
        print('一共经过{}轮学习'.format(counter))
        print('学习时间一共为{}'.format(t2 - t1))
        print('学习后的隐层到输出层权值为{}'.format(w))
        print('学习后的隐层径向基函数的尺度系数为{}'.format(beta))
        return w, beta

    def test(self):
        x_test = np.random.randint(0, 2, (10, 2))
        y_test = np.logical_xor(x_test[:, 0], x_test[:, 1])
        w, beta = self.BP()
        y_out = []
        for i in range(len(x_test)):
            dis = []
            gauss = []
            for j in range(self.hidden_neurons):
                dis_j = np.linalg.norm(x_test[i] - self.center[j])  # 2范数，样本到第j个隐层中心的模长
                dis.append([dis_j])  # q*1
                gauss_j = np.array(np.exp(-beta[j] * dis_j))
                gauss.append(gauss_j[0])  # q*1
            out_y = np.mat(w).T * np.mat(gauss)  # 1*1
            y_out.append(np.array(out_y)[0])
        print('测试样本{}'.format(x_test))
        print('测试样本的异或值{}'.format(y_test))
        print('模型的输出值{}'.format(y_out))


RBF = RBF(x, y, 5, 0.1, 0.0001)
RBF.test()



