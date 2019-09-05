import numpy as np


Data = np.array([
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

# m = len(Data)  # 样本个数
# n = len(Data[0]) - 3  # 属性个数，这里只处理离散变量，连续变量不做考虑
# y = Data[:, -1]
# test = [1, 1, 1, 1, 1, 1, 0.697, 0.460]  # 测试用例
# positive_dic = [{} for _ in range(n)]  # 记录正类中不同属性的个数
# p_positive = [{} for _ in range(n)]  # 记录正类不同属性的概率
#
# for i, d in enumerate(positive_dic):
#     for attr in Data[:, i]:
#         d[attr] = 0
#     for attr in Data[:8, i]:  # 分类为正的元素
#         d[attr] = d.get(attr, 0) + 1
# for i, d in enumerate(p_positive):
#     for attr in positive_dic[i]:
#         d[attr] = (positive_dic[i][attr] + 1) / (len(y) + len(np.unique(y)) * len(np.unique(Data[:, i])))
#
# negative_dic = [{} for i in range(n)]  # 记录负类中不同属性的个数
# p_negative = [{} for i in range(n)]  # 记录负类不同属性的概率
# for i, d in enumerate(negative_dic):
#     for attr in Data[:, i]:
#         d[attr] = 0
#     for attr in Data[8:, i]:  # 分类为负的元素
#         d[attr] = d.get(attr, 0) + 1
# for i, d in enumerate(p_negative):
#     for attr in negative_dic[i]:
#         d[attr] = (negative_dic[i][attr] + 1) / (len(y) + len(np.unique(y)) * len(np.unique(Data[:, i])))
# print(p_positive)
# print(p_negative)

n = len(Data[0]) - 3  # 属性个数，这里只处理离散变量，连续变量不做考虑
y = Data[:, -1]
test = [1, 1, 1, 1, 1, 1, 0.697, 0.460]  # 测试用例

p_positive = 0  # 正例的概率
p_negative = 0  # 负例的概率

# 遍历6个离散变量
for i in range(6):
    D_c_xi = [0, 0]  # 计算D_c_xi 下标0代表分类为0 下标1代表分类1
    for j in range(len(Data)):  # 遍历所有样本
        if Data[j, i] == test[i]:
            D_c_xi[int(y[j])] += 1
    p_positive_condition = 1  # 计算p(xj|c,xi)
    p_negative_condition = 1  # 计算p(xj|c,xi)
    for j in range(6):
        D_c_xi_xj = [0, 0]  # 计算D_c_xi_xj 下标0代表分类为0 下标1代表分类1
        for k in range(len(Data)):
            if Data[k, i] == test[i] and Data[k, j] == test[j]:
                D_c_xi_xj[int(y[j])] += 1
        p_positive_condition *= (D_c_xi_xj[1] + 1) / (D_c_xi[1] + len(np.unique(Data[:, j])))
        p_negative_condition *= (D_c_xi_xj[0] + 1) / (D_c_xi[0] + len(np.unique(Data[:, j])))
    p_positive += (D_c_xi[1] + 1) / (len(Data) + 2 * len(np.unique(Data[:, i]))) * p_positive_condition
    p_negative += (D_c_xi[0] + 1) / (len(Data) + 2 * len(np.unique(Data[:, i]))) * p_negative_condition
print('判断为正例的概率是{}，判断为负例的概率是{}'.format(p_positive, p_negative))
if p_positive > p_negative:
    print('判断为正例')
else:
    print('判断为负例')