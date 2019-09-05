import numpy as np
import pandas as pd


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


m = len(Data)  # 样本个数
n = len(Data[0]) - 1  # 属性个数
y = Data[:, -1]
class_dic = {}
for label in y:
    class_dic[label] = class_dic.get(label, 0) + 1
# print(class_dic)
p_dic = {}  # 先验概率
for label in class_dic:
    p_dic[label] = (class_dic[label] + 1) / (len(y) + len(np.unique(y)))

# 最后两个为连续属性，需要另外处理
positive_dic = [{} for i in range(n-2)]  # 记录正类中不同属性的个数
p_positive = [{} for i in range(n-2)]  # 记录正类不同属性的概率
for i, d in enumerate(positive_dic):
    for attr in Data[:, i]:
        d[attr] = 0
    for attr in Data[:8, i]:  # 分类为正的元素
        d[attr] = d.get(attr, 0) + 1
for i, d in enumerate(p_positive):
    for attr in positive_dic[i]:
        d[attr] = (positive_dic[i][attr] + 1) / (8 + len(np.unique(Data[:, i])))

negative_dic = [{} for i in range(n-2)]  # 记录负类中不同属性的个数
p_negative = [{} for i in range(n-2)]  # 记录负类不同属性的概率
for i, d in enumerate(negative_dic):
    for attr in Data[:, i]:
        d[attr] = 0
    for attr in Data[8:, i]:  # 分类为负的元素
        d[attr] = d.get(attr, 0) + 1
for i, d in enumerate(p_negative):
    for attr in negative_dic[i]:
        d[attr] = (negative_dic[i][attr] + 1) / (9 + len(np.unique(Data[:, i])))


def p_continous(x, data):
    mean = np.mean(data)
    var = pd.Series(data).var()
    p = np.exp(-(x - mean) ** 2 * 0.5 / var) / (np.sqrt(2 * np.pi * var))
    return p


test = [1, 1, 1, 1, 1, 1, 0.697, 0.460]  # the predict sample
p_positive.append(p_continous(test[-2], Data[:8, 6]))
p_positive.append(p_continous(test[-1], Data[:8, 7]))
p_negative.append(p_continous(test[-2], Data[8:, 6]))
p_negative.append(p_continous(test[-1], Data[8:, 7]))

result = [p_dic[0], p_dic[1]]
for i, attr in enumerate(test[:-2]):
    result[0] *= p_positive[i][attr]
    result[1] *= p_negative[i][attr]
result[0] *= p_positive[-2] * p_positive[-1]
result[1] *= p_negative[-2] * p_negative[-1]
print('类先验概率是{}'.format(p_dic))
print('正例的类条件概率{}'.format(p_positive))
print('负例的类田间概率{}'.format(p_negative))
print('判断为正例的概率是{}，判断为负例的概率是{}'.format(result[0], result[1]))
if result[0] > result[1]:
    print('该测试样本判断为正例')
else:
    print('该测试样本判断为负例')