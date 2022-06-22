import pandas 
import numpy
# data = pandas.read_csv('online_shoppers_intention.csv')
# numpy.save('1_image.npy',data)

data1 = pandas.read_csv('new.csv')
print(data1)
numpy.save('4_label.npy',data1) 

# data1 = numpy.load('4_label.npy')
# print(data1)
# [row, col] = data1.shape

# new = numpy.zeros(row)

# for i in range(0,row):
#     new[i] = data1[i][0]
print(new)
numpy.save('4_labels.npy',new) 




# data = numpy.load('data3.npy')

# print(data[4])

# class kmeans():
#     def __init__(self, data, K=2, method='random'):
#     # 初始化聚类中心，method可选 random|distance|random+distance 三种方法
#     def init_centers(self, data, method):
#     # 返回各个类中的所有样本和聚类中心的欧式距离平方总和
#     def get_distance(self, data):
#     # 返回样本所属聚类集
#     def get_cluster(self, data):
#     # 返回新的聚类中心集
#     def get_center(self, data, clusters):
#     # 模型训练过程
#     def train(self, data):
#     # 返回实验所需要的NMI和J值
#     def ret(self, data, labels):
