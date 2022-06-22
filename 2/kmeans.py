import ssl
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
epochs = 20


class kmeans():
    def __init__(self, data, K=2, method='random'):
        self.K = K
        self.centers = None
        # 初始化聚类中心
        self.init_centers(data, method)

    # 初始化聚类中心，method可选 random|distance|random+distance 三种方法
    def init_centers(self, data, method):
        # 生成一个随机化的序列作为索引
        shuff_num = np.arange(data.shape[0])
        np.random.shuffle(shuff_num)

        if method == 'random':
            # 随机选K个作为聚类中心
            self.centers = np.zeros((self.K, data.shape[1]))
            for i in range(self.K):
                self.centers[i] = data[shuff_num[i]]

        elif method == 'distance':
            self.centers = np.zeros((1, data.shape[1]))
            # 记录已选的样本序号
            visted = []
            # 先随机选一个样本为第一个聚类中心，
            self.centers[0] = data[shuff_num[0]]
            visted.append(shuff_num[0])
            # 再选K-1个离当前聚类中心最远的样本为聚类中心
            for i in range(1, self.K):
                distance_arr = self.get_distance(data)
                sum_dis = np.sum(distance_arr, axis=1)
                # 从大到小排序距离
                shuff_num = np.argsort(-sum_dis)
                for i in shuff_num:
                    if i not in visted:
                        self.centers = np.insert(
                            self.centers, obj=self.centers.shape[0], values=data[i], axis=0)
                        visted.append(i)
                        break

        elif method == 'random+distance':
            self.centers = np.zeros((1, data.shape[1]))
            visted = np.zeros(data.shape[0], dtype=int)
            # 先随机选一个样本为第一个聚类中心，
            self.centers[0] = data[shuff_num[0]]
            visted[shuff_num[0]] = 1
            # 然后在离当前聚类中心最远的点中 随机选 新的聚类中心
            for i in range(1, self.K):
                # 没选的样本序列
                notvisted = np.argsort(visted)
                notvisted = notvisted[:-i]
                # 计算没选的样本离聚类中心的总距离
                distance_arr = self.get_distance(data[notvisted, :])
                sum_dis = np.sum(distance_arr, axis=1)
                # 从远到近排序
                shuff_num = np.argsort(-sum_dis)
                while True:
                    i = notvisted[random.randint(0, notvisted.size-1)]
                    # 随机选点，如果没被选过，而且距离超过一半的点就是新的中心
                    if i in notvisted[shuff_num[:int(np.ceil(float(shuff_num.size)/2))]]:
                        self.centers = np.insert(
                            self.centers, obj=self.centers.shape[0], values=data[i], axis=0)
                        visted[i] = 1
                        break

    # 返回各个类中的所有样本和聚类中心的欧式距离平方总和
    def get_distance(self, data):
        distance_arr = np.zeros((data.shape[0], self.centers.shape[0]))
        for i in range(data.shape[0]):
            distance_arr[i] = np.sum((self.centers - data[i])**2, axis=1)**0.5
        return distance_arr

    # 返回样本所属聚类集
    def get_cluster(self, data):
        distance_arr = self.get_distance(data)
        clusters = np.argmin(distance_arr, axis=1)
        sum_dis = np.sum(np.min(distance_arr, axis=1))
        return clusters, sum_dis

    # 返回新的聚类中心集
    def get_center(self, data, clusters):
        now_cluster = np.zeros((data.shape[0], self.K))
        now_cluster[clusters[:, None] == np.arange(self.K)] = 1
        # print(now_cluster)
        return np.dot(now_cluster.T, data)/np.sum(now_cluster, axis=0).reshape((-1, 1))

    # 模型训练过程
    def train(self, data):
        clusters, _ = self.get_cluster(data)
        newcenters = self.get_center(data, clusters)
        dif = np.sum((newcenters-self.centers)**2)**0.5
        self.centers = newcenters
        return dif

    # 返回实验所需要的NMI和J值
    def ret(self, data, labels):
        clusters, sum_dis = self.get_cluster(data)
        return normalized_mutual_info_score(clusters, labels), sum_dis


data_num = 4

images = np.load('data/'+str(data_num)+'_image.npy')
raw_label = np.load('data/'+str(data_num)+'_label.npy')

row = len(raw_label)
labels = np.zeros(row, dtype=np.int64)

# 从给的标签中找出正确的类大小
K_true = 0
for i in range(0, row):
    labels[i] = raw_label[i]-1
    if(labels[i] > K_true):
        K_true = labels[i]
K_true = K_true+1
print(K_true)

J_arr = []

# # 聚类个数设定为正确类个数情况下，三种不同初始化方法版本的聚类效果
# plt.figure()
# plt.xlabel("epochs")
# plt.ylabel("NMI")
# plt.xticks(range(1,epochs+1))

# method_str = ['random','distance','random+distance']
# color_str = ['red','blue','black']
# for r in range(3):
#     nmi_arr = []
#     print('method = ' + method_str[r] + ' : ')
#     # 类设定为正确类个数
#     train_model = kmeans(data=images, K=K_true, method=method_str[r])
#     # 训练 epochs 轮，记录每轮的 NMI 值
#     for i in range(epochs):
#         dif = train_model.train(images)
#         nmi_val, distance = train_model.ret(images, labels)
#         print('J = {:.4f} NMI = {:.4f}'.format(distance , nmi_val))
#         nmi_arr.append( nmi_val )
#     # 将该method的 NMI 值变化加入图中
#     plt.plot(range(1,epochs+1), nmi_arr, c=color_str[r], label=method_str[r])

# plt.legend()
# plt.grid()
# plt.savefig('nmi_'+str(data_num)+'.jpg', dpi = 1800)
# plt.show()

# 设置不同聚类个数K的情况下，三种不同初始化方法版本的 目标函数J 随着聚类个数变化的曲线。
plt.figure()
plt.xlabel("K")
plt.ylabel("J_val")
plt.xticks(range(1, epochs+1))

method_str = ['random', 'distance', 'random+distance']
color_str = ['red', 'blue', 'black']
for r in range(3):
    J_arr = []
    print('method = ' + method_str[r] + ' : ')
    # 聚类个数变化在 2~K
    for K_num in range(2, K_true+1):
        train_model = kmeans(data=images, K=K_num, method=method_str[r])
        for i in range(epochs):
            dif = train_model.train(images)
            # 差别小到一定程度，结束训练
            if dif < 1e-6:
                endi = epochs-i-1
                break
        nmi_val, j_val = train_model.ret(images, labels)
        print('J = {:.4f} NMI = {:.4f}'.format(j_val, nmi_val))
        J_arr.append(j_val)

    plt.plot(range(2, K_true+1), J_arr, c=color_str[r], label=method_str[r])

plt.legend()
plt.grid()
plt.savefig('J'+str(data_num)+'.jpg', dpi=1800)
# plt.show()
