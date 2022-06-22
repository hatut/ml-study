# 19335084 黄梓浩 hatut
# collaborative filtering
from math import sqrt
from operator import itemgetter
import re
import csv
import random


class my_collaborative_filtering:
    def __init__(self, data):
        self.user_dataset = {}  # 以用户为键，记录每个用户对电影的评分
        self.item_dataset = {}  # 以电影为键，记录每个电影下的评分用户的评分
        for sub in data:
            self.user_dataset.setdefault(sub[0], {})
            self.user_dataset[sub[0]].setdefault(sub[1], sub[2])
            self.item_dataset.setdefault(sub[1], {})
            self.item_dataset[sub[1]].setdefault(sub[0], sub[2])

    def get_avg_rating(self, data):  # 求得该项的字典数据下的平均值
        ret = 0.0
        for (item, rating) in data.items():
            ret += float(rating)
        return ret / float(len(data))

# 基于用户的方法
    def get_user_pearson(self, user1, user2):  # 用pearson相关性求用户间的相似度
        user1_data = self.user_dataset[user1]
        user2_data = self.user_dataset[user2]
        avg_x = self. get_avg_rating(user1_data)
        avg_y = self. get_avg_rating(user2_data)
        # get common movie
        common_list = []
        tmp = {}
        for (m, r) in user1_data.items():
            tmp.setdefault(m, 0)
            tmp[m] += 1
        for (m, r) in user2_data.items():
            tmp.setdefault(m, 0)
            tmp[m] += 1
        for (m, count) in tmp.items():
            if(count == 2):
                common_list.append(m)
        cov = sig_x = sig_y = 0.0
        for item in common_list:
            x = user1_data[item]
            y = user2_data[item]
            cov += (x - avg_x) * (y - avg_y)
            sig_x += (x - avg_x) ** 2
            sig_y += (y - avg_y) ** 2
        sig_x = sqrt(sig_x)
        sig_y = sqrt(sig_y)
        if (sig_x*sig_y == 0):
            return 0.0
        return cov / (sig_x * sig_y)

    def get_user_k_near_neigh(self, user, k):  # 返回k个最近邻的相似用户
        neigh = []
        for (u, data) in self.user_dataset.items():
            if (u != user):
                p = self.get_user_pearson(user, u)
                neigh.append([u, p])
        return (sorted(neigh, key=itemgetter(1), reverse=True))[:k]

    def user_based_predict(self, user, item, k):  # 通过基于用户的方法预测某用户对某物品的评分
        # 先取k个最相似的
        k_near_neigh = self.get_user_k_near_neigh(user, k)
        valid_neigh = []
        # 在k个中找对该电影评了分的项
        for nei in k_near_neigh:
            if (item in self.user_dataset[nei[0]].keys()):
                valid_neigh.append(nei)
        if (len((valid_neigh)) == 0):
            return 0.0
        # ∑(Wi,1)*(rating i,item) / ∑(Wi,1)，
        up = down = 0.0
        for nei in valid_neigh:
            rating = self.user_dataset[nei[0]][item]
            up += nei[1] * rating
            down += nei[1]
        if (down == 0.0):
            return 0.0
        return up / down

# 基于物品的方法
    def get_item_pearson(self, item1, item2):
        item1_data = self.item_dataset[item1]
        item2_data = self.item_dataset[item2]
        avg_x = self. get_avg_rating(item1_data)
        avg_y = self. get_avg_rating(item2_data)
        # get common user
        common_list = []
        tmp = {}
        for (u, r) in item1_data.items():
            tmp.setdefault(u, 0)
            tmp[u] += 1
        for (u, r) in item2_data.items():
            tmp.setdefault(u, 0)
            tmp[u] += 1
        for (u, count) in tmp.items():
            if(count == 2):
                common_list.append(u)
        cov = sig_x = sig_y = 0.0
        for item in common_list:
            x = item1_data[item]
            y = item2_data[item]
            cov += (x - avg_x) * (y - avg_y)
            sig_x += (x - avg_x) ** 2
            sig_y += (y - avg_y) ** 2
        sig_x = sqrt(sig_x)
        sig_y = sqrt(sig_y)
        if(sig_x == 0 or sig_y == 0):
            return 0.0
        return cov / (sig_x * sig_y)

    def get_item_k_near_neigh(self, item, k):
        neigh = []
        for (m, data) in self.item_dataset.items():
            if (m != item):
                p = self.get_item_pearson(item, m)
                neigh.append([m, p])
        return sorted(neigh, key=itemgetter(1), reverse=True)[:k]

    def item_based_predict(self, user, item, k):  # 通过基于物品的方法预测某用户对某物品的评分
        k_near_neigh = self.get_item_k_near_neigh(item, k)
        valid_neigh = []
        for nei in k_near_neigh:
            if (user in self.item_dataset[nei[0]].keys()):
                valid_neigh.append(nei)
        if (len((valid_neigh)) == 0):
            return 0.0
        # ∑(Wi,1)*(rating i,item) / ∑(Wi,1)
        up = down = 0.0
        for nei in valid_neigh:
            rating = self.item_dataset[nei[0]][user]
            up += nei[1] * rating
            down += nei[1]
        if (down == 0.0):
            return 0.0
        return up / down

# main:


with open('data/ml-100k/u.data', encoding='utf-8') as file:
    content = file.read()
    content = re.split(r'\s', content)
l = len(content)
l = int(l/4)
data = []

for i in range(l):
    a = int(content[4*i])
    b = int(content[4*i+1])
    c = int(content[4*i+2])
    data.append([a,b,c])


# data = []
# l = 0
# with open('data/ml-latest-small/ratings.csv', encoding='utf-8') as file:
#     f_csv = csv.reader(file)
#     for row in f_csv:
#         if (row[0] == 'userId'):
#             continue
#         a = int(float(row[0]))
#         b = int(float(row[1]))
#         c = int(float(row[2]))
#         data.append([a, b, c])
#         l = l + 1


random.shuffle(data)
test_rate = 0.01
train_data = data[0:(l-int(l*test_rate))]
test_data = data[(l-int(l*test_rate)):l]

cf = my_collaborative_filtering(train_data)

print(len(test_data))

# print(cf.user_dataset[196])
# print(cf.user_based_predict(train_data[0][0],train_data[0][1],800))
# print(train_data[0][2])


# for i in range(len(train_data)):
#     if ( (train_data[i][0] in cf.user_dataset) and (train_data[i][1] in cf.item_dataset) ):
#         print(cf.user_based_predict(train_data[i][0],train_data[i][1],5))
#         print(train_data[i][2])

rmse1 = rmse2 = -1
k1 = k2 = 0


print("user_based_result:")

for k in [250,300,350]:
# for k in [400, 450, 500, 550]:
    # for k in [525,575,600]:
    s1 = s2 = 0.0
    user_pred_rating = []
    l = 0
    for i in range(len(test_data)):
        if ((test_data[i][0] in cf.user_dataset) and (test_data[i][1] in cf.item_dataset)):
            user_pred_rating.append(cf.user_based_predict(
                test_data[i][0], test_data[i][1], k))
            s1 += ((user_pred_rating[l]-test_data[i][2])) * \
                (user_pred_rating[l]-test_data[i][2])
            l = l+1
    rm1 = s1 / float(len(user_pred_rating))
    if(rm1 < rmse1) or (rmse1 == -1):
        rmse1 = rm1
        k1 = k
    print(str(k) + " : "+str(rmse1))

print("* best rmse:", rmse1)
print("* best k:", k1)

print("item_based_result:")

for k in [3500,4000,4500]:
# for k in [850, 1000, 1150, 1300]:
    # for k in [1350,1400,1450]:
    s1 = s2 = 0.0
    item_pred_rating = []
    l = 0
    for i in range(len(test_data)):
        if ((test_data[i][0] in cf.user_dataset) and (test_data[i][1] in cf.item_dataset)):
            item_pred_rating.append(cf.item_based_predict(
                test_data[i][0], test_data[i][1], k))
            s2 += ((item_pred_rating[l]-test_data[i][2])) * \
                (item_pred_rating[l]-test_data[i][2])
            l = l+1
    rm2 = s2 / float(len(item_pred_rating))
    if(rm2 < rmse2) or (rmse2 == -1):
        rmse2 = rm2
        k2 = k
    print(str(k) + " : "+str(rmse2))


print("* best rmse:", rmse2)
print("* best k:", k2)
