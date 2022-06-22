from operator import itemgetter
import re
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np


class fast_unfolding:
    def __init__(self, nodes, edges):
        # 节点，边，节点的初始数量
        [self.nodes, self.edges, self.node_num] = [nodes, edges, len(nodes)]
        self.node_com = [n for n in nodes]    # 节点所在的社区,初始节点的社区是自己编号
        self.node_in = [0 for n in nodes]     # 社区节点的内部边的权重和,初始社区内没有边，为0
        self.node_tot = [0 for n in nodes]    # 节点的连边的权重和
        self.com_dete = []  # 初始节点的社区划分
        self.m = 0  # m（网络的边权值和）
        n_edges = [[] for n in nodes]
        for e in edges:
            self.m += e[2]
            self.node_tot[e[0]] += e[2]
            self.node_tot[e[1]] += e[2]
            n_edges[e[0]].append((e[1], e[2]))
            n_edges[e[1]].append((e[0], e[2]))
        self.node_edges = n_edges   # 节点对应的边

    # 第一步 根据模块度增益决定节点要加入到哪个社区中
    def step1_fun(self):
        # 算出每个社区节点的 Σin和 Σtot
        self.sig_in = [self.node_in[node]*2 for node in self.nodes]
        self.sig_tot = [self.node_tot[node] for node in self.nodes]
        # 现在的社区划分
        now_com = [[node] for node in self.nodes]
        stop_flag = True 
        while True:
            sp1_stop_flag = True
            # 按连边的权重和大小来遍历节点
            temp = [[node, self.node_tot[node]] for node in self.nodes]
            temp = sorted(temp, key=itemgetter(1), reverse=False)
            seq = [i[0] for i in temp]

            for node in seq:
                node_c = self.node_com[node]    # 节点所在的社区
                max_dq = 0
                remove_in = 0
                join_c = node_c     # 节点将要加入的社区
                fin_enter_in = 0   # 最后加入的时候的度
                # 找和这个节点相邻的节点
                neigh_node_set = set()  # 相邻节点的集合
                try:
                    nei_edge = self.node_edges[node]
                except KeyError:
                    continue
                for e in nei_edge:
                    neigh_node_set.add(e[0])
                    if self.node_com[e[0]] == node_c:   # 如果在同一社区
                        remove_in += e[1]
                
                # 把这个节点移出社区的模块度增益
                dq1 = -2 * (remove_in + self.node_in[node]) +self.sig_tot[node_c] * \
                    self.node_tot[node]/self.m - self.node_tot[node]**2/(2*self.m)

                visited_com = set() # 已经尝试过把节点加入的社区集合
                for neigh in neigh_node_set:
                    neigh_c = self.node_com[neigh]  
                    if (neigh_c in visited_com) or (neigh_c == node_c):
                        continue
                    visited_com.add(neigh_c)
                    enter_in = 0
                    for e in self.node_edges[node]:
                        if self.node_com[e[0]] == neigh_c: # 如果在同一社区
                            enter_in += e[1]
                    # 把这个节点 加入 这个社区的模块度增益
                    dq2 = 2 * enter_in + 2 * self.node_in[node]-self.sig_tot[neigh_c] * \
                        self.node_tot[node]/self.m - self.node_tot[node]**2/(2*self.m)

                    dq = dq1 + dq2 # 总的模块度增益
                    if dq > max_dq:
                        max_dq = dq
                        fin_enter_in = enter_in
                        join_c = neigh_c
                # ΔQ没有收敛的话
                if max_dq > 1e-8:
                    # 把节点加入到新社区，更新相关值
                    stop_flag = sp1_stop_flag = False
                    now_com[node_c].remove(node)
                    now_com[join_c].append(node)
                    self.node_com[node] = join_c
                    self.sig_in[node_c] -= 2 * (remove_in + self.node_in[node])
                    self.sig_tot[node_c] -= self.node_tot[node]
                    self.sig_in[join_c] += 2 * (fin_enter_in + self.node_in[node])
                    self.sig_tot[join_c] += self.node_tot[node]
            if sp1_stop_flag:
                return stop_flag, now_com

    # 第二步 合并同一个社区中的节点变为一个节点
    def step2_fun(self, coms):
        nodes_n = [i for i in range(len(coms))]         # 社区变为节点的序号
        nodes_c = [0 for i in range(self.node_num)]     # 原始节点对应的社区
        self.node_com = [n for n in nodes_n]
        l = len(coms)
        for i in range(l):
            for item in coms[i]:
                nodes_c[item] = i
        new_node_in = [0 for n in nodes_n]
        for n in self.nodes:
            new_node_in[ nodes_c[n] ] += self.node_in[n]
        # 更新边
        edge_set = {}
        for e in self.edges:
            c1 = nodes_c[e[0]]
            c2 = nodes_c[e[1]]
            if c1 == c2:   # 这条边的两个点在同一社区
                new_node_in[c1] += e[2]
            else:   # 不在同一社区，添边
                tup = ( (c1, c2) if(c1<c2) else (c2,c1) )
                try:
                    edge_set[tup] += e[2]
                except KeyError:
                    edge_set[tup] = e[2]
        # 从集合中取边为默认格式
        new_edges = [[x[0], x[1], y] for x, y in edge_set.items()]
        new_node_edges = [ [] for i in range(self.node_num)]
        self.node_tot = [2*new_node_in[node] for node in nodes_n]
        for e in new_edges:
            self.node_tot[e[0]] += e[2]
            self.node_tot[e[1]] += e[2]
            new_node_edges[e[0]].append((e[1], e[2]))
            new_node_edges[e[1]].append((e[0], e[2]))
        # 更新类的各个参数
        self.nodes = nodes_n
        self.edges = new_edges
        self.node_in = new_node_in
        self.node_edges = new_node_edges

    # 迭代进行第一步和第二步，进行社区发现，输出
    def run(self):
        while True:
            stop_flag, coms = self.step1_fun()
            if stop_flag:
                break
            coms = [c for c in coms if c]
            if self.com_dete:
                com_dete = []
                for c in coms:
                    temp = []
                    for node in c:
                        temp.extend(self.com_dete[node])
                    com_dete.append(temp)
                self.com_dete = com_dete
            else:
                self.com_dete = coms
            self.step2_fun(coms)

        # 计算并输出模块度
        q = 0
        for i in range(len(self.sig_tot)):
            q += (self.sig_in[i]-self.sig_tot[i]**2/(2*self.m))/(2*self.m)
        print('Q: ', q)
        # 输出预测出的各个节点的社区 labels
        prelabel = np.zeros(self.node_num, dtype=int)
        l = len(self.com_dete)
        for i in range(l):
            for item in self.com_dete[i]:
                prelabel[item] = i+1
        # print(prelabel)
        return prelabel


# 主程序 输入输出，文件处理

nodes = set()
edges = []


with open('data/com-dblp.ungraph.txt', encoding='utf-8') as file:
    content = file.read()
    content = re.split(r'\s', content)
l = len(content)
l = int(l/2)

ditny = {}
no = 0
for i in range(l):
    a = int(content[2*i])
    b = int(content[2*i+1])

    if a not in ditny:
        ditny[a] = no
        no += 1
    if b not in ditny:
        ditny[b] = no
        no += 1

    nodes.add(ditny[a])
    nodes.add(ditny[b])
    edges.append([ditny[a], ditny[b], 1])
    edges.append([ditny[b], ditny[a], 1])

fu = fast_unfolding(nodes, edges)

plabel = fu.run()


truelabel = np.zeros(len(nodes), dtype=int)
# with open('data/email-Eu-core-department-labels.txt', encoding='utf-8') as file:
#     content = file.read()
#     content = re.split(r'\s', content)
# l = len(content)
# l = int(l/2)

# for i in range(l):
#     a = int(content[2*i])
#     b = int(content[2*i+1])
#     truelabel[ditny[a]] = b+1

com_num = 1
with open('data/com-dblp.all.cmty.txt', encoding='utf-8') as file:
    while True:
        content = file.readline()
        content = re.split(r'\s', content)
        if (content[0]==''):
            break
        l = len(content)

        for i in range(l):
            try:
                a = int(content[i])
            except ValueError:
                break
            truelabel[ditny[a]] = com_num
        com_num += 1
# print(truelabel)

print('NMI: ', normalized_mutual_info_score(truelabel, plabel))
