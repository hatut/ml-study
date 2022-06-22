# 19335084
import numpy as np
from matplotlib import pyplot as plt

label_num = 10
L1_lamb = 0 
L2_lamb = 0

data_num = 1

test_rate = 0.3
K = 10
epochs = 100
learning_rate = 0.005
batch_size = 512

# 记录不同分类法下的ACC
linear_acc= []
nonlinear_acc= []
L1_acc= []
L2_acc= []


def softmax(matrix):
    bigval = matrix.max(axis = 1)-1
    bigval = np.reshape(bigval, (bigval.size, 1))
    temp = np.exp(matrix - bigval)
    col_sum = temp.sum(axis=1)
    col_sum = np.reshape(col_sum, (col_sum.size,1))
    return temp/col_sum

def train(W, train_X, train_Y, learning_rate):
    temp = softmax(np.dot(train_X, W)) - train_Y
    dW = (1/train_X.shape[0])*np.dot(train_X.T, temp) 
    # 添加正则项，控制L1，L2的lambda至少一个为0，实现不同正则方法
    dW += L1_lamb*np.abs(W)
    dW += 2*L2_lamb*W
    W -= learning_rate*dW
    return W


# 得测试集的ACC
def get_acc(W, test_X, labels_Y):
    temp = softmax(np.dot(test_X, W))
    Y = temp.argmax(axis=1)
    num = 0
    for i in range(Y.size):
        if (labels_Y[i, Y[i]] == 1):
            num += 1
    return num/Y.size

def train_proce(train_X,train_Y,test_X,test_Y):
    # 随机初始化 W 矩阵
    W = np.random.rand(train_X.shape[1], label_num)
    # 训练
    for i in range(epochs):
        # 打乱训练集
        state = np.random.get_state()
        np.random.shuffle(train_X)
        np.random.set_state(state)
        np.random.shuffle(train_Y)
        size = int(train_X.shape[0]/K)
        for k in range(K):
            block_X = np.r_[train_X[:k*size, :], train_X[(k+1)*size:, :]]
            block_Y = np.r_[train_Y[:k*size, :], train_Y[(k+1)*size:, :]]
            # mini-batch
            for i in range(int(block_X.shape[0]/batch_size)):
                # 这一batch中的起始和结束位置
                l = i*batch_size
                r = l+batch_size
                if (r > block_X.shape[0]):
                    r = block_X.shape[0]
                W = train(W, block_X[l:r,:], block_Y[l:r,:],  learning_rate)

    return get_acc(W, test_X, test_Y)


def main():

    # 读对应序号的数据集 npy文件
    data_image = np.load('mfeat/image'+str(data_num)+'.npy')
    data_label = np.load('mfeat/label'+str(data_num)+'.npy')
    # 打乱数据集
    [row, col] = data_image.shape
    state = np.random.get_state()
    np.random.shuffle(data_image)
    np.random.set_state(state)
    np.random.shuffle(data_label)
    # 按比例分配训练集和测试集
    r2 = int(round(row*test_rate))
    r1 = row-r2

    train_image = np.zeros((r1,col))
    train_label = np.zeros(r1,dtype=np.int64)
    test_image = np.zeros((r2,col))
    test_label = np.zeros(r2,dtype=np.int64)
    for i in range(0,r1):
        train_image[i] = data_image[i]
        train_label[i] = data_label[i]
    for i in range(r1,row):
        test_image[i-r1] = data_image[i]
        test_label[i-r1] = data_label[i]

    # 特征添加偏置1，同时标签转换为one-hot格式
    extend = np.ones(train_image.shape[0])
    train_X = np.c_[extend, train_image]
    train_Y = np.zeros((train_label.size, label_num))
    for i in range(train_label.size):
        train_Y[i, train_label[i]] = 1

    extend = np.ones(test_image.shape[0])
    test_X = np.c_[extend, test_image]
    test_Y = np.zeros((test_label.size, label_num))
    for i in range(test_label.size):
        test_Y[i, test_label[i]] = 1

    # 特征标准化
    mean_X = []
    std_X = []
    for i in range(train_X.shape[1]):
        mean_X.append(np.mean(train_X[:, i]))
        std_X.append(np.std(train_X[:,i]))
        if std_X[i] != 0:
            train_X[:, i] = (train_X[:,i]-mean_X[i])/std_X[i]
    for i in range(test_X.shape[1]):
        if std_X[i] != 0:
            test_X[:, i] = (test_X[:,i]-mean_X[i])/std_X[i]



    # 纯线性分类
    L1_lamb = 0
    L2_lamb = 0
    acc = train_proce(train_X,train_Y,test_X,test_Y)
    print('    linear: ACC= %.6f'%acc)
    linear_acc.append(acc)
    
    # 非线性分类(exp函数作为基函数)
    train_X = np.exp(train_X)
    test_X = np.exp(test_X)

    acc = train_proce(train_X,train_Y,test_X,test_Y)
    print('non-linear: ACC= %.6f'%acc)
    nonlinear_acc.append(acc)

    # L1正则化非线性
    L1_lamb = 1
    L2_lamb = 0
    
    acc = train_proce(train_X,train_Y,test_X,test_Y)
    print('L1-regular: ACC= %.6f'%acc)
    L1_acc.append(acc)

    # L2正则化非线性
    L1_lamb = 0
    L2_lamb = 1

    acc = train_proce(train_X,train_Y,test_X,test_Y)
    print('L2-regular: ACC= %.6f'%acc)
    L2_acc.append(acc)


if __name__=='__main__':

    # 读不同数据集进行测试
    for tt in range(0,6):
        data_num = tt+1 
        print('data num =%d'%data_num)

        main()

    plt.figure()
    plt.xlabel("data_num")
    plt.ylabel("acc")
    plt.xticks(range(1, data_num+1))

    plt.plot(range(1,data_num+1), linear_acc, c='red', label='linear')
    plt.plot(range(1,data_num+1), nonlinear_acc, c='blue', label='non-linear')
    plt.plot(range(1,data_num+1), L1_acc, c='black', label='L1')
    plt.plot(range(1,data_num+1), L2_acc, c='green', label='L2')

    plt.legend()
    plt.grid()
    plt.savefig('acc.jpg', dpi = 1800)
    plt.show()
