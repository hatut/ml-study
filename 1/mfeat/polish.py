import numpy as np
import random

file = open("mfeat-mor",mode="r")

para_num = 6

data1_image = np.zeros((2000,para_num),dtype=np.float64)
data1_label = np.zeros(2000,dtype=np.float64)
for i in range (0, 2000):
    line = file.readline()
    line = line.strip().split()
    for j in range(0,para_num):
        line[j] = float(line[j])
    data1_image[i,:] = line[:]

    data1_label[i] = i/200
file.close()
print(data1_image)
for i in range (1,2000):
    j = 2000-i
    r = random.randint(0,j)
    data1_image[[j,r],:] = data1_image[[r,j],:]
    data1_label[[j,r]] = data1_label[[r,j]]
print(data1_image)

np.save('image6.npy',data1_image)
np.save('label6.npy',data1_label)
