#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import math as mh
import matplotlib.pyplot as plt

def sigmoid(theta,x):
    return (1 / (1 + np.exp(-(np.dot(x,theta)))))

def binary_cross_entropy(actual, predicted):
    sum_score = 0.0
    for i in range(len(actual)):
        sum_score += actual[i] * mh.log(1e-15 + predicted[i])
    mean_sum_score = 1.0 / len(actual) * sum_score
    return -mean_sum_score
    

def cal_loss(a,b,x):
    return np.dot(np.transpose(a-b),x)
    
def logis(x,lr,it):
    data = np.delete(x,-1,-1)
    op1 = x[:,-1]
    op = op1.reshape(np.shape(x)[0],1)
    wt = np.random.uniform(low=0.01, high=0.1, size=(np.shape(x)[1],1))
    size = np.shape(x)[0]
    dt = np.insert(data, 0, 1, axis=1)
    for j in range(it):
        t1 = 0
        temp = np.zeros(shape=(np.shape(wt)[1],1))
        ans = sigmoid(wt,dt)
        dloss = cal_loss(ans,op,dt)
        wt = wt - (lr * np.transpose(dloss))
        ce_pq = binary_cross_entropy(op,ans)
        print("The cross entropy loss for pass ",j,"is :",ce_pq)
    b = np.where(ans>0.5, 1, ans)
    c = np.where(b<0.5, 0, b)
    
        

def main():
    col_list1 = [0,1,2,3,4,5]
    logis_data = pd.read_csv("caesarian.csv",sep = ",",skiprows=0, header=None, nrows=80, usecols=col_list1)
    d = np.array(logis_data)
    a = logis(d,0.00007,4000)

if __name__ == "__main__": 
    main()


# In[7]:


import pandas as pd
import numpy as np
import math as mh
import matplotlib.pyplot as plt

def sigmoid(theta,x):
    return (1 / (1 + np.exp(-(np.dot(x,theta)))))

def cal_loss(a,b,x):
    return np.dot(np.transpose(a-b),x)

def binary_cross_entropy(actual, predicted):
    sum_score = 0.0
    for i in range(len(actual)):
        sum_score += actual[i] * mh.log(1e-15 + predicted[i])
    mean_sum_score = 1.0 / len(actual) * sum_score
    return -mean_sum_score
    
def logis(x,lr,it):
    trdt = np.delete(x,-1,-1)
    op1 = x[:,-1]
    op = op1.reshape(np.shape(x)[0],1)
    wt = np.random.uniform(low=0.01, high=0.1, size=(np.shape(x)[1],1))
    size = np.shape(x)[0]
    x1 = trdt[:,[0]]
    x2 = np.power(trdt[:,[1]],2)
    x3 = np.power(trdt[:,[2]],3)
    x4 = np.power(trdt[:,[3]],4)
    x5 = np.power(trdt[:,[4]],5)
    t1 = np.insert(x1, 0, 1, axis=1)
    t2 = np.concatenate((t1, x2), axis=1)
    t3 = np.concatenate((t2, x3), axis=1)
    t4 = np.concatenate((t3, x4), axis=1)
    dt = np.concatenate((t4, x5), axis=1)
    for j in range(it):
        t1 = 0
        temp = np.zeros(shape=(np.shape(wt)[1],1))
        ans = sigmoid(wt,dt)
        derivative_loss = cal_loss(ans,op,dt)
        wt = wt - (lr * np.transpose(derivative_loss))
        ce_pq = binary_cross_entropy(op,ans)
        print("The cross entropy loss for pass ",j,"is :",ce_pq)
    b = np.where(ans>0.5, 1, ans)
    c = np.where(b<0.5, 0, b)
    
def main():
    col_list1 = [0,1,2,3,4,5]
    logis_data = pd.read_csv("caesarian.csv",sep = ",",skiprows=0, header=None, nrows=80, usecols=col_list1)
    d = np.array(logis_data)
    a = logis(d,0.00006,3000)

if __name__ == "__main__": 
    main()


# In[19]:


import pandas as pd
import numpy as np
import math as mh
import matplotlib.pyplot as plt

def sigmoid(theta,x):
    return (1 / (1 + np.exp(-(np.dot(x,theta)))))

def binary_cross_entropy(actual, predicted):
    sum_score = 0.0
    for i in range(len(actual)):
        sum_score += actual[i] * mh.log(1e-15 + predicted[i])
    mean_sum_score = 1.0 / len(actual) * sum_score
    return -mean_sum_score

def multi(x,y,lr,it):
    op = y.reshape(np.shape(x)[0],1)
    size = np.shape(x)[0]
    feat = np.shape(x)[1]+1
    data = np.insert(x, 0, 1, axis=1)
    w1 = np.random.uniform(low=0.01, high=0.1, size=(feat,1))
    w2 = np.random.uniform(low=0.01, high=0.1, size=(feat,1))
    w3 = np.random.uniform(low=0.01, high=0.1, size=(feat,1))
    for i in range(it):
        ypred1 = sigmoid (w1,data)
        ypred2 = sigmoid (w2,data)
        ypred3 = sigmoid (w3,data)
        grad_w1 = np.dot(np.transpose(data),(ypred1-op))
        grad_w2 = np.dot(np.transpose(data),(ypred2-op))
        grad_w3 = np.dot(np.transpose(data),(ypred3-op))
        w1 = w1 - (lr*grad_w1)
        w2 = w2 - (lr*grad_w2)
        w3 = w3 - (lr*grad_w3)
        t1 = np.concatenate((ypred1, ypred2), axis=1)
        t2 = np.concatenate((t1, ypred3), axis=1)
        ans = np.argmax(t2, axis=1)
        ce_pq = binary_cross_entropy(op,ans)
        print("The binary cross entropy loss for pass",i,"is: ",ce_pq)
        w1 = np.array(w1,dtype=np.float)
        w2 = np.array(w2,dtype=np.float)
        w3 = np.array(w3,dtype=np.float)
    
    
    

def main():
    col_list1 = [0,1,2,3]
    col_list2 = [4]
    dt = pd.read_csv("iris.data",sep = ",",skiprows=0, header=None, nrows=150, usecols=col_list1)
    a = np.array(dt)
    y = pd.read_csv("iris.data",sep = ",",skiprows=0, header=None, nrows=150, usecols=col_list2)
    temp = np.array(y)
    b = np.where(temp=='Iris-setosa', 0, temp)
    c = np.where(b=='Iris-versicolor', 1, b)
    op = np.where(c=='Iris-virginica', 2, c)
    ans = multi(a,op,0.00001,1000)
    
    
if __name__ == "__main__": 
    main()


# In[14]:


import pandas as pd
import numpy as np
import math as mh
import matplotlib.pyplot as plt

def binary_cross_entropy(actual, predicted):
    sum_score = 0.0
    for i in range(len(actual)):
        sum_score += actual[i] * mh.log(1e-15 + predicted[i])
    mean_sum_score = 1.0 / len(actual) * sum_score
    return -mean_sum_score

def multi(x,y,lr):
    size = np.shape(x)[0]
    feat = np.shape(x)[1]+1
    y1 = np.concatenate((y, y), axis=1)
    op = np.concatenate((y1, y), axis=1)
    k = 3
    units = 64
    data = np.insert(x, 0, 1, axis=1)
    w1 = np.random.uniform(low=0.01, high=0.1, size=(feat,units))
    w2 = np.random.uniform(low=0.01, high=0.1, size=(units,k))
    for t in range(1000):
        h = 1/(1+np.exp(-data.dot(w1)))
        ypred = 1/(1+np.exp(-h.dot(w2)))
        grad_ypred = 2.0*(ypred-y)
        grad_w2 = h.T.dot(grad_ypred * ypred * (1-ypred))
        grad_h = grad_ypred.dot(w2.T)
        grad_w1 = data.T.dot(grad_h * h * (1-h))
        w1 = w1 -(lr*grad_w1)
        w1 = np.array(w1,dtype=np.float)
        w2 = w2 -(lr*grad_w2)
        w2 = np.array(w2,dtype=np.float)
        ans = np.argmax(ypred, axis=1)
        ce_pq = binary_cross_entropy(y,ans)
        print("The binary cross entropy loss for pass",t,"is: ",ce_pq)
    


def main():
    col_list1 = [0,1,2,3]
    col_list2 = [4]
    dt = pd.read_csv("iris.data",sep = ",",skiprows=0, header=None, nrows=150, usecols=col_list1)
    a = np.array(dt)
    y = pd.read_csv("iris.data",sep = ",",skiprows=0, header=None, nrows=150, usecols=col_list2)
    temp = np.array(y)
    b = np.where(temp=='Iris-setosa', 0, temp)
    c = np.where(b=='Iris-versicolor', 1, b)
    op = np.where(c=='Iris-virginica', 2, c)
    ans = multi(a,op,0.00005)
    
    
if __name__ == "__main__": 
    main()


# In[ ]:




