#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import math as mh
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def uni_perf(a,b):
    size = np.shape(a)[0]
    err = 0
    tp = 0
    tn =0 
    fp =0 
    fn = 0
    for i in range(size):
        err += (a[i]-b[i])**2
        if(b[i]==1):
            if(a[i]==b[i]):
                tp += 1
            else:
                fn += 1
        if(b[i]==2):
            if(a[i]==b[i]):
                tn += 1
            else:
                fp += 1
    pr = tp/(tp+fp)
    re = tp/(tp+fn)
    acc = (tp+tn)/(tp+tn+fp+fn)
    f = (2*pr*re)/(pr+re)
    err = err/size
    print("The L2 loss is: ",err)
    print("The precision is:",pr," The recall is:",re," The accuracy is:",acc," The F measure is:",f)
    
    
    
def uni_dis(x,v1,v2,m1,m2,c1,c2):
    ans = np.zeros(shape= np.shape(x))
    size = np.shape(x)[0]
    for i in range(size):
        g1 = (- mh.log(mh.sqrt(v1)))-((1/2*v1)*(x[i]-m1)**2)+ mh.log(c1)
        g2 = (- mh.log(mh.sqrt(v2)))-((1/2*v2)*(x[i]-m2)**2)+ mh.log(c2)
        if(g1>g2):
            ans[i] = 1
        if(g2>g1):
            ans[i] = 2
    return ans


def uni(x):
    data = x[:,0]
    op = x[:,-1]
    m = np.shape(x)[0]
    t1 = 0
    t2 = 0
    v1 = 0
    v2 = 0
    c1 = 0
    c2 = 0
    for i in range(m):
        if(op[i] == 1):
            t1 += data[i]
            c1 += 1
        else:
            t2 += data[i]
            c2 += 1
    mean1 = t1/c1
    mean2 = t2/c2
    a1 = c1/m
    a2 = c2/m
    for j in range(m):
        v1 += np.power(data[i]-mean1,2)
        v2 += np.power(data[i]-mean2,2)
    v1 = v1/m
    v2 = v2/m
    trgt = uni_dis(data,v1,v2,mean1,mean2,a1,a2)
    print(trgt)
    uni_perf(trgt,op)
    return 0
    
    
def main():
    uni_data = np.loadtxt(fname = "Skin_NonSkin.txt")
    a = uni(uni_data)
    

    
if __name__ == "__main__": 
    main()
    


# In[8]:




def multi_perf(a,b):
    size = np.shape(a)[0]
    err = 0
    tp = 0
    tn =0 
    fp =0 
    fn = 0
    for i in range(size):
        err += (a[i]-b[i])**2
        if(b[i]==1):
            if(a[i]==b[i]):
                tp += 1
            else:
                fn += 1
        if(b[i]==2):
            if(a[i]==b[i]):
                tn += 1
            else:
                fp += 1
    pr = tp/(tp+fp)
    re = tp/(tp+fn)
    acc = (tp+tn)/(tp+tn+fp+fn)
    f = (2*pr*re)/(pr+re)
    err = err/size
    print("The L2 loss is: ",err)
    print("The precision is:",pr," The recall is:",re," The accuracy is:",acc," The F measure is:",f)

    
def multi_dis(x,e1,e2,m1,m2,a1,a2):
    ans = np.zeros(shape= (np.shape(x)[0],1))
    size = np.shape(x)[0]
    for i in range(size):
        d1= (-0.5*mh.log(np.linalg.det(e1)))-(0.5*(np.dot(np.dot(x[i]-m1,np.linalg.inv(e1)),np.transpose(x[i]-m1))))+mh.log(a1)
        d2= (-0.5*mh.log(np.linalg.det(e2)))-(0.5*(np.dot(np.dot(x[i]-m2,np.linalg.inv(e2)),np.transpose(x[i]-m2))))+mh.log(a2)
        if(d1>d2):
            ans[i]=1
        else:
            ans[i]=2
    return ans


def multi(x):
    op = x[:,-1]
    data = np.delete(x,-1,-1)
    m = np.shape(x)[0]
    mean1 = np.zeros(shape= (1,np.shape(data)[1]))
    mean2 = np.zeros(shape= (1,np.shape(data)[1]))
    c1=0
    c2=0
    for i in range(m):
        if(op[i]==1):
            mean1 += data[i]
            c1 += 1
        else:
            mean2 += data[i]
            c2 += 1
    mean1 = mean1/c1
    mean2 = mean2/c2
    t1=np.zeros(shape= (np.shape(data)[1],np.shape(data)[1]))
    t2=np.zeros(shape= (np.shape(data)[1],np.shape(data)[1]))
    for i in range(m):
        if(op[i]==1):
            t1 += np.dot(np.transpose(data[i]-mean1),data[i]-mean1)
        else:
            t2+= np.dot(np.transpose(data[i]-mean2),data[i]-mean2)
    e1 = t1/c1
    e2 = t2/c2
    a1 = c1/m
    a2 = c2/m
    fin = multi_dis(data,e1,e2,mean1,mean2,a1,a2)
    multi_perf(fin,op)
    return 0
    


def main():
    uni_data = np.loadtxt(fname = "Skin_NonSkin.txt")
    a = multi(uni_data)
    

    
if __name__ == "__main__": 
    main()


# In[3]:


import pandas as pd
import numpy as np
import math as mh
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def multi_perf(a,b):
    size = np.shape(a)[0]
    err = 0
    tp = 0
    tn =0 
    fp =0 
    fn = 0
    for i in range(size):
        err += (a[i]-b[i])**2
        if(b[i]==1):
            if(a[i]==b[i]):
                tp += 1
            else:
                fn += 1
        if(b[i]==2):
            if(a[i]==b[i]):
                tn += 1
            else:
                fp += 1
    pr = tp/(tp+fp)
    re = tp/(tp+fn)
    acc = (tp+tn)/(tp+tn+fp+fn)
    f = (2*pr*re)/(pr+re)
    err = err/size
    print("The L2 loss is: ",err)
    print("The precision is:",pr," The recall is:",re," The accuracy is:",acc," The F measure is:",f)
    
    
def multi_dis(x,e1,e2,e3,m1,m2,m3,a1,a2,a3):
    ans = np.zeros(shape= (np.shape(x)[0],1))
    size = np.shape(x)[0]
    for i in range(size):
        d1= (-0.5*mh.log(np.linalg.det(e1)))-(0.5*(np.dot(np.dot(x[i]-m1,np.linalg.inv(e1)),np.transpose(x[i]-m1))))+mh.log(a1)
        d2= (-0.5*mh.log(np.linalg.det(e2)))-(0.5*(np.dot(np.dot(x[i]-m2,np.linalg.inv(e2)),np.transpose(x[i]-m2))))+mh.log(a2)
        d3= (-0.5*mh.log(np.linalg.det(e3)))-(0.5*(np.dot(np.dot(x[i]-m3,np.linalg.inv(e3)),np.transpose(x[i]-m3))))+mh.log(a3)
        dec = np.array([d1,d2,d3])
        c = np.where(dec == np.amax(dec))
        ans[i]=c[0]

    
    return ans


def multi(x,y):
    data = x
    op = y
    m = np.shape(x)[0]
    mean1 = np.zeros(shape= (1,np.shape(data)[1]))
    mean2 = np.zeros(shape= (1,np.shape(data)[1]))
    mean3 = np.zeros(shape= (1,np.shape(data)[1]))
    c1=0
    c2=0
    c3=0
    for i in range(m):
        if(op[i]==0):
            mean1 += data[i]
            c1 += 1
        elif(op[i]==1):
            mean2 += data[i]
            c2 += 1
        else:
            mean3 += data[i]
            c3 +=1
    mean1 = mean1/c1
    mean2 = mean2/c2
    mean3 = mean3/c3
    t1=np.zeros(shape= (np.shape(data)[1],np.shape(data)[1]))
    t2=np.zeros(shape= (np.shape(data)[1],np.shape(data)[1]))
    t3=np.zeros(shape= (np.shape(data)[1],np.shape(data)[1]))
    for i in range(m):
        if(op[i]==0):
            t1 += np.dot(np.transpose(data[i]-mean1),data[i]-mean1)
        elif(op[i]==1):
            t2+= np.dot(np.transpose(data[i]-mean2),data[i]-mean2)
        else:
            t3+= np.dot(np.transpose(data[i]-mean3),data[i]-mean3)
    e1 = t1/c1
    e2 = t2/c2
    e3 = t3/c3
    a1 = c1/m
    a2 = c2/m
    a3= c3/m
    fin = multi_dis(data,e1,e2,e3,mean1,mean2,mean3,a1,a2,a3)
    multi_perf(fin,op)
    return 0
    
    
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
    ans = multi(a,op)
    
    
    
    
if __name__ == "__main__": 
    main()


# In[9]:


import pandas as pd
import numpy as np
import math as mh
import matplotlib.pyplot as plt

def bern_perf(a,b):
    size = np.shape(a)[0]
    err = 0
    tp = 0
    tn =0 
    fp =0 
    fn = 0
    for i in range(size):
        err += (a[i]-b[i])**2
        if(b[i]==0):
            if(a[i]==b[i]):
                tp += 1
            else:
                fn += 1
        if(b[i]==1):
            if(a[i]==b[i]):
                tn += 1
            else:
                fp += 1
    pr = tp/(tp+fp)
    re = tp/(tp+fn)
    acc = (tp+tn)/(tp+tn+fp+fn)
    f = (2*pr*re)/(pr+re)
    err = err/size
    print("The L2 loss is: ",err)
    print("The precision is:",pr," The recall is:",re," The accuracy is:",acc," The F measure is:",f)
    return 0
    
    
def bern_dis(x,m1,m2,a1,a2):
    ans = np.zeros(shape= (np.shape(x)[0],1))
    size = np.shape(x)[0]
    feat = np.shape(x)[1]
    for i in range(size):
        g1 = 0
        g2 = 0
        for j in range(feat):
            g1 += (x[i][j]*mh.log(m1[0][j]))+((1-x[i][j])*mh.log(1-m1[0][j]))
            g2 += (x[i][j]*mh.log(m2[0][j]))+((1-x[i][j])*mh.log(1-m2[0][j]))
        g1 = g1+mh.log(a1)
        g2 = g2+mh.log(a2)
        if(g1>g2):
            ans[i]=1
        elif(g2>g1):
            ans[i]=0
    
    return ans

def bern(x):
    op = x[:,-1]
    data = np.delete(x,-1,-1)
    m = np.shape(x)[0]
    mean1 = np.zeros(shape= (1,np.shape(data)[1]))
    mean2 = np.zeros(shape= (1,np.shape(data)[1]))
    c1=0
    c2=0
    for i in range(m):
        if(op[i]==1):
            mean1 += data[i]
            c1 += 1
        else:
            mean2 += data[i]
            c2 += 1
    mean1 = mean1/c1
    mean2 = mean2/c2
    a1 = c1/m
    a2 = c2/m
    fin = bern_dis(data,mean1,mean2,a1,a2)
    res = bern_perf(fin,op)
    return 0
    
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def main():
    data = pd.read_csv("spambase.data",sep = ",",skiprows=0, header=None, nrows=4600)
    a = np.array(data)
    b = np.delete(a,[48,49,50,51,52,53,54,55,56],1)
    c = np.where(b>0,1,b)
    ans = bern(c)
    
if __name__ == "__main__": 
    main()


# In[23]:


import pandas as pd
import numpy as np
import math as mh
import matplotlib.pyplot as plt

def bino_perf(a,b):
    size = np.shape(a)[0]
    err = 0
    tp = 0
    tn =0 
    fp =0 
    fn = 0
    for i in range(size):
        err += (a[i]-b[i])**2
        if(b[i]==0):
            if(a[i]==b[i]):
                tp += 1
            else:
                fn += 1
        if(b[i]==1):
            if(a[i]==b[i]):
                tn += 1
            else:
                fp += 1
    pr = tp/(tp+fp)
    re = tp/(tp+fn)
    acc = (tp+tn)/(tp+tn+fp+fn)
    f = (2*pr*re)/(pr+re)
    err = err/size
    print("The L2 loss is: ",err)
    print("The precision is:",pr," The recall is:",re," The accuracy is:",acc," The F measure is:",f)
    return 0

def comb(n,r):
    f = mh.factorial
    a=(f(n)// f(r)*f(n-r))
    return a

def bino_dis(x,m1,m2,a1,a2,d):
    ans = np.zeros(shape= (np.shape(x)[0],1))
    size = np.shape(x)[0]
    feat = np.shape(x)[1]
    for i in range(size):
        g1 = 0
        g2 = 0
        for j in range(feat):
            g1 += mh.log(comb(d[i],x[i][j]))+(x[i][j]*mh.log(m1[0][j]))+(d[i]-x[i][j])*mh.log(1-m1[0][j])
            g2 += mh.log(comb(d[i],x[i][j]))+(x[i][j]*mh.log(m2[0][j]))+(d[i]-x[i][j])*mh.log(1-m2[0][j])
        g1 = g1+mh.log(a1)
        g2 = g2+mh.log(a2)
        if(g1>g2):
            ans[i]=1
        elif(g2>g1):
            ans[i]=0
    
    return ans

def bino(x):
    op = x[:,-1]
    de = np.delete(x,-1,-1)
    dt = de*10
    data = dt.astype(int)
    m = np.shape(x)[0]
    d = np.zeros(shape= (np.shape(data)[0],1))
    for l in range(m):
        d[l]=np.sum(data[l,:])
    mean1 = np.zeros(shape= (1,np.shape(data)[1]))
    mean2 = np.zeros(shape= (1,np.shape(data)[1]))
    c1=0
    c2=0
    tc2=0
    tc1=0
    for i in range(m):
        if(op[i]==1):
            mean1 += data[i]
            tc1 += d[i]
            c1 += 1
        else:
            mean2 += data[i]
            tc2 += d[i]
            c2 += 1
    
    mean1 = mean1/tc1
    mean2 = mean2/tc2
    a1 = c1/m
    a2 = c2/m
    fin = bino_dis(data,mean1,mean2,a1,a2,d)
    res = bino_perf(fin,op)
    return 0


def main():
    data = pd.read_csv("spambase.data",sep = ",",skiprows=0, header=None, nrows=4600)
    a = np.array(data)
    b = np.delete(a,[48,49,50,51,52,53,54,55,56],1)
    ans = bino(b)
    
if __name__ == "__main__": 
    main()


# In[ ]:




