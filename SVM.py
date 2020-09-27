#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from matplotlib import pyplot
from pandas import DataFrame
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import numpy as np
import math as mh
import pandas as pd
from sklearn.svm import SVC 

def linsep_data():
    X, y = make_moons(n_samples=500, noise=0.1)
    # scatter plot, dots colored by class value
    df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    colors = {0:'red', 1:'blue'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()
    y1 = np.where(y>0, 1, y)
    y2 = np.where(y1<=0, -1, y)
    return X,y2

def nonlin_data():
    X, y = make_circles(n_samples=500, noise=0.05)
    # scatter plot, dots colored by class value
    df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    colors = {0:'red', 1:'blue'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()
    y1 = np.where(y>0, 1, y)
    y2 = np.where(y1<=0, -1, y)
    return X,y2

def hard_margin_svm(X,y):
    m,n = X.shape
    y = y.reshape(-1,1) * 1.
    X_dash = y * X
    H = np.dot(X_dash , X_dash.T) * 1.
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(m))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas

def soft_margin_svm(X,y):
    C = 10
    m,n = X.shape
    y = y.reshape(-1,1) * 1.
    X_dash = y * X
    H = np.dot(X_dash , X_dash.T) * 1.
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas

def compute_params_hms(X,y,alphas):
    y = y.reshape(-1,1) * 1.
    w = ((y * alphas).T @ X).reshape(-1,1)
    S = (alphas > 1e-4).flatten()
    b = y[S] - np.dot(X[S], w)
    print('Alphas = ',alphas[alphas > 1e-4])
    print('w = ', w.flatten())
    print('b = ', b[0])
    y_pred = np.dot(X,w) + b
    y_pred1 = np.where(y_pred>0, 1, y_pred)
    y_pred2 = np.where(y_pred1<=0, -1, y_pred1)
    return w,b,y_pred2

def compute_params_sms(X,y,alphas):
    y = y.reshape(-1,1) * 1.
    w = ((y * alphas).T @ X).reshape(-1,1)
    S = (alphas > 1e-4).flatten()
    b = y[S] - np.dot(X[S], w)
    print('Alphas = ',alphas[alphas > 1e-4])
    print('w = ', w.flatten())
    print('b = ', b[0])
    y_pred = np.dot(X,w) + b
    y_pred1 = np.where(y_pred>0, 1, y_pred)
    y_pred2 = np.where(y_pred1<=0, -1, y_pred1)
    return w,b,y_pred2

def perf(a,b):
    size = np.shape(a)[0]
    tp = 0
    tn =0 
    fp =0 
    fn = 0
    for i in range(size):
        if(b[i]==-1):
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
    print("The precision is:",pr," The recall is:",re," The accuracy is:",acc," The F measure is:",f)

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def gauss_svm(X,y):
    C = 10
    m,n = X.shape
    K = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            K[i,j] = gaussian_kernel(X[i], X[j])
    y = y.reshape(-1,1) * 1.
    P = cvxopt_matrix(np.outer(y,y) * K)
    q = cvxopt_matrix(np.ones(m) * -1)
    A = cvxopt_matrix(y, (1,m))
    b = cvxopt_matrix(0.0)
    tmp1 = np.diag(np.ones(m) * -1)
    tmp2 = np.identity(m)
    G = cvxopt_matrix(np.vstack((tmp1, tmp2)))
    tmp1 = np.zeros(m)
    tmp2 = np.ones(m) * C
    h = cvxopt_matrix(np.hstack((tmp1, tmp2)))
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas


def main():
    print("Hard Margin SVM:")
    lin_x,lin_y = nonlin_data()
    alph = hard_margin_svm(lin_x,lin_y)
    wt,b,ypred = compute_params_hms(lin_x,lin_y,alph)
    perf(ypred,lin_y)
    print("Soft Margin SVM:")
    alph_sms = soft_margin_svm(lin_x,lin_y)
    wt1,b1,ypred1 = compute_params_sms(lin_x,lin_y,alph_sms)
    perf(ypred1,lin_y)
    print("Gaussian SVM:")
    alph_gauss = gauss_svm(lin_x,lin_y)
    wt2,b2,ypred2 = compute_params_sms(lin_x,lin_y,alph_gauss)
    perf(ypred2,lin_y)
    
if __name__ == "__main__": 
    main()


# In[ ]:


from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from matplotlib import pyplot
from pandas import DataFrame
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import numpy as np
import math as mh
import pandas as pd

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def gauss_svm(X,y):
    C = 10
    m,n = X.shape
    K = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            K[i,j] = gaussian_kernel(X[i], X[j])
    y = y.reshape(-1,1) * 1.
    P = cvxopt_matrix(np.outer(y,y) * K)
    q = cvxopt_matrix(np.ones(m) * -1)
    A = cvxopt_matrix(y, (1,m))
    b = cvxopt_matrix(0.0)
    tmp1 = np.diag(np.ones(m) * -1)
    tmp2 = np.identity(m)
    G = cvxopt_matrix(np.vstack((tmp1, tmp2)))
    tmp1 = np.zeros(m)
    tmp2 = np.ones(m) * C
    h = cvxopt_matrix(np.hstack((tmp1, tmp2)))
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas

def compute_params_sms(X,y,alphas):
    y = y.reshape(-1,1) * 1.
    w = ((y * alphas).T @ X).reshape(-1,1)
    S = (alphas > 1e-4).flatten()
    b = y[S] - np.dot(X[S], w)
    print('Alphas = ',alphas[alphas > 1e-4])
    print('w = ', w.flatten())
    print('b = ', b[0])
    y_pred = np.dot(X,w) + b
    y_pred1 = np.where(y_pred>0, 1, y_pred)
    y_pred2 = np.where(y_pred1<=0, -1, y_pred1)
    return w,b,y_pred2

def main():
    a = np.loadtxt(fname = "Skin_NonSkin.txt")
    data = a[:,:3]
    data_normed = data / data.max(axis=0)
    op = a[:,-1]
    op1 = np.where(op==1,-1,op)
    op2 = np.where(op1==2,1,op1)
    print("SkinNonSkin:")
    alph_gauss = gauss_svm(data_normed,op2)
    wt,b,ypred = compute_params_sms(data_normed,op2,alph_gauss)
    perf(ypred,op2)
    train_data = pd.read_csv("haberman.txt")
    b = np.array(train_data)
    data1 = b[:,:3]
    data_normed1 = data1 / data1.max(axis=0)
    op3 = b[:,-1]
    op4 = np.where(op3==1,-1,op3)
    op5 = np.where(op4==2,1,op4)
    print("haberman:")
    alph_gauss1 = gauss_svm(data_normed1,op5)
    wt1,b1,ypred1 = compute_params_sms(data_normed1,op5,alph_gauss1)
    perf(ypred1,op5)
    
if __name__ == "__main__": 
    main()


# In[ ]:





# In[ ]:




