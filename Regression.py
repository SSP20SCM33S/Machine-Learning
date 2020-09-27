#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import math as mh
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plot_data(x, y):
    plt.scatter(x, y)
    plt.show()
       
    
def find_par(x, y):
    p0 = 0
    p1 = 0
    avg_tr_err = 0
    avg_te_err = 0
    avg_r_sq = 0
    k=10
    for i in range(0,180,20):
        temp1 = 0
        temp2 = 0
        temp3 = 0
        tr_err = 0
        te_err = 0
        mean_err = 0
        index = range(i,i+20)
        trdt = np.delete(x, index)
        trval = np.delete(y, index)
        tedt = x[i:i+20]
        teval = y[i:i+20]
        size = np.size(trdt)
        te_size = np.size(tedt)
        mean_data = np.mean(trdt)
        mean_output = np.mean(trval)
        deviation_xy = np.sum(trval*trdt) - size * mean_output * mean_data 
        deviation_xx = np.sum(trdt*trdt) - size * mean_data * mean_data
        p_1 = deviation_xy / deviation_xx 
        p_0 = mean_output - p_1*mean_data
        for m in range(size):
            tr_err = ((p_0+p_1*trdt[m])-trval[m])**2
            mean_err = ((p_0+p_1*trdt[m])-mean_output)**2
            temp1 = temp1 + tr_err
            temp3 = temp3 + mean_err
        for n in range(20):
            te_err = ((p_0+p_1*tedt[n])-teval[n])**2
            temp2 = temp2 + te_err
        avg_tr_err = avg_tr_err + temp1/size
        avg_te_err = avg_te_err + temp2/te_size
        avg_r_sq = avg_r_sq + (1-(temp1/temp3))
        p0 = p0 + p_0
        p1 = p1 + p_1
        
    p0 = p0/k
    p1= p1/k
    avg_tr_err = avg_tr_err/k
    avg_te_err = avg_te_err/k
    avg_r_sq = avg_r_sq/k
    return(p0, p1,avg_tr_err,avg_te_err,avg_r_sq) 


def compare(x,y,z):
    model = LinearRegression()
    model.fit(x, y)
    print('p0 of inbuilt model:', model.intercept_)
    print('p1 of inbuilt model:', model.coef_)
    r_sq = model.score(x, y)
    print("The R-Square of the inbuilt model is:", r_sq)
    print("The parameters of my model are ",z[:2])
    print("The average training error of my model is",z[2])
    print("The average testing error of my model is",z[3])
    print("The R-Square of my model is",z[4])
   
  
def main():  
    col_list1 = [1]
    col_list2 = [2]
    data1 = pd.read_csv("svar-set1.dat",sep = " ",skiprows=6, header=None, nrows=200, usecols=col_list1)
    x1 = np.array(data1)
    op1 = pd.read_csv("svar-set1.dat",sep = " ",skiprows=6, header=None, nrows=200, usecols=col_list2)
    y1 = np.array(op1)
    plot_data(x1,y1) 
    ans1 = find_par(x1, y1) 
    compare(x1,y1,ans1)
    
    data2 = pd.read_csv("svar-set2.dat",sep = " ",skiprows=6, header=None, nrows=200, usecols=col_list1)
    x2 = np.array(data2)
    op2 = pd.read_csv("svar-set2.dat",sep = " ",skiprows=6, header=None, nrows=200, usecols=col_list2)
    y2 = np.array(op2)
    plot_data(x2,y2) 
    ans2 = find_par(x2, y2) 
    compare(x2,y2,ans2)
    
    data3 = pd.read_csv("svar-set3.dat",sep = " ",skiprows=6, header=None, nrows=200, usecols=col_list1)
    x3 = np.array(data3)
    op3 = pd.read_csv("svar-set3.dat",sep = " ",skiprows=6, header=None, nrows=200, usecols=col_list2)
    y3 = np.array(op3)
    plot_data(x3,y3) 
    ans3 = find_par(x3, y3) 
    compare(x3,y3,ans3)
    
    data4 = pd.read_csv("svar-set4.dat",sep = " ",skiprows=6, header=None, nrows=200, usecols=col_list1)
    x4 = np.array(data4)
    op4 = pd.read_csv("svar-set4.dat",sep = " ",skiprows=6, header=None, nrows=200, usecols=col_list2)
    y4 = np.array(op4)
    plot_data(x4,y4) 
    ans4 = find_par(x4, y4) 
    compare(x4,y4,ans4)
    
 
  
if __name__ == "__main__": 
    main() 



# In[ ]:




