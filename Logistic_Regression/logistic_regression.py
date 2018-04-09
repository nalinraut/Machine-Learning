#Nalin Yatin Raut
import os
import csv
import scipy.stats
from collections import Counter
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math
import random


def warn(*args, **kwargs):
    '''This function supresses the warning in output console'''
    pass
def dataPreprocess(data):
    '''This function processes the data '''
    m,n = np.shape(data)
    mode = []
    new_data = np.zeros((m,n))
    for j in range(n):
        for i in range(m):
            if data[i,j] == 'normal':
                data[i,j] = 1
            if data[i,j] == 'abnormal':
                data[i,j] = 0

            if data[i,j] == 'present':
                data[i,j] = 1
            if data[i,j] == 'notpresent':
                data[i,j] = 0

            if data[i,j] == 'yes':
                data[i,j] = 1
            if data[i,j] == 'no':
                data[i,j] = 0

            if data[i,j] == 'good':
                data[i,j] = 1
            if data[i,j] == 'poor':
                data[i,j] = 0

            if data[i,j]=='ckd':
                data[i,j] = 1
            if data[i,j] == 'notckd':
                data[i,j] = 0

        feature = data[:,j]

        most_common,num_most_common = Counter(feature).most_common(1)[0]
        if most_common == '?':
            most_common,num_most_common = Counter(feature).most_common(2)[1]

        for k in range(m):
            if data[k,j] == '?':
                data[k,j]=most_common
            new_data[k,j] = float(data[k,j])

        if j<n-1:
            feature = np.array(new_data[:,j])
            mean = np.mean(feature)
            rnge = np.amax(feature)-np.amin(feature)
            new_data[:,j] = (feature-mean)/rnge
    return new_data
def plotiing(ar1, ar2, ar3):

    plt.figure()
    plt.plot(ar1,ar3,'r-',label='Training Data')
    plt.scatter(ar1,ar3, color='r', s=5,linewidth=5)
    plt.plot(ar1,ar2,'g-',label='Test Data')
    plt.scatter(ar1,ar2, color='g', s=5,linewidth=5)
    plt.xlabel('Regularization Parameter- Lambda')
    plt.ylabel('F-measure')
    plt.title('F-measure  vs Regularization Parameter-Lambda')
    plt.legend()
    plt.show()

    

def sigmoid(z):
    sig=1.0 / ( 1.0 + np.e**(-z) ) 
    return sig

def cost_fn(x,y, theta,lamda,m):
    
    temp=list(theta)
    temp[0]=0
    temp=np.array(temp)
    h = sigmoid(np.dot(x, theta)) #get the hypothesis
    try:
        m = len(h)
        J=-1.0/m*(np.sum(y*(np.log(h)) + (1.0-y)*(np.log(1.0-h)))+(float(lamda)/(2*m)*np.sum(temp**2)))
    except RuntimeWarning:
        m = len(h)
        J=-1.0/m*(np.sum(y*(np.log(h)) + (1.0-y)*(np.log(1.0-h)))+(float(lamda)/(2*m)*np.sum(temp**2)))

    return J

def gradients( x,y, theta,lamda,m):

    zrd=list(theta)
    zrd[0]=0
    zrd=np.array(zrd)
    h = sigmoid(np.dot(x, theta)) #get the hypothesis
    G= ((1.0/m ) * (np.dot(x.T,( h - y ))) - ((float(lamda)/m) * zrd))
    return G

def descent(x,y, it,theta, lamda, m,alpha):
    s=(x[0]).size
    
    cost=[]
    for i in range(it):
        cost.append(cost_fn(x,y, theta,lamda,m))
        if len(cost)>1:
           if abs(cost[i])>abs(cost[i-1]) or abs(cost[i])==abs(cost[i-1]) :break
        gradientz =gradients(x,y, theta,lamda, m)
        #Update thetas based on "gradientz"
        for k in range (s):
            theta = theta - alpha * gradientz
    print(i)
    return(cost, theta)

def prediction(x_test, y_test, vartheta):

        pred = np.dot(x_test, vartheta)
        pred[ pred >= 0] = 1
        pred[ pred < 0 ] = 0
        
        tp=0
        fp=0
        fn=0
        

        for i in range(pred.size):
            if pred[i]==1 and y_test[i]==1:
                tp+=1

            elif pred[i]==1 and y_test[i]==0:
                fp+=1
            elif pred[i]==0 and y_test[i]==1:
                fn+=1
        

        pr=float(tp)/float(tp+fp)

        re=float(tp)/float(tp+fn)
        fmeasure= 2*pr*re/(pr+re)

        return(fmeasure)

def main():
   
    file = []
    with open("realdata1.csv","rb") as f:
        data = csv.reader(f)
        for row in data:
                file.append(row)
    file = np.array(file)
    file = file[1:,:]
    file = dataPreprocess(file)
    np.random.shuffle(file)
    
    y = file[:,24]
    x = file[:,:24]
    #normalize the data.
    for i in range(400):
        x[i,:] = (x[i,:]-np.mean(x[i,:],axis=0))/(np.amax(x[i,:],axis=0).T-np.amin(x[i,:],axis=0).T)
    
    

    x_train = x[0:320,:]
    y_train = y[0:320]
    x_test = file[320:401,:]
    y_test = y[320:401]

    fmeasures_test=[]
    fmeasures_train=[]
    trainedData=int(x.shape[0]*80/100)

    #Training data
    x_train = x[:trainedData,:]
    y_train = y[:trainedData]

    #Testing data
    x_test = x[trainedData:x.shape[0]+1]
    y_test = y[trainedData:y.shape[0]+1]

    theta=np.array([0]*24)
    m=320
    alpha=0.1
    it=50000
    fmeasures1=[]
    fmeasures2=[]
    lamda=list(np.linspace(-2,5,36))
    
    for i in lamda:

        a,b=descent(x_train,y_train,it,theta,i, m,alpha)
        fm1=prediction(x_test,y_test, b)
        fmeasures1.append(fm1)
        fm2=prediction(x_train,y_train,b)
        fmeasures2.append(fm2)
    plotiing(lamda, fmeasures1,fmeasures2)
main()