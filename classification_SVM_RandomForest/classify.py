'''Nalin Yatin Raut
AI Assignment 04, Problem 4'''
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def warn(*args, **kwargs):
    '''This function supresses the warning in output console'''
    pass


def printingPlotting(a,b):
    print('AI Assignment 04 - Problem 4:\n')
    print('F measure values for Train Data (80%): ')
    print('F measure value with SVM-Linear_kernel: ',a[2][0] )
    print('F measure value with SVM-RBF_kernel: ', a[2][1])
    print('F measure value with Random_Forest_Classifier: ', a[2][2])

    print('\nF measure values for Test Data (20%): ')
    print('F measure value with SVM-Linear_kernel: ',b[2][0] )
    print('F measure value with SVM-RBF_kernel: ', b[2][1])
    print('F measure value with Random_Forest_Classifier: ', b[2][2])

    met=['SVM-Linear','SVM-RBF','Random Forest']
    y=[0.5,1,1.5]

    plt.figure(1)
    plt.xticks(y, met)
    plt.plot(y,a[0][:],'r-',label='Trained Data')
    plt.plot(y,a[0][:],'ro')
    plt.plot(y,b[0][:],'g-', label='Test Data')
    plt.plot(y,b[0][:],'g^')
    plt.xlabel('Method')
    plt.ylabel('F measure')
    plt.title(' F measure values with 20% trained data')
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.xticks(y, met)
    plt.plot(y,a[1][:],'r-', label='Trained Data')
    plt.plot(y,a[1][:],'ro')
    plt.plot(y,b[1][:],'g-', label='Test Data')
    plt.plot(y,b[1][:],'g^')
    plt.xlabel('Method')
    plt.ylabel('F measure')
    plt.title(' F measure values with 50% trained data')
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.xticks(y, met)
    plt.plot(y,a[2][:],'r-', label='Trained Data')
    plt.plot(y,a[2][:],'ro')
    plt.plot(y,b[2][:],'g-', label='Test Data')
    plt.plot(y,b[2][:],'g^')
    plt.xlabel('Method')
    plt.ylabel('F measure')
    plt.title(' F measure values with 80% trained data')
    plt.legend()
    plt.show()



def fMeasure(pred,pred1,y_test,y_train):
    '''This function computes the f-measure values given the predicted values and trained and tested class values'''
    numMatches = 0
    tp = 0
    fp = 0
    fn = 0
    for i in range(0,80):
        if pred[i] == y_test[i]:
            numMatches +=1
        if pred[i] == 1 and y_test[i] ==1:
          tp += 1
        if pred[i] == 1 and y_test[i] ==0:
          fp += 1
        if pred[i] == 0 and y_test[i] ==1:
            fn +=1
    pre = float(tp)/float(tp+fp)
    rec = float(tp)/float(tp+fn)
    fm = 2 * pre * rec/ (pre+rec)
    numMatches1 = 0
    tp1 = 0
    fp1 = 0
    fn1 = 0
    for i in range(0,80):
        if pred1[i] == y_train[i]:
            numMatches1 +=1
        if pred1[i] == 1 and y_train[i] ==1:
          tp1 += 1
        if pred1[i] == 1 and y_train[i] ==0:
          fp1 += 1
        if pred1[i] == 0 and y_train[i] ==1:
            fn1 +=1
    pre1 = float(tp1)/float(tp1+fp1)
    rec1 = float(tp1)/float(tp1+fn1)
    fm1 = 2 * pre1 * rec1/ (pre1+rec1)
    return numMatches, numMatches1, fm,fm1

def predictions(classifier,x_train,y_train, x_test, y_test):
    ''' Predicts the values of class given the test data '''
    classifier.fit(x_train,y_train)
    pred=[]
    for i in range(0,x_test.shape[0]):
        temp=classifier.predict(x_test[i])
        pred.append(temp[0])
    pred1=[]
    for i in range(0,x_train.shape[0]):
        temp1=classifier.predict(x_train[i])
        pred1.append(temp1[0])
    return pred, pred1
    


def svmLinear(x_train,y_train, x_test, y_test):
    '''Uses the SVM-Linear classifier'''
    pred, pred1=predictions(svm.SVC(kernel = 'linear'), x_train,y_train, x_test, y_test)
    numMatches, numMatches1,fm, fm1=fMeasure(pred, pred1,y_test, y_train)
    return numMatches,numMatches1, fm, fm1


def svmRbf(x_train,y_train, x_test, y_test):
    '''Uses the SVM-Rbf classifier'''
    pred, pred1=predictions(svm.SVC(kernel = 'rbf'),x_train,y_train, x_test, y_test)
    numMatches,numMatches1, fm, fm1=fMeasure(pred, pred1, y_test, y_train)
    return numMatches, numMatches1, fm, fm1

def svmRandomForest(x_train,y_train, x_test, y_test):
    '''Uses the SVM-Random Forest classifier'''
    pred, pred1 = predictions(RandomForestClassifier(max_depth=2, random_state=0),x_train,y_train, x_test, y_test)
    numMatches,numMatches1,fm, fm1=fMeasure(pred,pred1,y_test, y_train)
    return numMatches, numMatches1, fm, fm1

def main():
   
    warnings.warn = warn
    df=pd.read_csv('original_data.csv')
    df = df.replace({'\t':''}, regex=True)
    df=df.replace(r'\?+', np.nan, regex=True)
    df[['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']] = df[['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']].apply(pd.to_numeric)
    df.replace('yes', 1, inplace = True)
    df.replace('no', 0, inplace = True)
    df.replace('notpresent', 1, inplace = True)
    df.replace('present', 0, inplace = True)
    df.replace('abnormal', 1, inplace = True)
    df.replace('normal', 0, inplace = True)
    df.replace('poor', 1, inplace = True)
    df.replace('good', 0, inplace = True)
    df.replace('ckd', 1, inplace = True)
    df.replace('notckd', 0, inplace = True)
    cols = df.columns
    df= df.astype(float)

    #Shuffling the data
    df = df.sample(frac=1)

    #finding the means of all the columns to replace the missing features
    means = []
    for i in range(0,25):
        temp = df[df.columns[i]].mean()
        means.append(temp)
        df[cols[i]].replace(np.nan, temp, inplace=True)
    
    df.to_csv('updated_clean.csv', sep='\t')

    df = pd.read_table('updated_clean.csv')
    df = df.astype(float)

    df =df.drop(['Unnamed: 0'], axis=1)

    df1 = df.iloc[:,24]
    df2 =df.drop(['class'], axis=1)

    #Preparing data for classifiers
    y = np.array(df1.values)
    x = np.array(df2.values)
    x =x.reshape(-1,24)
    y.reshape(-1,1)
    fmeasures_test=[]
    fmeasures_train=[]
    percentTrainedData=list(np.linspace(20,80,3))
    
    for i in percentTrainedData:

        trainedData=int(x.shape[0]*i/100)
        #Training data
        x_train = x[:trainedData,:]
        y_train = y[:trainedData]

        #Testing data
        x_test = x[trainedData:x.shape[0]+1]
        y_test = y[trainedData:y.shape[0]+1]
        numM_1,numM1_1,f_1, f1_1=svmLinear(x_train,y_train, x_test, y_test)
        numM_2 ,numM1_2, f_2, f1_2=svmRbf(x_train,y_train, x_test, y_test)
        numM_3, numM1_3, f_3, f1_3=svmRandomForest(x_train,y_train, x_test, y_test)
    
        fmeasures_test.append([f_1,f_2,f_3])
        fmeasures_train.append([f1_1,f1_2,f1_3])

    printingPlotting(fmeasures_train, fmeasures_test)

    

main()