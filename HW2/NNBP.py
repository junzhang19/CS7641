import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn import svm
#---additional import by Zhiyuan---#
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import ensemble
import sklearn
import time
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve

############ Program Starts #############
def read_data(dataPath):
    #df = pd.read_csv(dataPath + 'african_crises.csv')
    df = pd.read_csv('african_crises_original.csv')
    return df

def create_features(data):
    # data['exch_usd'].hist(bins=200)
    n=len(data)
    temp1=list(data['exch_usd'])
    #temp2=list(data['gdp_weighted_default'])
    temp3=list(data['inflation_annual_cpi'])
    for i in range(n):
        if temp1[i]<1:
            temp1[i]=0
        elif temp1[i]<10:
            temp1[i]=1
        elif temp1[i]<100:
            temp1[i]=2
        else:
            temp1[i]=3
        if temp3[i]<0:
            temp3[i]=0
        elif temp3[i]<4:
            temp3[i]=1
        elif temp3[i]<10:
            temp3[i]=2
        else:
            temp3[i]=3
    #'exch_usd' <1;<10;<100;>=100
    #'gdp_weighted_default'
    #'inflation_annual_cpi' <0,<4,<10,>10
    temp1s=pd.Series(temp1)
    #temp1s=rename("abc")
    temp3s=pd.Series(temp3)
    
    x=data.drop(['case','cc3','country','year','gdp_weighted_default','banking_crisis','exch_usd','inflation_annual_cpi'],axis=1)
    #x = pd.append([x,temp1s.to_frame().T])
    x=pd.concat([x,temp1s,temp3s],axis=1)
    #x.rename(columns = {"inflation_crises":'3','1':'4'})
    x.columns=['systemic','domestic','external','indep','currency','inflation','exch','inflation_cpi'];
    y=data['banking_crisis']
    return [x,y]
    
def main():
    OUTPUT_FILE = 'nnbp_result.csv'
    datapath = '../data/'
    data = read_data(datapath)
    [X,y]=create_features(data)
    
    iteration_pool = [10,20,50,100,200,500,1000,2000,5000]
    accuracies = []
    training_times = []
    for iteration in iteration_pool:
        print("calculating for iteration = %d" % iteration)
        mlp = MLPClassifier(hidden_layer_sizes=(9),max_iter = iteration)
        t0=time.time()
        mlp.fit(X,y)
        t1=time.time()
        training_time = t1-t0
        print('training_time = ',training_time)
        predictions = mlp.predict(X)
        correct = 0
        for i in range(len(predictions)):
            if y[i]==predictions[i]:
                correct = correct+1
        accuracy1 = correct/len(y)*100
        
        accuracies.append(accuracy1)
        training_times.append(training_time)
        #print(accuracy1)
        
    with open(OUTPUT_FILE, "w") as outFile:
        for i in range(1):
            outFile.write(','.join([
                "iterations",
                "bp_accuracy",
                "bp_training_time"]) + '\n')
        for i in range(len(iteration_pool)):
            outFile.write(','.join([
                str(iteration_pool[i]),
                str(accuracies[i]),
                str(training_times[i])]) + '\n')
    print("the end of the program")
       
if __name__ == "__main__":
    main()



