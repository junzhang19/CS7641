import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import ensemble
import sklearn
import time
#from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve

############ Program Starts #############
def read_data(dataPath):
    df = pd.read_csv(dataPath + 'african_crises.csv')
    #df = pd.read_csv(dataPath + 'online_shoppers_intention_revised.csv')
    return df


# create features for African_Crises
def create_features(data):
    n = len(data)
    exch_usd = list(data['exch_usd'])
    inf_cpi = list(data['inflation_annual_cpi'])
    #classify exch_usd (<1,<10,<100,100+) and inflation_annual_cpi(<0,<4,<10,10+)
    for i in range(n):
        if exch_usd[i] < 1:
            exch_usd[i] = 0
        elif exch_usd[i] < 10:
            exch_usd[i] = 1
        elif exch_usd[i] < 100:
            exch_usd[i] = 2
        else:
            inf_cpi[i] = 3
        if inf_cpi[i] < 0:
            inf_cpi[i] = 0
        elif inf_cpi[i] < 4:
            inf_cpi[i] = 1
        elif inf_cpi[i] < 10:
            inf_cpi[i] = 2
        else:
            inf_cpi[i] = 3

    exch_usd_series = pd.Series(exch_usd)
    inf_cpi_series = pd.Series(inf_cpi)
    
    x = data.drop(['case','cc3','country','year','gdp_weighted_default','banking_crisis','exch_usd','inflation_annual_cpi'],axis=1)
    x = pd.concat([x,exch_usd_series,inf_cpi_series],axis=1)
    x.columns = ['systemic','domestic','external','indep','currency','inflation','exch','inflation_cpi'];
    y = data['banking_crisis']
    return [x,y]


# create features for Online Shopping Intention
'''
def create_features(data):
    data_x_na = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    x = data_x_na.drop(['Revenue'], axis = 1)
    y = data_x_na['Revenue']
    return [x, y]
'''
    
def plot_learning_curve(model,title,X_train,y_train,cv=None):
    step=np.linspace(1/cv,1.0,cv)
    plt.figure()
    plt.title(title)
    plt.xlabel("Training Samples")
    plt.ylabel("Score")
    train_sizes,train_scores,test_scores = learning_curve(model,X_train,y_train,cv=cv,train_sizes=step)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color='g')
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color='b')
    plt.plot(train_sizes,train_scores_mean,'o-',color='g',label="Training Score")
    plt.plot(train_sizes,test_scores_mean,'o-',color='b',label="Cross-validation Score")
    #plt.ylim((0.9,1.02))
    plt.legend(loc='best')
    plt.show()
    return plt
    
def nn(X_train, X_test, y_train, y_test):
    # grid search
    paramlist=[0.1,0.5,1,1.5,2,2.5,3,3.5,4]
    score_training = []
    score_testing  = []
    for param in paramlist:
        clf=MLPClassifier(alpha = param, hidden_layer_sizes=(10,10), max_iter=500)
        clf.fit(X_train,y_train)
        score_training.append(clf.score(X_train,y_train))
        score_testing.append(clf.score(X_test,y_test))
    
    plt.figure()
    plt.xlabel("Alpha")
    plt.ylabel("Score")
    plt.plot(paramlist, score_training,'o-',color='g',label="Training Sample")
    plt.plot(paramlist,score_testing,'o-',color='b',label="Testing Sample")
    plt.legend(loc='best')
    plt.show()
    
    index = score_testing.index(max(score_testing))
    
    mlp = MLPClassifier(alpha = paramlist[index], hidden_layer_sizes=(10,10), max_iter=500)
    t0=time.time()
    mlp.fit(X_train,y_train)
    t1=time.time()
    training_time = t1 - t0
    print('training_time = ',training_time)
    predictions = mlp.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    # return mlp.predict(X_test)
    #----plot---#
    plot_learning_curve(mlp,'Learning Curve of Neural Network',X_train,y_train,cv=10)

def svm(X_train, X_test, y_train, y_test):
    # grid search
    paramlist=[1,2,3,4,5,6,7,8,9,10]
    #paramlist=[0.1,0.3,0.5,1,1.5]
    score_training = []
    score_testing  = []
    for param in paramlist:
        clf=SVC(C = param)
        clf.fit(X_train,y_train)
        score_training.append(clf.score(X_train,y_train))
        score_testing.append(clf.score(X_test,y_test))
    
    plt.figure()
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.plot(paramlist, score_training,'o-',color='g',label="Training Sample")
    plt.plot(paramlist,score_testing,'o-',color='b',label="Testing Sample")
    plt.legend(loc='best')
    plt.show()
    
    index = score_testing.index(max(score_testing))
    print(index)
    
    clf = SVC(C = 4)  # 'rbf','linear','poly'
    t0 = time.time()
    clf.fit(X_train, y_train)
    t1 = time.time()
    training_time = t1 - t0
    print('training_time = ',training_time)
    predictions = clf.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    #----plot---#
    plot_learning_curve(clf,'Learning Curve of SVM',X_train,y_train,cv=5)


def dt(X_train, X_test, y_train, y_test):
    # grid search
    paramlist = [1,2,3,4,5,6,7,8,9,10]
    score_training = []
    score_testing  = []
    for param in paramlist:
        clf=DecisionTreeClassifier(max_depth=param,min_samples_leaf=3)
        clf.fit(X_train,y_train)
        score_training.append(clf.score(X_train,y_train))
        score_testing.append(clf.score(X_test,y_test))
    
    plt.figure()
    plt.xlabel("max_depth")
    plt.ylabel("Score")
    plt.plot(paramlist, score_training,'o-',color='g',label="Training Sample")
    plt.plot(paramlist,score_testing,'o-',color='b',label="Testing Sample")
    plt.legend(loc='best')
    plt.show()
    
    index = score_testing.index(max(score_testing))
    print(index)
    
    tree = DecisionTreeClassifier(max_depth=paramlist[index],min_samples_leaf=3)
    t0 = time.time()
    tree.fit(X_train, y_train)
    t1 = time.time()
    training_time = t1 - t0
    print('training_time = ',training_time)
    y_pred = tree.predict(X_test)
    print ("Accuracy is ", accuracy_score(y_test, y_pred)*100)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    #----plot---#
    plot_learning_curve(tree,'Learning Curve of Decision Tree',X_train,y_train,cv=10)
    
def boost(X_train, X_test, y_train, y_test):
    # grid search
    paramlist=[10, 50, 100, 200, 500, 1000]
    score_training = []
    score_testing  = []
    for param in paramlist:
        clf=AdaBoostClassifier(n_estimators=param,learning_rate=0.01)
        clf.fit(X_train,y_train)
        score_training.append(clf.score(X_train,y_train))
        score_testing.append(clf.score(X_test,y_test))
    
    plt.figure()
    plt.xlabel("n_estimators")
    plt.ylabel("Score")
    plt.plot(paramlist, score_training,'o-',color='g',label="Training Sample")
    plt.plot(paramlist,score_testing,'o-',color='b',label="Testing Sample")
    plt.legend(loc='best')
    plt.show()
    
    index = score_testing.index(max(score_testing))
    
    clf = AdaBoostClassifier(n_estimators=200,learning_rate=0.01)
    t0 = time.time()
    clf.fit(X_train,y_train)
    t1 = time.time()
    training_time = t1 - t0
    print('training_time = ',training_time)
    predictions = clf.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    #----plot---#
    plot_learning_curve(clf,'Learning Curve of AdaBoost',X_train,y_train,cv=10)

def knn(X_train, X_test, y_train, y_test):
        # grid search
    paramlist=[]
    for i in range(30):
        i = i+1
        paramlist.append(i)
        
    score_training = []
    score_testing  = []
    for param in paramlist:
        clf=KNeighborsClassifier(n_neighbors=param)
        clf.fit(X_train,y_train)
        score_training.append(clf.score(X_train,y_train))
        score_testing.append(clf.score(X_test,y_test))
    
    plt.figure()
    plt.xlabel("n_neighbors")
    plt.ylabel("Score")
    plt.plot(paramlist, score_training,'o-',color='g',label="Training Sample")
    plt.plot(paramlist,score_testing,'o-',color='b',label="Testing Sample")
    plt.legend(loc='best')
    plt.show()
    
    index = score_testing.index(max(score_testing))
    
    knn = KNeighborsClassifier(n_neighbors=paramlist[index])
    t0 = time.time()
    knn.fit(X_train, y_train)
    t1 = time.time()
    training_time = t1 - t0
    print('training_time = ',training_time)
    # predict the response for test dataset
    predictions = knn.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    #----plot---#
    plot_learning_curve(knn,'Learning Curve of K-Nearest-Neighbors',X_train,y_train,cv=10)
    
def main():
    datapath = '../data/'
    data = read_data(datapath)
    [X,y] = create_features(data)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=7641)
    nn(X_train, X_test, y_train, y_test)
    svm(X_train, X_test, y_train, y_test)
    dt(X_train, X_test, y_train, y_test)
    boost(X_train, X_test, y_train, y_test)
    knn(X_train, X_test, y_train, y_test)    

if __name__ == "__main__":
    main()


