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

rand_state = 100

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
	plt.grid()
	plt.savefig('../plots/'+ title)

class NN:
	def __init__(self, X_train, X_test, y_train, y_test, dim_param, title, plot=False):
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test
		self.dim_param = dim_param
		self.gen_plot = plot
		self.title = title

	def tester(self):
		# grid search
		score_training = []
		score_testing  = []
		for param in self.dim_param:
			clf=MLPClassifier(alpha = param, hidden_layer_sizes=(10,10), max_iter=500, random_state=rand_state)
			clf.fit(self.X_train,self.y_train)
			score_training.append(clf.score(self.X_train,self.y_train))
			score_testing.append(clf.score(self.X_test,self.y_test))
		
		if self.gen_plot:
			plt.figure()
			plt.title(self.title + '-NN-Model_Score')
			plt.xlabel("Alpha")
			plt.ylabel("Score")
			plt.plot(self.dim_param, score_training,'o-',color='g',label="Training Sample")
			plt.plot(self.dim_param,score_testing,'o-',color='b',label="Testing Sample")
			plt.legend(loc='best')
			plt.savefig('../plots/'+ self.title + '-NN-Model_Score')
    
		index = score_testing.index(max(score_testing))
		print('Best alpha = ', self.dim_param[index])
        
		mlp = MLPClassifier(alpha = self.dim_param[index], hidden_layer_sizes=(10,10), max_iter=500, random_state=rand_state)
		t0=time.time()
		mlp.fit(self.X_train,self.y_train)
		t1=time.time()
		training_time = t1 - t0
		print('training_time = ',training_time)
		#predictions = mlp.predict(self.X_test)
		#print(confusion_matrix(self.y_test, predictions))
		#print(classification_report(self.y_test, predictions))
		# return mlp.predict(X_test)
		#----plot---#
		if self.gen_plot:
			plot_learning_curve(mlp, self.title + '-NN-Learning_Curve', self.X_train, self.y_train,cv=10)