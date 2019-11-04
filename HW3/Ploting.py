import pandas as pd
import matplotlib.pyplot as plt

""" For Part3 """

class plot_all:
	def __init__(self, ds_title, title, paramlist):
		self.ds_title = ds_title
		self.title = title
		self.paramlist = paramlist
		
		
	def run(self):
		#plot silhouette scores - KM
		df1 = pd.read_csv('../data/plots/' + self.ds_title + '-KM-silhouette_scores.csv')
		df2 = pd.read_csv('../data/plots/' + self.title + '-KM-silhouette_scores.csv')
		plt.figure()
		plt.grid()
		plt.title(self.title + '-KM-Silhouette Scores')
		plt.xlabel('n_components')
		plt.ylabel('Silhouette Scores')
		plt.plot(self.paramlist,df1.iloc[:,[1]],'o-',color='g',label=self.ds_title)
		plt.plot(self.paramlist,df2.iloc[:,[1]],'o-',color='b',label=self.title)
		plt.legend(loc='best')
		plt.grid()
		plt.savefig('../plots/'+ self.title + '-KM-silhouette_scores')
		
		#plot accuracy scores -KM
		df1 = pd.read_csv('../data/plots/' + self.ds_title + '-KM-accuracy_scores.csv')
		df2 = pd.read_csv('../data/plots/' + self.title + '-KM-accuracy_scores.csv')
		plt.figure()
		plt.grid()
		plt.title(self.title + '-KM-Accuracy Scores')
		plt.xlabel('n_components')
		plt.ylabel('Accuracy Scores')
		plt.plot(self.paramlist,df1.iloc[:,[1]],'o-',color='g',label=self.ds_title)
		plt.plot(self.paramlist,df2.iloc[:,[1]],'o-',color='b',label=self.title)
		plt.legend(loc='best')
		plt.grid()
		plt.savefig('../plots/'+ self.title + '-KM-accuracy_scores')
		
		#plot silhouette scores - EM
		df1 = pd.read_csv('../data/plots/' + self.ds_title + '-EM-silhouette_scores.csv')
		df2 = pd.read_csv('../data/plots/' + self.title + '-EM-silhouette_scores.csv')
		plt.figure()
		plt.grid()
		plt.title(self.title + '-EM-Silhouette Scores')
		plt.xlabel('n_components')
		plt.ylabel('Silhouette Scores')
		plt.plot(self.paramlist,df1.iloc[:,[1]],'o-',color='g',label=self.ds_title)
		plt.plot(self.paramlist,df2.iloc[:,[1]],'o-',color='b',label=self.title)
		plt.legend(loc='best')
		plt.grid()
		plt.savefig('../plots/'+ self.title + '-EM-silhouette_scores')
		
		#plot accuracy scores -EM
		df1 = pd.read_csv('../data/plots/' + self.ds_title + '-EM-accuracy_scores.csv')
		df2 = pd.read_csv('../data/plots/' + self.title + '-EM-accuracy_scores.csv')
		plt.figure()
		plt.grid()
		plt.title(self.title + '-EM-Accuracy Scores')
		plt.xlabel('n_components')
		plt.ylabel('Accuracy Scores')
		plt.plot(self.paramlist,df1.iloc[:,[1]],'o-',color='g',label=self.ds_title)
		plt.plot(self.paramlist,df2.iloc[:,[1]],'o-',color='b',label=self.title)
		plt.legend(loc='best')
		plt.grid()
		plt.savefig('../plots/'+ self.title + '-EM-accuracy_scores')