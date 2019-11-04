import pandas as pd
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

from collections import defaultdict
from itertools import product
from matplotlib.ticker import MaxNLocator
from scipy.linalg import pinv

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import pairwise_distances

rand_state=20

class PCA_DR():
	def __init__(self, data, target, dim_param, title, plot=False):
		self.data = data
		self.target = target
		self.dim_param = dim_param
		self.gen_plot = plot
		self.title = title
	
	def run(self, param):
		pca = PCA(n_components=param, random_state=rand_state)
		newdata = pca.fit_transform(self.data)
		return newdata
	
	def tester(self):
		for k in self.dim_param:
			pca = PCA(n_components=k)
			pca.fit_transform(self.data)
			eigenvalue = pca.explained_variance_
			explainedRatio = pca.explained_variance_ratio_
			
		if self.gen_plot:
			self.plot(eigenvalue, explainedRatio)

	def plot(self, eigenvalue, explainedRatio):
		#Eigenvalue
		ax = plt.figure().gca()
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		plt.plot(self.dim_param, eigenvalue, 'o-')
		plt.xlabel('n_components')
		plt.ylabel('Eigenvalue')
		plt.title(self.title + '-PCA-Eigenvalue')
		plt.savefig('../plots/'+ self.title + '-PCA-Eigenvalue')
		
		#ratios explained
		ax = plt.figure().gca()
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		plt.plot(self.dim_param, explainedRatio, 'o-')
		plt.xlabel('n_components')
		plt.ylabel('Explained Varience Ratio')
		plt.title(self.title + '-PCA-Explained_Ratio')
		plt.savefig('../plots/'+ self.title + '-PCA-Explained_Ratio')
		
	
class ICA_DR():
	def __init__(self, data, target, dim_param, title, plot=False):
		self.data = data
		self.target = target
		self.dim_param = dim_param
		self.gen_plot = plot
		self.title = title
	
	def run(self, param):
		ica = FastICA(n_components=param, random_state=rand_state)
		newdata = ica.fit_transform(self.data)
		return newdata	
	
	def tester(self):
		kurtosis = {}
		for k in self.dim_param:
			ica = FastICA(n_components=k)
			tmp = ica.fit_transform(self.data)
			tmp = pd.DataFrame(tmp)
			tmp = tmp.kurt(axis=0)
			kurtosis[k] = tmp.abs().mean()
			
		kurtosis = pd.Series(kurtosis)
		
		if self.gen_plot:
			self.plot(kurtosis)		
	
	def plot(self, kurtosis):
		#Kurtosis 
		ax = plt.figure().gca()
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		plt.plot(self.dim_param, kurtosis, 'o-')
		plt.xlabel('n_components')
		plt.ylabel('Kurtosis')
		plt.title(self.title + '-ICA-Kurtosis')
		plt.savefig('../plots/'+ self.title + '-ICA-Kurtosis')
		
	
class RP_DR(): ##tried different random states
	def __init__(self, data, target, dim_param, title, plot=False):
		self.data = data
		self.target = target
		self.dim_param = dim_param
		self.gen_plot = plot
		self.title = title

	def run(self, param):
		rp = SparseRandomProjection(n_components=param, random_state=rand_state)
		newdata = rp.fit_transform(self.data)
		return newdata	
	
	#refer to Chad's code
	def pairwise_dist_corr(self, x1, x2):
		assert x1.shape[0] == x2.shape[0]
		d1 = pairwise_distances(x1)
		d2 = pairwise_distances(x2)
		return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]
		
	def reconstruction_error(self, projection, x):
		w = projection.components_
		if sps.issparse(w):
			w = w.todense()
		p = pinv(w)
		reconstructed = ((p@w)@(x.T)).T  #Unproject projected data
		x = np.matrix(x)
		errors = np.square(x - reconstructed)
		return np.nanmean(errors)
	
	def tester(self):
		corr = defaultdict(dict)
		err = defaultdict(dict)
		for i, k in product(range(5), self.dim_param):
			rp = SparseRandomProjection(random_state=i, n_components=k)
			corr[k][i] = self.pairwise_dist_corr(rp.fit_transform(self.data), self.data)
			err[k][i] = self.reconstruction_error(rp, self.data)
			rp.components_
		corr = pd.DataFrame(corr).T
		err = pd.DataFrame(err).T
		#print(corr[corr.columns[1]])
		#print(err)
		if self.gen_plot:
			self.plot(err, corr)
		
	def plot(self, err, corr):
		#Reconstruction err
		ax = plt.figure().gca()
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		plt.plot(self.dim_param, err, 'o-')
		plt.xlabel('n_components')
		plt.ylabel('Reconstruction Error')
		plt.title(self.title + '-RP-Reconstruction_Err')
		plt.savefig('../plots/'+ self.title + '-RP-Reconstruction_Err')
		
		#Pairwise Dist Correlation
		ax = plt.figure().gca()
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		plt.plot(self.dim_param, corr, 'o-')
		plt.xlabel('n_components')
		plt.ylabel('Pairwise Dist Corr')
		plt.title(self.title + '-RP-Pairwise_Dist_Corr')
		plt.savefig('../plots/'+ self.title + '-RP-Pairwise_Dist_Corr')
	
class RF_DR():
	def __init__(self, data, target, dim_param, title, plot=False):
		self.data = data
		self.target = target
		self.dim_param = dim_param
		self.gen_plot = plot
		self.title = title
	
	def run(self, threshold):
		rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=rand_state, n_jobs=1)
		imp = rfc.fit(self.data, self.target).feature_importances_
		imp = pd.DataFrame(imp,columns=['Feature Importance'], index=self.data.columns)
		imp.sort_values(by=['Feature Importance'],inplace=True,ascending=False)
		imp['CumSum'] = imp['Feature Importance'].cumsum()
		imp = imp[imp['CumSum']<=threshold]
		top_cols = imp.index.tolist()
		newdata = self.data[top_cols]
		return newdata

	def tester(self):
		rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=rand_state, n_jobs=1)
		rfc = rfc.fit(self.data, self.target)
		importances = rfc.feature_importances_
		
		if self.gen_plot:
			self.plot(importances)

	def plot(self, importances):
		#ratios explained
		ax = plt.figure().gca()
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		plt.plot(range(1, self.data.shape[1]+1), importances, 'o-')
		plt.xlabel('n_components')
		plt.ylabel('Feature Importance')
		plt.title(self.title + '-RF-Feature_Importance')
		plt.savefig('../plots/'+ self.title + '-RF-Feature_Importance')