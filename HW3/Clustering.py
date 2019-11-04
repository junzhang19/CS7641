import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from scipy.spatial.distance import cdist
from matplotlib.ticker import MaxNLocator
from collections import Counter

def cluster_predictions(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target   
    return pred

rand_state = 20

class KMCluster():
    def __init__(self, data, target, num_clusters, title, plot=False):
        self.data = data
        self.target = target
        self.num_clusters = num_clusters
        self.gen_plot = plot
        self.title = title
    
    def run(self, param):
        km = KMeans(n_clusters=param, max_iter=500, random_state=rand_state,init='k-means++')
        newdata = km.fit_transform(self.data)
        return newdata

    def tester(self):
        meandist=[]
        homogeneity_scores=[]
        completeness_scores=[]
        accuracy_scores=[]
        silhouette_scores=[]
        km = KMeans(max_iter=500, random_state=rand_state,init='k-means++')

        for k in self.num_clusters:
                km = km.set_params(n_clusters=k)
                km.fit_transform(self.data)
                predicts = km.labels_
                
                min = np.min(np.square(cdist(self.data, km.cluster_centers_, 'euclidean')), axis = 1)
                value = np.mean(min)
                meandist.append(value)
                homogeneity_scores.append(metrics.homogeneity_score(self.target, predicts))
                completeness_scores.append(metrics.completeness_score(self.target, predicts))
                silhouette_scores.append(metrics.silhouette_score(self.data, predicts))
                y_pred = cluster_predictions(self.target, predicts)
                accuracy_scores.append(metrics.accuracy_score(self.target, y_pred))
        df_sil = pd.DataFrame(silhouette_scores)
        df_acc = pd.DataFrame(accuracy_scores)
        df_sil.to_csv('../data/plots/'+ self.title + '-KM-silhouette_scores.csv')
        df_acc.to_csv('../data/plots/' + self.title + '-KM-accuracy_scores.csv')
        
        if self.gen_plot:
            self.plot(meandist, homogeneity_scores, completeness_scores)

    def plot(self, meandist, homogeneity, completeness):
		#Plot SSE; Elbow method
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(self.num_clusters, meandist, 'o-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Average within cluster SSE')
        plt.title(self.title + '-Kmeans-Average within cluster SSE')
        plt.savefig('../plots/'+ self.title + '-Kmeans-Average_within_cluster_SSE')
		
		#Plot homogeneity score; Elbow Method
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(self.num_clusters, homogeneity, 'o-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Homogeneity Score')
        plt.title(self.title + '-Kmeans-Homogeneity Score')
        plt.savefig('../plots/'+ self.title + '-Kmeans-Homogeneity_Score')
		
		#Plot completeness score; Elbow Method
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(self.num_clusters, completeness, 'o-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Completeness Score')
        plt.title(self.title + '-Kmeans-Completeness Score')
        plt.savefig('../plots/'+ self.title + '-Kmeans-Completeness_Score')

class EMCluster():
    def __init__(self, data, target, num_clusters, title, plot=False):
        self.data = data
        self.target = target
        self.num_clusters = num_clusters
        self.gen_plot = plot
        self.title = title
        
    def run(self, param):
        em = GaussianMixture(n_components=param, covariance_type = 'diag', random_state=rand_state)
        w = em.fit(self.data).covariances_
        newdata=self.data@w.T
        return newdata

    def tester(self):
        model_scores=[]
        homogeneity_scores=[]
        completeness_scores=[]
        aic=[]
        bic=[]
        accuracy_scores=[]
        silhouette_scores=[]
        em = GaussianMixture(covariance_type = 'diag', random_state=rand_state)

        for k in self.num_clusters:
            em = em.set_params(n_components=k)
            em.fit(self.data)
            predicts = em.predict(self.data)
			
            model_scores.append(em.score(self.data))
            homogeneity_scores.append(metrics.homogeneity_score(self.target, predicts))
            completeness_scores.append(metrics.completeness_score(self.target, predicts))
            bic.append(em.bic(self.data))
            aic.append(em.aic(self.data))
            silhouette_scores.append(metrics.silhouette_score(self.data, predicts))
            y_pred = cluster_predictions(self.target, predicts)
            accuracy_scores.append(metrics.accuracy_score(self.target, y_pred))
        df_sil = pd.DataFrame(silhouette_scores)
        df_acc = pd.DataFrame(accuracy_scores)
        df_sil.to_csv('../data/plots/'+ self.title + '-EM-silhouette_scores.csv')
        df_acc.to_csv('../data/plots/'+ self.title + '-EM-accuracy_scores.csv')

        if self.gen_plot:
            self.plot(model_scores, homogeneity_scores, completeness_scores, bic, aic)

    def plot(self, model_scores, homogeneity, completeness,  bic, aic):
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.num_clusters, model_scores, 'o-')
            plt.xlabel('Number of clusters')
            plt.ylabel('Log Probablility')
            plt.title(self.title + '-EM-Log Probability')
            plt.savefig('../plots/'+ self.title + '-EM-Log_Probability')
            
            #Plot Homogeneity Score; Elbow method
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.num_clusters, homogeneity, 'o-')
            plt.xlabel('Number of clusters')
            plt.ylabel('Homogeneity Score')
            plt.title(self.title + '-EM-Homogeneity Score')
            plt.savefig('../plots/'+ self.title + '-EM-Homogeneity_Score')
            
            #Plot Completeness Score; Elbow method
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.num_clusters, completeness, 'o-')
            plt.xlabel('Number of clusters')
            plt.ylabel('Completeness Score')
            plt.title(self.title + '-EM-Completeness Score')
            plt.savefig('../plots/'+ self.title + '-EM-Completeness_Score')
         
            #Plot BIC Score; Elbow method
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.num_clusters, bic, 'o-')
            plt.xlabel('Number of clusters')
            plt.ylabel('BIC Score')
            plt.title(self.title + '-EM-BIC Score')
            plt.savefig('../plots/'+ self.title + '-EM-BIC_Score')

            #Plot AIC Score; Elbow method
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.num_clusters, aic, 'o-')
            plt.xlabel('Number of clusters')
            plt.ylabel('AIC Score')
            plt.title(self.title + '-EM-AIC Score')
            plt.savefig('../plots/'+ self.title + '-EM-AIC_Score')