import numpy as np
import pandas as pd

from Clustering import EMCluster, KMCluster
from DimensionReduction import PCA_DR, ICA_DR, RP_DR, RF_DR
from Ploting import plot_all
from NeuralNetwork import NN
from sklearn.model_selection import train_test_split
############ Program Starts #############
def read_data(dataPath):
    df = pd.read_csv(dataPath + 'african_crises.csv')
    return df

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

def main():
	datapath = '../data/'
	data = read_data(datapath)
	[X,y] = create_features(data)
	
	part1 = False
	part2 = False
	part3 = False
	part4 = True
	part5 = True
	
	'''---- Clustering ---'''
	paramlist_clustering = list(np.arange(2,15,1))
	model_KM = KMCluster(data=X, target=y, num_clusters=paramlist_clustering, plot=True, title='African_Crises')
	model_EM = EMCluster(data=X, target=y, num_clusters=paramlist_clustering, plot=True, title='African_Crises')
	
	if part1:
		model_KM.tester()
		model_EM.tester()
	
	'''---- Dimension Reduction ---'''
	paralist_dr = list(np.arange(1,7,1))
	model_PCA = PCA_DR(data=X, target=y, dim_param=paralist_dr, plot=True, title='African_Crises')
	model_ICA = ICA_DR(data=X, target=y, dim_param=paralist_dr, plot=True, title='African_Crises')
	model_RP = RP_DR(data=X, target=y, dim_param=paralist_dr, plot=True, title='African_Crises')
	model_RF = RF_DR(data=X, target=y, dim_param=paralist_dr, plot=True, title='African_Crises')	
	
	if part2:
		model_PCA.tester()
		model_ICA.tester()
		model_RP.tester()
		model_RF.tester()
    
	'''---- Clustering After DR ---'''
	pcaX = model_PCA.run(2)
	icaX = model_ICA.run(3)
	rpX = model_RP.run(2)
	rfX = model_RF.run(0.95) #threshold = 0.95

	if part3:
		title = 'African_Crises-PCA'
		model_KM_PCA = KMCluster(data=pcaX, target=y, num_clusters=paramlist_clustering, plot=True, title=title)
		model_KM_PCA.tester()
		model_EM_PCA = EMCluster(data=pcaX, target=y, num_clusters=paramlist_clustering, plot=True, title=title)
		model_EM_PCA.tester()
		plot_all(ds_title='African_Crises', title=title, paramlist=paramlist_clustering).run()
	
		title = 'African_Crises-ICA'
		model_KM_ICA = KMCluster(data=icaX, target=y, num_clusters=paramlist_clustering, plot=True, title=title)
		model_KM_ICA.tester()	
		model_EM_ICA = EMCluster(data=icaX, target=y, num_clusters=paramlist_clustering, plot=True, title=title)
		model_EM_ICA.tester()
		plot_all(ds_title='African_Crises', title=title, paramlist=paramlist_clustering).run()
		
		title = 'African_Crises-RP'
		model_KM_RP = KMCluster(data=rpX, target=y, num_clusters=paramlist_clustering, plot=True, title=title)
		model_KM_RP.tester()	
		model_EM_RP = EMCluster(data=rpX, target=y, num_clusters=paramlist_clustering, plot=True, title=title)
		model_EM_RP.tester()
		plot_all(ds_title='African_Crises', title=title, paramlist=paramlist_clustering).run()
		
		title = 'African_Crises-RF'
		model_KM_RF = KMCluster(data=rfX, target=y, num_clusters=paramlist_clustering, plot=True, title=title)
		model_KM_RF.tester()	
		model_EM_RF = EMCluster(data=rfX, target=y, num_clusters=paramlist_clustering, plot=True, title=title)
		model_EM_RF.tester()
		plot_all(ds_title='African_Crises', title=title, paramlist=paramlist_clustering).run()
		
	'''---- NN After DR ---'''
	alphalist=list(np.arange(0.5,4,0.5))
	X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(pcaX, y, test_size=0.20)
	X_ica_train, X_ica_test, y_ica_train, y_ica_test = train_test_split(icaX, y, test_size=0.20)
	X_rp_train, X_rp_test, y_rp_train, y_rp_test = train_test_split(rpX, y, test_size=0.20)
	X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(np.array(rfX), y, test_size=0.20)
    
	if part4:
		title = 'African_Crises-PCA'
		model_NN_PCA = NN(X_pca_train, X_pca_test, y_pca_train, y_pca_test, dim_param = alphalist, plot=True, title=title)
		model_NN_PCA.tester()
		
		title = 'African_Crises-ICA'
		model_NN_ICA = NN(X_ica_train, X_ica_test, y_ica_train, y_ica_test, dim_param = alphalist, plot=True, title=title)
		model_NN_ICA.tester()
		
		title = 'African_Crises-RP'
		model_NN_RP = NN(X_rp_train, X_rp_test, y_rp_train, y_rp_test, dim_param = alphalist, plot=True, title=title)
		model_NN_RP.tester()
		
		title = 'African_Crises-RF'
		model_NN_RF = NN(X_rf_train, X_rf_test, y_rf_train, y_rf_test, dim_param = alphalist, plot=True, title=title)
		model_NN_RF.tester()
	
	'''---- NN After Clustering ---'''
	kmX = model_KM.run(2)
	emX = model_EM.run(2)
	
	X_km_train, X_km_test, y_km_train, y_km_test = train_test_split(kmX, y, test_size=0.20)
	X_em_train, X_em_test, y_em_train, y_em_test = train_test_split(emX, y, test_size=0.20)
	
	if part5:
		title = 'African_Crises-KM'
		model_NN_KM = NN(X_km_train, X_km_test, y_km_train, y_km_test, dim_param = alphalist, plot=True, title=title)
		model_NN_KM.tester()
		
		title = 'African_Crises-EM'
		model_NN_EM = NN(X_em_train, X_em_test, y_em_train, y_em_test, dim_param = alphalist, plot=True, title=title)
		model_NN_EM.tester()

if __name__ == "__main__":
    main()