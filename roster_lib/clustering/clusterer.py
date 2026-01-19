import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import entropy

from roster_lib.utils.colinearity_handler import ColinearityHandler
from roster_lib.clustering.pca import find_n_PC
from roster_lib.constants import PREPROC_DATA_PATH
from id_dict import pid2name, name2pid, pid2pos_bref

SCALERS = {'minmax' : MinMaxScaler, 'robust': RobustScaler, 'standard': StandardScaler}
PENALTY_RATE = 1

EVRS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
N_CLUSTS = range(2,15)
METHODS = ['kmeans',
            'agg_ward',
            'agg_average',
            'agg_complete',
            # 'agg_single'
            ]
SCALINGS = ['standard', 'robust', 'minmax']
FEATURES_SELECTIONS = ['incl', 'excl', 'autoexcl']

DEFAULT_ALPHA = 1
METRICS_ASC = {'silhouette_score': False,
               'davies_bouldin': True,
               'calinski_harabasz':False,	
               'entropy': False,
               'population_std': True}

class Clusterer :
    
    def __init__(self, use_positions:bool=True, alpha:float = DEFAULT_ALPHA):
        self.preproc_path = PREPROC_DATA_PATH / 'clustering'
        self.last_version = max([int(f.split('_')[-1][1:-4]) for f in os.listdir(self.preproc_path) if 'clustering' in f])
        self.use_positions = use_positions
        self.alpha = alpha
        self.rdf = self.load_results()
        
    def load_results(self, version:int = None):
        if version is None :
            version = self.last_version
        fp = self.preproc_path / f"ARO_clustering_v{version}.csv"
        df = pd.read_csv(fp, index_col = 0)
        df['entropy_silhouette'] = df['silhouette_score'] * df['entropy'] ** self.alpha
        return df

    def run_comparison(self,
                       features_selections: list = FEATURES_SELECTIONS,
                       scaling_methods: list = SCALINGS,
                       evr_targets: list = EVRS,
                       n_clusts: list = N_CLUSTS,
                       methods: list = METHODS):
        
        version = self.last_version +1 
        fp = self.preproc_path / f"ARO_clustering_v{version}.csv"

        colinearity_handler = self._manage_colinearity_handler()
        
        n_exps = len (features_selections) * len(scaling_methods) * len(evr_targets) * len(n_clusts) * len(methods)
        n_exps_per_fs = len(scaling_methods) * len(evr_targets) * len(n_clusts) * len(methods)
        n_exps_per_scaler = len(evr_targets) * len(n_clusts) * len(methods)
        n_exps_per_evr = len(n_clusts) * len(methods)
        n_exps_per_method = len(n_clusts)
        results = []
        best_score, best_penalized_score, best_entropy_weighted_score = 0, 0, 0
        for m, fs in enumerate(features_selections):
            for i, scaling_method in enumerate(scaling_methods) :
                scaler = SCALERS[scaling_method]()
                df = colinearity_handler.get_data(feature_selection=fs)
                df_scaled = scaler.fit_transform(df)
                basis_pca = PCA().fit(df_scaled)
                for j, evr in enumerate(evr_targets) :
                    n_PCA = find_n_PC(basis_pca.explained_variance_ratio_, evr, display_fig= False, verbose = False)
                    _PCA = PCA(n_components=n_PCA)
                    _PCA.fit(df_scaled)
                    _X_proj = _PCA.transform(df_scaled) 
                    for k, method in enumerate(methods):
                        for l, n_clust in enumerate(n_clusts):
                            labels = self.clusterize(X_proj= _X_proj, n_clust= n_clust, method=method)
                            silhouette = silhouette_score(_X_proj, labels) 
                            db = davies_bouldin_score(_X_proj, labels)
                            ch = calinski_harabasz_score(_X_proj, labels)
                            pop_std = self._pop_std(labels)
                            entropy_score = self.normalized_entropy(labels)
                            results.append({'feature_selection':fs,
                                            'method': method, 
                                            'scaling': scaling_method, 
                                            'evr': evr, 
                                            'n_PC': n_PCA , 
                                            'n_clust': n_clust, 
                                            'silhouette_score': silhouette,
                                            'davies_bouldin': db,
                                            'calinski_harabasz' : ch,
                                            'entropy': entropy_score,
                                            'population_std': pop_std})
                            counter = m* n_exps_per_fs + i * n_exps_per_scaler + j * n_exps_per_evr + k * n_exps_per_method + l
                            best_score = max(best_score, silhouette)
                            best_entropy_weighted_score = max(best_entropy_weighted_score, silhouette * entropy_score ** self.alpha)
                            best_penalized_score = max(best_penalized_score, silhouette - pop_std * PENALTY_RATE)
                            
                            msg = f"Processing experiment n° {counter+1:>4} of {n_exps} | "
                            msg += f"Best Silhouette Score = {best_score:.3f} | "
                            msg += f"Best Entropy-weighted Silhouette Score = {best_entropy_weighted_score:.3f} | "
                            msg += f"Best Penalized Score = {best_penalized_score:.3f}"
                            print (msg, end = '\r')
        df = pd.DataFrame(results)
        df.to_csv(fp)
        return df       
   
    def customized_clustering(self,
                            evr:float = 1.0,
                            n_clust:int = 2,
                            method:str = 'kmeans',
                            scaling:str = 'standard',
                            feature_selection:str = 'incl',
                            verbose:bool = True):
        
        colinearity_handler = self._manage_colinearity_handler()
            
        selected_df = colinearity_handler.get_data(feature_selection=feature_selection)
        if scaling is not None:
            scaler = SCALERS[scaling]()
            df_scaled = scaler.fit_transform(selected_df)
        else :
            df_scaled = selected_df
        
        basis_pca = PCA().fit(df_scaled)
        n_PCA = find_n_PC(basis_pca.explained_variance_ratio_, evr, display_fig= False, verbose = False)
        pca = PCA(n_components=n_PCA)
        pca.fit(df_scaled)
        X_proj = pca.transform(df_scaled) 
        labels = self.clusterize(X_proj, n_clust, method)
        silhouette = silhouette_score(X_proj, labels)
        db = davies_bouldin_score(X_proj, labels)
        ch = calinski_harabasz_score(X_proj, labels)
        pop_std = self._pop_std(labels)
        entropy_score = self.normalized_entropy(labels)
        entropy_silhouette = silhouette * entropy_score ** self.alpha
        if verbose :
            print(f"Silhouette: {silhouette:.3f} | Davies-Bouldin: {db:.3f} | Calinski-Harabasz: {ch:.3f} | Normalized Entropy : {entropy_score:.3f} | Population STD: {pop_std:.3f} | Entropy-weighted Silhouette: {entropy_silhouette:.3f}")
        return X_proj, labels  
    
    def plot_clustering(self,
                        axs = None, 
                        evr:float = 1.0,
                        n_clust:int = 2,
                        method:str = 'kmeans',
                        scaling:str = 'standard',
                        feature_selection:str = 'incl',
                        verbose:bool = True, 
                        return_data:bool = False):  
            X_proj, labels = self.customized_clustering(evr, n_clust, method, scaling, feature_selection, verbose)
            if axs == None :
                fig, axs = plt.subplots(1,3,figsize = (18,5));
            sns.scatterplot(x = X_proj[:,0], y = X_proj[:,1], hue = labels, alpha = 0.5, palette='bright', ax=axs[0], legend=False);
            axs[0].set_xlabel("PC1");
            axs[0].set_ylabel("PC2");
            axs[0].set_title("Visual inspection of clustering along 2 first PCs");
            _, counts = np.unique(labels, return_counts= True)
            counts[::-1].sort()
            sns.barplot(data = counts, ax = axs[1]);
            axs[1].set_title("Cluster population histogram");
            axs[1].set_xlabel("Cluster");
            axs[1].set_ylabel("Count");
            silhouette = silhouette_score(X_proj, labels)
            entropy = self.normalized_entropy(labels)
            entropy_silhouette = silhouette * entropy ** self.alpha
            _metrics = pd.DataFrame([[silhouette, entropy, entropy_silhouette]], columns = ['Silhouette','Weighted Entropy', 'EW Silhouette'])
            sns.barplot(data = _metrics, ax = axs[2]);
            axs[2].set_title("Metrics score")
            if return_data:
                return X_proj, labels
            
    def plot_data(self, scalings:list = ['minmax','standard','robust'], feature_selections:list = ['incl','excl','autoexcl']):

        colinearity_handler = self._manage_colinearity_handler()
        fig, axs = plt.subplots(len(feature_selections), len(scalings), figsize = (5* len(scalings), 5* len(feature_selections)))
        
        for i, fs in enumerate(feature_selections):
            _df = colinearity_handler.get_data(fs)
            for j, sc in enumerate(scalings):      
                scaler = SCALERS[sc]()
                _df_scaled = scaler.fit_transform(_df)
                _X_proj = PCA().fit_transform(_df_scaled)
                _X_proj = pd.DataFrame(_X_proj, columns = [f"PC_{i+1}" for i in range(_X_proj.shape[1])])
                sns.scatterplot(data = _X_proj, x = 'PC_1', y = 'PC_2', alpha = 0.5, ax = axs[i,j]);
                axs[i,j].set_title(f"{sc} scaling, feature selection : {fs}");
                     
    def get_scaled_data(self, scaling:str = 'robust', feature_selection:str= 'incl', retrieve_name:bool=True):
        
        colinearity_handler = self._manage_colinearity_handler()
        _df = colinearity_handler.get_data(feature_selection)
        sc = SCALERS[scaling]()
        _df_scaled = sc.fit_transform(_df)
        _X_proj = PCA().fit_transform(_df_scaled)
        _X_proj = pd.DataFrame(_X_proj, columns = [f"PC_{i+1}" for i in range(_X_proj.shape[1])], index = _df.index)
        if retrieve_name:
            _X_proj.index = [pid2name[int(x.split("_")[0])] + "_" +  x.split("_")[1] for x in _X_proj.index]
        
        return _X_proj
   
    def _manage_colinearity_handler(self):
        if hasattr(self, 'colinearity_handler'):
            colinearity_handler = self.colinearity_handler
        else : 
            colinearity_handler = ColinearityHandler(verbose = False, use_positions =self.use_positions)
            self.colinearity_handler = colinearity_handler
        return colinearity_handler        
         
    @staticmethod    
    def clusterize(X_proj, n_clust:int, method:str):
        if '_' in method :
            linkage = method.split('_')[1]
        clustering = KMeans(n_clusters= n_clust).fit(X_proj) if method == 'kmeans' else AgglomerativeClustering(n_clusters= n_clust, linkage= linkage).fit(X_proj)
        labels = clustering.labels_
        return labels 
    
    @staticmethod
    def _pop_std(labels:np.array):
        _, counts = np.unique(labels, return_counts= True)
        return (counts / counts.sum()).std() 
    
    @staticmethod       
    def normalized_entropy(labels:np.array):
        unique, counts = np.unique(labels, return_counts= True)
        freq = counts / counts.sum()
        return entropy(freq) / np.log(unique.shape[0])
    
    
    
if __name__ == '__main__' :
    Clusterer(use_positions=False).run_comparison()