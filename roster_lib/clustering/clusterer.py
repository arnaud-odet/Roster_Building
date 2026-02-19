import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import sys
import json

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples

from roster_lib.utils.feature_handler import FeatureHandler
from roster_lib.clustering.pca import find_n_PC
from roster_lib.utils.clustering import silhouetteW, normalized_entropy, ball_hall
from roster_lib.clustering.sphere import SphericalKMeans
from roster_lib.constants import PREPROC_DATA_PATH
from roster_lib.id_dict import pid2name, name2pid, pid2pos_bref

SCALERS = {'minmax' : MinMaxScaler, 'robust': RobustScaler, 'standard': StandardScaler}
PENALTY_RATE = 1

EVRS = [0.6, 0.8, 0.9, 0.95, 0.98]
N_CLUSTS = range(2,13)
METHODS = {
    'kmeans': KMeans,
    'spherical-kmeans': SphericalKMeans,
    'spectral': SpectralClustering,
    'agg_ward': AgglomerativeClustering,
    'agg_average': AgglomerativeClustering,
    'agg_complete': AgglomerativeClustering,
    'agg_single': AgglomerativeClustering
}
SCALINGS = ['standard', 'robust', 'minmax']
FEATURES_SELECTIONS = ['incl', 'excl', 'autoexcl']

DEFAULT_ALPHA, DEFAULT_BETA = 1, 1
METRICS_LEVELS = {
    'Silhouette': [0.25, 0.5],
    'SilhouetteW': [0.25, 0.5],
    'Entropy': [0.5, 0.8],
    'EW Silh.': [0.3, 0.4],
    'EVR-EW Silh.': [0.2, 0.3] 
}

COLORS = ['darkred', 'darkorange', 'seagreen']

class Clusterer :
    
    def __init__(self, use_positions:bool=True, load_feature_version:int = None, alpha:float = DEFAULT_ALPHA, beta:float = DEFAULT_BETA):
        self.preproc_path = PREPROC_DATA_PATH / 'clustering'
        self.last_version = max([int(f.split('_')[-1][1:-4]) for f in os.listdir(self.preproc_path) if 'clustering' in f])
        self.use_positions = use_positions
        self.load_feature_version = load_feature_version
        self.alpha = alpha
        self.beta = beta
        self.rdf = self.load_results()
        self.metrics = {
            'silhouette': {'function': silhouette_score, 'ascending' : False},
            'silhouetteW': {'function': silhouetteW, 'ascending' : False},
            'davies_bouldin': {'function': davies_bouldin_score, 'ascending' : True},
            'calinski_harabasz': {'function': calinski_harabasz_score, 'ascending' : False},
            'normalized_entropy': {'function': normalized_entropy, 'ascending' : False},            
            # 'ball_hall': {'function': self._ball_hall, 'ascending' : True},
        }
        
    def load_results(self, version:int = None):
        if version is None :
            version = self.last_version
        fp = self.preproc_path / f"ARO_clustering_v{version}.csv"
        df = pd.read_csv(fp, index_col = 0)
        df['e_w_silhouette'] = df['silhouette'] * df['entropy'] ** self.alpha
        df['evr_e_w_silhouette'] = df['e_w_silhouette'] * df['evr'] ** self.beta
        return df

    def run_comparison(self,
                       n_runs:int = 5,
                       features_selections: list = FEATURES_SELECTIONS,
                       scaling_methods: list = SCALINGS,
                       evr_targets: list = EVRS,
                       n_clusts: list = N_CLUSTS,
                       methods: list = list(METHODS.keys())):
        
        version = self.last_version +1 
        fp = self.preproc_path / f"ARO_clustering_v{version}.csv"
        n_exps = len (features_selections) * len(scaling_methods) * len(evr_targets) * len(n_clusts) * len(methods)
        n_exps_per_fs = len(scaling_methods) * len(evr_targets) * len(n_clusts) * len(methods)
        n_exps_per_scaler = len(evr_targets) * len(n_clusts) * len(methods)
        n_exps_per_evr = len(n_clusts) * len(methods)
        n_exps_per_method = len(n_clusts)

        print(f"Running clustering comparison version {version}: {n_exps} experiments consisting in {n_runs} runs each.")
        print(f"Data extraction in progress ...", end = '\r')
        feature_handler = self._manage_feature_handler()
        print(f"Data extraction COMPLETED.     ")
        self._record_features(version=version)
        
        results = []
        for m, fs in enumerate(features_selections):
            df = feature_handler.get_data(feature_selection=fs)
            print(f"Processing feature selection mode '{fs}' - n° feature considered : {df.shape[1]}")
            for i, scaling_method in enumerate(scaling_methods) :
                best_si, best_evr_ew_si, best_ew_si, best_sw = 0, 0, 0, 0  
                scaler = SCALERS[scaling_method]()
                df_scaled = scaler.fit_transform(df)
                basis_pca = PCA().fit(df_scaled)
                print(f"    Processing scaling '{scaling_method}':")
                for j, evr in enumerate(evr_targets) :
                    n_PCA = find_n_PC(basis_pca.explained_variance_ratio_, evr, display_fig= False, verbose = False)
                    _PCA = PCA(n_components=n_PCA)
                    _PCA.fit(df_scaled)
                    _X_proj = _PCA.transform(df_scaled) 
                    for k, method in enumerate(methods):
                        for l, n_clust in enumerate(n_clusts):
                            if self._check_exp_validity(method, n_clust, evr):
                                _metrics_lists = {key : [] for key in self.metrics.keys()}
                                _metrics_lists.update({'entropy': []})
                                for _ in range(n_runs):
                                    labels = self.clusterize(X_proj= _X_proj, n_clust= n_clust, method=method)
                                    _metrics = self._compute_metrics(_X_proj, labels)
                                    for key,val in _metrics_lists.items():
                                        val.append(_metrics[key])
                                _exp_metrics = {key: np.mean(val) for key,val in _metrics_lists.items()}
                                _exp_params = {'feature_selection':fs,'method': method, 'scaling': scaling_method, 'evr': evr, 'n_PC': n_PCA , 'n_clust': n_clust}
                                _exp_params.update(_exp_metrics)
                                results.append(_exp_params)
                            counter = m* n_exps_per_fs + i * n_exps_per_scaler + j * n_exps_per_evr + k * n_exps_per_method + l
                            best_si = max(best_si, _exp_metrics['silhouette'])
                            best_sw = max(best_sw, _exp_metrics['silhouetteW'])
                            best_ew_si = max(best_ew_si, _exp_metrics['silhouette'] * _exp_metrics['entropy'] ** self.alpha)
                            best_evr_ew_si = max(best_evr_ew_si, _exp_metrics['silhouette'] * evr ** self.beta * _exp_metrics['entropy'] ** self.alpha)
                            msg = f"        Processing experiment n° {counter+1:>4} of {n_exps}. Best scores: "
                            msg += f"Silhouette Index = {best_si:.3f} | "
                            msg += f"SilhouetteW = {best_sw:.3f} | "
                            msg += f"Entropy-weighted Silhouette = {best_ew_si:.3f} | "
                            msg += f"EVR-EW Silhouette = {best_evr_ew_si:.3f} | "
                            print (msg, end = '\r' if counter - (m * n_exps_per_fs + i* n_exps_per_scaler) + 1 < n_exps_per_scaler else '\n')
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
        
        feature_handler = self._manage_feature_handler()
            
        selected_df = feature_handler.get_data(feature_selection=feature_selection)
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
        _metrics = self._compute_metrics(X_proj, labels)
        if verbose :
            metrics_str = ("| ").join([f"{k}: {v:.3f} " for k,v in _metrics.items()])
            print(metrics_str)
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
            _, counts = np.unique(labels, return_counts= True)
            counts[::-1].sort()
            cluster_ids = [int(i+1) for i in range(counts.shape[0])]

            if axs == None :
                fig, axs = plt.subplots(1,4,figsize = (24,5));
            # print(X_proj.shape)
            sns.scatterplot(x = X_proj[:,0], y = X_proj[:,1], hue = labels, alpha = 0.4, palette='bright', ax=axs[0], legend=False);
            axs[0].set_xlabel("PC1");
            axs[0].set_ylabel("PC2");
            axs[0].set_title("Visual inspection of clustering along 2 first PCs");
            
            sns.barplot(data = counts, ax = axs[1]);
            axs[1].set_title("Cluster population histogram");
            axs[1].set_xlabel("Cluster");
            axs[1].set_xticks(range(len(cluster_ids)))
            axs[1].set_xticklabels(cluster_ids);
            axs[1].set_ylabel("Count");

            _metrics = self._compute_metrics(X_proj, labels)
            silhouette = _metrics['silhouette']
            silhouetteW = _metrics['silhouetteW']
            entropy = _metrics['normalized_entropy']
            entropy_silhouette = silhouette * entropy ** self.alpha
            evr_ew_silhouette = entropy_silhouette * evr ** self.beta
            metrics = ['Silhouette', 'SilhouetteW' ,'Entropy', 'EW Silh.', 'EVR-EW Silh.']
            scores = [silhouette, silhouetteW, entropy, entropy_silhouette, evr_ew_silhouette]
            colors = [COLORS[sum([score > threshold for threshold in METRICS_LEVELS[k]])] for score, k in zip(scores, metrics)]
            bar_container = axs[2].bar(metrics, scores, color = colors)
            axs[2].set(ylabel='Metric value', title='Clustering metrics', ylim=(0, 1))
            axs[2].bar_label(bar_container, fmt='{:,.3f}')
            
            cluster_df = self.feature_handler.df.copy()
            cluster_df['id'] = [int(x.split('_')[0]) for x in cluster_df.index]
            cluster_df['name'] = cluster_df['id'].map(pid2name)
            cluster_df['Position'] = cluster_df.index.map(pid2pos_bref)
            cluster_df['Cluster'] = labels 
            cluster_df['Cluster'] = cluster_df['Cluster'].map( (cluster_df['Cluster'].value_counts() + cluster_df['Cluster'].value_counts().index / 1000 ).rank(ascending=False)) # re-ordering clusters ID, and adding a small delta to diffeentiate equailities 
            sns.heatmap(cluster_df.pivot_table(index = 'Cluster', columns = ['Position'], values = 'id', aggfunc= 'count').fillna(0).astype(int)[['PG','SG','SF','PF','C']],
                annot= True, fmt = 'd', cmap = 'coolwarm', cbar=False, ax = axs[3]);
            axs[3].set_title("Cluster repartition vs positions");
            axs[3].set_yticklabels(cluster_ids)
            
            if return_data:
                return X_proj, labels
            
    def _compute_metrics(self, X, labels):
        _metrics = {k : v['function'](X, labels) for k,v in self.metrics.items()}
        return _metrics


    # Data handling        
    def plot_data(self, scalings:list = ['standard','robust','minmax'], feature_selections:list = ['incl','excl','autoexcl',None]):

        feature_handler = self._manage_feature_handler()
        fig, axs = plt.subplots(len(scalings), len(feature_selections), figsize = (5* len(feature_selections), 5* len(scalings)))
        
        for j, fs in enumerate(feature_selections):      
            _df = feature_handler.get_data(fs)
            for i, sc in enumerate(scalings):
                scaler = SCALERS[sc]()
                _df_scaled = scaler.fit_transform(_df)
                _X_proj = PCA().fit_transform(_df_scaled)
                _X_proj = pd.DataFrame(_X_proj, columns = [f"PC_{i+1}" for i in range(_X_proj.shape[1])], index= _df.index)
                _X_proj['position'] = _X_proj.index.map(pid2pos_bref)
                
                sns.scatterplot(data = _X_proj, x = 'PC_1', y = 'PC_2', alpha = 0.5, ax = axs[i,j], hue = 'position', legend= i == 0 and j ==0);
                axs[i,j].set_title(f"{sc} scaling, feature selection : {fs}");
                     
    def get_data(self, 
                scaling:str = None, 
                feature_selection:str= None, 
                perform_PCA:bool=True, 
                target_evr:float = None,
                retrieve_name:bool=False, 
                retrieve_position:bool=False, 
                new_instance:bool=False):
        
        if (not perform_PCA) and (target_evr is not None) :
            print(f"perform_PCA is set to False, ignoring parameter target_evr")
        
        feature_handler = self._manage_feature_handler(new_instance=new_instance)
        data = feature_handler.get_data(feature_selection).copy()
        _index = data.index
        _columns = data.columns
        if scaling is not None :
            sc = SCALERS[scaling]()
            data = sc.fit_transform(data)
            data = pd.DataFrame(data,index = _index, columns= _columns)
        if perform_PCA :    
            basis_pca = PCA().fit(data)
            if target_evr is not None :
                n_PCA = find_n_PC(basis_pca.explained_variance_ratio_, target_evr, display_fig= False, verbose = False)
                _PCA = PCA(n_components=n_PCA)
                data = _PCA.fit_transform(data)                
            else :
                data = basis_pca.fit_transform(data)
            data = pd.DataFrame(data, columns = [f"PC_{i+1}" for i in range(data.shape[1])], index = _index)
        if retrieve_position:
            data['position'] = data.index.map(pid2pos_bref)
        if retrieve_name:
            data.index = [pid2name[int(x.split("_")[0])] + "_" +  x.split("_")[1] for x in data.index]
        
        return data

    def get_ACP_matrix(self,
                    scaling:str = None,
                    feature_selection:str=None,
                    return_PCA:bool=False):
        
        _df = self.get_data(scaling=scaling, feature_selection=feature_selection, retrieve_name=False, retrieve_position=False, perform_PCA=False)
        pca = PCA().fit(_df)
        W = pca.components_
        W = pd.DataFrame(W.T,
                        index=_df.columns,
                        columns=[f'PC_{i}' for i in range(1, _df.shape[1]+1)])
        if return_PCA :
            return W, pca
        return W
    
    def plot_PCA(self,
                n_features:int = 20,
                scaling:str = None,
                feature_selection:str = None,
                ax=None):
        W = self.get_ACP_matrix(scaling=scaling, feature_selection=feature_selection)
        W = W[['PC_1','PC_2']]
        W['norm'] = W['PC_1'] **2 + W['PC_2']**2
        W.sort_values(by = 'norm', ascending=False, inplace=True)
        W = W.reset_index().loc[:n_features]
        if ax ==None:
            fig, ax = plt.subplots(1,1,figsize = (8,6))   
        sns.scatterplot(data = W, x = 'PC_1', y = 'PC_2', ax = ax);
        bottom_x, top_x = ax.get_xlim()  
        bottom_y, top_y = ax.get_ylim()
        ax.plot([min(bottom_x,0),max(top_x,0)],[0,0], c = 'black');
        ax.plot([0,0],[min(bottom_y,0),max(top_y,0)], c = 'black');
        for index, row in W.iterrows():
            ax.annotate(row['index'], xy = (row['PC_1'], row['PC_2']));
                 

    # Features handling
    def _manage_feature_handler(self, new_instance:bool=False):
        if hasattr(self, 'feature_handler') and not new_instance:
            feature_handler = self.feature_handler
        else : 
            feature_handler = FeatureHandler(verbose = False, 
                                            use_positions =self.use_positions, 
                                            feature_version= None if new_instance else self.load_feature_version)
            self.feature_handler = feature_handler
        return feature_handler        

    def _record_features(self, version:int = None):
        feature_handler = self._manage_feature_handler()
        features = {
            "incl": feature_handler.incl_dict,
            "excl": feature_handler.excl_dict,
            "autoexcl": feature_handler.autoexcl_dict
        }
        if version == None :
            version = self.last_version
        filepath = self.preproc_path / f'features_v{version}.json'
        with open(filepath, "w") as f:
            json.dump(features, f, indent= 4)
    
    def load_features(self, version:int = None):
        if version == None :
            version = self.last_version
        filepath = self.preproc_path / f'features_v{version}.json'
        with open(filepath, "r") as f:
            data = json.load(f)
        return data        
    
    def _check_exp_validity(self, method:str, n_clust:int, evr:float):
        # Prevents experiments considered too expensive
        if method == 'spectral':
            return False
        return True
    
    @staticmethod    
    def clusterize(X_proj, n_clust:int, method:str):
        cl_args = {'n_clusters':n_clust}
        if '_' in method :
            cl_args['linkage'] = method.split('_')[1]
        if method == 'spectral':
            cl_args['eigen_solver'] = 'amg'
            cl_args['assign_labels'] = 'discretize'            
        clustering = METHODS[method](**cl_args).fit(X_proj)
        labels = clustering.labels_
        return labels 
    
  
if __name__ == '__main__' :
    Clusterer(use_positions=False, 
                load_feature_version= None, 
                alpha=0.5,
                beta = 1).run_comparison(n_runs= 1, 
                                            scaling_methods=['standard','minmax'],
                                            methods= ['kmeans','spherical-kmeans','agg_ward','agg_average','agg_complete'],
                                            features_selections=['incl','excl'])
    
