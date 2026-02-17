import pandas as pd
import numpy as np
import os
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances, silhouette_score, silhouette_samples

from roster_lib.clustering.clusterer import Clusterer
from roster_lib.constants import PREPROC_DATA_PATH
from roster_lib.utils.clustering import silhouetteW

class PartitionHDBSCAN:
    
    def __init__(self, 
                min_cluster_size: int = 5,
                min_samples: int = None,
                cluster_selection_epsilon: float = 0.0,
                max_cluster_size: int = None):
        
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.max_cluster_size = max_cluster_size
        
    @staticmethod
    def _create_centroids(X, labels):
        centroids = np.zeros(((np.unique(labels) > -1).sum(), X.shape[1]))
        for label in np.unique(labels):
            if label != -1 :
                centroids[label, :] = X.values[labels == label].mean(axis = 0)         
        return centroids
        
    @staticmethod
    def _allocate_to_centroids(X, labels, centroids):
        if X[labels == -1].shape[0] > 0 :
            final_labels = np.copy(labels)
            pdist = pairwise_distances(X[labels == -1], centroids)
            closest_cluster = np.argmin(pdist, axis = 1)
            final_labels[labels == -1] = closest_cluster
            return final_labels
    
    def fit(self, X, verbose:bool = True):
        self.data = X
        clusterer = HDBSCAN(min_cluster_size = self.min_cluster_size,
                            min_samples = self.min_samples,
                            cluster_selection_epsilon = self.cluster_selection_epsilon,
                            max_cluster_size = self.max_cluster_size)
        
        clusterer.fit(X)
        labels = clusterer.labels_
        self.hdbscan_labels_ = labels
        self.hdbscan_allocated_ = labels != -1
        if (labels == -1).sum() == X.shape[0]:
            self.labels_ = labels
            self.n_clusters_ = 0
            self.silhouette_score = -1
            self.silhouetteW_score = -1
            if verbose :
                print("Clustering failed, no clusters were identified.")
        else :
            if (labels == -1).sum() == 0 :
                self.labels_ = labels
                final_labels = labels
            else :     
                centroids = self._create_centroids(X, labels)
                final_labels = self._allocate_to_centroids(X, labels, centroids)
                self.labels_ = final_labels
            n_clust = np.unique(final_labels).shape[0]
            self.n_clusters_ = n_clust
            self.silhouette_score = silhouette_score(X, final_labels) if n_clust > 1 else -1
            self.silhouetteW_score = silhouetteW(X, final_labels) if n_clust > 1 else -1
        
        
class P_HDB_GridSearch:
    
    def __init__(self,
                ref_metric:str = 'silhouette',
                scalings:list = ['standard'],
                feature_selection: list = [None],
                target_evrs:list = [1],
                min_cluster_sizes: list = [5],
                min_samples: list = [None],
                cluster_selection_epsilons:list = [0],
                max_cluster_sizes:list = [None],
                clusterer_alpha:float = 0.5,
                clusterer_beta:float = 0.5,
                use_positions:bool = False):
        
            self.ref_metric = ref_metric 
            self.scalings = scalings
            self.feature_selection = feature_selection
            self.target_evrs = target_evrs
            self.min_cluster_sizes = min_cluster_sizes
            self.min_samples = min_samples 
            self.cluster_selection_epsilons = cluster_selection_epsilons 
            self.max_cluster_sizes = max_cluster_sizes 
            n_scalings = len(scalings)
            n_fs = len(self.feature_selection) 
            n_te = len(target_evrs)
            n_min_sizes = len(min_cluster_sizes) 
            n_min_samples = len(min_samples)
            n_cluster_eps = len(cluster_selection_epsilons)
            n_max_sizes = len(max_cluster_sizes)
            self.n_exps = n_scalings * n_fs * n_te * n_min_sizes * n_min_samples * n_cluster_eps * n_max_sizes
            self.preproc_path = PREPROC_DATA_PATH / 'clustering' 
            self.version = max([int(f.split('_')[-1][1:-4]) for f in os.listdir(self.preproc_path) if 'partition_hdbscan' in f]) +1 
            self.filepath = self.preproc_path / f'partition_hdbscan_v{self.version}.csv'
            
            self.clusterer = Clusterer(alpha= clusterer_alpha, beta = clusterer_beta, use_positions=use_positions)
            
    def fit(self, verbose:bool = True):
        
        _results = []
        best_score = 0
        counter = 0
        for fs in self.feature_selection:
            for sc in self.scalings:
                for evr in self.target_evrs:
                    base_X = self.clusterer.get_data(scaling=sc, feature_selection=fs , perform_PCA=True, target_evr= evr, retrieve_name=False, retrieve_position=False)
                    for mcs in self.min_cluster_sizes:
                        for mss in self.min_samples:
                            for cse in self.cluster_selection_epsilons:
                                for mx in self.max_cluster_sizes:
                                    counter += 1 
                                    if verbose :
                                        print(f"Processing exp n° {counter:>4} out of {self.n_exps:>4} | Best {self.ref_metric} reached = {best_score:.3f}", end = '\r' if counter < self.n_exps else '\n')
                                    X = base_X.copy()
                                    p_hdb = PartitionHDBSCAN(min_cluster_size=mcs, 
                                                        min_samples=mss, 
                                                        cluster_selection_epsilon =cse,
                                                        max_cluster_size = mx,
                                                        )
                                    p_hdb.fit(X, verbose = False)
                                    exp_details = {'feature_selection':fs,
                                                    'scaling': sc,
                                                    'evr': evr, 
                                                    'min_cluster_size':mcs, 
                                                    'min_samples':mss, 
                                                    'cluster_selection_epsilon':cse, 
                                                    'max_cluster_size':mx, 
                                                    'n_clust': p_hdb.n_clusters_}
                                    try : 
                                        exp_metrics = self.clusterer._compute_metrics(X, p_hdb.labels_)
                                    except :
                                        exp_metrics = {'silhouette': p_hdb.silhouette_score, 'silhouetteW':p_hdb.silhouetteW_score}
                                    exp_details.update(exp_metrics)
                                    _results.append(exp_details)
                                    best_score = max(best_score, exp_details[self.ref_metric])
        results_df = pd.DataFrame(_results).fillna(0) # na corresponds to the except above, in case metrics are not computed, or None values of parameters
        results_df.to_csv(self.filepath)
        return results_df
        
if __name__ == '__main__':
    grid = P_HDB_GridSearch(
        scalings= ['minmax', 'standard'],
        feature_selection= ['incl','excl', None],
        target_evrs= [0.6, 0.8, 0.9, 0.95, 0.98],
        min_cluster_sizes= [4, 6, 8, 10, 12 ,14, 16, 18, 20, 30, 40, 50],
        min_samples= list(range(3,16)),
        cluster_selection_epsilons= [0],
        max_cluster_sizes= [None]
    )
    grid.fit(verbose=True)

                                        