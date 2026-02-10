import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from sklearn.cluster import DBSCAN

class DCIC:
    
    def __init__(self, 
                eps_start:float,
                eps_max:float = None, 
                n_samples:int = 5, 
                eps_factor:float=2,
                max_iter:int = 1000):
        self.n_samples = n_samples
        self.factor = eps_factor
        self.max_iter = max_iter
        self.iter_count = 0
        self.start_eps = eps_start
        self.eps = eps_start
        self.limit = eps_max
    
    def _distance_based_allocation(self, _cluster_df):
        _new_iter_clust = _cluster_df.copy()
        _existing_clusters = _new_iter_clust[_new_iter_clust[self.last_iter]>-1][self.last_iter].unique()
        for cl in _existing_clusters:
            _distances = np.sum(np.array([(_new_iter_clust[col] - _new_iter_clust.loc[cl,col])**2 for col in self.fit_columns]),axis=0)
            _new_iter_clust[cl] = _distances
        _new_iter_clust['allocation'] = _existing_clusters[np.argmin(_new_iter_clust[_existing_clusters], axis=1)]
        _new_iter_clust
        return _new_iter_clust, _existing_clusters
    
    @staticmethod
    def _labels_manager(index ,old_label:int, new_label:int, direct_mapping:dict, distant_mapping:dict):
        if old_label > -1 :
            return old_label
        if new_label in direct_mapping.keys():
            return direct_mapping[new_label]
        if new_label in distant_mapping.keys():
            return distant_mapping[new_label].loc[index,'allocation']
        else :
            return -1

    def _keep_iterating(self):
        if self.iter_count == 0:
            return True        
        if self.clustering[self.clustering[self.new_iter] == -1].shape[0] ==0:
            return False
        if self.limit is not None :
            return self.eps < self.limit and self.iter_count < self.max_iter
        else :
            return self.iter_count < self.max_iter

    def _prepare_data(self, X):    
        X = pd.DataFrame(X) if type(X) != pd.core.frame.DataFrame else X.reset_index(drop=True).copy() 
        self.input_data = X.copy()
        self.cluster_history = {}
        self.fit_columns = list(X.columns)  
        X['cl_iter_0'] = -1
        self.clustering = X.copy()
        return X   
    
    def _record_state(self):      
        self.n_clust = (self.clustering[self.new_iter].unique() > -1).sum()
        self.n_singles = self.clustering[self.clustering[self.new_iter] == -1].shape[0]
    
    def _display_state(self):
        print(f"Iteration n° {self.iter_count}, distance = {self.eps:.2e} | n non-singletons clusters : {self.n_clust}, n singletons : {self.n_singles}")   
        
    def _assign_labels(self, X, labels):
        X[self.new_iter] = labels
        return X

    # Revoir pour tenir compte des clusters iter-1
    def _manage_ids(self, X):
        max_label = X[self.new_iter].max()
        X_old_clusters = X[X[self.last_iter]>-1].copy()
        allocated_dict = {index : int(row[self.new_iter]) for index, row in X_old_clusters.iterrows()}
        X_new_clusters = X[(X[self.new_iter]>-1)&(X[self.last_iter]==-1)].copy()
        new_allocated_dict = {index : int(row[self.new_iter]) for index, row in X_new_clusters.iterrows()}
        allocated_dict.update(new_allocated_dict)
        X_single = X[(X[self.last_iter]==-1)&(X[self.new_iter] ==-1)].copy()
        unallocated_dict = {index : int(max_label + 1 + i) for i, index in enumerate(X_single.index)}
        allocated_dict.update(unallocated_dict)
        return allocated_dict

    def _create_new_input_data(self):
        X = self.clustering.copy()
        X_cluster = X[X[self.new_iter]>-1].copy()
        X_cluster = X_cluster[self.fit_columns + [self.new_iter]].groupby(self.new_iter).mean()
        X_cluster.reset_index(inplace=True)
        X_single = X[X[self.new_iter] ==-1][self.fit_columns + [self.new_iter]].copy()
        X_post = pd.concat((X_cluster, X_single), axis =0).reset_index(drop=True)
        return X_post

    def _manage_clusters(self, X, verbose :bool=True):
        next_clust_id = int(X[self.last_iter].max()+1)
        simple_mapping = {}
        distance_based_mapping = {}
        for lab in X[X[self.new_iter]>-1][self.new_iter].unique():
            _new_iter_clust = X[X[self.new_iter]== lab]
            if _new_iter_clust[[self.last_iter]].value_counts().loc[0:].shape[0] == 0 :
                if verbose :
                    print(f"    Cluster iter_{self.iter_count}:{lab} only contains previoulsy unclustered points allocated in new cluster iter_{self.iter_count}:{next_clust_id}")
                simple_mapping[lab] = next_clust_id
                self.cluster_history[next_clust_id]= self.iter_count
                next_clust_id+=1
            elif _new_iter_clust[[self.last_iter]].value_counts().loc[0:].shape[0] == 1 :
                old_clust = _new_iter_clust[[self.last_iter]].value_counts().loc[0:].index[0][0]
                if verbose:
                    print(f"    Cluster iter_{self.iter_count}:{lab} comprises previoulsy identified cluster iter_{self.cluster_history[old_clust]}:{old_clust}'s centroid and uclustered points. Merged with previous cluster iter_{self.cluster_history[old_clust]}:{old_clust}")
                simple_mapping[lab] = old_clust
            else :
                dba, existing_cl_list = self._distance_based_allocation(_new_iter_clust)
                if verbose :
                    print(f"    Cluster iter_{self.iter_count}:{lab} comprises centroids of previously allocated clusters {existing_cl_list}, using distance based allocation")        
                distance_based_mapping[lab] = dba
        X['restated'] = X.apply(lambda row : self._labels_manager(row.name, old_label= row[self.last_iter], new_label= row[self.new_iter], direct_mapping= simple_mapping, distant_mapping= distance_based_mapping), axis = 1)
        X[self.new_iter] = X['restated']
        return X.drop(columns = 'restated')
    
    def _assign_clusters(self, X, allocation_dict):
        last_iter_id = f"id_{self.iter_count-1}"
        new_iter_id = f"id_{self.iter_count}"
        if self.iter_count ==1:
            self.clustering[last_iter_id] = self.clustering.index  
        self.clustering = self.clustering.merge(X[self.new_iter], left_on = last_iter_id, right_index=True, how = 'left')
        self.clustering[new_iter_id] = self.clustering[last_iter_id].map(allocation_dict)
                          
    def _step(self, X, verbose:bool = True):
        self.last_iter = f"cl_iter_{self.iter_count}"
        self.iter_count += 1
        self.new_iter = f"cl_iter_{self.iter_count}"        
        dbscan = DBSCAN(self.eps,min_samples=self.n_samples).fit(X[self.fit_columns])
        labels = dbscan.labels_.copy()
        X = self._assign_labels(X, labels)
        X = self._manage_clusters(X, verbose = verbose) 
        allocation_dict = self._manage_ids(X) 
        self._assign_clusters(X, allocation_dict= allocation_dict)
        X = self._create_new_input_data()
        self._record_state()
        if verbose :
            self._display_state()
        self.eps = self.eps * self.factor         
        return X
    
    def fit(self, X, verbose:bool = True):
        X = self._prepare_data(X)
        while self._keep_iterating() :
            X = self._step(X, verbose = verbose)
            self.labels_ = self.clustering[self.new_iter].values
            
        
        
        
        
        
        