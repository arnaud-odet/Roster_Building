import pandas as pd
import numpy as np
import os
from roster_lib.clustering.clusterer import Clusterer
from roster_lib.constants import PREPROC_DATA_PATH
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

class GMM_Custom_GridSearch:
    
    def __init__(self,
                ref_metric:str = 'silhouette',
                scalings:list = ['standard'],
                feature_selection: list = [None],
                target_evrs:list = [1],
                n_components:list = [5],
                covariance_types:list = ['diag'],
                clusterer_alpha:float = 0.5,
                clusterer_beta:float = 0.5,
                use_positions:bool = False):
        
            self.ref_metric = ref_metric 
            self.scalings = scalings
            self.feature_selection = feature_selection
            self.target_evrs = target_evrs
            self.n_components = n_components
            self.covariance_types= covariance_types
            n_scalings = len(scalings)
            n_fs = len(self.feature_selection) 
            n_te = len(target_evrs)
            n_nc = len(n_components)
            n_ct = len(covariance_types)
            self.n_exps = n_scalings * n_fs * n_te * n_nc * n_ct
            self.preproc_path = PREPROC_DATA_PATH / 'clustering' 
            try :
                self.version = max([int(f.split('_')[-1][1:-4]) for f in os.listdir(self.preproc_path) if 'gmm' in f]) +1 
            except :
                self.version = 0
            self.filepath = self.preproc_path / f'gmm_v{self.version}.csv'
            
            self.clusterer = Clusterer(alpha= clusterer_alpha, beta = clusterer_beta, use_positions=use_positions)
            
    def fit(self, verbose:bool = True):
        
        _results = []
        best_score = 0
        counter = 0
        for fs in self.feature_selection:
            for sc in self.scalings:
                for evr in self.target_evrs:
                    X = self.clusterer.get_data(scaling=sc, feature_selection=fs , perform_PCA=True, target_evr= evr, retrieve_name=False, retrieve_position=False)
                    for nc in self.n_components :
                        for ct in self.covariance_types :
                            counter += 1 
                            if verbose :
                                print(f"Processing exp n° {counter:>4} out of {self.n_exps:>4} | Best {self.ref_metric} reached = {best_score:.3f}", end = '\r' if counter < self.n_exps else '\n')
                            model = GaussianMixture(n_components= nc, covariance_type= ct).fit(X)
                            labels = model.predict(X)
                            exp_details = {'feature_selection':fs,
                                            'scaling': sc,
                                            'evr': evr, 
                                            'n_components':nc, 
                                            'covariance_type':ct,  
                                            'n_clust': np.unique(labels).shape[0]}
                            exp_metrics = self.clusterer._compute_metrics(X, labels)
                            exp_details.update(exp_metrics)
                            _results.append(exp_details)
                            best_score = max(best_score, exp_details[self.ref_metric])
        results_df = pd.DataFrame(_results).fillna(0)
        results_df.to_csv(self.filepath)
        return results_df
    
if __name__ == '__main__':
    gs_gmm = GMM_Custom_GridSearch(
        scalings= ['minmax', 'standard'],
        feature_selection= ['incl','excl', None],
        target_evrs= [0.6, 0.8, 0.9, 0.95, 0.98],
        n_components= list(range(2,11)),
        covariance_types= ["spherical", "tied", "full","diag"]        
    ).fit()