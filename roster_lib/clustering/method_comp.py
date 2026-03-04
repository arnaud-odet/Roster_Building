from roster_lib.clustering.clusterer import Clusterer
from roster_lib.clustering.partition_hdbscan import P_HDB_GridSearch
from roster_lib.clustering.gmm import GMM_Custom_GridSearch

FS = ['excl']
SC = ['standard','minmax','robust']
EVRS = [0.8, 0.9, 0.95, 0.98]
N_CLS = list(range(2,11))

for time_norm in [True, False]:
    for min_minute_per_g in [0,4,8]:
        print("===============================================================================")
        print(f"======== Processing Agg & K-Means with {min_minute_per_g} min minutes, time_norm = {time_norm} ========")
        print("===============================================================================")
        Clusterer(use_positions= False, 
                time_norm= time_norm).run_comparison(n_runs=1,
                                                        scaling_methods= SC,
                                                        methods = ['kmeans','spherical-kmeans','agg_ward','agg_average','agg_complete'],
                                                        evr_targets= EVRS,
                                                        n_clusts= N_CLS,
                                                        features_selections= FS)
        print("===============================================================================")
        print(f"======== Processing Part HDBSCAN with {min_minute_per_g} min minutes, time_norm = {time_norm} ========")
        print("===============================================================================")
        grid = P_HDB_GridSearch(
            time_norm= time_norm, 
            scalings= SC,
            feature_selection= FS,
            target_evrs= EVRS,
            min_cluster_sizes= [4, 7, 10, 15, 20, 40],
            min_samples= [3,5,7,10,15],
            cluster_selection_epsilons= [0.2, 0.5, 1, 2],
            max_cluster_sizes= [None,400, 800]
            )
        grid.fit(verbose=True)
        print("===============================================================================")
        print(f"======== Processing Gaussian MM with {min_minute_per_g} min minutes, time_norm = {time_norm} ========")
        print("===============================================================================")        
        gs_gmm = GMM_Custom_GridSearch(
            scalings= SC,
            feature_selection= FS,
            target_evrs= EVRS,
            n_components= N_CLS,
            time_norm= time_norm,
            covariance_types= ["spherical", "tied", "full","diag"]        
            ).fit()