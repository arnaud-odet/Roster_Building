from roster_lib.clustering.clusterer import Clusterer
from roster_lib.clustering.partition_hdbscan import P_HDB_GridSearch
from roster_lib.clustering.gmm import GMM_Custom_GridSearch

FS = ['incl','excl', None, 'autoexcl']
SC = ['standard','minmax','robust']
EVRS = [0.6, 0.8, 0.9, 0.95, 0.98]
N_CLS = list(range(2,11))

for time_norm in [True, False]:
    for min_minute_per_g in [4,8]:
        print("===============================================================================")
        print(f"==== Processing Agg & K-Means with {min_minute_per_g} min minutes, time_norm = {time_norm} ====")
        print("===============================================================================")
        Clusterer(use_positions= False, 
                time_norm= time_norm, 
                load_feature_version= 13).run_comparison(n_runs=1,
                                                        scaling_methods= SC,
                                                        methods = ['kmeans','spherical-kmeans','agg_ward','agg_average','agg_complete'],
                                                        evr_targets= EVRS,
                                                        n_clusts= N_CLS,
                                                        features_selections= FS)
        print("===============================================================================")
        print(f"==== Processing Part HDBSCAN with {min_minute_per_g} min minutes, time_norm = {time_norm} ====")
        print("===============================================================================")
        grid = P_HDB_GridSearch(
            time_norm= time_norm, 
            scalings= SC,
            feature_selection= FS,
            target_evrs= EVRS,
            min_cluster_sizes= [4, 6, 8, 10, 12, 16, 20, 30, 40],
            min_samples= list(range(3,16)),
            cluster_selection_epsilons= [0],
            max_cluster_sizes= [None,400]
        )
        grid.fit(verbose=True)
        print("===============================================================================")
        print(f"==== Processing Gaussian MM with {min_minute_per_g} min minutes, time_norm = {time_norm} ====")
        print("===============================================================================")        
        gs_gmm = GMM_Custom_GridSearch(
        scalings= SC,
        feature_selection= FS,
        target_evrs= EVRS,
        n_components= N_CLS,
        time_norm= time_norm,
        covariance_types= ["spherical", "tied", "full","diag"]        
        ).fit()