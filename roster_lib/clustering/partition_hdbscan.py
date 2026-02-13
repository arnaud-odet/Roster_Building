import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances, silhouette_score, silhouette_samples

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
    
    @staticmethod
    def _silhouetteW(X, labels):
        """
        Compute weighted silhouette score (SilhouetteW metric).
        Args:
            X: array-like of shape (n_samples, n_features) - The data
            labels: array-like of shape (n_samples,) - Cluster labels
        
        Returns:
            float: SilhouetteW score
        """
        ssa = silhouette_samples(X, labels)
        # Get unique clusters and their counts
        # unique_labels: (n_clusters,), inverse: (n_samples,), counts: (n_clusters,)
        unique_labels, inverse_indices, counts = np.unique(
            labels, return_inverse=True, return_counts=True
        )
        
        nns = np.sum(counts > 1)
        if nns == 0:
            return 0.0
        
        populations = counts[inverse_indices]
        return np.sum(ssa / populations) / nns
    
    def fit(self, X):
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
            print("Clustering failed, no clusters were identified.")
        else :
            if (labels == -1).sum() == 0 :
                self.labels_ = labels
            else :     
                centroids = self._create_centroids(X, labels)
                final_labels = self._allocate_to_centroids(X, labels, centroids)
                self.labels_ = final_labels
            n_clust = np.unique(final_labels).shape[0]
            self.n_clusters_ = n_clust
            self.silhouette_score = silhouette_score(X, final_labels) if n_clust > 1 else -1
            self.silhouetteW_score = self._silhouetteW(X, final_labels) if n_clust > 1 else -1
        
        
        
         