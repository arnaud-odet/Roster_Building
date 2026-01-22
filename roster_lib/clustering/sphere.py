from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np


class SphericalKMeans :
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def fit(self, X):
                
        # Normalize data to unit sphere (L2 normalization)
        X_normalized = normalize(X, norm='l2', axis=1)  # Shape: (100, 50)

        # Standard K-Means on normalized data = Spherical K-Means
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',  # Works well for spherical data
            n_init=10,
            max_iter=300,
            random_state=42
        )

        labels = kmeans.fit_predict(X_normalized)

        # IMPORTANT: Normalize cluster centers too
        self.cluster_centers_ = normalize(kmeans.cluster_centers_, norm='l2', axis=1)
        self.labels_ = labels
        return self