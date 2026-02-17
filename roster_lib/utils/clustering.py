import numpy as np
from sklearn.metrics import silhouette_samples
from scipy.stats import entropy

def silhouetteW(X, labels):
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


def ball_hall(X, labels):
    """
    Compute the Ball-Hall index.
    
    Args:
        X: array-like of shape (n_samples, n_features) - The data points
        labels: array-like of shape (n_samples,) - Cluster labels for each point
    
    Returns:
        float: Ball-Hall index (negative value, higher is better, closer to 0)
    """
    
    # Get unique clusters and remap labels to contiguous indices
    # unique_labels: (n_clusters,)
    # inverse: (n_samples,) - contiguous indices [0, 1, 2, ...]
    # counts: (n_clusters,)
    unique_labels, inverse, counts = np.unique(
        labels, return_inverse=True, return_counts=True
    )
    
    n_clusters = len(unique_labels)
    n_features = X.shape[1]
    
    # Compute centroids using bincount (very fast)
    # centroids: (n_clusters, n_features)
    centroids = np.zeros((n_clusters, n_features), dtype=np.float64)
    
    for j in range(n_features):
        # Sum all feature values per cluster
        centroids[:, j] = np.bincount(inverse, weights=X[:, j], minlength=n_clusters)
    
    # Divide by counts to get means
    # counts[:, None]: (n_clusters, 1) for broadcasting
    centroids /= counts[:, None]
    
    # Get centroid for each point
    # point_centroids: (n_samples, n_features)
    point_centroids = centroids[inverse]
    
    # Compute squared distances
    # squared_distances: (n_samples,)
    squared_distances = np.sum((X - point_centroids) ** 2, axis=1)
    
    # Sum squared distances per cluster using bincount
    # wcss_per_cluster: (n_clusters,)
    wcss_per_cluster = np.bincount(inverse, weights=squared_distances, minlength=n_clusters)
    
    # Compute Ball-Hall index
    return np.sum(wcss_per_cluster / counts)


def normalized_entropy(X, labels:np.array):
    unique, counts = np.unique(labels, return_counts= True)
    freq = counts / counts.sum()
    return entropy(freq) / np.log(unique.shape[0])