import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.preprocessing import StandardScaler


class OptimalKFinder:
    """
    A class to find the optimal number of clusters using multiple methods.
    
    Args:
        k_range (range): Range of k values to test (default: range(2, 11))
        random_state (int): Random state for reproducibility (default: 42)
    """
    
    def __init__(self, k_range=range(2, 30), random_state=42):
        self.k_range = k_range
        self.random_state = random_state
        self.scores = {
            "inertia": [],           # Lower is better
            "silhouette": [],        # Higher is better  [-1, 1]
            "davies_bouldin": [],    # Lower is better   [0, +inf]
            "calinski_harabasz": []  # Higher is better  [0, +inf]
        }
        self.gap_scores = []
        self.gap_stds = []

    # -------------------------------------------------------------------------
    def fit(self, X):
        """
        Compute all clustering scores for each k in k_range.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
        """
        print("Computing clustering scores...")
        
        for k in self.k_range:
            # Fit KMeans for current k
            # X shape: (n_samples, n_features)
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)  # labels shape: (n_samples,)
            
            # --- Elbow Method (Inertia) ---
            # Sum of squared distances to closest cluster center
            self.scores["inertia"].append(kmeans.inertia_)
            
            # --- Silhouette Score ---
            # Measures how similar a point is to its own cluster vs other clusters
            self.scores["silhouette"].append(silhouette_score(X, labels))
            
            # --- Davies-Bouldin Score ---
            # Measures the average similarity between clusters
            self.scores["davies_bouldin"].append(davies_bouldin_score(X, labels))
            
            # --- Calinski-Harabasz Score ---
            # Ratio of between-cluster dispersion to within-cluster dispersion
            self.scores["calinski_harabasz"].append(calinski_harabasz_score(X, labels))
            
            print(f"  k={k} ✓")
        
        return self

    # -------------------------------------------------------------------------
    def fit_gap_statistic(self, X, n_refs=10):
        """
        Compute the Gap Statistic for each k.
        Compares inertia of the clustering to inertia of random data.
        
        Args:
            X         (np.ndarray): Input data of shape (n_samples, n_features)
            n_refs    (int)       : Number of random reference datasets (default: 10)
        """
        print("Computing Gap Statistic (this may take a while)...")
        
        # Get the bounding box of the data
        # X shape: (n_samples, n_features)
        data_min = X.min(axis=0)  # shape: (n_features,)
        data_max = X.max(axis=0)  # shape: (n_features,)
        
        for k in self.k_range:
            # --- Compute inertia on actual data ---
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            actual_inertia = np.log(kmeans.inertia_)  # scalar
            
            # --- Compute inertia on random reference datasets ---
            ref_inertias = []
            for _ in range(n_refs):
                # Generate random data with same shape and bounds as X
                # random_data shape: (n_samples, n_features)
                random_data = np.random.uniform(
                    low=data_min,
                    high=data_max,
                    size=X.shape
                )
                ref_kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                ref_kmeans.fit(random_data)
                ref_inertias.append(np.log(ref_kmeans.inertia_))
            
            # ref_inertias shape: (n_refs,)
            ref_inertias = np.array(ref_inertias)
            
            # Gap = mean(ref_inertias) - actual_inertia
            gap = ref_inertias.mean() - actual_inertia  # scalar
            std = ref_inertias.std() * np.sqrt(1 + 1 / n_refs)  # scalar (corrected std)
            
            self.gap_scores.append(gap)
            self.gap_stds.append(std)
            
            print(f"  k={k} | Gap={gap:.4f} ± {std:.4f} ✓")
        
        return self

    # -------------------------------------------------------------------------
    def plot(self):
        """
        Plot all computed scores in a grid layout.
        """
        k_values = list(self.k_range)
        has_gap = len(self.gap_scores) > 0
        
        n_plots = 4 + (1 if has_gap else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        fig.suptitle("Optimal Number of Clusters - Diagnostics", fontsize=14, fontweight="bold")
        
        # --- Plot 1: Elbow Method ---
        axes[0].plot(k_values, self.scores["inertia"], "bo-")
        axes[0].set_title("Elbow Method (Inertia)")
        axes[0].set_xlabel("Number of Clusters (k)")
        axes[0].set_ylabel("Inertia")
        axes[0].grid(True)
        
        # --- Plot 2: Silhouette Score ---
        best_k_sil = k_values[np.argmax(self.scores["silhouette"])]
        axes[1].plot(k_values, self.scores["silhouette"], "go-")
        axes[1].axvline(x=best_k_sil, color="r", linestyle="--", label=f"Best k={best_k_sil}")
        axes[1].set_title("Silhouette Score")
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Score")
        axes[1].legend()
        axes[1].grid(True)
        
        # --- Plot 3: Davies-Bouldin Score ---
        best_k_db = k_values[np.argmin(self.scores["davies_bouldin"])]
        axes[2].plot(k_values, self.scores["davies_bouldin"], "mo-")
        axes[2].axvline(x=best_k_db, color="r", linestyle="--", label=f"Best k={best_k_db}")
        axes[2].set_title("Davies-Bouldin Score")
        axes[2].set_xlabel("Number of Clusters (k)")
        axes[2].set_ylabel("Score")
        axes[2].legend()
        axes[2].grid(True)
        
        # --- Plot 4: Calinski-Harabasz Score ---
        best_k_ch = k_values[np.argmax(self.scores["calinski_harabasz"])]
        axes[3].plot(k_values, self.scores["calinski_harabasz"], "co-")
        axes[3].axvline(x=best_k_ch, color="r", linestyle="--", label=f"Best k={best_k_ch}")
        axes[3].set_title("Calinski-Harabasz Score")
        axes[3].set_xlabel("Number of Clusters (k)")
        axes[3].set_ylabel("Score")
        axes[3].legend()
        axes[3].grid(True)
        
        # --- Plot 5: Gap Statistic (if computed) ---
        if has_gap:
            best_k_gap = k_values[np.argmax(self.gap_scores)]
            axes[4].plot(k_values, self.gap_scores, "ro-", label="Gap")
            axes[4].fill_between(
                k_values,
                np.array(self.gap_scores) - np.array(self.gap_stds),
                np.array(self.gap_scores) + np.array(self.gap_stds),
                alpha=0.2, color="r"
            )
            axes[4].axvline(x=best_k_gap, color="b", linestyle="--", label=f"Best k={best_k_gap}")
            axes[4].set_title("Gap Statistic")
            axes[4].set_xlabel("Number of Clusters (k)")
            axes[4].set_ylabel("Gap")
            axes[4].legend()
            axes[4].grid(True)
        
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    def summary(self):
        """
        Print a summary of the best k for each method.
        """
        k_values = list(self.k_range)
        
        print("\n" + "="*45)
        print("         OPTIMAL K SUMMARY")
        print("="*45)
        print(f"  {'Method':<25} {'Best k':>10}")
        print("-"*45)
        print(f"  {'Elbow (visual)':<25} {'N/A':>10}")
        print(f"  {'Silhouette ↑':<25} {k_values[np.argmax(self.scores['silhouette'])]:>10}")
        print(f"  {'Davies-Bouldin ↓':<25} {k_values[np.argmin(self.scores['davies_bouldin'])]:>10}")
        print(f"  {'Calinski-Harabasz ↑':<25} {k_values[np.argmax(self.scores['calinski_harabasz'])]:>10}")
        
        if self.gap_scores:
            print(f"  {'Gap Statistic ↑':<25} {k_values[np.argmax(self.gap_scores)]:>10}")
        
        print("="*45)