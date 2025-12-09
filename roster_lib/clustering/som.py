import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import silhouette_score
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class SelfOrganizingMap(nn.Module):
    """
    GPU-accelerated Self-Organizing Map (SOM) implementation using PyTorch
    with silhouette score tracking during training
    """
    
    def __init__(self, map_size: Tuple[int, int], input_dim: int, 
                 learning_rate: float = 0.1, sigma: float = 1.0, 
                 decay_function: str = 'exponential', device: str = 'cuda'):
        """
        Initialize GPU-accelerated SOM
        
        Args:
            map_size: (height, width) of the output map
            input_dim: dimension of input vectors
            learning_rate: initial learning rate
            sigma: initial neighborhood radius
            decay_function: decay function for learning rate and sigma
            device: 'cuda' or 'cpu'
        """
        super(SelfOrganizingMap, self).__init__()
        
        self.map_size = map_size
        self.input_dim = input_dim
        self.initial_learning_rate = learning_rate
        self.initial_sigma = sigma
        self.decay_function = decay_function
        
        # Set device (GPU/CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize weight vectors as learnable parameters
        # Shape: (map_height * map_width, input_dim)
        self.n_neurons = map_size[0] * map_size[1]
        self.weights = nn.Parameter(
            torch.randn(self.n_neurons, input_dim, device=self.device) * 0.1
        )
        
        # Initialize bias terms to handle dead neurons
        self.bias = nn.Parameter(
            torch.zeros(self.n_neurons, device=self.device)
        )
        
        # Create coordinate grid for neurons (fixed, not learnable)
        self.register_buffer('neuron_coordinates', 
                           self._create_neuron_coordinates())
        
        # Training statistics including silhouette scores
        self.training_silhouette = []
        
    def _create_neuron_coordinates(self) -> torch.Tensor:
        """
        Create coordinate grid for neurons
        
        Returns:
            coordinates tensor (n_neurons, 2) on device
        """
        coords = []
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                coords.append([i, j])
        return torch.tensor(coords, dtype=torch.float32, device=self.device)
    
    def _euclidean_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate Euclidean distances between single input and all neurons
        
        Args:
            x: single input vector (input_dim,)
            
        Returns:
            distance vector (n_neurons,)
        """
        # x shape: (input_dim,)
        # self.weights shape: (n_neurons, input_dim)
        # Output shape: (n_neurons,)
        
        # Calculate squared distances
        squared_distances = torch.sum((self.weights - x.unsqueeze(0)) ** 2, dim=1)
        
        # Apply bias terms (add to distances to reduce winning probability)
        biased_distances = squared_distances + self.bias
        
        return torch.sqrt(biased_distances + 1e-8)  # Add small epsilon for numerical stability
    
    def _find_winner(self, x: torch.Tensor) -> int:
        """
        Find Best Matching Unit (BMU) for a single input
        
        Args:
            x: single input vector (input_dim,)
            
        Returns:
            winner index (scalar)
        """
        # Calculate distances: (n_neurons,)
        distances = self._euclidean_distance(x)
        
        # Find winner
        winner = torch.argmin(distances)
        
        return int(winner)
    
    def _neighborhood_function(self, winner_idx: int, sigma: float) -> torch.Tensor:
        """
        Calculate neighborhood function for a single winner
        
        Args:
            winner_idx: winner neuron index
            sigma: neighborhood radius
            
        Returns:
            neighborhood influence (n_neurons,)
        """
        # Get winner coordinates: (2,)
        winner_coords = self.neuron_coordinates[winner_idx]
        
        # Calculate distances between winner position and all neurons
        # winner_coords shape: (2,)
        # neuron_coordinates shape: (n_neurons, 2)
        
        # Calculate squared distances in coordinate space
        coord_distances = torch.sum(
            (self.neuron_coordinates - winner_coords.unsqueeze(0)) ** 2, 
            dim=1
        )  # (n_neurons,)
        
        # Apply Gaussian neighborhood function
        neighborhood = torch.exp(-coord_distances / (2 * sigma ** 2))
        
        return neighborhood
    
    def _update_weights(self, x: torch.Tensor, winner_idx: int,
                       learning_rate: float, sigma: float):
        """
        Update weights using single-sample SOM learning rule
        
        Args:
            x: single input vector (input_dim,)
            winner_idx: winner neuron index
            learning_rate: current learning rate
            sigma: current neighborhood radius
        """
        # Calculate neighborhood influences: (n_neurons,)
        neighborhood = self._neighborhood_function(winner_idx, sigma)
        
        # Calculate weight updates
        # x shape: (input_dim,)
        # self.weights shape: (n_neurons, input_dim)
        
        # Calculate error vectors: (n_neurons, input_dim)
        error_vectors = x.unsqueeze(0) - self.weights
        
        # Apply neighborhood weighting: (n_neurons, 1)
        neighborhood_expanded = neighborhood.unsqueeze(1)
        
        # Calculate weighted updates: (n_neurons, input_dim)
        weighted_updates = learning_rate * neighborhood_expanded * error_vectors
        
        # Apply updates to weights
        with torch.no_grad():
            self.weights.data += weighted_updates
    
    def _update_bias(self, winner_idx: int, bias_increment: float = 0.01, 
                    bias_decay: float = 0.99):
        """
        Update bias terms for single winner
        
        Args:
            winner_idx: winner neuron index
            bias_increment: increment for winning neuron bias
            bias_decay: decay factor for all bias terms
        """
        with torch.no_grad():
            # Increase bias for winning neuron
            self.bias.data[winner_idx] += bias_increment
            
            # Decay all bias terms
            self.bias.data *= bias_decay
    
    def _decay_parameters(self, epoch: int, total_epochs: int) -> Tuple[float, float]:
        """
        Decay learning rate and sigma over time
        
        Args:
            epoch: current epoch
            total_epochs: total number of epochs
            
        Returns:
            (current_learning_rate, current_sigma)
        """
        if self.decay_function == 'exponential':
            decay_factor = torch.exp(torch.tensor(-epoch / total_epochs))
        else:
            decay_factor = 1.0 - (epoch / total_epochs)
        
        learning_rate = self.initial_learning_rate * decay_factor
        sigma = self.initial_sigma * decay_factor
        
        return float(learning_rate), float(sigma)
    
    def predict(self, X):
        """
        Predict cluster assignments for input data
        
        Args:
            X: input data (n_samples, input_dim) - can be numpy array or torch tensor
            
        Returns:
            cluster assignments (n_samples,) as torch tensor
        """
        
        # Convert to tensor if needed and ensure it's on correct device
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        else:
            X = X.to(self.device)
        
        # Set to evaluation mode (this returns None, don't chain it!)
        # self.eval()

        predictions = torch.zeros(X.shape[0], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            for i in range(X.shape[0]):
                x = X[i]  # Shape: (input_dim,)
                winner_idx = self._find_winner(x)
                predictions[i] = winner_idx
        
        return predictions


    def train(self, X, epochs: int = 1000, verbose: bool = True):
        """
        Train the SOM with silhouette score tracking
        
        Args:
            X: training data (n_samples, input_dim) - can be numpy array or torch tensor
            epochs: number of training epochs
            verbose: whether to print progress
        """
        # Convert to tensor if needed and ensure it's on correct device
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        else:
            X = X.to(self.device)
        
        n_samples = X.shape[0]
        
        # Initialize silhouette score tracking
        self.training_silhouette = []
        
        print(f"Training SOM on {self.device} with {n_samples} samples...")
        
        for epoch in range(epochs):
            # Decay parameters
            learning_rate, sigma = self._decay_parameters(epoch, epochs)
            
            # Shuffle data for each epoch
            indices = torch.randperm(n_samples, device=self.device)
            
            # Process each sample
            for idx in indices:
                x = X[idx]  # Shape: (input_dim,)
                
                # Find winner neuron
                winner_idx = self._find_winner(x)
                
                # Update weights
                self._update_weights(x, winner_idx, learning_rate, sigma)
                
                # Update bias to prevent dead neurons
                self._update_bias(winner_idx)
            
            # Calculate silhouette score
            # Note: silhouette_score from sklearn expects numpy arrays on CPU
            with torch.no_grad():
                predictions = self.predict(X)
                
                # Convert to numpy for silhouette score calculation
                X_cpu = X.cpu().numpy()
                predictions_cpu = predictions.cpu().numpy()
                
                # Only calculate silhouette score if we have more than 1 cluster
                unique_clusters = np.unique(predictions_cpu)
                if len(unique_clusters) > 1:
                    silh_score = silhouette_score(X_cpu, predictions_cpu)
                else:
                    silh_score = -1.0  # Invalid score for single cluster
                
                self.training_silhouette.append(silh_score)
            
            if verbose:
                print(f"Epoch {epoch + 1:>4}/{epochs:>4}, "
                    f"Learning Rate: {learning_rate:.3e}, "
                    f"Sigma: {sigma:.3e}, "
                    f"Silhouette score: {silh_score:.3e}",
                    end='\r')
        
        if verbose:
            print()  # New line after training
        
        print("SOM training completed!")
        return self.training_silhouette
    
    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get cluster centers (weight vectors)
        
        Returns:
            cluster centers (n_neurons, input_dim)
        """
        return self.weights.detach()

            
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class SelfOrganizingMap:
    """
    Self-Organizing Map (SOM) implementation for clustering basketball players
    Based on the methodology described in the paper
    """
    
    def __init__(self, map_size: Tuple[int, int], input_dim: int, 
                 learning_rate: float = 0.1, sigma: float = 1.0, 
                 decay_function: str = 'exponential'):
        """
        Initialize SOM
        
        Args:
            map_size: (height, width) of the output map
            input_dim: dimension of input vectors
            learning_rate: initial learning rate
            sigma: initial neighborhood radius
            decay_function: decay function for learning rate and sigma
        """
        self.map_size = map_size
        self.input_dim = input_dim
        self.initial_learning_rate = learning_rate
        self.initial_sigma = sigma
        self.decay_function = decay_function
        
        # Initialize weight vectors randomly
        # Shape: (map_height, map_width, input_dim)
        self.weights = np.random.random((map_size[0], map_size[1], input_dim))
        
        # Initialize bias terms to handle dead neurons
        self.bias = np.zeros((map_size[0], map_size[1]))
        
        # Create coordinate grid for neurons
        self.neuron_coordinates = np.array([
            [i, j] for i in range(map_size[0]) for j in range(map_size[1])
        ])
        
    def _euclidean_distance(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean distance between input and weight vectors
        
        Args:
            x: input vector (input_dim,)
            weights: weight matrix (map_height, map_width, input_dim)
            
        Returns:
            distance matrix (map_height, map_width)
        """
        # x shape: (input_dim,), weights shape: (map_height, map_width, input_dim)
        # Output shape: (map_height, map_width)
        return np.sqrt(np.sum((weights - x) ** 2, axis=2))
    
    def _find_winner(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Find the Best Matching Unit (BMU) - winner neuron
        
        Args:
            x: input vector (input_dim,)
            
        Returns:
            coordinates of winner neuron (i, j)
        """
        # Calculate distances: shape (map_height, map_width)
        distances = self._euclidean_distance(x, self.weights)
        
        # Apply bias term to handle dead neurons
        distances = distances + self.bias
        
        # Find winner coordinates
        winner_idx = np.unravel_index(np.argmin(distances), self.map_size)
        return winner_idx
    
    def _neighborhood_function(self, winner_pos: Tuple[int, int], 
                             sigma: float) -> np.ndarray:
        """
        Calculate neighborhood function (Gaussian)
        
        Args:
            winner_pos: position of winner neuron (i, j)
            sigma: neighborhood radius
            
        Returns:
            neighborhood influence matrix (map_height, map_width)
        """
        # Create coordinate arrays
        i_coords, j_coords = np.meshgrid(
            np.arange(self.map_size[0]), 
            np.arange(self.map_size[1]), 
            indexing='ij'
        )
        
        # Calculate distances from winner
        distances = ((i_coords - winner_pos[0]) ** 2 + 
                    (j_coords - winner_pos[1]) ** 2)
        
        # Apply Gaussian neighborhood function
        # Shape: (map_height, map_width)
        return np.exp(-distances / (2 * sigma ** 2))
    
    def _update_weights(self, x: np.ndarray, winner_pos: Tuple[int, int], 
                       learning_rate: float, sigma: float):
        """
        Update weight vectors using SOM learning rule
        
        Args:
            x: input vector (input_dim,)
            winner_pos: position of winner neuron
            learning_rate: current learning rate
            sigma: current neighborhood radius
        """
        # Calculate neighborhood influence: shape (map_height, map_width)
        neighborhood = self._neighborhood_function(winner_pos, sigma)
        
        # Update weights for all neurons
        # self.weights shape: (map_height, map_width, input_dim)
        # neighborhood shape: (map_height, map_width)
        # x shape: (input_dim,)
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                # Wi,j = Wi,j + η * φ(i,j) * (x - Wi,j)
                self.weights[i, j] += (learning_rate * neighborhood[i, j] * 
                                     (x - self.weights[i, j]))
    
    def _update_bias(self, winner_pos: Tuple[int, int], bias_decay: float = 0.99):
        """
        Update bias terms to prevent dead neurons
        
        Args:
            winner_pos: position of winner neuron
            bias_decay: decay factor for bias
        """
        # Increase bias for winner (reduce chance of winning again)
        self.bias[winner_pos] += 0.01
        
        # Decay all bias terms
        self.bias *= bias_decay
    
    def _decay_parameters(self, epoch: int, total_epochs: int) -> Tuple[float, float]:
        """
        Decay learning rate and sigma over time
        
        Args:
            epoch: current epoch
            total_epochs: total number of epochs
            
        Returns:
            (current_learning_rate, current_sigma)
        """
        if self.decay_function == 'exponential':
            # Exponential decay
            decay_factor = np.exp(-epoch / total_epochs)
        else:
            # Linear decay
            decay_factor = 1.0 - (epoch / total_epochs)
        
        learning_rate = self.initial_learning_rate * decay_factor
        sigma = self.initial_sigma * decay_factor
        
        return learning_rate, sigma
    
    def train(self, X: np.ndarray, epochs: int = 1000, verbose: bool = True):
        """
        Train the SOM
        
        Args:
            X: training data (n_samples, input_dim)
            epochs: number of training epochs
            verbose: whether to print progress
        """
        n_samples = X.shape[0]
        self.training_silhouette = []
        
        for epoch in range(epochs):
            # Decay parameters
            learning_rate, sigma = self._decay_parameters(epoch, epochs)
            
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x = X[idx]  # Shape: (input_dim,)
                
                # Find winner neuron
                winner_pos = self._find_winner(x)
                
                # Update weights
                self._update_weights(x, winner_pos, learning_rate, sigma)
                
                # Update bias to prevent dead neurons
                self._update_bias(winner_pos)
            
            silhouette = silhouette_score(X, self.predict(X))
            self.training_silhouette.append(silhouette)
            
            if verbose :
                print(f"Epoch n° {epoch + 1:>4} of {epochs:>4}, "
                      f"Learning Rate: {learning_rate:.3e}, "
                      f"Sigma: {sigma:.3e}",
                      f"Silhouette Score: {silhouette:.3e}",
                      end = '\n' if ((epoch+1)/epochs*10)%1==0 else '\r' 
                      )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for input data
        
        Args:
            X: input data (n_samples, input_dim)
            
        Returns:
            cluster assignments (n_samples,)
        """
        clusters = []
        
        for x in X:
            winner_pos = self._find_winner(x)
            # Convert 2D position to 1D cluster ID
            cluster_id = winner_pos[0] * self.map_size[1] + winner_pos[1]
            clusters.append(cluster_id)
        
        return np.array(clusters)
    
    def get_cluster_centers(self) -> np.ndarray:
        """
        Get cluster centers (weight vectors)
        
        Returns:
            cluster centers (n_clusters, input_dim)
        """
        return self.weights.reshape(-1, self.input_dim)

    
    def plot_silhouette_evolution(self):
        """
        Plot evolution of silhouette score during training
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_silhouette, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score Evolution During SOM Training')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        if self.training_silhouette:
            max_score = max(self.training_silhouette)
            max_epoch = self.training_silhouette.index(max_score)
            final_score = self.training_silhouette[-1]
            
            print(f"Maximum silhouette score: {max_score:.4f} at epoch {max_epoch + 1}")
            print(f"Final silhouette score: {final_score:.4f}")