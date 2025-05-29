import numpy as np
import torch
from scipy.stats import norm, truncnorm


class BinaryCategoricalEncoder:
    """
    Encoder/Decoder for datasets with only binary categorical variables.
    
    This class encodes binary categorical data into continuous data using truncated Gaussian sampling,
    and decodes continuous data back to binary categorical using threshold indicators.
    """
    
    def __init__(self, k=10, device='cpu'):
        """
        Args:
            k: Number of truncated Gaussian samples per categorical sample
            device: Device to use for computations ('cpu' or 'cuda')
        """
        self.k = k
        self.marginal_probs = None
        self.thresholds = None
        self.device = device
    
    def fit(self, X_cat):
        """
        Calculate marginal probabilities for each binary variable.
        
        Args:
            X_cat: Binary categorical data, shape (n_samples, n_features)
        """
        if isinstance(X_cat, torch.Tensor):
            X_cat = X_cat.cpu().numpy()
        
        # Calculate empirical Bernoulli parameters (marginal probabilities)
        self.marginal_probs = np.mean(X_cat, axis=0)
        
        # Calculate thresholds Phi^{-1}(1 - p_i)
        self.thresholds = norm.ppf(1 - self.marginal_probs)
        
        # Handle edge cases where p_i = 0 or 1
        self.thresholds[np.isnan(self.thresholds)] = 0
        self.thresholds[np.isinf(self.thresholds)] = np.sign(self.thresholds[np.isinf(self.thresholds)]) * 5
        
        return self
    
    def encode(self, X_cat):
        """
        Encode binary categorical data to continuous data using truncated Gaussian sampling.
        
        Args:
            X_cat: Binary categorical data, shape (n_samples, n_features)
            
        Returns:
            Z: Continuous encoded data, shape (n_samples * k, n_features)
        """
        if isinstance(X_cat, torch.Tensor):
            X_cat = X_cat.cpu().numpy()
            
        n_samples, n_features = X_cat.shape
        Z = np.zeros((n_samples * self.k, n_features))
        
        for i in range(n_features):
            threshold = self.thresholds[i]
            
            # Get indices for 0s and 1s
            zeros_idx = np.where(X_cat[:, i] == 0)[0]
            ones_idx = np.where(X_cat[:, i] == 1)[0]
            
            # Sample for zeros (truncated below threshold)
            if len(zeros_idx) > 0:
                samples_zeros = truncnorm.rvs(
                    a=-np.inf,
                    b=threshold,
                    loc=0,
                    scale=1,
                    size=(len(zeros_idx), self.k)
                )
                for j, idx in enumerate(zeros_idx):
                    Z[idx*self.k:(idx+1)*self.k, i] = samples_zeros[j]
            
            # Sample for ones (truncated above threshold)
            if len(ones_idx) > 0:
                samples_ones = truncnorm.rvs(
                    a=threshold,
                    b=np.inf,
                    loc=0,
                    scale=1,
                    size=(len(ones_idx), self.k)
                )
                for j, idx in enumerate(ones_idx):
                    Z[idx*self.k:(idx+1)*self.k, i] = samples_ones[j]
        
        return Z
    
    def decode(self, Z):
        """
        Decode continuous data back to binary categorical using threshold indicators.
        
        Args:
            Z: Continuous data, shape (n_samples, n_features)
            
        Returns:
            X_cat: Binary categorical data, shape (n_samples, n_features)
        """
        if isinstance(Z, torch.Tensor):
            Z = Z.cpu().numpy()
            
        # Apply threshold indicator: X_i = 1_{Z_i > threshold_i}
        X_cat = (Z > self.thresholds[np.newaxis, :]).astype(int)
        
        return X_cat
    
    def to(self, device):
        """Move encoder to specified device."""
        self.device = device
        return self 