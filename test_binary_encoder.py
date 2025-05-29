import numpy as np
from tabdiff.utils import BinaryCategoricalEncoder

# Create some test binary data
np.random.seed(42)
n_samples = 100
n_features = 5

# Generate binary data with different marginal probabilities
probs = [0.2, 0.5, 0.7, 0.9, 0.05]
X_cat = np.zeros((n_samples, n_features), dtype=int)
for i, p in enumerate(probs):
    X_cat[:, i] = np.random.binomial(1, p, n_samples)

print("Original binary data shape:", X_cat.shape)
print("First 5 samples:")
print(X_cat[:5])
print("\nMarginal probabilities (empirical):", np.mean(X_cat, axis=0))

# Test encoder
encoder = BinaryCategoricalEncoder(k=10)
encoder.fit(X_cat)

print("\nCalculated marginal probabilities:", encoder.marginal_probs)
print("Calculated thresholds:", encoder.thresholds)

# Encode data
Z_encoded = encoder.encode(X_cat)
print("\nEncoded continuous data shape:", Z_encoded.shape)
print("First 5 encoded samples (for first data point):")
print(Z_encoded[:5, :])

# Decode data
X_decoded = encoder.decode(Z_encoded[:n_samples])  # Decode first k samples (should recover first data point)
print("\nDecoded binary data shape:", X_decoded.shape) 
print("First 5 decoded samples:")
print(X_decoded[:5])

# Check reconstruction accuracy (on the first copy of each sample)
accuracy = np.mean(X_cat == X_decoded)
print(f"\nReconstruction accuracy: {accuracy:.2%}")

# Clean up
import os
os.remove("test_binary_encoder.py") 