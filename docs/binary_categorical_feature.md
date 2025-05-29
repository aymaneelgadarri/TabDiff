# Binary Categorical Feature for TabDiff

This document describes the new binary categorical feature added to TabDiff, which enables training on datasets that contain only binary categorical variables.

## Overview

The binary categorical feature allows TabDiff to handle datasets with only binary (0/1) categorical variables by:

1. **Encoding**: Converting binary categorical data to continuous data using truncated Gaussian sampling
2. **Training**: Running TabDiff on the encoded continuous dataset
3. **Decoding**: Converting generated continuous samples back to binary categorical data

## Mathematical Details

### Encoding Process

For each binary variable X_i with empirical Bernoulli parameter p_i:

1. Calculate the marginal probability: `p_i = mean(X_i)`
2. Calculate the threshold: `threshold_i = Φ^(-1)(1 - p_i)`, where Φ^(-1) is the inverse CDF of a standard Gaussian
3. For each sample:
   - If X_i = 0: Sample Z_i from truncated Gaussian N(0,1) restricted to (-∞, threshold_i)
   - If X_i = 1: Sample Z_i from truncated Gaussian N(0,1) restricted to (threshold_i, ∞)
4. Repeat step 3 k times per sample to generate k continuous samples

### Decoding Process

For a continuous sample Z = (Z_1, ..., Z_m):
- Decode using indicator function: `X_i = 1{Z_i > threshold_i}`

## Usage

### Command Line Arguments

- `--binary_cat_only`: Enable the binary categorical encoding feature
- `--binary_encoding_k`: Number of truncated Gaussian samples per categorical sample (default: 10)

### Example Commands

Train on a binary categorical dataset (e.g., disease dataset):
```bash
python main.py --dataname disease --mode train --binary_cat_only --binary_encoding_k 10
```

Test a trained model:
```bash
python main.py --dataname disease --mode test --binary_cat_only
```

## Implementation Details

### Files Modified/Added

1. **tabdiff/utils/binary_categorical_encoder.py**: Core encoder/decoder implementation
2. **utils_train.py**: Modified TabDiffDataset to support binary encoding
3. **tabdiff/main.py**: Added support for binary encoding parameters
4. **tabdiff/trainer.py**: Modified to handle decoding during sampling
5. **main.py**: Added command line arguments

### Key Classes

- `BinaryCategoricalEncoder`: Handles encoding/decoding of binary categorical data
  - `fit()`: Calculate marginal probabilities and thresholds
  - `encode()`: Convert binary to continuous using truncated Gaussian sampling
  - `decode()`: Convert continuous back to binary using threshold indicators

## Considerations

1. **Memory Usage**: The encoding process expands the dataset by a factor of k
2. **Suitable Datasets**: This feature is designed for datasets with only binary categorical variables
3. **Reconstruction**: Due to the stochastic nature of encoding, perfect reconstruction is not guaranteed

## Example Dataset

The disease dataset is an example of a dataset with only binary categorical variables:
- 131 binary features representing various symptoms
- 1 binary target variable for disease classification
- All values are either 0 or 1 