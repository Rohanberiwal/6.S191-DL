## Batch Normalization (BN)

- **Description**: Batch normalization normalizes the activations of each layer across the mini-batch during training.
- **Purpose**: Improves training stability and accelerates convergence by stabilizing the distribution of activations.
- **Operation**: Calculates mean and standard deviation of activations in each mini-batch and normalizes using these statistics.
- **Learnable Parameters**: Scale and shift parameters allow adaptive adjustment of normalized activations.

## Layer Normalization (LN)

- **Description**: Layer normalization normalizes the activations of each layer across the entire layer rather than across mini-batches.
- **Purpose**: Improves training stability, less sensitive to batch size variations, and computationally efficient.
- **Operation**: Calculates mean and standard deviation of activations along each feature dimension and normalizes using these statistics.
- **Learnable Parameters**: Does not introduce learnable parameters, making it computationally efficient.
