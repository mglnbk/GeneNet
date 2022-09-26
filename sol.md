# Gene Network

## Data Resources

- Public dataset from *TCGA*
    - Copy Number Variation
    - Somatic Mutation Information (extracted)

## Model architecture

- Input:
    - High-dimensional data (channel, number_sample, feature_counts)

- First Layer (Feature Selection and dimenality reduction)
    - Input: (channel, number_sample, features)
    - Output: (1, number_sample, low_dim_features)
    - **Note**: Low_dim_features represent the different genes 

- Biological
