# Summary of the results we have so far
## Tested Architectures

We evaluated several architectures, including:
- **cnn1d**
- **cnn3d** (deprecated early due to poor performance)
- **gnn**
- **cnn1d+gnn**

## Input Transformation Process

We discovered that modifying the inputs significantly improves results. Below is a step-by-step breakdown of the transformation:

### 1. LiDAR Input
- **Input Shape**: `x_lidar_input` with shape `(20, 200, 10)`:
    - 20×200 grid cells, each with 10 channels.

### 2. One-Hot Transformation
- For each grid cell, compress the 10 channels into 4 binary flags:
    - **BS present**: `1` if any channel equals `-2`.
    - **Vehicle present**: `1` if any channel equals `-1`.
    - **Obstacle present**: `1` if any channel equals `1`.
    - **Free**: `1` if all channels are `0`.
- **Output Shape**: `one_hot_lidar` with shape `(20, 200, 4)`.

### 3. Flatten Grid
- Reshape the grid into a sequence:
    - `(20, 200, 4)` → `(4000, 4)`.

### 4. GPS/Coordinate Input
- **Input Shape**: `x_coord_input` with shape `(2,)` (2D coordinate per sample).
- **Tiling**:
    - Expand `(2,)` to `(1, 2)`.
    - Tile to `(4000, 2)`.

### 5. Combine Features
- Concatenate the LiDAR flags `(4000, 4)` with the tiled coordinates `(4000, 2)`:
    - **Output Shape**: `combined` with shape `(4000, 6)`.

### Summary
This process transforms each LiDAR cell into four binary flags (`BS/vehicle/obstacle/free`), appends the same GPS/coordinate info to every cell, and produces a sequence of 4000 elements, each with 6 features, for downstream modeling.

## Performance Overview

After defining this common input structure, we started testing the cnn1d and gnn architectures. In general, the gnn architecture has shown better performance, although the it is more computationally expensive.

The cnn1d+gnn architecture was tested but did not yield better results than the gnn alone or the cnn1d alone. It was not worth the extra complexity.

I also tested skip connections, but they did not improve performance and just added complexity. I thought they might help with gradient flow, but it seems unnecessary for our relatively shallow networks. Additionally, it seems the problem of the accuracy plateauing is more related to the dataset itself rather than the architecture.

The dataset, as we already discussed, is very imbalanced and, considering the complexity of the task, it might be too small to achieve better results. Using the s008 for training and the s009 for validation, we got good (not always) values for the training accuracy, but the validation accuracy plateaued around 0.6-0.63. But when using NAS to combine the datasets, the validation accuracy improved to around 0.99-1.0, and the training accuracy had the same behavior. I dont think it is overfitting as both the training and validation accuracies are similar.

Moreover, i let the NAS find the best ratio to use of the s009 dataset in the training set, and it found that using 0.4 was best. This is exactly the number of NLOS cases in the s008 dataset. This is definitely not a coincidence, and it shows how important it is to have a balanced dataset.

| Dataset | # of Samples | LOS   | NLOS  | NLOS Percentage |
|---------|--------------|-------|-------|-----------------|
| S008    | 11,194       | 6,482 | 4,712 | 42%             |
| S009    | 9,638        | 1,473 | 8,165 | 85%             |



## What is left to do

1) maybe explore a bit more the cnn1d+gnn architecture, but I dont think it will yield better results. NOT WORTH IT.
2) maybe explore more the skip connections, but again, I dont think it will yield better results. NOT WORTH IT.
3) Get some related works models and train with the balanced dataset. For that, I let a NAS code find the best ratio, then save the numpy arrays, and then use them to train the models from related works. I have to do this way, as i dont understand the dataset data. And the raw data is very unorganized and hard to understand. WORTH IT.

I think that is it. And of course, write everything in a paper.