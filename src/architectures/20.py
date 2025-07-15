from _imports import *

def build_model(trial: optuna.Trial, hparams: dict, show_summary: bool = True, t_seed: int = 42) -> tf.keras.Model:
    x_lidar_input = layers.Input(shape=(20, 200, 10), name="lidar_input")

    # Inline one-hot encoding of semantic values
    one_hot_lidar = layers.Lambda(
        lambda x: tf.concat(
            [
                # “Is there a BS anywhere in the 10 channels?” → 1 channel
                tf.cast(tf.reduce_any(tf.equal(x, -2), axis=-1, keepdims=True), tf.float32),
                # “Vehicle?” → 1 channel
                tf.cast(tf.reduce_any(tf.equal(x, -1), axis=-1, keepdims=True), tf.float32),
                # “Obstacle?” → 1 channel
                tf.cast(tf.reduce_any(tf.equal(x, 1), axis=-1, keepdims=True), tf.float32),
                # “Free?” → 1 channel (all channels zero)
                tf.cast(tf.reduce_all(tf.equal(x, 0), axis=-1, keepdims=True), tf.float32),
            ],
            axis=-1,
        ),
        #! Lambda has deserialization issues, so providing the output shape is necessary
        output_shape=(20, 200, 4),
        name="lidar_transform_to_one_hot",
    )(x_lidar_input)
    # -> (batch, 20, 200, 4)

    # Flatten the 20×200 grid into a 4000-length sequence with the 4 channels
    x_lidar_flat: layers.Layer = layers.Reshape((20 * 200, 4), name="lidar_flatten_4_channels")(
        one_hot_lidar
    )

    # ———————————————————————————————— GPS Input ———————————————————————————————— #
    # Input for coordinate data (e.g., shape: (2,))
    x_coord_input = layers.Input(shape=(2,), name="coord_input")

    # Turn (batch,2) → (batch,1,2) → tile to (batch,4000,2)'
    x_coord: layers.Layer = layers.Lambda(
        lambda x: tf.tile(tf.expand_dims(x, axis=1), [1, 20 * 200, 1]),
        #! Lambda has deserialization issues, so providing the output shape is necessary
        output_shape=(20 * 200, 2),
        name="coord_tile_flat",
    )(x_coord_input)

    # ————————————————————————————— Combine Branches ————————————————————————————— #
    # Fuse channels:  (batch,4000,4) + (batch,4000,2) → (batch,4000,6)
    combined = layers.Concatenate(axis=-1, name="combine_lidar_coord")([x_lidar_flat, x_coord])

    # ———————————————————————————————— Initializer ———————————————————————————————— #
    initializer = tf.keras.initializers.GlorotUniform(
        # seed=t_seed,
    )

    # ———————————————————————————————————— GNN ——————————————————————————————————— #
    from spektral.layers import (
        GCNConv,
        GATConv,
        GraphSageConv,
        ChebConv,
        GraphMasking,
        GlobalAvgPool,
    )
    import scipy.sparse as sp
    from spektral.utils import convolution

    def build_grid_adjacency(rows: int, cols: int) -> tf.sparse.SparseTensor:
        n = rows * cols
        A = sp.lil_matrix((n, n), dtype=np.float32)

        # connect each node to its up/down/left/right neighbours
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        jdx = rr * cols + cc
                        A[idx, jdx] = 1.0

        A = A.tocsr()
        # make symmetric
        A = A + A.T
        A[A > 1] = 1.0

        # apply GCN normalization (D⁻½ A D⁻½)
        A_norm = convolution.gcn_filter(A)

        # convert to tf.sparse.SparseTensor
        coo = A_norm.tocoo()
        indices = np.vstack((coo.row, coo.col)).T
        return tf.sparse.reorder(tf.sparse.SparseTensor(indices, coo.data, coo.shape))

    def build_knn_adjacency(rows: int, cols: int, k: int) -> tf.sparse.SparseTensor:
        from sklearn.neighbors import NearestNeighbors

        n = rows * cols
        # grid‐cell centers in 2D
        coords = np.array([(i // cols, i % cols) for i in range(n)], np.float32)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
        _, indices = nbrs.kneighbors(coords)
        A = sp.lil_matrix((n, n), dtype=np.float32)
        for i in range(n):
            for j in indices[i][1:]:  # skip self at indices[i][0]
                A[i, j] = 1.0
                A[j, i] = 1.0
        A_norm = convolution.gcn_filter(A.tocsr())
        coo = A_norm.tocoo()
        idx = np.vstack((coo.row, coo.col)).T
        return tf.sparse.reorder(tf.sparse.SparseTensor(idx, coo.data, coo.shape))

    A_grid = build_grid_adjacency(rows=20, cols=200)

    # choose adjacency type
    adj_type = trial.suggest_categorical("adjacency_type", ["grid", "knn"])
    if adj_type == "grid":
        A = A_grid
    else:
        # let Optuna pick k for KNN
        k = trial.suggest_int("knn_k", 4, 16, step=4)
        A = build_knn_adjacency(rows=20, cols=200, k=k)

    # Mask out padded nodes if any (here none), and pair with A
    x_graph, a_graph = GraphMasking()([combined, A])
    x = x_graph

    max_gnn_layers = trial.suggest_int("num_gcn_layers", 1, 4)
    for i in range(max_gnn_layers):
        # first slot cannot be “none” to guarantee ≥1 layer
        if i == 0:
            layer_type = trial.suggest_categorical(
                f"layer_{i}_type",
                ["gcn", "gat", "cheb"],
            )
        else:
            layer_type = trial.suggest_categorical(
                f"layer_{i}_type",
                ["none", "gcn", "gat", "cheb"],
            )
        if layer_type == "none":
            continue

        layer_name = f"{layer_type}_{i}"
        units = trial.suggest_categorical(f"{layer_name}_units", [64, 128, 256])
        activation = hparams.get_activation(trial, f"{layer_name}_activation")
        dropout = trial.suggest_float(f"{layer_name}_dropout", 0.0, 0.5, step=0.1)

        if layer_type == "gcn":
            x = GCNConv(
                units,
                activation=activation,
                kernel_initializer=initializer,
                name=layer_name,
            )([x, a_graph])

        elif layer_type == "gat":
            heads = trial.suggest_int(f"{layer_name}_heads", 1, 8)
            x = GATConv(
                units,
                attn_heads=heads,
                concat=False,
                activation=activation,
                kernel_initializer=initializer,
                name=layer_name,
            )([x, a_graph])

        elif layer_type == "cheb":
            K = trial.suggest_int(f"{layer_name}_K", 2, 5)
            x = ChebConv(
                units,
                K=K,
                activation=activation,
                kernel_initializer=initializer,
                name=layer_name,
            )([x, a_graph])

        if dropout > 0:
            x = layers.Dropout(dropout, name=f"{layer_name}_dropout")(x)

    # global pooling to get graph-level embedding
    x = GlobalAvgPool(name="global_pool")(x)

    # ———————————————————————————— Extra dense layers ———————————————————————————— #
    num_dense_layers = trial.suggest_int("num_dense_layers", 2, 3)

    for i in range(num_dense_layers):
        # Dense layer with dropout
        x = build_dnn(
            trial=trial,
            hparams=hparams,
            x=x,
            name_prefix=f"dense_{i}",
            # Units
            units_range=(250, 400),
            units_step=50,
            # Dropout
            dropout_rate_range=(0.0, 0.4),
            dropout_rate_step=0.2,
            # Other parameters
            kernel_initializer=initializer,
        )

    # —————————————————————————————————— Output —————————————————————————————————— #
    outputs = layers.Dense(
        256,
        activation="softmax",
        name="output",
        kernel_initializer=initializer,
    )(x)

    # —————————————————————————— Set Inputs and Outputs —————————————————————————— #
    model = Model(inputs=(x_lidar_input, x_coord_input), outputs=(outputs,))
    if show_summary:
        model.summary()
    return model
