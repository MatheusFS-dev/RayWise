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

    # ———————————————————————————————— Conv Layers ——————————————————————————————— #
    initializer = tf.keras.initializers.GlorotUniform(seed=t_seed)

    # First Conv1D layer
    x = build_cnn1d(
        trial=trial,
        hparams=hparams,
        x=combined,
        name_prefix="conv1d_0",
        # Filters
        filters_range=trial.suggest_categorical("conv1d_0_filters", [128, 256, 512]),
        # filters_step=40,
        # Kernel size
        kernel_size_range=(2, 3),
        kernel_size_step=1,
        # Other parameters
        # strides=trial.suggest_int(f"conv1d_{i}_strides", 1, 2),
        kernel_initializer=initializer,
    )
    pool_size = 2
    x = layers.MaxPooling1D(pool_size=pool_size, name="max_pool_0")(x)

    # Second Conv1D layer
    x = build_cnn1d(
        trial=trial,
        hparams=hparams,
        x=x,
        name_prefix="conv1d_1",
        # Filters
        filters_range=trial.suggest_categorical("conv1d_1_filters", [128, 256, 512]),
        # filters_step=40,
        # Kernel size
        kernel_size_range=(2, 4),
        kernel_size_step=1,
        # Other parameters
        # strides=trial.suggest_int(f"conv1d_1_strides", 1, 2),
        kernel_initializer=initializer,
    )
    pool_size = trial.suggest_categorical("pool_size_1", [2, 5])
    x = layers.MaxPooling1D(pool_size=pool_size, name="max_pool_1")(x)

    # Third Conv1D layer
    x = build_cnn1d(
        trial=trial,
        hparams=hparams,
        x=x,
        name_prefix="conv1d_2",
        # Filters
        filters_range=trial.suggest_categorical("conv1d_2_filters", [64, 128, 256, 512]),
        # filters_step=40,
        # Kernel size
        kernel_size_range=(2, 8),
        kernel_size_step=1,
        # Other parameters
        # strides=trial.suggest_int(f"conv1d_2_strides", 1, 2),
        kernel_initializer=initializer,
    )
    pool_size = trial.suggest_categorical("pool_size_2", [2, 3, 4])
    x = layers.MaxPooling1D(pool_size=pool_size, name="max_pool_2")(x)

    # Fourth Conv1D layer
    x = build_cnn1d(
        trial=trial,
        hparams=hparams,
        x=x,
        name_prefix="conv1d_3",
        # Filters
        filters_range=trial.suggest_categorical("conv1d_3_filters", [64, 128, 256]),
        # filters_step=20,
        # Kernel size
        kernel_size_range=(2, 5),
        kernel_size_step=1,
        # Other parameters
        # strides=trial.suggest_int(f"conv1d_3_strides", 1, 2),
        kernel_initializer=initializer,
    )
    pool_size = trial.suggest_categorical("pool_size_3", [2, 3, 4])
    x = layers.MaxPooling1D(pool_size=pool_size, name="max_pool_3")(x)

    # ————————————————————————————————— Attention ———————————————————————————————— #
    # num_heads × key_dim = d_model
    # d_model is the dimensionality of the input to the attention layer
    C = x.shape[-1]

    # Prepare coords into the same shape as the CNN output
    coord_projection = layers.Dense(
        units=C,
        name="coord_projection",
        kernel_initializer=initializer,
    )(x_coord_input)
    coord_seq = layers.RepeatVector(x.shape[1], name="coord_seq")(coord_projection)

    num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
    key_dim = trial.suggest_categorical("key_dim", [32, 64, 128])

    # Coords -> Lidar
    attn1 = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=0.1,
        kernel_initializer=initializer,
        name="multihead_self_attention_coords_to_lidar",
    )(
        query=coord_seq,
        value=x,
        key=x,
    )
    attn1 = layers.Add(name="residual_attention_1")([coord_seq, attn1])
    attn1 = layers.LayerNormalization(name="norm_attention_1")(attn1)

    # Lidar -> Coords
    attn2 = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=0.1,
        kernel_initializer=initializer,
        name="multihead_self_attention_lidar_to_coords",
    )(
        query=x,
        value=coord_seq,
        key=coord_seq,
    )
    attn2 = layers.Add(name="residual_attention_2")([x, attn2])
    attn2 = layers.LayerNormalization(name="norm_attention_2")(attn2)

    # fuse attended streams
    x = layers.Concatenate(name="fuse_attended")([attn1, attn2])
    x = layers.Dense(C, name="project_fusion")(x)

    # Feed-forward network after attention
    ff = layers.Dense(
        units=trial.suggest_categorical("ffn_units", [64, 128, 256, 512]),
        activation=get_activation(
            trial.suggest_categorical("ffn_activation", hparams.activation_choices)
        ),
        name="ffn_after_attention",
        kernel_initializer=initializer,
    )(x)
    ff = layers.Dense(x.shape[-1], name="ffn2_after_attention", kernel_initializer=initializer)(ff)
    x = layers.Add(name="residual_ffn")([x, ff])
    x = layers.LayerNormalization(name="norm_ffn")(x)

    # ———————————————————————————— Extra dense layers ———————————————————————————— #
    # (batch, length, channels) -> (batch, length * channels)
    x = layers.Flatten(name="flatten")(x)

    num_dense_layers = trial.suggest_int("num_dense_layers", 0, 2)

    for i in range(num_dense_layers):
        # Dense layer with dropout
        x = build_dnn(
            trial=trial,
            hparams=hparams,
            x=x,
            name_prefix=f"dense_{i}",
            # Units
            units_range=(200, 400),
            units_step=50,
            # Dropout
            dropout_rate_range=(0.0, 0.5),
            dropout_rate_step=0.1,
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
