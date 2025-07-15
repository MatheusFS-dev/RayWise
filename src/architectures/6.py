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
    conv_layers = 4
    max_pool_dim1 = math.floor((20 * 200) ** (1.0 / conv_layers))

    for i in range(conv_layers):
        filters = trial.suggest_categorical(f"conv1d_{i}_filters", [128, 256, 512])

        x = build_cnn1d(
            trial=trial,
            hparams=hparams,
            x=combined if i == 0 else x,
            name_prefix=f"conv1d_{i}",
            # Filters
            filters_range=filters,
            # filters_step=80,
            # Kernel size
            kernel_size_range=(1, 10),
            kernel_size_step=1,
            # Other parameters
            # strides=trial.suggest_int(f"conv1d_{i}_strides", 1, 2),
        )

        # Add MaxPooling1D after each Conv1D layer
        pool_size = trial.suggest_int(f"pool_size_{i}", 2, max_pool_dim1)
        x = layers.MaxPooling1D(pool_size=pool_size, name=f"max_pool_{i}")(x)

    # ———————————————————————————— Extra dense layers ———————————————————————————— #
    dense_layers = 2

    strategy = trial.suggest_categorical(
        "output_strategy",
        [
            "flatten",
            "global_average_pooling",
        ],
    )

    if strategy == "flatten":
        # (batch, length, channels) -> (batch, length * channels)
        x = layers.Flatten(name="flatten")(x)
    elif strategy == "global_average_pooling":
        # (batch, length, channels) -> (batch, channels)
        x = layers.GlobalAveragePooling1D(name="global_average_pooling", data_format="channels_last")(x)

    for i in range(dense_layers):
        x = build_dnn(
            trial=trial,
            hparams=hparams,
            x=x,
            name_prefix=f"dense_{i}",
            # Units
            units_range=(150, 300),
            units_step=50,
            # Dropout
            dropout_rate_range=(0.0, 0.5),
            dropout_rate_step=0.1,
        )

    # —————————————————————————————————— Output —————————————————————————————————— #
    outputs = layers.Dense(256, activation="softmax", name="output")(x)

    # —————————————————————————— Set Inputs and Outputs —————————————————————————— #
    model = Model(inputs=(x_lidar_input, x_coord_input), outputs=(outputs,))
    if show_summary:
        model.summary()
    return model
