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

    # ———————————————————————————————— CNN Layers ———————————————————————————————— #
    max_conv_layers = 4

    for i in range(max_conv_layers):
        x = build_cnn1d(
            name_prefix=f"conv1d_layer_{i}",
            trial=trial,
            hparams=hparams,
            x=combined if i == 0 else x,
            # Filters
            filters_range=100,
            filters_step=10,
            # Kernel
            kernel_size_range=(1, 5),
            kernel_size_step=1,
            # Regularization
            trial_kernel_reg=False,
            trial_bias_reg=False,
            trial_activity_reg=False,
            # Other parameters
            strides=1,
            use_batch_norm=True,
            # kernel_initializer=trial.suggest_categorical("initializer_layer_1", initializer_options),
        ) # min output shape: (batch, 250, filters)

    # ——————————————————————————————— Dense Layers ——————————————————————————————— #
    # Using conv1d as a dense layer

    max_dense_layers = trial.suggest_int("max_dense_layers", 1, 3)

    for i in range(max_dense_layers):
        x = build_dense_as_conv1d(
            name_prefix=f"dense_as_conv1d_layer_{i}",
            trial=trial,
            hparams=hparams,
            x=x,
            # Filters
            filters_range=1000,
            filters_step=50,
        )

    # ———————————————————— Decide how to deal with the output ———————————————————— #
    strategy = trial.suggest_categorical(
        "output_strategy",
        [
            "flatten",
            "global_average_pooling",
            "max_pooling_concat",
        ],
    )
    strategy = "flatten"

    if strategy == "flatten":
        # (batch, length, channels) -> (batch, length/pool_size, channels)
        x = layers.MaxPooling1D(
            pool_size=1, name="pool_before_flatten"
        )(x)
        # (batch, length, channels) -> (batch, length * channels)
        x = layers.Flatten(name="flatten")(x)
    elif strategy == "global_average_pooling":
        # (batch, length, channels) -> (batch, channels)
        x = layers.GlobalAveragePooling1D(name="global_average_pooling", data_format="channels_last")(x)
    elif strategy == "max_pooling_concat":
        pool_size = trial.suggest_int("max_pool_size_concat", 2, 5, step=1)
        p1 = layers.MaxPooling1D(pool_size=pool_size, name="p1")(x)
        p2 = layers.MaxPooling1D(pool_size=pool_size, name="p2")(p1)
        p3 = layers.MaxPooling1D(pool_size=pool_size, name="p3")(p2)
        # Concatenate the pooled outputs
        x = layers.Concatenate(axis=1, name="concat_pooled_outputs")([x, p1, p2, p3])
        # output shape: (batch, length/pool_size^3, channels)

        x = build_dense_as_conv1d(
            name_prefix="dense_as_conv1d_after_concat",
            trial=trial,
            hparams=hparams,
            x=x,
            # Filters
            filters_range=(20, 200),
            filters_step=20,
        )

        x = layers.Flatten(name="flatten_after_concat")(x)

    # —————————————————————————————————— Output —————————————————————————————————— #
    outputs = layers.Dense(256, activation="softmax", name="output")(x)

    # —————————————————————————— Set Inputs and Outputs —————————————————————————— #
    model = Model(inputs=(x_lidar_input, x_coord_input), outputs=(outputs,))
    if show_summary:
        model.summary()
    return model
