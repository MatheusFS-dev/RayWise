from _imports import *

def build_model(trial: optuna.Trial, hparams: dict, show_summary: bool = True, t_seed: int = 42) -> tf.keras.Model:
    x_lidar_input = layers.Input(shape=(20, 200, 10))

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
        name="lidar_one_hot",
    )(x_lidar_input)
    # -> (batch, 20, 200, 4)

    # Merge channels per the trial’s choice
    # ? This was the best performing option in the previous trials
    lidar_channel_option: int = 4  # trial.suggest_categorical("lidar_channel_option", [4, 14])
    if lidar_channel_option == 14:
        x_lidar_preproc = layers.Concatenate(name="lidar_concat")(
            [x_lidar_input, one_hot_lidar]
        )  # → (batch,20,200,14)
    else:
        x_lidar_preproc = one_hot_lidar  # → (batch,20,200,4)

    # Flatten the 20×200 grid into a 4000-length sequence with the chosen channels
    x_lidar_flat: layers.Layer = layers.Reshape((20 * 200, lidar_channel_option), name="lidar_flatten")(
        x_lidar_preproc
    )

    # ———————————————————————————————— GPS Input ———————————————————————————————— #
    # Input for coordinate data (e.g., shape: (2,))
    x_coord_input = layers.Input(shape=(2,))

    # Turn (batch,2) → (batch,1,2) → tile to (batch,4000,2)'
    x_coord: layers.Layer = layers.Lambda(
        lambda x: tf.tile(tf.expand_dims(x, axis=1), [1, 20 * 200, 1]),
        #! Lambda has deserialization issues, so providing the output shape is necessary
        output_shape=(20 * 200, 2),
        name="coord_tile_flat",
    )(x_coord_input)

    # ————————————————————————————— Combine Branches ————————————————————————————— #
    # Fuse channels: (batch,20,200,10) + (batch,20,200,2) → (batch,20,200,12)
    combined = layers.Concatenate(axis=-1)([x_lidar_flat, x_coord])

    max_layers = trial.suggest_int("num_layers", 3, 6)

    # Calculate max pool size
    max_pool_dim1 = math.floor((20 * 200) ** (1.0 / max_layers))

    # ? Tip: after finding max_layers, for each layer call a different builder, then optimize it individually
    x = model = build_cnn1d(
        trial=trial,
        hparams=hparams,
        x=combined,
        max_layers=max_layers,
        max_filters=512,
        min_filters=32,
        filters_step=32,
        max_kernel_size=5,
        min_kernel_size=1,
        min_pool_size=1,
        max_pool_size=max_pool_dim1,
        data_format="channels_last",
        padding="same",
        strides=1,
        dilation_rate=1,
        groups=1,
        # kernel_initializer=trial.suggest_categorical("initializer_layer_1", initializer_options),
        trial_batch_norm=True,
        trial_kernel_reg=False,
        trial_bias_reg=False,
        trial_activity_reg=False,
        regularizer_positions=None,
        trial_skip_connections=False,
        share_activation=False,
        name_prefix="cnn1d",
    )

    # ———————————————————————————— Flatten the Output ———————————————————————————— #
    x = layers.Flatten(name="flatten")(x)

    # ——————————————————————————————— Dense Layers ——————————————————————————————— #
    # ? This was the best performing option in the previous trials
    num_dense_layers = 2 # trial.suggest_int("num_dense_layers", 0, 3)
    for i in range(num_dense_layers):
        x = build_dnn(
            trial=trial,
            hparams=hparams,
            x=x,
            max_layers=1,
            max_units=500,
            min_units=50,
            units_step=50,
            # kernel_initializer=trial.suggest_categorical("initializer_layer_1", initializer_options),
            min_dropout_rate=0.0,
            max_dropout_rate=0.5,
            dropout_rate_step=0.1,
            dropout_positions=None,  # None for all layers
            regularizer_positions=None,  # None for all layers
            trial_batch_norm=False,
            trial_kernel_reg=False,
            trial_bias_reg=False,
            trial_activity_reg=False,
            trial_skip_connections=False,
            share_activation=False,
            name_prefix=f"extra_dense_{i}",
        )

    # —————————————————————————————————— Output —————————————————————————————————— #
    outputs = layers.Dense(256, activation="softmax")(x)

    # —————————————————————————— Set Inputs and Outputs —————————————————————————— #
    model = Model(inputs=(x_lidar_input, x_coord_input), outputs=(outputs,))
    if show_summary:
        model.summary()
    return model
