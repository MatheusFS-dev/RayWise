from _imports import *

def build_model(trial: optuna.Trial, hparams: dict, show_summary: bool = True, t_seed: int = 42) -> tf.keras.Model:
    x_lidar_input = layers.Input(shape=(20, 200, 10), name="lidar_input")

    # Inline one-hot encoding of semantic values
    one_hot_lidar = layers.Lambda(
        lambda x: tf.concat(
            [
                # “Is there a BS anywhere in the 10 channels?” -> 1 channel
                tf.cast(tf.reduce_any(tf.equal(x, -2), axis=-1, keepdims=True), tf.float32),
                # “Vehicle?” -> 1 channel
                tf.cast(tf.reduce_any(tf.equal(x, -1), axis=-1, keepdims=True), tf.float32),
                # “Obstacle?” -> 1 channel
                tf.cast(tf.reduce_any(tf.equal(x, 1), axis=-1, keepdims=True), tf.float32),
                # “Free?” -> 1 channel (all channels zero)
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

    # Turn (batch,2) -> (batch,1,2) -> tile to (batch,4000,2)'
    x_coord: layers.Layer = layers.Lambda(
        lambda x: tf.tile(tf.expand_dims(x, axis=1), [1, 20 * 200, 1]),
        #! Lambda has deserialization issues, so providing the output shape is necessary
        output_shape=(20 * 200, 2),
        name="coord_tile_flat",
    )(x_coord_input)

    # ———————————————————————————————— Conv Layers ——————————————————————————————— #
    initializer = tf.keras.initializers.GlorotUniform(
        # seed=t_seed,
    )

    # First Conv1D layer
    x = build_cnn1d(
        trial=trial,
        hparams=hparams,
        x=x_lidar_flat,
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

    # (batch, length, channels) -> (batch, length * channels)
    x = layers.Flatten(name="flatten")(x)

    # ————————————————————————————————— Attention ———————————————————————————————— #
    H = trial.suggest_categorical("mhf_heads",[2,3,4])
    head_dim = trial.suggest_categorical("head_dim",[32,64])
    embed_dim = head_dim * H

    e_lidar = layers.Dense(embed_dim, name="lidar_embed")(x)  # -> (batch, embed_dim)
    e_gps = layers.Dense(embed_dim, name="gps_embed")(x_coord_input)  # -> (batch, embed_dim)

    Q = layers.Dense(embed_dim, name="q_proj")(e_gps)    # -> (batch,embed_dim)
    K = layers.Dense(embed_dim, name="k_proj")(e_lidar) # -> (batch,embed_dim)
    V = layers.Dense(embed_dim, name="v_proj")(e_lidar) # -> (batch,embed_dim)

    # split heads
    Qh = layers.Lambda(lambda t: tf.reshape(t,(-1,H,head_dim)), name="reshape_q")(Q) # -> (batch,H,head_dim)
    Kh = layers.Lambda(lambda t: tf.reshape(t,(-1,H,head_dim)), name="reshape_k")(K) # -> (batch,H,head_dim)
    Vh = layers.Lambda(lambda t: tf.reshape(t,(-1,H,head_dim)), name="reshape_v")(V) # -> (batch,H,head_dim)

    # dot-prod scores & softmax
    scores = layers.Lambda(
        lambda z: tf.reduce_sum(z[0]*z[1],axis=-1)/tf.sqrt(tf.cast(head_dim,tf.float32)),
        name="mhf_scores"
    )([Qh,Kh])  # -> (batch,H)
    attn = layers.Activation("softmax", name="mhf_attn")(scores)  # -> (batch,H)
    attn = layers.Lambda(lambda t: tf.expand_dims(t,-1), name="mhf_attn_expand")(attn)  # -> (batch,H,1)

    # weight values and recombine
    fused_heads = layers.Multiply(name="mhf_weight")([attn,Vh])  # -> (batch,H,head_dim)
    fused = layers.Lambda(lambda t: tf.reshape(t,(-1,embed_dim)), name="mhf_concat")(fused_heads)  # -> (batch,embed_dim)
    fused = layers.Dense(embed_dim, name="mhf_out_proj")(fused)  # -> (batch,embed_dim)

    x = layers.Activation("relu", name="fusion_act")(fused)  # → (batch,embed_dim)
    x = layers.Dense(trial.suggest_int("head_units",64,512,step=64), activation="relu", name="head_dense")(x)  # → (batch, head_units)

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
