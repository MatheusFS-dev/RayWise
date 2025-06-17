import numpy as np
import tensorflow as tf

# Generate random training data
print("Generating random training data...")
num_samples = 10000
input_shape = (32, 32, 3)  # RGB images
num_classes = 10

X_train = np.random.randn(num_samples, *input_shape).astype(np.float32)
y_train = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, num_samples), num_classes)

X_val = np.random.randn(2000, *input_shape).astype(np.float32)
y_val = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 2000), num_classes)

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")

# Create CNN model
print("Creating CNN model...")
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Model summary:")
model.summary()

# Train model
print("Starting training...")
history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_val, y_val), verbose=1)

print("Training completed!")

# Test profiling function with different batch sizes
print("\nTesting profiling function:")

from araras.keras.utils.profiler import  get_memory_and_time

mem, inf1 = get_memory_and_time(model, batch_size=32, device="GPU:0", warmup_runs=0, test_runs=1)

print(f"Peak memory usage: {mem / (1024 ** 2):.2f} MB")
print(f"Average inference: {inf1:.4f} seconds")
