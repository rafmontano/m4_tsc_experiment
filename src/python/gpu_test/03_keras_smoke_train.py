# scripts/03_keras_smoke_train.py
import tensorflow as tf
import numpy as np

print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# Simple synthetic classification
n = 20000
d = 64

rng = np.random.default_rng(123)
X = rng.normal(size=(n, d)).astype(np.float32)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int32)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(d,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    X, y,
    epochs=5,
    batch_size=256,
    validation_split=0.2,
    verbose=2,
)

print("Final val_accuracy:", history.history["val_accuracy"][-1])