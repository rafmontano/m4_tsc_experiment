# scripts/02_tf_matmul.py
import time
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)

size = 4096
a = tf.random.normal([size, size])
b = tf.random.normal([size, size])

# Warm-up
_ = tf.matmul(a, b)

t0 = time.time()
c = tf.matmul(a, b)
_ = c.numpy()
t1 = time.time()

print("Matmul result shape:", c.shape)
print("Elapsed seconds:", round(t1 - t0, 4))