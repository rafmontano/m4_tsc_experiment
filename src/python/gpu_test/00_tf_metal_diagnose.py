# src/00_tf_metal_diagnose.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_METAL_LOGGING"] = "1"

import tensorflow as tf

print("TF version:", tf.__version__)
print("CPUs:", tf.config.list_physical_devices("CPU"))
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("Pluggable:", tf.config.list_physical_devices("PLUGGABLE_DEVICE"))

try:
    from tensorflow.python.framework import config as tf_config
    print("Logical devices:", tf.config.list_logical_devices())
except Exception as e:
    print("Could not list logical devices:", e)

tf.debugging.set_log_device_placement(True)

# A small op that should show device placement
a = tf.random.normal([2048, 2048])
b = tf.random.normal([2048, 2048])
c = tf.matmul(a, b)
_ = c.numpy()

print("matmul completed")