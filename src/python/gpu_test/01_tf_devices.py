# scripts/01_tf_devices.py
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())

cpus = tf.config.list_physical_devices("CPU")
gpus = tf.config.list_physical_devices("GPU")

print("CPUs:", cpus)
print("GPUs:", gpus)

if gpus:
    for i, gpu in enumerate(gpus):
        details = tf.config.experimental.get_device_details(gpu)
        print(f"GPU[{i}] details:", details)
else:
    print("No GPU device detected by TensorFlow.")