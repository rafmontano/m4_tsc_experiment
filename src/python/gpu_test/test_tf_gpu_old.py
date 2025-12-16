# test_tf_gpu_old.py
# Simple script to check TensorFlow GPU visibility and run a small GPU op

import tensorflow as tf

def main():
    print("TensorFlow version:", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs visible to TensorFlow:", gpus)

    if not gpus:
        print("No GPU detected by TensorFlow. Check CUDA/cuDNN and versions.")
        return

    with tf.device("/GPU:0"):
        a = tf.random.normal((2000, 2000))
        b = tf.random.normal((2000, 2000))
        c = tf.matmul(a, b)

    print("Matmul result shape:", c.shape)
    print("TensorFlow GPU test completed successfully.")

if __name__ == "__main__":
    main()
