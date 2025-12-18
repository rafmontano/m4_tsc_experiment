# src/python/gpu_test/10_inceptiontime_smoke_docker.py

import tensorflow as tf
from sktime.datasets import load_unit_test
from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier

def main():
    print("TF:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)

    clf = InceptionTimeClassifier(n_epochs=3, batch_size=16, verbose=True)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print("Smoke-train OK. Test accuracy:", acc)

if __name__ == "__main__":
    main()
