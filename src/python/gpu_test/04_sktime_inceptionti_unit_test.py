# scripts/04_sktime_inceptiontime_unit_test.py
import tensorflow as tf
from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier
from sktime.datasets import load_unit_test
from sklearn.metrics import accuracy_score

print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

X_train, y_train = load_unit_test(split="train")
X_test, y_test = load_unit_test(split="test")

clf = InceptionTimeClassifier(
    n_epochs=20,
    batch_size=16,
    verbose=True,
)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)