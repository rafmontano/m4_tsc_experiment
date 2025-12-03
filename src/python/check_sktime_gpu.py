from sktime.datasets import load_unit_test
from sktime.classification.deep_learning import InceptionTimeClassifier
import tensorflow as tf

print("TF GPUs:", tf.config.list_physical_devices("GPU"))

X_train, y_train = load_unit_test(split="train")
X_test, y_test = load_unit_test(split="test")

clf = InceptionTimeClassifier(
    n_epochs=2,
    batch_size=16,
    verbose=True,
)

clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))
