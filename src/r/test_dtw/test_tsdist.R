#test_tsdist

# Load libraries
library(dtw)
library(proxy)
library(TSdist)

# 1. Prepare sample data
# Create two classes of time series data
# Class A (e.g., sine waves)
ts_A <- lapply(1:20, function(i) sin(seq(0, 2*pi, length.out = 50)) + rnorm(50, 0, 0.2))
# Class B (e.g., cosine waves)
ts_B <- lapply(1:20, function(i) cos(seq(0, 2*pi, length.out = 50)) + rnorm(50, 0, 0.2))

# Combine into training set (matrix where each row is a time series)
train_data <- do.call(rbind, c(ts_A[1:15], ts_B[1:15]))
train_labels <- c(rep("A", 15), rep("B", 15))

# Combine into testing set
test_data <- do.call(rbind, c(ts_A[16:20], ts_B[16:20]))
test_labels <- c(rep("A", 5), rep("B", 5))

# Register DTW as a distance measure in the 'proxy' package (needed for OneNN to find it)
# The 'TSdist' package does this automatically, but it's good practice.
pr_DB$set_entry(FUN = dtw::dtw, names = "DTW", method = "distance")

# 2. Perform 1NN classification with DTW distance
# The 'OneNN' function handles the distance matrix calculation and classification
predictions <- OneNN(
  train_data,
  test_data,
  train_labels,
  distance = "DTW" # Specify the DTW distance measure
)

# 3. Evaluate the results
# Create a confusion matrix or calculate accuracy
actual_labels <- factor(test_labels)
predicted_labels <- factor(predictions)

# View the results
confusion_matrix <- table(Actual = actual_labels, Predicted = predicted_labels)
print(confusion_matrix)

accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("\nClassification Accuracy:", accuracy, "\n")
