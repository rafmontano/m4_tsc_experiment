# =====================================================================
# 10_eval_xgb.R
# Evaluation of saved XGBoost model using caret::confusionMatrix
#
# Assumes in 00_main.R:
#   - LABEL_ID       (1..4)
#   - TAG            (e.g. "y", "q", "m", "w", "d", "h")
#   - features_file  (e.g. "data/all_windows_with_features_q.rds")
#
# Model + meta paths must match 09_hyper_xgb.R:
#   - models/xgb/xgb_l{LABEL_ID}_{TAG}.model
#   - models/xgb/xgb_l{LABEL_ID}_{TAG}_meta.rds
# =====================================================================

library(dplyr)
library(xgboost)
library(caret)

# ---------------------------------------------------------------------
# 0) LABEL_ID, TAG defined in 00_main.R
# ---------------------------------------------------------------------

if (!exists("LABEL_ID")) {
  stop("LABEL_ID is not defined. Run 00_main.R first.")
}
if (!exists("TAG")) {
  stop("TAG is not defined. Ensure freq_tag(TARGET_PERIOD) was computed in 00_main.R.")
}
if (!exists("features_file")) {
  stop("features_file is not defined in 00_main.R.")
}

label_col <- paste0("l", LABEL_ID)

model_dir  <- file.path("models", "xgb")
model_path <- file.path(model_dir, sprintf("xgb_l%d_%s.model", LABEL_ID, TAG))
meta_path  <- file.path(model_dir, sprintf("xgb_l%d_%s_meta.rds", LABEL_ID, TAG))

cat("Evaluating model for label:", label_col, "\n")
cat("Model path:", model_path, "\n")
cat("Meta path :", meta_path, "\n\n")

# ---------------------------------------------------------------------
# 1) Load data, model, and metadata
# ---------------------------------------------------------------------

if (!file.exists(features_file)) {
  stop("Data file not found: ", features_file,
       "\nRun 07_ts_features.R and 09_hyper_xgb.R first.")
}

all_windows_with_features <- readRDS(features_file)

# Remove list columns if present
if ("x"  %in% names(all_windows_with_features)) all_windows_with_features$x  <- NULL
if ("xx" %in% names(all_windows_with_features)) all_windows_with_features$xx <- NULL

if (!file.exists(model_path)) stop("Model file not found: ", model_path)
if (!file.exists(meta_path))  stop("Metadata file not found: ", meta_path)

model_xgb <- xgb.load(model_path)
meta_xgb  <- readRDS(meta_path)

predictor_cols <- meta_xgb$predictor_cols
split_seed     <- meta_xgb$split_seed

if (!label_col %in% names(all_windows_with_features)) {
  stop("Label column ", label_col, " not found in dataset.")
}

# ---------------------------------------------------------------------
# 2) Recreate train/test split
# ---------------------------------------------------------------------

set.seed(split_seed)
ids <- unique(all_windows_with_features$series_id)

train_ids <- sample(ids, size = floor(0.8 * length(ids)))
test_ids  <- setdiff(ids, train_ids)

test_df <- subset(all_windows_with_features, series_id %in% test_ids)

X_test <- as.matrix(test_df[, predictor_cols, drop = FALSE])
y_test <- as.integer(test_df[[label_col]])   # numeric 0/1/2

cat("Test set windows:", nrow(test_df), "\n")

# ---------------------------------------------------------------------
# 3) Predict with saved model
# ---------------------------------------------------------------------

dtest     <- xgb.DMatrix(X_test)
pred_prob <- predict(model_xgb, dtest)

num_classes <- length(pred_prob) / nrow(X_test)
num_classes <- as.integer(round(num_classes))

pred_mat   <- matrix(pred_prob, ncol = num_classes, byrow = TRUE)
pred_class <- max.col(pred_mat) - 1L   # convert 1..K → 0..K-1

# ---------------------------------------------------------------------
# 4) Evaluation using caret
# ---------------------------------------------------------------------

# Convert to factors with consistent levels
y_test_factor <- factor(y_test,   levels = c(0, 1, 2))
pred_factor   <- factor(pred_class, levels = c(0, 1, 2))

cm <- confusionMatrix(pred_factor, y_test_factor)

cat("\n====================== Caret Evaluation ======================\n")
print(cm)

# ---------------------------------------------------------------------
# 5) Macro F1 + nicely formatted metrics
# ---------------------------------------------------------------------

caret_stats <- cm$byClass

precision <- caret_stats[, "Precision"]
recall    <- caret_stats[, "Recall"]
f1        <- caret_stats[, "F1"]

macro_f1 <- mean(f1, na.rm = TRUE)

cat("\nMacro F1:", round(macro_f1, 4), "\n")

metrics_df <- data.frame(
  class     = c(0, 1, 2),
  precision = precision,
  recall    = recall,
  f1        = f1
)

cat("\nPer-class metrics (Caret):\n")
print(metrics_df)

cat("\nOverall Accuracy:", round(cm$overall["Accuracy"], 4), "\n")
cat("Kappa:",            round(cm$overall["Kappa"], 4), "\n")
cat("Macro F1:",         round(macro_f1, 4), "\n")

