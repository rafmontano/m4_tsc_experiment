# =====================================================================
# 10_eval_xgb.R
# Evaluation of saved XGBoost model using caret::confusionMatrix
#
# Uses test split artefacts created by 08_split_export.R
# (Single source of truth, aligned with 09_hyper_xgb.R)
#
# Alignment principle:
#   - model_path + meta_path come from 0_common.R (do NOT redefine here)
# =====================================================================

library(xgboost)
library(caret)

# ---------------------------------------------------------------------
# 0) Required globals (from 00_main.R + 0_common.R)
# ---------------------------------------------------------------------

req <- c("LABEL_ID", "TAG", "model_path", "meta_path")
missing <- req[!vapply(req, exists, logical(1))]
if (length(missing) > 0L) {
  stop("Missing required globals. Run 00_main.R first. Missing: ", paste(missing, collapse = ", "))
}

label_col <- paste0("l", LABEL_ID)

# Split artefact paths (from Script 08 convention)
split_dir <- file.path("data", "splits")
test_rds_path <- file.path(split_dir, sprintf("test_l%d_%s.rds", LABEL_ID, TAG))
meta_rds_path <- file.path(split_dir, sprintf("split_meta_l%d_%s.rds", LABEL_ID, TAG))

cat("Evaluating model for label:", label_col, "\n")
cat("Model path:", model_path, "\n")
cat("Meta  path:", meta_path, "\n")
cat("Test  RDS :", test_rds_path, "\n")
cat("Split meta:", meta_rds_path, "\n\n")

# ---------------------------------------------------------------------
# 1) Load model + metadata + split artefacts
# ---------------------------------------------------------------------

if (!file.exists(model_path)) stop("Model file not found: ", model_path)
if (!file.exists(meta_path))  stop("Metadata file not found: ", meta_path)

if (!file.exists(test_rds_path)) stop("Test split RDS not found: ", test_rds_path, "\nRun 08_split_export.R first.")
if (!file.exists(meta_rds_path)) stop("Split meta RDS not found: ", meta_rds_path, "\nRun 08_split_export.R first.")

model_xgb  <- xgb.load(model_path)
meta_xgb   <- readRDS(meta_path)
split_meta <- readRDS(meta_rds_path)

predictor_cols <- meta_xgb$predictor_cols
if (is.null(predictor_cols) || length(predictor_cols) == 0L) {
  stop("predictor_cols missing/empty in model metadata: ", meta_path)
}

test_df <- readRDS(test_rds_path)

if (!label_col %in% names(test_df)) {
  stop("Label column ", label_col, " not found in test split RDS: ", test_rds_path)
}

missing_preds <- setdiff(predictor_cols, names(test_df))
if (length(missing_preds) > 0L) {
  stop(
    "Test split is missing predictor columns required by the model:\n",
    paste(missing_preds, collapse = ", ")
  )
}

cat("Test set windows:", nrow(test_df), "\n")

# ---------------------------------------------------------------------
# 2) Build X/y for evaluation (aligned with training)
# ---------------------------------------------------------------------

X_test <- as.matrix(test_df[, predictor_cols, drop = FALSE])
y_test <- as.integer(test_df[[label_col]])

# xgboost cannot handle Inf / NaN
X_test[!is.finite(X_test)] <- NA_real_

# Multiclass expects labels in {0,1,...,K-1}
min_y <- min(y_test, na.rm = TRUE)
if (is.finite(min_y) && min_y == 1L) {
  y_test <- y_test - 1L
}

# ---------------------------------------------------------------------
# 3) Predict with saved model
# ---------------------------------------------------------------------

dtest     <- xgb.DMatrix(X_test)
pred_prob <- predict(model_xgb, dtest)

# Prefer class count from metadata if available; otherwise infer
if (!is.null(meta_xgb$class_counts)) {
  num_classes <- length(meta_xgb$class_counts)
} else {
  num_classes <- as.integer(round(length(pred_prob) / nrow(X_test)))
}

pred_mat   <- matrix(pred_prob, ncol = num_classes, byrow = TRUE)
pred_class <- max.col(pred_mat) - 1L

# ---------------------------------------------------------------------
# 4) Evaluation using caret
# ---------------------------------------------------------------------

levels_all <- 0:(num_classes - 1)

y_test_factor <- factor(y_test, levels = levels_all)
pred_factor   <- factor(pred_class, levels = levels_all)

cm <- confusionMatrix(pred_factor, y_test_factor)

cat("\n====================== Caret Evaluation ======================\n")
print(cm)

# ---------------------------------------------------------------------
# 5) Macro F1 + per-class metrics
# ---------------------------------------------------------------------

caret_stats <- cm$byClass

# Coerce binary edge case to matrix (not expected here, but safe)
if (is.null(dim(caret_stats))) {
  caret_stats <- matrix(caret_stats, nrow = 1)
  colnames(caret_stats) <- names(cm$byClass)
}

precision <- caret_stats[, "Precision"]
recall    <- caret_stats[, "Recall"]

# Some caret versions provide F1; if not, compute it
if ("F1" %in% colnames(caret_stats)) {
  f1 <- caret_stats[, "F1"]
} else {
  f1 <- 2 * precision * recall / (precision + recall)
}

macro_f1 <- mean(f1, na.rm = TRUE)

metrics_df <- data.frame(
  class     = levels_all,
  precision = as.numeric(precision),
  recall    = as.numeric(recall),
  f1        = as.numeric(f1)
)

cat("\nMacro F1:", round(macro_f1, 4), "\n")
cat("\nPer-class metrics (Caret):\n")
print(metrics_df)

cat("\nOverall Accuracy:", round(as.numeric(cm$overall["Accuracy"]), 4), "\n")
cat("Kappa:",            round(as.numeric(cm$overall["Kappa"]), 4), "\n")
cat("Macro F1:",         round(macro_f1, 4), "\n")