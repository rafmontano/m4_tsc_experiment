# =====================================================================
# 11_eval_real_xgb.R
# Evaluate XGBoost model (l1–l4) on REAL M4 last-window data
# and export data/predictions for Python comparison
#
# Assumes in 00_main.R:
#   - LABEL_ID          (1..4)
#   - WINDOW_SIZE
#   - TAG               (e.g. "y", "q", "m", "w", "d", "h")
#   - subset_clean_file (e.g. "data/M4_subset_clean_q.rds")
#   - threshold_file    (e.g. "data/label_threshold_c_q.rds")
#
# Model + meta paths must match 09_hyper_xgb.R:
#   - models/xgb/xgb_l{LABEL_ID}_{TAG}.model
#   - models/xgb/xgb_l{LABEL_ID}_{TAG}_meta.rds
# =====================================================================

library(dplyr)
library(xgboost)
library(tsfeatures)
library(forecast)
library(tibble)
library(caret)
library(parallel)

source("src/r/utils.R")     # infer_frequency, scale_pair_minmax_std, label_from_z_int, etc.
source("src/r/features.R")  # calc_features()

# ---------------------------------------------------------------------
# 0) Check required globals from 00_main.R
# ---------------------------------------------------------------------

if (!exists("LABEL_ID")) {
  stop("LABEL_ID not defined. Please set it in 00_main.R before running this script.")
}
if (!exists("WINDOW_SIZE")) {
  stop("WINDOW_SIZE not defined. Please set it in 00_main.R before running this script.")
}
if (!exists("TAG")) {
  stop("TAG not defined. Ensure freq_tag(TARGET_PERIOD) was computed in 00_main.R.")
}
if (!exists("subset_clean_file")) {
  stop("subset_clean_file not defined in 00_main.R.")
}
if (!exists("threshold_file")) {
  stop("threshold_file not defined in 00_main.R.")
}

label_col   <- paste0("l", LABEL_ID)
window_size <- WINDOW_SIZE

model_dir  <- file.path("models", "xgb")
model_path <- file.path(model_dir, sprintf("xgb_l%d_%s.model", LABEL_ID, TAG))
meta_path  <- file.path(model_dir, sprintf("xgb_l%d_%s_meta.rds", LABEL_ID, TAG))

cat("Evaluating REAL DATA for label:", label_col, "\n")
cat("Model path:", model_path, "\n")
cat("Meta path :", meta_path, "\n\n")

# ---------------------------------------------------------------------
# 1) Load model, metadata, training threshold c, and cleaned M4 data
# ---------------------------------------------------------------------

if (!file.exists(model_path)) stop("Model file not found: ", model_path)
if (!file.exists(meta_path))  stop("Metadata file not found: ", meta_path)
if (!file.exists(subset_clean_file)) stop("Clean M4 subset not found: ", subset_clean_file)
if (!file.exists(threshold_file))    stop("Threshold file not found: ", threshold_file)

model_xgb <- xgb.load(model_path)
meta_xgb  <- readRDS(meta_path)
M4_clean  <- readRDS(subset_clean_file)
c_value   <- readRDS(threshold_file)

predictor_cols <- meta_xgb$predictor_cols

# Infer frequency from the subset's period (e.g., "Quarterly" → 4)
period <- M4_clean[[1]]$period
freq   <- infer_frequency(period)

cat("Detected period:", as.character(period), "→ frequency =", freq, "\n\n")

# ---------------------------------------------------------------------
# 2) Label helper: generic z function for label IDs 1..4
#     (must match training logic in 06_labels.R)
# ---------------------------------------------------------------------



# label_from_z_int(z, c) comes from utils.R

# ---------------------------------------------------------------------
# 3) Build evaluation dataset: LAST window per M4 series (parallel)
# ---------------------------------------------------------------------

n_series <- length(M4_clean)

n_cores <- max(1, detectCores(logical = FALSE))
cat("Using", n_cores, "cores for REAL evaluation.\n")

#cl <- makeCluster(n_cores)
cl <- makeForkCluster(n_cores)

# Make sure each worker has the same environment
clusterEvalQ(cl, {
  library(tibble)
  library(dplyr)
  library(forecast)
  library(tsfeatures)
  source("src/r/utils.R")
  source("src/r/features.R")
})

# Export objects needed inside the worker function
clusterExport(
  cl,
  c("M4_clean", "window_size", "LABEL_ID", "c_value", "freq",
    "calc_features", "compute_z_generic", "label_from_z_int"),
  envir = environment()
)

clusterSetRNGStream(cl, 123)

rows <- parLapply(
  cl,
  X = seq_len(n_series),
  fun = function(i) {
    s <- M4_clean[[i]]
    
    x_full <- as.numeric(s$x)
    xx     <- as.numeric(s$xx)
    
    # Skip too-short series
    if (length(x_full) < window_size) return(NULL)
    
    # Last window
    x_win <- tail(x_full, window_size)
    
    # Joint scaling of window + future horizon
    scaled <- scale_pair_minmax_std(x_win, xx)
    x_std  <- scaled$x_std
    xx_std <- scaled$xx_std
    
    # Label using same logic as training
    z_val    <- compute_z_generic(x_std, xx_std, LABEL_ID)
    true_lbl <- label_from_z_int(z_val, c_value)
    
    # Safe feature extraction
    feats <- tryCatch(
      {
        calc_features(list(x = ts(x_std, frequency = freq)))$features
      },
      error = function(e) {
        message("Skipping series ", s$st,
                " (worker ", i, ") due to feature error: ", e$message)
        NULL
      }
    )
    if (is.null(feats)) return(NULL)
    
    tibble(
      st         = s$st,
      true_label = true_lbl
    ) %>%
      bind_cols(feats)
  }
)

stopCluster(cl)

real_eval_df <- dplyr::bind_rows(rows)
cat("Built REAL evaluation dataset with", nrow(real_eval_df), "rows.\n")

# Optional: save for inspection (R-native, TAG-aware)
real_eval_rds <- sprintf("data/real_eval_l%d_%s_df.rds", LABEL_ID, TAG)
saveRDS(real_eval_df, real_eval_rds)
cat("Saved REAL evaluation RDS → ", real_eval_rds, "\n", sep = "")

# 4) Align predictor columns (missing → 0)
# ---------------------------------------------------------------------

for (col in predictor_cols) {
  if (!col %in% names(real_eval_df)) {
    real_eval_df[[col]] <- 0
  }
}

X_real <- as.matrix(real_eval_df[, predictor_cols, drop = FALSE])
y_real <- as.integer(real_eval_df$true_label)

# 4a) Sanity check: remove NA / Inf before XGBoost --------------------

# ensure numeric matrix
storage.mode(X_real) <- "double"

na_count  <- sum(is.na(X_real))
inf_count <- sum(is.infinite(X_real))

if (na_count > 0L || inf_count > 0L) {
  cat(
    "\n[WARN] X_real contains non-finite values:",
    na_count, "NA and", inf_count, "Inf.\n",
    "       Imputing them with column means (or 0 if all bad).\n"
  )
  
  for (j in seq_len(ncol(X_real))) {
    col <- X_real[, j]
    bad <- !is.finite(col)
    if (any(bad)) {
      good_vals <- col[is.finite(col)]
      repl <- if (length(good_vals) > 0L) mean(good_vals) else 0
      col[bad] <- repl
      X_real[, j] <- col
    }
  }
  
  # quick assert
  stopifnot(!any(is.na(X_real)), !any(is.infinite(X_real)))
}


# ---------------------------------------------------------------------
# 4b) Export evaluation dataset for Python / sktime / sklearn
# ---------------------------------------------------------------------

export_dir <- file.path("data", "export")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

real_data_export <- cbind(
  st         = real_eval_df$st,
  true_label = y_real,
  X_real
)

real_data_path <- file.path(
  export_dir,
  sprintf("real_eval_l%d_%s_data.csv", LABEL_ID, TAG)
)

predictor_path <- file.path(
  export_dir,
  sprintf("predictors_l%d_%s.csv", LABEL_ID, TAG)
)

write.csv(real_data_export, real_data_path, row.names = FALSE)

# predictor list (same as from training; overwriting is fine)
write.csv(
  data.frame(feature = predictor_cols),
  predictor_path,
  row.names = FALSE
)

cat("\nExported REAL evaluation data for Python:\n")
cat("  → ", real_data_path, "\n", sep = "")
cat("  → ", predictor_path, "\n", sep = "")

# ---------------------------------------------------------------------
# 5) Predict class probabilities
# ---------------------------------------------------------------------

dreal     <- xgb.DMatrix(X_real)
pred_prob <- predict(model_xgb, dreal)

num_classes <- length(pred_prob) / nrow(X_real)
num_classes <- as.integer(round(num_classes))

pred_mat   <- matrix(pred_prob, ncol = num_classes, byrow = TRUE)
pred_class <- max.col(pred_mat) - 1L  # convert to 0/1/2

# ---------------------------------------------------------------------
# 5b) Export predictions (labels + probabilities) for Python
# ---------------------------------------------------------------------

probs_df <- as.data.frame(pred_mat)
colnames(probs_df) <- paste0("prob_", 0:(num_classes - 1))

real_pred_export <- cbind(
  st         = real_eval_df$st,
  true_label = y_real,
  pred_class = pred_class,
  probs_df
)

real_pred_path <- file.path(
  export_dir,
  sprintf("real_eval_l%d_%s_preds.csv", LABEL_ID, TAG)
)

write.csv(real_pred_export, real_pred_path, row.names = FALSE)

cat("  → ", real_pred_path, "\n", sep = "")

# ---------------------------------------------------------------------
# 6) Evaluation using caret (Confusion, Accuracy, per-class P/R/F1)
# ---------------------------------------------------------------------

y_factor    <- factor(y_real,     levels = c(0, 1, 2))
pred_factor <- factor(pred_class, levels = c(0, 1, 2))

cm <- confusionMatrix(pred_factor, y_factor)

cat("\n====================== Caret Evaluation (REAL) ======================\n")
print(cm)

conf_mat <- cm$table

cat("\nConfusion Matrix (REAL data) — label ", label_col, ":\n", sep = "")
print(conf_mat)

overall_acc <- as.numeric(cm$overall["Accuracy"])
kappa_val   <- as.numeric(cm$overall["Kappa"])

cat("\nAccuracy on REAL data:", round(overall_acc, 4), "\n")
cat("Kappa on REAL data   :", round(kappa_val, 4), "\n")

by_class <- cm$byClass
if (is.null(dim(by_class))) {
  by_class <- matrix(by_class, nrow = 1)
  rownames(by_class) <- levels(y_factor)
}

precision <- by_class[, "Precision"]
recall    <- by_class[, "Recall"]

if ("F1" %in% colnames(by_class)) {
  f1 <- by_class[, "F1"]
} else {
  f1 <- 2 * precision * recall / (precision + recall)
}

metrics_df <- data.frame(
  class     = c(0, 1, 2),
  precision = as.numeric(precision),
  recall    = as.numeric(recall),
  f1        = as.numeric(f1)
)

macro_f1 <- mean(metrics_df$f1, na.rm = TRUE)

cat("\nPer-class metrics (REAL, caret) for", label_col, ":\n")
print(metrics_df)

cat("\nMacro F1 (REAL):", round(macro_f1, 4), "\n")

# ---------------------------------------------------------------------
# 7) Save REAL evaluation results (confusion matrix + metrics) to disk
# ---------------------------------------------------------------------

results_dir <- file.path("results", "xgb")
if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)

# 7a) Confusion matrix as a data frame
conf_df <- as.data.frame.matrix(conf_mat)
conf_df$Predicted_class <- rownames(conf_df)
conf_df <- conf_df[, c("Predicted_class", setdiff(colnames(conf_df), "Predicted_class"))]

conf_path_csv <- file.path(
  results_dir,
  sprintf("real_confusion_l%d_%s.csv", LABEL_ID, TAG)
)
conf_path_rds <- file.path(
  results_dir,
  sprintf("real_confusion_l%d_%s.rds", LABEL_ID, TAG)
)

write.csv(conf_df, conf_path_csv, row.names = FALSE)
saveRDS(conf_mat, conf_path_rds)

# 7b) Per-class metrics + overall summary
metrics_df$label_id         <- LABEL_ID
metrics_df$tag              <- TAG
metrics_df$overall_accuracy <- overall_acc
metrics_df$macro_f1         <- macro_f1
metrics_df$kappa            <- kappa_val

metrics_path_csv <- file.path(
  results_dir,
  sprintf("real_metrics_l%d_%s.csv", LABEL_ID, TAG)
)
metrics_path_rds <- file.path(
  results_dir,
  sprintf("real_metrics_l%d_%s.rds", LABEL_ID, TAG)
)

write.csv(metrics_df, metrics_path_csv, row.names = FALSE)
saveRDS(metrics_df, metrics_path_rds)

cat("\nSaved REAL evaluation results for", label_col, "to:\n")
cat("  Confusion (csv): ", conf_path_csv,  "\n", sep = "")
cat("  Confusion (rds): ", conf_path_rds,  "\n", sep = "")
cat("  Metrics   (csv): ", metrics_path_csv, "\n", sep = "")
cat("  Metrics   (rds): ", metrics_path_rds, "\n", sep = "")

