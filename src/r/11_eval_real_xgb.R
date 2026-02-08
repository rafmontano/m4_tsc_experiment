# =====================================================================
# 11_eval_real_xgb.R
# Evaluate XGBoost model (l1–l4) on REAL M4 last-window data
# and export data/predictions for Python comparison
#
# Alignment principles:
#   - Uses c_train from threshold_file_train (05_compute_c.R output)
#   - Uses predictor_cols from model metadata (09_hyper_xgb.R output)
#   - Uses scaling + z + labeling functions consistent with training (utils.R)
#   - Uses model_path/meta_path from 0_common.R (single source of truth)
# =====================================================================

library(dplyr)
library(xgboost)
library(tsfeatures)
library(forecast)
library(tibble)
library(caret)
library(parallel)

source("src/r/utils.R")     # infer_frequency, scale_pair_minmax_std, label_from_z_int, compute_z_generic, etc.
source("src/r/features.R")  # calc_features()

# ---------------------------------------------------------------------
# 0) Required globals from 00_main.R / 0_common.R
# ---------------------------------------------------------------------

req_objs <- c(
  "LABEL_ID", "WINDOW_SIZE", "TAG",
  "subset_clean_file", "threshold_file_train",
  "model_path", "meta_path"
)

missing_objs <- req_objs[!vapply(req_objs, exists, logical(1))]
if (length(missing_objs) > 0L) {
  stop("Missing required globals. Run 00_main.R first. Missing: ", paste(missing_objs, collapse = ", "))
}

label_col   <- paste0("l", LABEL_ID)
window_size <- as.integer(WINDOW_SIZE)

cat("Evaluating REAL DATA for label:", label_col, "\n")
cat("Model path    :", model_path, "\n")
cat("Meta path     :", meta_path, "\n")
cat("Subset clean  :", subset_clean_file, "\n")
cat("c_train file  :", threshold_file_train, "\n\n")

# ---------------------------------------------------------------------
# 1) Load model, metadata, c_train, and cleaned M4 data
# ---------------------------------------------------------------------

if (!file.exists(model_path))           stop("Model file not found: ", model_path)
if (!file.exists(meta_path))            stop("Metadata file not found: ", meta_path)
if (!file.exists(subset_clean_file))    stop("Clean M4 subset not found: ", subset_clean_file)
if (!file.exists(threshold_file_train)) stop("Threshold file not found: ", threshold_file_train)

model_xgb <- xgb.load(model_path)
meta_xgb  <- readRDS(meta_path)
M4_clean  <- readRDS(subset_clean_file)
c_value   <- readRDS(threshold_file_train)

predictor_cols <- meta_xgb$predictor_cols
if (is.null(predictor_cols) || length(predictor_cols) == 0L) {
  stop("predictor_cols missing/empty in metadata: ", meta_path)
}

# Hard validation: required functions must exist
req_fns <- c(
  "infer_frequency", "scale_pair_minmax_std",
  "label_from_z_int", "calc_features", "compute_z_generic"
)

missing_fns <- req_fns[!vapply(req_fns, exists, logical(1), mode = "function")]
if (length(missing_fns) > 0L) {
  stop("Missing required functions after sourcing utils/features: ", paste(missing_fns, collapse = ", "))
}

# Infer frequency from period
period <- M4_clean[[1]]$period
freq   <- infer_frequency(period)
cat("Detected period:", as.character(period), "→ frequency =", freq, "\n\n")

# ---------------------------------------------------------------------
# 2) Build REAL evaluation dataset (last window per series, parallel)
# ---------------------------------------------------------------------

n_series <- length(M4_clean)

n_cores <- max(1, detectCores(logical = FALSE) - 1)
cat("Using", n_cores, "cores for REAL evaluation.\n")

cl <- NULL

# PSOCK is safest when running via source() / RStudio
cl <- makeCluster(n_cores, type = "PSOCK", outfile = "")
doParallel::registerDoParallel(cl)

on.exit({
  if (!is.null(cl)) {
    try(stopCluster(cl), silent = TRUE)
  }
}, add = TRUE)

clusterEvalQ(cl, {
  library(tibble)
  library(dplyr)
  library(forecast)
  library(tsfeatures)
  source("src/r/utils.R")
  source("src/r/features.R")
  NULL
})

clusterExport(
  cl,
  c(
    "M4_clean", "window_size", "LABEL_ID", "c_value", "freq",
    "calc_features", "compute_z_generic", "label_from_z_int",
    "scale_pair_minmax_std"
  ),
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
    
    if (length(x_full) < window_size) return(NULL)
    
    x_win <- tail(x_full, window_size)
    
    scaled <- scale_pair_minmax_std(x_win, xx)
    x_std  <- scaled$x_std
    xx_std <- scaled$xx_std
    
    z_val    <- compute_z_generic(x_std, xx_std, LABEL_ID)
    true_lbl <- label_from_z_int(z_val, c_value)
    
    feats <- tryCatch(
      {
        calc_features(list(x = ts(x_std, frequency = freq)))$features
      },
      error = function(e) NULL
    )
    if (is.null(feats)) return(NULL)
    
    tibble(
      st         = s$st,
      true_label = as.integer(true_lbl)
    ) %>% bind_cols(feats)
  }
)

real_eval_df <- dplyr::bind_rows(rows)
cat("Built REAL evaluation dataset with", nrow(real_eval_df), "rows.\n")

real_eval_rds <- sprintf("data/real_eval_l%d_%s_df.rds", LABEL_ID, TAG)
saveRDS(real_eval_df, real_eval_rds)
cat("Saved REAL evaluation RDS → ", real_eval_rds, "\n", sep = "")

if (nrow(real_eval_df) == 0L) {
  stop("[11] REAL evaluation dataset is empty. Check window_size, feature extraction, or M4 subset integrity.")
}

# ---------------------------------------------------------------------
# 3) Align predictor columns to training schema
# ---------------------------------------------------------------------

for (col in predictor_cols) {
  if (!col %in% names(real_eval_df)) {
    real_eval_df[[col]] <- 0
  }
}

X_real <- as.matrix(real_eval_df[, predictor_cols, drop = FALSE])
y_real <- as.integer(real_eval_df$true_label)

storage.mode(X_real) <- "double"
X_real[!is.finite(X_real)] <- NA_real_

min_y <- min(y_real, na.rm = TRUE)
if (is.finite(min_y) && min_y == 1L) {
  y_real <- y_real - 1L
}

# ---------------------------------------------------------------------
# 4) Export REAL evaluation dataset for Python
# ---------------------------------------------------------------------

export_dir <- file.path("data", "export")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

real_data_export <- cbind(
  st         = real_eval_df$st,
  true_label = y_real,
  X_real
)

real_data_path <- file.path(export_dir, sprintf("real_eval_l%d_%s_data.csv", LABEL_ID, TAG))
predictor_path <- file.path(export_dir, sprintf("predictors_l%d_%s.csv", LABEL_ID, TAG))

write.csv(real_data_export, real_data_path, row.names = FALSE)
write.csv(data.frame(feature = predictor_cols), predictor_path, row.names = FALSE)

cat("\nExported REAL evaluation data for Python:\n")
cat("  → ", real_data_path, "\n", sep = "")
cat("  → ", predictor_path, "\n", sep = "")

# ---------------------------------------------------------------------
# 5) Predict probabilities and classes
# ---------------------------------------------------------------------

dreal     <- xgb.DMatrix(X_real)
pred_prob <- predict(model_xgb, dreal)

if (!is.null(meta_xgb$class_counts)) {
  num_classes <- length(meta_xgb$class_counts)
} else {
  num_classes <- as.integer(round(length(pred_prob) / nrow(X_real)))
}

pred_mat   <- matrix(pred_prob, ncol = num_classes, byrow = TRUE)
pred_class <- max.col(pred_mat) - 1L

probs_df <- as.data.frame(pred_mat)
colnames(probs_df) <- paste0("prob_", 0:(num_classes - 1))

real_pred_export <- cbind(
  st         = real_eval_df$st,
  true_label = y_real,
  pred_class = pred_class,
  probs_df
)

real_pred_path <- file.path(export_dir, sprintf("real_eval_l%d_%s_preds.csv", LABEL_ID, TAG))
write.csv(real_pred_export, real_pred_path, row.names = FALSE)
cat("  → ", real_pred_path, "\n", sep = "")

# ---------------------------------------------------------------------
# 6) Evaluation using caret
# ---------------------------------------------------------------------

levels_all <- 0:(num_classes - 1)

y_factor    <- factor(y_real,     levels = levels_all)
pred_factor <- factor(pred_class, levels = levels_all)

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

f1 <- if ("F1" %in% colnames(by_class)) by_class[, "F1"] else (2 * precision * recall / (precision + recall))

metrics_df <- data.frame(
  class     = levels_all,
  precision = as.numeric(precision),
  recall    = as.numeric(recall),
  f1        = as.numeric(f1)
)

macro_f1 <- mean(metrics_df$f1, na.rm = TRUE)

cat("\nPer-class metrics (REAL, caret) for", label_col, ":\n")
print(metrics_df)
cat("\nMacro F1 (REAL):", round(macro_f1, 4), "\n")

# ---------------------------------------------------------------------
# 7) Save REAL evaluation results
# ---------------------------------------------------------------------

results_dir <- file.path("results", "xgb")
if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)

conf_df <- as.data.frame.matrix(conf_mat)
conf_df$Predicted_class <- rownames(conf_df)
conf_df <- conf_df[, c("Predicted_class", setdiff(colnames(conf_df), "Predicted_class"))]

conf_path_csv <- file.path(results_dir, sprintf("real_confusion_l%d_%s.csv", LABEL_ID, TAG))
conf_path_rds <- file.path(results_dir, sprintf("real_confusion_l%d_%s.rds", LABEL_ID, TAG))

write.csv(conf_df, conf_path_csv, row.names = FALSE)
saveRDS(conf_mat, conf_path_rds)

metrics_df$label_id         <- LABEL_ID
metrics_df$tag              <- TAG
metrics_df$overall_accuracy <- overall_acc
metrics_df$macro_f1         <- macro_f1
metrics_df$kappa            <- kappa_val

metrics_path_csv <- file.path(results_dir, sprintf("real_metrics_l%d_%s.csv", LABEL_ID, TAG))
metrics_path_rds <- file.path(results_dir, sprintf("real_metrics_l%d_%s.rds", LABEL_ID, TAG))

write.csv(metrics_df, metrics_path_csv, row.names = FALSE)
saveRDS(metrics_df, metrics_path_rds)

cat("\nSaved REAL evaluation results for", label_col, "to:\n")
cat("  Confusion (csv): ", conf_path_csv,  "\n", sep = "")
cat("  Confusion (rds): ", conf_path_rds,  "\n", sep = "")
cat("  Metrics   (csv): ", metrics_path_csv, "\n", sep = "")
cat("  Metrics   (rds): ", metrics_path_rds, "\n", sep = "")

