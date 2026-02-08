# =====================================================================
# 09_hyper_xgb.R
# Hyperparameter search + training for XGBoost (labels l1–l4)
# Multiclass (0 = Down, 1 = Neutral, 2 = Up)
#
# Uses train/test split artefacts created by 08_split_export.R
# =====================================================================

library(dplyr)
library(xgboost)
library(rBayesianOptimization)
library(parallel)

# ---------------------------------------------------------------------
# 0) Required globals + paths
# ---------------------------------------------------------------------

# Required globals from 00_main.R + 0_common.R
req <- c("LABEL_ID", "TAG", "features_file", "model_dir", "model_path", "meta_path", "FORCE_RERUN")
missing <- req[!vapply(req, exists, logical(1))]
if (length(missing) > 0L) {
  stop("Missing required globals. Run 00_main.R first. Missing: ", paste(missing, collapse = ", "))
}

label_col <- paste0("l", LABEL_ID)

# model_dir/model_path/meta_path are defined in 0_common.R (single source of truth)
if (!dir.exists(model_dir)) dir.create(model_dir, recursive = TRUE)

best_param_path <- file.path(model_dir, sprintf("xgb_l%d_%s_bestpar.rds", LABEL_ID, TAG))

split_dir <- file.path("data", "splits")
train_rds_path <- file.path(split_dir, sprintf("train_l%d_%s.rds", LABEL_ID, TAG))
test_rds_path  <- file.path(split_dir, sprintf("test_l%d_%s.rds",  LABEL_ID, TAG))
meta_rds_path  <- file.path(split_dir, sprintf("split_meta_l%d_%s.rds", LABEL_ID, TAG))

cat("Training XGBoost model for label:", label_col, "\n")
cat("Model will be saved to:", model_path, "\n")
cat("Metadata will be saved to:", meta_path, "\n\n")

# ---------------------------------------------------------------------
# 1) Load full dataset (only to derive predictor_cols consistently)
# ---------------------------------------------------------------------

if (!file.exists(features_file)) {
  stop("File not found: ", features_file, "\nRun 07_ts_features.R first.")
}

all_windows_with_features <- readRDS(features_file)
cat("Loaded dataset:", nrow(all_windows_with_features), "rows from", features_file, "\n")

if ("x"  %in% names(all_windows_with_features)) all_windows_with_features$x  <- NULL
if ("xx" %in% names(all_windows_with_features)) all_windows_with_features$xx <- NULL

if (!label_col %in% names(all_windows_with_features)) {
  stop("Label column ", label_col, " not found in dataset.")
}

predictor_cols <- setdiff(
  colnames(all_windows_with_features),
  c("l1", "l2", "l3", "l4", "series_id", "series_name")
)
cat("Number of predictor features before filtering:", length(predictor_cols), "\n")

# ---------------------------------------------------------------------
# 2) Load train/test split (single source of truth)
# ---------------------------------------------------------------------

if (!file.exists(train_rds_path) || !file.exists(test_rds_path) || !file.exists(meta_rds_path)) {
  stop(
    "Missing split artefacts in data/splits.\n",
    "Expected:\n",
    " - ", train_rds_path, "\n",
    " - ", test_rds_path,  "\n",
    " - ", meta_rds_path,  "\n",
    "Run 08_split_export.R first."
  )
}

train_df   <- readRDS(train_rds_path)
test_df    <- readRDS(test_rds_path)
split_meta <- readRDS(meta_rds_path)

cat("[09] Loaded split artefacts:\n")
cat("  -> ", train_rds_path, "\n", sep = "")
cat("  -> ", test_rds_path,  "\n", sep = "")
cat("  -> ", meta_rds_path,  "\n", sep = "")
cat("[09] Train windows:", nrow(train_df), " | Test windows:", nrow(test_df), "\n")

if (!label_col %in% names(train_df) || !label_col %in% names(test_df)) {
  stop("Label column ", label_col, " missing from train/test split RDS.")
}

# ---------------------------------------------------------------------
# 3) Drop constant features based ONLY on train set
# ---------------------------------------------------------------------

feature_df_train <- train_df[, predictor_cols, drop = FALSE]
feature_vars  <- sapply(feature_df_train, function(col) var(col, na.rm = TRUE))
constant_cols <- names(feature_vars[feature_vars == 0 | is.na(feature_vars)])

if (length(constant_cols) > 0L) {
  message("Removing constant columns: ", paste(constant_cols, collapse = ", "))
  predictor_cols <- setdiff(predictor_cols, constant_cols)
}

X_train <- as.matrix(train_df[, predictor_cols, drop = FALSE])
X_test  <- as.matrix(test_df[,  predictor_cols, drop = FALSE])

# REQUIRED: xgboost cannot handle Inf / NaN
X_train[!is.finite(X_train)] <- NA_real_
X_test[!is.finite(X_test)]   <- NA_real_

y_train <- as.integer(train_df[[label_col]])
y_test  <- as.integer(test_df[[label_col]])

if (length(y_train) == 0L) stop("[09] Empty training set after split.")
if (length(unique(y_train)) < 2L) stop("[09] y_train has < 2 classes. CV/BO cannot run.")

# REQUIRED: xgboost multiclass expects labels in {0,1,...,K-1}
min_y <- min(y_train, na.rm = TRUE)
if (is.finite(min_y) && min_y == 1L) {
  y_train <- y_train - 1L
  y_test  <- y_test  - 1L
}

cat("Final number of predictor features:", length(predictor_cols), "\n")

# ---------------------------------------------------------------------
# 4) Class weights
# ---------------------------------------------------------------------

class_counts <- table(y_train)
K <- length(class_counts)
N <- length(y_train)

class_weights <- N / (K * as.numeric(class_counts))
names(class_weights) <- names(class_counts)

cat("\nClass distribution (train):\n")
print(class_counts)
cat("Class weights used:\n")
print(class_weights)

w_train <- class_weights[as.character(y_train)]

# ---------------------------------------------------------------------
# 4.5) Export TRAIN/TEST matrices to CSV (unchanged behaviour)
# ---------------------------------------------------------------------

export_dir <- file.path("data", "export")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

train_path     <- file.path(export_dir, sprintf("train_l%d_%s.csv",      LABEL_ID, TAG))
test_path      <- file.path(export_dir, sprintf("test_l%d_%s.csv",       LABEL_ID, TAG))
predictor_path <- file.path(export_dir, sprintf("predictors_l%d_%s.csv", LABEL_ID, TAG))

write.csv(cbind(label = y_train, X_train), train_path, row.names = FALSE)
write.csv(cbind(label = y_test,  X_test),  test_path,  row.names = FALSE)
write.csv(data.frame(feature = predictor_cols), predictor_path, row.names = FALSE)

cat("\nExported data for Python/sklearn/sktime:\n")
cat(" → ", train_path, "\n", sep = "")
cat(" → ", test_path,  "\n", sep = "")
cat(" → ", predictor_path, "\n", sep = "")

# ---------------------------------------------------------------------
# 5) Bayesian Optimization hyperparameter search
# ---------------------------------------------------------------------

xgb_hyperparameter_search <- function(
    X, y, w,
    nfold = 3,
    init_points = 4,
    n_iter = 6,
    nrounds_max = 250,
    nthread = max(1, detectCores() - 1)
) {
  y <- as.integer(y)
  dtrain <- xgb.DMatrix(data = X, label = y, weight = w)
  
  xgb_cv_bayes <- function(max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree) {
    
    params <- list(
      booster          = "gbtree",
      objective        = "multi:softprob",
      eval_metric      = "mlogloss",
      num_class        = length(unique(y)),
      max_depth        = as.integer(max_depth),
      eta              = eta,
      gamma            = gamma,
      min_child_weight = min_child_weight,
      subsample        = subsample,
      colsample_bytree = colsample_bytree,
      nthread          = nthread
    )
    
    cv <- tryCatch(
      xgb.cv(
        params                = params,
        data                  = dtrain,
        nrounds               = nrounds_max,
        nfold                 = nfold,
        showsd                = TRUE,
        verbose               = FALSE,
        early_stopping_rounds = 15
      ),
      error = function(e) NULL
    )
    
    # REQUIRED: objective must always return a scalar Score
    if (is.null(cv) || is.null(cv$evaluation_log) || nrow(cv$evaluation_log) == 0L) {
      return(list(Score = -1e6 + runif(1, -1e-3, 1e-3), Pred = 1L))
    }
    
    # If best_iteration is missing, fall back to the min test loss row
    if (is.null(cv$best_iteration) || !is.finite(cv$best_iteration)) {
      j <- which.min(cv$evaluation_log$test_mlogloss_mean)
      val <- cv$evaluation_log$test_mlogloss_mean[j]
      if (!is.finite(val)) return(list(Score = -1e6 + runif(1, -1e-3, 1e-3), Pred = 1L))
      return(list(Score = -val, Pred = as.integer(j)))
    }
    
    val <- cv$evaluation_log$test_mlogloss_mean[cv$best_iteration]
    if (!is.finite(val) || length(val) == 0L) {
      return(list(Score = -1e6 + runif(1, -1e-3, 1e-3), Pred = 1L))
    }
    
    list(Score = -val, Pred = as.integer(cv$best_iteration))
  }
  
  bounds <- list(
    max_depth        = c(8L, 12L),
    eta              = c(0.08, 0.15),
    gamma            = c(0, 1),
    min_child_weight = c(1, 6),
    subsample        = c(0.8, 1.0),
    colsample_bytree = c(0.6, 0.85)
  )
  
  BayesianOptimization(
    FUN         = xgb_cv_bayes,
    bounds      = bounds,
    init_points = init_points,
    n_iter      = n_iter,
    acq         = "ucb",
    kappa       = 2.576,
    verbose     = TRUE
  )
}

if (file.exists(best_param_path) && !isTRUE(FORCE_RERUN)) {
  cat("\n[09] Found saved hyperparameters → loading from file and skipping search.\n")
  best_params <- readRDS(best_param_path)
} else {
  cat("\n[09] Running BayesianOptimization for XGBoost hyperparameters...\n")
  
  opt_res <- xgb_hyperparameter_search(
    X = X_train,
    y = y_train,
    w = w_train,
    nfold = 3,
    init_points = 4,
    n_iter = 6,
    nrounds_max = 250,
    nthread = 8
  )
  
  best_params <- opt_res$Best_Par
  cat("\nBest parameters:\n")
  print(best_params)
  
  saveRDS(best_params, best_param_path)
  cat("[09] Saved best_params → ", best_param_path, "\n", sep = "")
}

# ---------------------------------------------------------------------
# 6) Train final model using best parameters + CV for best nrounds
# ---------------------------------------------------------------------

dtrain_final <- xgb.DMatrix(X_train, label = y_train, weight = w_train)

params_final <- list(
  booster          = "gbtree",
  objective        = "multi:softprob",
  eval_metric      = "mlogloss",
  num_class        = length(unique(y_train)),
  max_depth        = as.integer(best_params["max_depth"]),
  eta              = best_params["eta"],
  gamma            = best_params["gamma"],
  min_child_weight = best_params["min_child_weight"],
  subsample        = best_params["subsample"],
  colsample_bytree = best_params["colsample_bytree"],
  nthread          = 8
)

cv_final <- tryCatch(
  xgb.cv(
    params                = params_final,
    data                  = dtrain_final,
    nrounds               = 1000,
    nfold                 = 3,
    verbose               = FALSE,
    early_stopping_rounds = 15
  ),
  error = function(e) NULL
)

if (is.null(cv_final) || is.null(cv_final$evaluation_log) || nrow(cv_final$evaluation_log) == 0L) {
  stop("[09] Final CV failed; cannot select best_nrounds. Inspect features for NA/Inf and label distribution.")
}

best_nrounds <- cv_final$best_iteration

# REQUIRED: if best_iteration is NULL, pick the iteration with min test loss
if (is.null(best_nrounds) || !is.finite(best_nrounds)) {
  best_nrounds <- which.min(cv_final$evaluation_log$test_mlogloss_mean)
}

best_nrounds <- as.integer(best_nrounds)
cat("\nSelected best nrounds:", best_nrounds, "\n")

final_model <- xgb.train(
  params  = params_final,
  data    = dtrain_final,
  nrounds = best_nrounds,
  verbose = 0
)

# ---------------------------------------------------------------------
# 7) Evaluation on test set (sanity check)
# ---------------------------------------------------------------------

dtest     <- xgb.DMatrix(X_test)
pred_prob <- predict(final_model, dtest)

num_classes <- length(unique(y_train))
pred_mat   <- matrix(pred_prob, ncol = num_classes, byrow = TRUE)
pred_class <- max.col(pred_mat) - 1L

conf_mat <- table(Pred = pred_class, True = y_test)
cat("\nConfusion matrix (", label_col, "):\n", sep = "")
print(conf_mat)

accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
cat("\nHeld-out accuracy for", label_col, ":", round(accuracy, 4), "\n")

# ---------------------------------------------------------------------
# 8) Save model + metadata
# ---------------------------------------------------------------------

xgb.save(final_model, model_path)

meta_xgb <- list(
  predictor_cols = predictor_cols,
  constant_cols  = constant_cols,
  split_seed     = split_meta$split_seed,
  label_id       = LABEL_ID,
  label_col      = label_col,
  tag            = TAG,
  features_file  = features_file,
  class_counts   = class_counts,
  class_weights  = class_weights,
  best_nrounds   = best_nrounds
)

saveRDS(meta_xgb, meta_path)

cat("\nSaved model to ", model_path, "\n", sep = "")
cat("Saved metadata to ", meta_path, "\n", sep = "")

