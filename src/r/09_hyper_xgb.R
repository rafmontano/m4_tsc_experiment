# =====================================================================
# 09_hyper_xgb.R
# Hyperparameter search + training for XGBoost (labels l1–l4)
# Multiclass (0 = Down, 1 = Neutral, 2 = Up)
#
# Assumes in 00_main.R:
#   - LABEL_ID       (1..4)
#   - TAG            (e.g. "y", "q", "m", "w", "d", "h")
#   - features_file  (e.g. "data/all_windows_with_features_q.rds")
#
# Outputs:
#   - models/xgb/xgb_l{LABEL_ID}_{TAG}.model
#   - models/xgb/xgb_l{LABEL_ID}_{TAG}_meta.rds
#   - data/export/train_l{LABEL_ID}_{TAG}.csv
#   - data/export/test_l{LABEL_ID}_{TAG}.csv
#   - data/export/predictors_l{LABEL_ID}_{TAG}.csv
# =====================================================================

library(dplyr)
library(xgboost)
library(rBayesianOptimization)
library(parallel)

# ---------------------------------------------------------------------
# 0) Label + paths
# ---------------------------------------------------------------------

if (!exists("LABEL_ID")) {
  stop("LABEL_ID is not defined. Run 00_main.R first.")
}
if (!exists("TAG")) {
  stop("TAG is not defined. Ensure freq_tag(TARGET_PERIOD) was computed in 00_main.R.")
}

label_col <- paste0("l", LABEL_ID)

model_dir <- file.path("models", "xgb")
if (!dir.exists(model_dir)) dir.create(model_dir, recursive = TRUE)

model_path <- file.path(model_dir, sprintf("xgb_l%d_%s.model", LABEL_ID, TAG))
meta_path  <- file.path(model_dir, sprintf("xgb_l%d_%s_meta.rds", LABEL_ID, TAG))

cat("Training XGBoost model for label:", label_col, "\n")
cat("Model will be saved to:", model_path, "\n")
cat("Metadata will be saved to:", meta_path, "\n\n")

# ---------------------------------------------------------------------
# 1) Load the enriched dataset: windows + labels + features
# ---------------------------------------------------------------------

if (!exists("features_file")) {
  stop("features_file is not defined in 00_main.R")
}

if (!file.exists(features_file)) {
  stop("File not found: ", features_file,
       "\nRun 07_ts_features.R first.")
}

all_windows_with_features <- readRDS(features_file)
cat("Loaded dataset:", nrow(all_windows_with_features),
    "rows from", features_file, "\n")

# Drop list-columns if still present
if ("x"  %in% names(all_windows_with_features)) all_windows_with_features$x  <- NULL
if ("xx" %in% names(all_windows_with_features)) all_windows_with_features$xx <- NULL

if (!label_col %in% names(all_windows_with_features)) {
  stop("Label column ", label_col, " not found in dataset.")
}

# ---------------------------------------------------------------------
# 2) Define predictor columns (drop labels + metadata)
# ---------------------------------------------------------------------

predictor_cols <- setdiff(
  colnames(all_windows_with_features),
  c("l1", "l2", "l3", "l4", "series_id", "series_name")
)

cat("Number of predictor features before filtering:", length(predictor_cols), "\n")

# ---------------------------------------------------------------------
# 3) Train/test split based on series_id (avoid leakage)
# ---------------------------------------------------------------------

split_seed <- 123  # keep consistent with eval scripts
set.seed(split_seed)

all_ids   <- unique(all_windows_with_features$series_id)
train_ids <- sample(all_ids, size = floor(0.8 * length(all_ids)))
test_ids  <- setdiff(all_ids, train_ids)

train_df <- dplyr::filter(all_windows_with_features, series_id %in% train_ids)
test_df  <- dplyr::filter(all_windows_with_features, series_id %in% test_ids)

cat("Train windows:", nrow(train_df), " | Test windows:", nrow(test_df), "\n")

# ---------------------------------------------------------------------
# 4) Drop useless (constant) features — based ONLY on train set
# ---------------------------------------------------------------------

feature_df_train <- train_df[, predictor_cols, drop = FALSE]

feature_vars  <- sapply(feature_df_train, function(col) var(col, na.rm = TRUE))
constant_cols <- names(feature_vars[feature_vars == 0 | is.na(feature_vars)])

if (length(constant_cols) > 0) {
  message("Removing constant columns: ", paste(constant_cols, collapse = ", "))
  predictor_cols <- setdiff(predictor_cols, constant_cols)
}

# Apply feature set to both train and test
X_train <- as.matrix(train_df[, predictor_cols, drop = FALSE])
X_test  <- as.matrix(test_df[,  predictor_cols, drop = FALSE])

y_train <- as.integer(train_df[[label_col]])
y_test  <- as.integer(test_df[[label_col]])

cat("Final number of predictor features:", length(predictor_cols), "\n")

# ---------------------------------------------------------------------
# 4.1) Class weights for imbalance
# ---------------------------------------------------------------------

class_counts <- table(y_train)           # counts per class 0,1,2
K            <- length(class_counts)
N            <- length(y_train)

# Simple inverse-frequency weights: N / (K * n_k)
class_weights <- N / (K * as.numeric(class_counts))
names(class_weights) <- names(class_counts)

cat("\nClass distribution (train):\n")
print(class_counts)
cat("Class weights used:\n")
print(class_weights)

# Instance-level weights aligned with y_train
w_train <- class_weights[as.character(y_train)]

# ---------------------------------------------------------------------
# 4.5) Export TRAIN/TEST matrices to CSV for external tools (e.g. Python sktime)
# ---------------------------------------------------------------------

export_dir <- file.path("data", "export")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

train_export <- cbind(label = y_train, X_train)
test_export  <- cbind(label = y_test,  X_test)

train_path     <- file.path(export_dir, sprintf("train_l%d_%s.csv",      LABEL_ID, TAG))
test_path      <- file.path(export_dir, sprintf("test_l%d_%s.csv",       LABEL_ID, TAG))
predictor_path <- file.path(export_dir, sprintf("predictors_l%d_%s.csv", LABEL_ID, TAG))

write.csv(train_export, train_path, row.names = FALSE)
write.csv(test_export,  test_path,  row.names = FALSE)

write.csv(
  data.frame(feature = predictor_cols),
  predictor_path,
  row.names = FALSE
)

cat("\nExported data for Python/sklearn/sktime:\n")
cat(" → ", train_path, "\n", sep = "")
cat(" → ", test_path,  "\n", sep = "")
cat(" → ", predictor_path, "\n", sep = "")

# ---------------------------------------------------------------------
# 5) Hyperparameter search (Bayesian Optimization) with class weights
# ---------------------------------------------------------------------

xgb_hyperparameter_search <- function(
    X,
    y,
    w,
    nfold = 3,
    init_points = 4,
    n_iter = 6,
    nrounds_max = 250,
    nthread = max(1, detectCores() - 1)
) {
  y <- as.integer(y)
  dtrain <- xgb.DMatrix(data = X, label = y, weight = w)
  
  xgb_cv_bayes <- function(max_depth, eta, gamma, min_child_weight,
                           subsample, colsample_bytree) {
    
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
    
    cv <- xgb.cv(
      params                = params,
      data                  = dtrain,
      nrounds               = nrounds_max,
      nfold                 = nfold,
      showsd                = TRUE,
      verbose               = FALSE,
      early_stopping_rounds = 15
    )
    
    list(
      Score = -cv$evaluation_log$test_mlogloss_mean[cv$best_iteration],
      Pred  = cv$best_iteration
    )
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

opt_res <- xgb_hyperparameter_search(
  X = X_train,
  y = y_train,
  w = w_train,
  nfold = 3,
  init_points = 4,
  n_iter = 6,
  nrounds_max = 250
)

best_params <- opt_res$Best_Par
cat("\nBest parameters:\n")
print(best_params)

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
  nthread          = max(1, detectCores() - 1)
)

cv_final <- xgb.cv(
  params                = params_final,
  data                  = dtrain_final,
  nrounds               = 1000,
  nfold                 = 3,
  verbose               = FALSE,
  early_stopping_rounds = 15
)

best_nrounds <- cv_final$best_iteration
cat("\nSelected best nrounds:", best_nrounds, "\n")

final_model <- xgboost(
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
pred_class <- max.col(pred_mat) - 1L   # convert back to {0,1,2}

conf_mat <- table(Pred = pred_class, True = y_test)
cat("\nConfusion matrix (", label_col, "):\n", sep = "")
print(conf_mat)

accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
cat("\nHeld-out accuracy for", label_col, ":", round(accuracy, 4), "\n")

# ---------------------------------------------------------------------
# 8) Save model + metadata (including class_weights)
# ---------------------------------------------------------------------

xgb.save(final_model, model_path)

meta_xgb <- list(
  predictor_cols = predictor_cols,
  constant_cols  = constant_cols,
  split_seed     = split_seed,
  label_id       = LABEL_ID,
  label_col      = label_col,
  tag            = TAG,
  features_file  = features_file,
  class_counts   = class_counts,
  class_weights  = class_weights
)

saveRDS(meta_xgb, meta_path)

cat("\nSaved model to ", model_path, "\n", sep = "")
cat("Saved metadata to ", meta_path, "\n", sep = "")

