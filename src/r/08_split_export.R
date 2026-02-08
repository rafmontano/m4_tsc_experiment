# =====================================================================
# 08_split_export.R
# Single source of truth for train/test split + exports for R and Python
# =====================================================================

library(dplyr)
library(caret)

# ---------------------------------------------------------------------
# 0) Required globals
# ---------------------------------------------------------------------

if (!exists("LABEL_ID")) stop("LABEL_ID is not defined. Run 00_main.R first.")
if (!exists("TAG")) stop("TAG is not defined. Run 00_main.R first.")
if (!exists("features_file")) stop("features_file is not defined. Run 00_main.R first.")

label_col <- paste0("l", LABEL_ID)

# ---------------------------------------------------------------------
# 1) Split parameters (reproducible everywhere)
# ---------------------------------------------------------------------

SPLIT_SEED <- 123
TRAIN_FRAC <- 0.80

# ---------------------------------------------------------------------
# 2) Load features dataset from Script 07 (no mutation)
# ---------------------------------------------------------------------

if (!file.exists(features_file)) {
  stop("File not found: ", features_file, "\nRun 07_ts_features.R first.")
}

df <- readRDS(features_file)
cat("[08] Loaded features dataset:", nrow(df), "rows from", features_file, "\n")

if (!label_col %in% names(df)) stop("Label column not found: ", label_col)
#if (!all(c("series_id", "series_name") %in% names(df))) {
#  stop("Expected series_id and series_name columns not found.")
#}
if (!"x" %in% names(df)) {
  stop("Expected list-column 'x' not found. Cannot export TSC windows.")
}

# ---------------------------------------------------------------------
# 3) Train/test split (row split, stratified by label)
# ---------------------------------------------------------------------

set.seed(SPLIT_SEED)

train_idx <- caret::createDataPartition(
  y = df[[label_col]],
  p = TRAIN_FRAC,
  list = FALSE
)

train_df <- df[train_idx, , drop = FALSE]
test_df  <- df[-train_idx, , drop = FALSE]

cat("Train distribution:\n")
print(prop.table(table(train_df[[label_col]])))

cat("Test distribution:\n")
print(prop.table(table(test_df[[label_col]])))

cat("[08] Train windows:", nrow(train_df), " | Test windows:", nrow(test_df), "\n")

# ---------------------------------------------------------------------
# 4) Save split artefacts as RDS (easy reuse later in R)
# ---------------------------------------------------------------------

split_dir <- file.path("data", "splits")
if (!dir.exists(split_dir)) dir.create(split_dir, recursive = TRUE)

train_rds_path <- file.path(split_dir, sprintf("train_l%d_%s.rds", LABEL_ID, TAG))
test_rds_path  <- file.path(split_dir, sprintf("test_l%d_%s.rds",  LABEL_ID, TAG))
meta_rds_path  <- file.path(split_dir, sprintf("split_meta_l%d_%s.rds", LABEL_ID, TAG))

saveRDS(train_df, train_rds_path)
saveRDS(test_df,  test_rds_path)

# Store label distributions with fixed levels for stability
train_tab <- table(factor(train_df[[label_col]], levels = c(0, 1, 2)))
test_tab  <- table(factor(test_df[[label_col]],  levels = c(0, 1, 2)))

split_meta <- list(
  split_method = "row_stratified_label",
  split_seed   = SPLIT_SEED,
  train_frac   = TRAIN_FRAC,
  label_id     = LABEL_ID,
  label_col    = label_col,
  tag          = TAG,
  n_total      = nrow(df),
  n_train      = nrow(train_df),
  n_test       = nrow(test_df),
  train_label_dist = as.list(train_tab),
  test_label_dist  = as.list(test_tab)
)

saveRDS(split_meta, meta_rds_path)

cat("[08] Saved split RDS:\n")
cat("  -> ", train_rds_path, "\n", sep = "")
cat("  -> ", test_rds_path,  "\n", sep = "")
cat("  -> ", meta_rds_path,  "\n", sep = "")


# ---------------------------------------------------------------------
# 6) Export TSC-ready TRAIN/TEST CSV (07b-style) for Python
#     Format: label, t1..tW
#     IMPORTANT: uses the already-created train_df/test_df from Step 3
# ---------------------------------------------------------------------

export_dir <- file.path("data", "export")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

# Ensure required columns exist
if (!label_col %in% names(train_df) || !label_col %in% names(test_df)) {
  stop("Expected label column not found in train/test: ", label_col)
}
if (!"x" %in% names(train_df) || !"x" %in% names(test_df)) {
  stop("Expected list-column 'x' not found in train/test; cannot export TSC windows.")
}

# Convert labels to integer copy (do not modify original labels)
labels_to_int <- function(lbl) {
  if (is.factor(lbl)) {
    as.integer(lbl)
  } else {
    as.integer(lbl)
  }
}

train_labels_raw <- train_df[[label_col]]
test_labels_raw  <- test_df[[label_col]]

train_labels_int <- labels_to_int(train_labels_raw)
test_labels_int  <- labels_to_int(test_labels_raw)

# Detect window length (assumes fixed length)
tsc_window_length <- length(train_df$x[[1]])
cat("[08] Detected TSC window length:", tsc_window_length, "time steps.\n")

# Flatten list-column x into matrix (same idea as 07b)
train_X_mat <- matrix(
  unlist(train_df$x),
  nrow = nrow(train_df),
  ncol = tsc_window_length,
  byrow = TRUE
)

test_X_mat <- matrix(
  unlist(test_df$x),
  nrow = nrow(test_df),
  ncol = tsc_window_length,
  byrow = TRUE
)

# Build export data.frames: label, t1..tW
tsc_train_df <- data.frame(label = train_labels_int, train_X_mat, check.names = FALSE)
tsc_test_df  <- data.frame(label = test_labels_int,  test_X_mat,  check.names = FALSE)

colnames(tsc_train_df) <- c("label", paste0("t", seq_len(tsc_window_length)))
colnames(tsc_test_df)  <- c("label", paste0("t", seq_len(tsc_window_length)))

# Write CSVs
tsc_train_path <- file.path(export_dir, sprintf("windows_tsc_train_l%d_%s.csv", LABEL_ID, TAG))
tsc_test_path  <- file.path(export_dir, sprintf("windows_tsc_test_l%d_%s.csv",  LABEL_ID, TAG))

write.csv(tsc_train_df, tsc_train_path, row.names = FALSE)
write.csv(tsc_test_df,  tsc_test_path,  row.names = FALSE)

cat("[08] Exported TSC TRAIN/TEST CSVs:\n")
cat("  -> ", tsc_train_path, "\n", sep = "")
cat("  -> ", tsc_test_path,  "\n", sep = "")
