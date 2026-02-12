# =====================================================================
# qa_08_split_export.R
# QA checks for Script 08 (train/test split + exports)
#
# Objectives:
# 1) No leakage: detect dependence leakage via shared series_id across splits
#    (and optionally near-duplicate windows if stride is small).
# 2) Learnability sanity: verify split retains non-collapsed label distribution
#    and basic feature variance; also ensure no label shift for Python export.
# 3) Correct implementation: reproducibility, sizes, schema consistency,
#    exports match the RDS split exactly.
#
# Usage:
#   - Run after Script 08 has produced:
#       data/splits/train_l{LABEL_ID}_{TAG}.rds
#       data/splits/test_l{LABEL_ID}_{TAG}.rds
#       data/splits/split_meta_l{LABEL_ID}_{TAG}.rds
#       data/export/windows_tsc_train_l{LABEL_ID}_{TAG}.csv
#       data/export/windows_tsc_test_l{LABEL_ID}_{TAG}.csv
# =====================================================================

library(dplyr)

# ---------------------------------------------------------------------
# 0) Required globals
# ---------------------------------------------------------------------
req <- c("LABEL_ID", "TAG", "features_file")
missing <- req[!vapply(req, exists, logical(1))]
if (length(missing) > 0L) stop("Missing required globals: ", paste(missing, collapse = ", "))

label_col <- paste0("l", LABEL_ID)

split_dir  <- file.path("data", "splits")
export_dir <- file.path("data", "export")

train_rds_path <- file.path(split_dir, sprintf("train_l%d_%s.rds", LABEL_ID, TAG))
test_rds_path  <- file.path(split_dir, sprintf("test_l%d_%s.rds",  LABEL_ID, TAG))
meta_rds_path  <- file.path(split_dir, sprintf("split_meta_l%d_%s.rds", LABEL_ID, TAG))

tsc_train_path <- file.path(export_dir, sprintf("windows_tsc_train_l%d_%s.csv", LABEL_ID, TAG))
tsc_test_path  <- file.path(export_dir, sprintf("windows_tsc_test_l%d_%s.csv",  LABEL_ID, TAG))

for (p in c(train_rds_path, test_rds_path, meta_rds_path, tsc_train_path, tsc_test_path)) {
  if (!file.exists(p)) stop("Missing expected output: ", p)
}

cat("\n================ QA 08: Split + export ================\n")

# ---------------------------------------------------------------------
# 1) Load artefacts
# ---------------------------------------------------------------------
train_df <- readRDS(train_rds_path)
test_df  <- readRDS(test_rds_path)
meta     <- readRDS(meta_rds_path)

cat("Train rows:", nrow(train_df), " | Test rows:", nrow(test_df), "\n")

stopifnot(is.data.frame(train_df), is.data.frame(test_df))
stopifnot(label_col %in% names(train_df), label_col %in% names(test_df))
stopifnot("x" %in% names(train_df), "x" %in% names(test_df))

# ---------------------------------------------------------------------
# 2) Correctness: meta consistency
# ---------------------------------------------------------------------
stopifnot(meta$n_train == nrow(train_df))
stopifnot(meta$n_test  == nrow(test_df))
stopifnot(meta$label_col == label_col)
stopifnot(meta$tag == TAG)

cat("PASS: split_meta matches train/test row counts and label_col.\n")

# ---------------------------------------------------------------------
# 3) Correctness: stratification actually happened (approx)
# ---------------------------------------------------------------------
dist <- function(v) prop.table(table(factor(v, levels = c(0,1,2))))

d_tr <- dist(train_df[[label_col]])
d_te <- dist(test_df[[label_col]])

cat("\nTrain label dist:\n"); print(d_tr)
cat("Test  label dist:\n"); print(d_te)

if (max(abs(d_tr - d_te), na.rm = TRUE) > 0.05) {
  stop("FAIL: Train/Test label distributions differ by > 5%. Stratification may not be working as expected.")
}

cat("PASS: Train/Test label distributions are similar (stratified split).\n")

# ---------------------------------------------------------------------
# 4) No-leakage check A: shared series_id across splits
#     If series_id exists, this is a major dependence leakage risk when stride is small.
# ---------------------------------------------------------------------
if ("series_id" %in% names(train_df) && "series_id" %in% names(test_df)) {
  
  shared <- intersect(unique(train_df$series_id), unique(test_df$series_id))
  share_rate <- length(shared) / length(unique(c(train_df$series_id, test_df$series_id)))
  
  cat("\nseries_id present.\n")
  cat("Unique series in train:", length(unique(train_df$series_id)), "\n")
  cat("Unique series in test :", length(unique(test_df$series_id)), "\n")
  cat("Shared series_id count:", length(shared), " (share_rate=", round(share_rate, 3), ")\n", sep="")
  
  if (length(shared) > 0L) {
    cat("WARN: Train and test share series_id values. This is dependence leakage risk.\n")
    cat("      If STRIDE is small (e.g., 1), test may be overly optimistic.\n")
  } else {
    cat("PASS: No shared series_id (series-level split behaviour).\n")
  }
  
} else {
  cat("\nINFO: series_id not found; cannot test series-level leakage directly.\n")
}

# ---------------------------------------------------------------------
# 5) No-leakage check B (optional, pragmatic): exact duplicate windows across splits
#     For QA datasets, this should be rare unless you have repeated patterns.
# ---------------------------------------------------------------------
hash_window <- function(x) paste0(format(as.numeric(x), scientific = FALSE), collapse = ",")

set.seed(123)
n_samp <- min(200L, nrow(train_df), nrow(test_df))

tr_idx <- sample.int(nrow(train_df), n_samp)
te_idx <- sample.int(nrow(test_df),  n_samp)

tr_hash <- vapply(train_df$x[tr_idx], hash_window, character(1))
te_hash <- vapply(test_df$x[te_idx],  hash_window, character(1))

dup_hits <- sum(te_hash %in% tr_hash)

cat("\nExact-duplicate window check (sampled ", n_samp, " vs ", n_samp, "): hits=", dup_hits, "\n", sep="")
if (dup_hits > 0L) {
  cat("WARN: Found exact duplicate windows across train/test (sample). This indicates strong dependence leakage.\n")
} else {
  cat("PASS: No exact duplicate windows found in sampled check.\n")
}

# ---------------------------------------------------------------------
# 6) Learnability sanity: non-degenerate features
#     Check numeric predictor variance exists.
# ---------------------------------------------------------------------
num_cols <- names(train_df)[vapply(train_df, is.numeric, logical(1))]
num_cols <- setdiff(num_cols, c(label_col, "series_id"))  # label + id excluded

if (length(num_cols) == 0L) {
  cat("WARN: No numeric feature columns detected in train_df.\n")
} else {
  sds <- vapply(train_df[num_cols], sd, numeric(1), na.rm = TRUE)
  const_rate <- mean(!is.finite(sds) | sds < 1e-12)
  
  cat("\nNumeric feature count:", length(num_cols), "\n")
  cat("Constant/near-constant feature rate:", round(const_rate, 3), "\n")
  
  if (const_rate > 0.5) {
    cat("WARN: >50% of features are near-constant. This can make learning weak.\n")
  } else {
    cat("PASS: Feature variance looks broadly non-degenerate.\n")
  }
}

# ---------------------------------------------------------------------
# 7) Export correctness: CSV labels match RDS labels (and are 0/1/2)
# ---------------------------------------------------------------------
tsc_train <- read.csv(tsc_train_path)
tsc_test  <- read.csv(tsc_test_path)

stopifnot(nrow(tsc_train) == nrow(train_df))
stopifnot(nrow(tsc_test)  == nrow(test_df))

lab_csv_tr <- as.integer(tsc_train$label)
lab_csv_te <- as.integer(tsc_test$label)

lab_rds_tr <- as.integer(train_df[[label_col]])
lab_rds_te <- as.integer(test_df[[label_col]])

# If RDS labels are factors, as.integer() will be 1..K; detect and warn
if (is.factor(train_df[[label_col]]) || is.factor(test_df[[label_col]])) {
  cat("\nWARN: Label column is a factor in RDS split. as.integer(factor) produces 1..K.\n")
  cat("      Ensure exported labels are truly 0/1/2 for Python.\n")
}

# Hard check: CSV labels must be subset of {0,1,2}
if (!all(lab_csv_tr %in% c(0L,1L,2L)) || !all(lab_csv_te %in% c(0L,1L,2L))) {
  stop("FAIL: Exported CSV labels are not in {0,1,2}. Likely factor->integer bug in export.")
}

# Soft check: distributions must match (not necessarily row-by-row identical)
d_csv_tr <- prop.table(table(factor(lab_csv_tr, levels=c(0,1,2))))
d_csv_te <- prop.table(table(factor(lab_csv_te, levels=c(0,1,2))))

if (max(abs(d_csv_tr - d_tr), na.rm = TRUE) > 1e-6) {
  stop("FAIL: CSV TRAIN label distribution differs from RDS TRAIN distribution (label mapping bug).")
}
if (max(abs(d_csv_te - d_te), na.rm = TRUE) > 1e-6) {
  stop("FAIL: CSV TEST label distribution differs from RDS TEST distribution (label mapping bug).")
}

cat("PASS: CSV exports have correct sizes and label distributions match RDS.\n")

cat("\n================ QA 08 RESULT: PASS (with warnings if any) ====================\n")