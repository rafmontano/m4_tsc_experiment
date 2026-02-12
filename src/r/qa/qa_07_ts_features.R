# =====================================================================
# qa_07_ts_features.R
# QA checks for Script 07 (tsfeatures extraction + chunking + binding)
#
# Objectives:
# 1) No leakage: features must be computed from x only; ensure no future/label/id
#    fields appear in predictor set and that x/xx are not used as predictors.
# 2) Learnability sanity: features have variance; simple univariate separability
#    is non-degenerate; label distribution not collapsed.
# 3) Correct implementation: row alignment between labeled windows and features
#    is preserved; dimensions and finiteness checks pass.
#
# Usage:
#   - Run after Script 07 has produced features_file
#   - Requires globals: labeled_file, windows_std_file, features_file, LABEL_ID
# =====================================================================

library(dplyr)

req <- c("labeled_file", "windows_std_file", "features_file", "LABEL_ID")
missing <- req[!vapply(req, exists, logical(1))]
if (length(missing) > 0L) stop("Missing required globals: ", paste(missing, collapse = ", "))

if (!file.exists(labeled_file)) stop("Missing labeled_file: ", labeled_file)
if (!file.exists(windows_std_file)) stop("Missing windows_std_file: ", windows_std_file)
if (!file.exists(features_file)) stop("Missing features_file: ", features_file)

label_col <- paste0("l", as.integer(LABEL_ID))

cat("\n================ QA 07: tsfeatures dataset =================\n")

# ---------------------------------------------------------------------
# 1) Load artefacts
# ---------------------------------------------------------------------
W_lab <- readRDS(labeled_file)
W_std <- readRDS(windows_std_file)
F     <- readRDS(features_file)

cat("Rows labeled_file     :", nrow(W_lab), "\n")
cat("Rows windows_std_file :", nrow(W_std), "\n")
cat("Rows features_file    :", nrow(F), "\n")
cat("Cols features_file    :", ncol(F), "\n\n")

stopifnot(is.data.frame(W_lab), is.data.frame(W_std), is.data.frame(F))
stopifnot(nrow(W_lab) == nrow(W_std))
stopifnot(nrow(F) == nrow(W_lab))
stopifnot(all(c("x","xx") %in% names(W_std)))
stopifnot(all(c("x","xx", "l1","l2","l3","l4") %in% names(W_lab)))
stopifnot(label_col %in% names(F))

cat("PASS: basic structure and row counts match.\n")

# ---------------------------------------------------------------------
# 2) Correctness: alignment check (row-level checksum on x)
#    We verify that F's first columns still correspond to the same windows.
#    Since F was bind_cols(all_windows_labeled, feature_matrix),
#    'x' in F must match 'x' in W_lab exactly (list equality).
# ---------------------------------------------------------------------
same_x <- identical(F$x, W_lab$x)
same_xx <- identical(F$xx, W_lab$xx)

if (!same_x || !same_xx) {
  stop("FAIL: Row alignment mismatch. 'x'/'xx' in features_file do not exactly match labeled_file. Chunk reassembly or binding order likely broke.")
}

cat("PASS: features_file preserves row order (x/xx identical to labeled_file).\n")

# ---------------------------------------------------------------------
# 3) Correctness: finiteness of numeric feature columns
# ---------------------------------------------------------------------
# Identify candidate feature columns as numeric columns excluding known non-features
non_feature_cols <- c("x","xx","series_id","series_name","st","n","type","h","period","xx_real",
                      "l1","l2","l3","l4")
# Keep label_col in non_feature as well for predictor screening later
non_feature_cols <- unique(c(non_feature_cols, label_col))

num_cols <- names(F)[vapply(F, is.numeric, logical(1))]
feat_cols <- setdiff(num_cols, intersect(num_cols, non_feature_cols))

if (length(feat_cols) == 0L) stop("FAIL: No numeric feature columns detected in features_file.")

# Finite check
finite_ok <- all(vapply(feat_cols, function(cn) all(is.finite(F[[cn]])), logical(1)))
if (!finite_ok) {
  bad <- feat_cols[!vapply(feat_cols, function(cn) all(is.finite(F[[cn]])), logical(1))][1:20]
  stop("FAIL: Non-finite values detected in numeric feature columns. Examples: ", paste(bad, collapse = ", "))
}

cat("PASS: All numeric feature columns are finite.\n")

# ---------------------------------------------------------------------
# 4) Objective 1: leakage screening (predictor column hygiene)
# ---------------------------------------------------------------------
# Reject obvious identity/leakage-prone columns if they exist as numeric predictors.
# --- Leakage-only name screening (hard fail) ---
leak_suspects <- grep("(^series_id$|^st$|series_name|^id$|^window_id$|^row_id$|^uuid$)",
                      feat_cols, ignore.case = TRUE, value = TRUE)

if (length(leak_suspects) > 0L) {
  cat("\nFAIL: Leakage-prone identity columns detected among numeric predictors:\n")
  print(leak_suspects)
  stop("FAIL: Remove these columns from predictors.")
}

cat("PASS: No identity/leakage columns detected by name.\n")

# --- Allowed metadata / legit features that can look suspicious ---
allowed_meta <- intersect(feat_cols, c("hurst", "nperiods", "seasonal_period", "series_length"))

if (length(allowed_meta) > 0L) {
  cat("\nINFO: Columns that look suspicious by name but are allowed:\n")
  print(allowed_meta)
  
  # Check if they are constant (common with fixed WINDOW_SIZE/frequency)
  meta_sd <- vapply(F[allowed_meta], sd, numeric(1), na.rm = TRUE)
  const_meta <- names(meta_sd)[!is.finite(meta_sd) | meta_sd < 1e-12]
  
  if (length(const_meta) > 0L) {
    cat("INFO: These allowed columns are (near) constant and can be dropped safely:\n")
    print(const_meta)
  }
}

# Additionally ensure that label columns are not in feature list
label_cols_all <- c("l1","l2","l3","l4")
if (any(label_cols_all %in% feat_cols)) stop("FAIL: Label columns are included in predictors.")
cat("PASS: Labels not included among numeric feature predictors.\n")

# ---------------------------------------------------------------------
# 5) Objective 2: learnability sanity
#   A) Feature variance not degenerate
#   B) Very simple separability sanity (ANOVA p-values) on training label_col
# ---------------------------------------------------------------------
y <- as.integer(F[[label_col]])
tab <- prop.table(table(factor(y, levels = c(0,1,2))))
cat("\nLabel distribution (", label_col, "):\n", sep = "")
print(tab)

if (max(tab) > 0.98) stop("FAIL: Label distribution collapses (>98% one class). Learning will be extremely hard.")
cat("PASS: Label distribution is not collapsed.\n")

# Variance check: proportion of near-constant features
feat_sds <- vapply(F[feat_cols], function(v) sd(v), numeric(1))
near_const <- mean(!is.finite(feat_sds) | feat_sds < 1e-10)
cat("\nNear-constant feature rate:", round(near_const, 4), "\n")

if (near_const > 0.50) {
  stop("FAIL: Too many near-constant features (>50%). Feature extraction/NA->0 may be degenerating predictors.")
}
cat("PASS: Feature variance looks reasonable.\n")

# Simple univariate signal check (fast and pragmatic)
# We only test a subset if there are many features.
set.seed(123)
feat_subset <- feat_cols
if (length(feat_subset) > 64) feat_subset <- sample(feat_subset, 64)

pvals <- sapply(feat_subset, function(cn) {
  x <- F[[cn]]
  if (!is.finite(sd(x)) || sd(x) == 0) return(NA_real_)
  summary(aov(x ~ as.factor(y)))[[1]][["Pr(>F)"]][1]
})
pvals <- pvals[is.finite(pvals)]

if (length(pvals) == 0L) stop("FAIL: Could not compute any ANOVA p-values (all constant/non-finite).")

sig_rate <- mean(pvals < 1e-6)
cat("Univariate signal rate (p < 1e-6) on sampled features:", round(sig_rate, 4), "\n")

# We do NOT require “high signal”; we require “not dead”.
if (sig_rate < 0.05) {
  stop("FAIL: Very weak univariate separability across features. Possible misalignment or degenerate features.")
}

cat("PASS: Non-trivial univariate separability exists (sanity only).\n")

cat("\n================ QA 07 RESULT: PASS =====================\n")
