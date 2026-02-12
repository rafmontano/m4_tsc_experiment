# =====================================================================
# qa_05_compute_c.R
# QA checks for Script 05 (compute z-values and thresholds c_all / c_train)
#
# Objectives:
# 1) No leakage: c_train must be reproducible using TRAIN subset only
#    (reconstructed by C_SEED + C_TRAIN_FRAC) and must not depend on held-out windows.
# 2) Learnability sanity: c_train should not create a degenerate label distribution.
# 3) Correct implementation: values are finite, positive, reproducible, and consistent
#    with the compute_c() convention (tail probability ~ Q).
#
# Usage:
#   - Run after 05_compute_c.R has produced threshold_file_train and threshold_file_all
#   - Requires globals: windows_std_file, threshold_file_train, threshold_file_all,
#                       C_TRAIN_FRAC, C_BOOT_B, Q, C_SEED
#   - Requires utils.R loaded: compute_z(), compute_c(), label_from_z_int()
# =====================================================================

# ---------------------------------------------------------------------
# 0) Required globals
# ---------------------------------------------------------------------
req <- c(
  "windows_std_file", "threshold_file_train", "threshold_file_all",
  "C_TRAIN_FRAC", "C_BOOT_B", "Q", "C_SEED"
)
missing <- req[!vapply(req, exists, logical(1))]
if (length(missing) > 0L) stop("Missing required globals: ", paste(missing, collapse = ", "))

if (!exists("compute_z")) stop("compute_z() not found. Ensure src/r/utils.R is sourced.")
if (!exists("compute_c")) stop("compute_c() not found. Ensure src/r/utils.R is sourced.")
if (!exists("label_from_z_int")) stop("label_from_z_int() not found. Ensure src/r/utils.R is sourced.")

if (!file.exists(windows_std_file)) stop("Missing windows_std_file: ", windows_std_file)
if (!file.exists(threshold_file_train)) stop("Missing threshold_file_train: ", threshold_file_train)
if (!file.exists(threshold_file_all)) stop("Missing threshold_file_all: ", threshold_file_all)

cat("\n================ QA 05: Compute c ================\n")

# ---------------------------------------------------------------------
# 1) Load artefacts
# ---------------------------------------------------------------------
W <- readRDS(windows_std_file)
c_train_saved <- readRDS(threshold_file_train)
c_all_saved   <- readRDS(threshold_file_all)

cat("Windows (std):", nrow(W), "\n")
cat("Saved c_train:", c_train_saved, "\n")
cat("Saved c_all  :", c_all_saved, "\n\n")

stopifnot(is.data.frame(W), nrow(W) > 1L)
stopifnot(all(c("x", "xx") %in% names(W)))

# ---------------------------------------------------------------------
# 2) Correctness: c values finite and sane
# ---------------------------------------------------------------------
if (!is.finite(c_train_saved) || c_train_saved < 0) stop("FAIL: c_train is not finite non-negative.")
if (!is.finite(c_all_saved)   || c_all_saved   < 0) stop("FAIL: c_all is not finite non-negative.")
cat("PASS: c_train and c_all are finite and non-negative.\n")

# ---------------------------------------------------------------------
# 3) Recompute z_all and c_all, confirm exact reproducibility
# ---------------------------------------------------------------------
z_all <- compute_z(W)
c_all_recalc <- compute_c(z_all, q = Q)

if (!isTRUE(all.equal(c_all_saved, c_all_recalc, tolerance = 1e-12))) {
  stop("FAIL: c_all is not reproducible. Saved vs recomputed mismatch. Check compute_z/compute_c or upstream mutation.")
}
cat("PASS: c_all is reproducible from windows_std_file.\n")

# ---------------------------------------------------------------------
# 4) Reconstruct TRAIN subset indices used by Script 05 and recompute c_train
#     Script 05 uses:
#       set.seed(C_SEED)
#       n_train <- floor(C_TRAIN_FRAC*n)
#       train_idx <- sample.int(n, size=n_train, replace=FALSE)
#       c_train = median(bootstrapped compute_c(z_train))
# ---------------------------------------------------------------------
n <- nrow(W)
n_train <- max(1L, floor(C_TRAIN_FRAC * n))

set.seed(C_SEED)
train_idx <- sample.int(n, size = n_train, replace = FALSE)

W_train <- W[train_idx, , drop = FALSE]
z_train <- compute_z(W_train)

B <- as.integer(C_BOOT_B)
if (!is.finite(B) || B < 1L) stop("FAIL: C_BOOT_B must be positive integer.")
# Re-run EXACT RNG stream used in Script 05:
set.seed(C_SEED)

n <- nrow(W)
n_train <- max(1L, floor(C_TRAIN_FRAC * n))
train_idx <- sample.int(n, size = n_train, replace = FALSE)

W_train <- W[train_idx, , drop = FALSE]
z_train <- compute_z(W_train)

B <- as.integer(C_BOOT_B)
c_boot <- numeric(B)

# IMPORTANT: do NOT reset seed here (Script 05 does not)
for (b in seq_len(B)) {
  z_b <- sample(z_train, size = length(z_train), replace = TRUE)
  c_boot[b] <- compute_c(z_b, q = Q)
}

c_train_recalc <- median(c_boot, na.rm = TRUE)

if (!isTRUE(all.equal(c_train_saved, c_train_recalc, tolerance = 1e-10))) {
  stop(
    "FAIL: c_train not reproducible from TRAIN subset definition.\n",
    "Saved: ", c_train_saved, " | Recalc: ", c_train_recalc, "\n",
    "This indicates script divergence (seed usage/order), or thresholds were computed from different data."
  )
}
cat("PASS: c_train is reproducible from TRAIN subset-only procedure.\n")

# ---------------------------------------------------------------------
# 5) No-leakage signal check (weak but useful):
#    c_train should typically differ from c_all (unless dataset is huge/stationary).
# ---------------------------------------------------------------------
if (isTRUE(all.equal(c_train_saved, c_all_saved, tolerance = 1e-12))) {
  cat("WARN: c_train == c_all exactly. Not necessarily wrong, but unusual.\n")
} else {
  cat("PASS: c_train differs from c_all (expected: train-only vs oracle).\n")
}

# ---------------------------------------------------------------------
# 6) Validate compute_c() convention:
#    Your compute_c() uses quantile(abs(z), probs = 1 - q),
#    so the empirical tail rate P(|z| >= c) should be approx q.
# ---------------------------------------------------------------------
tail_rate_all <- mean(abs(z_all) >= c_train_saved, na.rm = TRUE)
tail_rate_tr  <- mean(abs(z_train) >= c_train_saved, na.rm = TRUE)

cat("\nTail-rate checks using c_train:\n")
cat("  Tail rate on ALL   :", round(tail_rate_all, 4), "\n")
cat("  Tail rate on TRAIN :", round(tail_rate_tr,  4), "\n")
cat("  Target q           :", Q, "\n")

# allow tolerance due to discreteness + bootstrap median
tol <- 0.05
if (abs(tail_rate_tr - Q) > tol) {
  stop("FAIL: Tail rate on TRAIN deviates too much from Q. Check compute_c() definition or Q meaning.")
}
cat("PASS: Tail rate on TRAIN approximately matches Q (consistent compute_c convention).\n")

# ---------------------------------------------------------------------
# 7) Learnability sanity: label distribution should not collapse
# ---------------------------------------------------------------------
y_all <- label_from_z_int(z_all, c_train_saved)
tab <- prop.table(table(y_all))

cat("\nLabel distribution induced by c_train on ALL windows:\n")
print(tab)

# Heuristic: avoid near-single-class collapse
if (max(tab) > 0.98) {
  stop("FAIL: Label distribution collapses (max class prevalence > 0.98). Learning will be extremely hard.")
}
cat("PASS: Label distribution is not collapsed.\n")

# ---------------------------------------------------------------------
# 8) Bootstrap stability (implementation sanity)
# ---------------------------------------------------------------------
iqr_boot <- IQR(c_boot, na.rm = TRUE)
cat("\nBootstrap c summary:\n")
print(summary(c_boot))
cat("Bootstrap IQR:", signif(iqr_boot, 4), "\n")

if (!is.finite(iqr_boot)) stop("FAIL: Bootstrap IQR not finite.")
if (iqr_boot == 0) cat("WARN: Bootstrap IQR is 0 (could happen with small/quantised z). Not necessarily wrong.\n")

cat("\n================ QA 05 RESULT: PASS ====================\n")

