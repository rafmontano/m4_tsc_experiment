# =====================================================================
# qa_04_transformations.R
# QA checks for Script 04 (min-max + standardisation)
#
# Objectives:
# 1) No leakage: scaling statistics must be computed from x ONLY (per paper),
#    not from c(x, xx). Detect and FAIL if xx influences the transform.
# 2) Learnability sanity: transformed series are non-degenerate (variance exists).
# 3) Correct implementation: lengths preserved, finite values, expected properties.
#
# Usage:
#   - Run after Script 04 has produced windows_std_file
#   - Requires globals: windows_raw_file, windows_std_file, WINDOW_SIZE, HORIZON
#   - Requires scale_pair_minmax_std() already loaded (utils.R sourced)
# =====================================================================

library(dplyr)

req <- c("windows_raw_file", "windows_std_file", "WINDOW_SIZE", "HORIZON")
missing <- req[!vapply(req, exists, logical(1))]
if (length(missing) > 0L) stop("Missing required globals: ", paste(missing, collapse = ", "))

if (!exists("scale_pair_minmax_std")) {
  stop("scale_pair_minmax_std() not found. Ensure src/r/utils.R is sourced before running this QA.")
}
if (!file.exists(windows_raw_file)) stop("Missing windows_raw_file: ", windows_raw_file)
if (!file.exists(windows_std_file)) stop("Missing windows_std_file: ", windows_std_file)

W_raw <- readRDS(windows_raw_file)
W_std <- readRDS(windows_std_file)

cat("\n================ QA 04: Transformations ================\n")
cat("Raw windows:", nrow(W_raw), " | Std windows:", nrow(W_std), "\n")
cat("WINDOW_SIZE:", WINDOW_SIZE, " HORIZON:", HORIZON, "\n\n")

stopifnot(nrow(W_raw) == nrow(W_std))
stopifnot(all(c("x","xx") %in% names(W_raw)))
stopifnot(all(c("x","xx") %in% names(W_std)))

# ---------------------------------------------------------------------
# 1) Correctness: lengths preserved
# ---------------------------------------------------------------------
len_x_raw  <- vapply(W_raw$x,  length, integer(1))
len_xx_raw <- vapply(W_raw$xx, length, integer(1))
len_x_std  <- vapply(W_std$x,  length, integer(1))
len_xx_std <- vapply(W_std$xx, length, integer(1))

if (any(len_x_raw != WINDOW_SIZE) || any(len_x_std != WINDOW_SIZE)) {
  stop("FAIL: x length mismatch (raw or std) vs WINDOW_SIZE.")
}
if (any(len_xx_raw != HORIZON) || any(len_xx_std != HORIZON)) {
  stop("FAIL: xx length mismatch (raw or std) vs HORIZON.")
}
cat("PASS: Lengths preserved (x=WINDOW_SIZE, xx=HORIZON).\n")

# ---------------------------------------------------------------------
# 2) Correctness: no NA/Inf introduced
# ---------------------------------------------------------------------
finite_list <- function(lst) all(vapply(lst, function(v) all(is.finite(as.numeric(v))), logical(1)))
if (!finite_list(W_std$x) || !finite_list(W_std$xx)) {
  stop("FAIL: Non-finite values found in transformed windows (NA/Inf).")
}
cat("PASS: No NA/Inf introduced by transformation.\n")

# ---------------------------------------------------------------------
# 3) No-leakage check: determine whether transform matches:
#    A) Paper definition: min/max/mean/sd computed from x only, applied to c(x,xx)
#    B) Leaky definition: min/max/mean/sd computed from c(x,xx)
#
# We do this by recomputing both candidate transforms and checking which matches
# the stored output. If it matches B, we FAIL (xx influenced scaling parameters).
# ---------------------------------------------------------------------

# Reference transform A (CURRENT intended, x-only stats):
# 1) Min-max using x only:
#    x_mm  = (x  - min(x)) / (max(x) - min(x))
#    xx_mm = (xx - min(x)) / (max(x) - min(x))
# 2) Z-score using x_mm only:
#    x_std  = (x_mm  - mean(x_mm)) / sd(x_mm)
#    xx_std = (xx_mm - mean(x_mm)) / sd(x_mm)
ref_transform_A <- function(x, xx) {
  x  <- as.numeric(x)
  xx <- as.numeric(xx)
  
  # Min–max using x only
  mn <- min(x, na.rm = TRUE)
  mx <- max(x, na.rm = TRUE)
  rng <- mx - mn
  
  if (!is.finite(rng) || rng == 0) {
    x_mm  <- rep(0, length(x))
    xx_mm <- rep(0, length(xx))
  } else {
    x_mm  <- (x  - mn) / rng
    xx_mm <- (xx - mn) / rng
  }
  
  # Standardise using x_mm only
  mu  <- mean(x_mm, na.rm = TRUE)
  sig <- sd(x_mm, na.rm = TRUE)
  
  if (!is.finite(sig) || sig == 0) {
    x_std  <- rep(0, length(x_mm))
    xx_std <- rep(0, length(xx_mm))
  } else {
    x_std  <- (x_mm  - mu) / sig
    xx_std <- (xx_mm - mu) / sig
  }
  
  list(x_std = x_std, xx_std = xx_std)
}

# Reference transform B (leaky-style):
# min/max computed from v=c(x,xx) rather than x only
ref_transform_B <- function(x, xx) {
  x <- as.numeric(x); xx <- as.numeric(xx)
  v <- c(x, xx)
  
  mn <- min(v); mx <- max(v)
  rng <- mx - mn
  if (!is.finite(rng) || rng == 0) {
    v_mm <- rep(0, length(v))
  } else {
    v_mm <- (v - mn) / rng
  }
  
  mu <- mean(v_mm); sig <- sd(v_mm)
  if (!is.finite(sig) || sig == 0) {
    v_z <- v_mm - mu
  } else {
    v_z <- (v_mm - mu) / sig
  }
  
  list(
    x_std  = v_z[seq_along(x)],
    xx_std = v_z[(length(x)+1):(length(x)+length(xx))]
  )
}

# Compare helper
vec_close <- function(a, b, tol = 1e-10) {
  a <- as.numeric(a); b <- as.numeric(b)
  if (length(a) != length(b)) return(FALSE)
  isTRUE(all.equal(a, b, tolerance = tol))
}

set.seed(123)
n_sample <- min(200L, nrow(W_raw))
idx <- sample.int(nrow(W_raw), n_sample)

matchA <- logical(n_sample)
matchB <- logical(n_sample)

for (k in seq_along(idx)) {
  i <- idx[k]
  x  <- W_raw$x[[i]]
  xx <- W_raw$xx[[i]]
  
  out <- list(x_std = W_std$x[[i]], xx_std = W_std$xx[[i]])
  
  a <- ref_transform_A(x, xx)
  b <- ref_transform_B(x, xx)
  
  matchA[k] <- vec_close(out$x_std, a$x_std) && vec_close(out$xx_std, a$xx_std)
  matchB[k] <- vec_close(out$x_std, b$x_std) && vec_close(out$xx_std, b$xx_std)
}

cat("\nTransform match rates on sample (n=", n_sample, "):\n", sep="")
cat("  Matches A (x-only stats, paper-style): ", mean(matchA), "\n", sep="")
cat("  Matches B (x+xx stats, leaky-style)  : ", mean(matchB), "\n", sep="")

if (mean(matchB) > 0.95) {
  stop("FAIL: Transformation strongly matches LEAKY variant (stats depend on xx). Fix scale_pair_minmax_std() to use x-only stats.")
}
if (mean(matchA) < 0.95) {
  stop("FAIL: Transformation does not match paper-style definition reliably. scale_pair_minmax_std() likely differs from intended formula.")
}
cat("PASS: Transformation matches paper-style x-only statistics (no within-row future leakage).\n")

# ---------------------------------------------------------------------
# 4) Learnability sanity: distribution not degenerate
#    Check global variance and per-row variance of transformed x
# ---------------------------------------------------------------------
x_sd <- vapply(W_std$x, function(v) sd(as.numeric(v)), numeric(1))
if (all(!is.finite(x_sd))) stop("FAIL: All transformed x have non-finite sd.")
if (median(x_sd, na.rm = TRUE) < 1e-6) {
  stop("FAIL: Transformed x is nearly constant (median sd too small). Likely degenerate -> model cannot learn.")
}
cat("PASS: Transformed x has non-trivial variance (median sd = ", signif(median(x_sd, na.rm = TRUE), 4), ").\n", sep="")

cat("\n================ QA 04 RESULT: PASS ====================\n")
