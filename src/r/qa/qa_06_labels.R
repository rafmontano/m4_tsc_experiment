# =====================================================================
# qa_06_labels.R
# QA checks for Script 06 (labels l1–l4)
#
# Objectives:
# 1) No leakage: labels must be computable from windows_std_file + c_train only,
#    and must match the formal z-definitions used in Script 06.
# 2) Learnability sanity: labels should not collapse to a single class; basic
#    directional consistency checks should hold.
# 3) Correct implementation: lengths preserved, finite values, reproducible.
#
# Usage:
#   Run after Script 06 has produced labeled_file.
#   Requires globals: windows_std_file, threshold_file_train, labeled_file, HORIZON
#   Requires utils.R loaded: label_from_z_int()
# =====================================================================

library(dplyr)

req <- c("windows_std_file", "threshold_file_train", "labeled_file", "HORIZON")
missing <- req[!vapply(req, exists, logical(1))]
if (length(missing) > 0L) stop("Missing required globals: ", paste(missing, collapse = ", "))

if (!exists("label_from_z_int")) stop("label_from_z_int() not found. Source src/r/utils.R first.")
if (!file.exists(windows_std_file)) stop("Missing windows_std_file: ", windows_std_file)
if (!file.exists(threshold_file_train)) stop("Missing threshold_file_train: ", threshold_file_train)
if (!file.exists(labeled_file)) stop("Missing labeled_file: ", labeled_file)

cat("\n================ QA 06: Labels (l1–l4) ================\n")

W_std   <- readRDS(windows_std_file)
c_train <- readRDS(threshold_file_train)
W_lab   <- readRDS(labeled_file)

cat("Rows windows_std:", nrow(W_std), "\n")
cat("Rows labeled    :", nrow(W_lab), "\n")
cat("Loaded c_train  :", c_train, "\n")
cat("HORIZON         :", HORIZON, "\n\n")

# ---------------------------------------------------------------------
# 1) Correctness: structure and invariants
# ---------------------------------------------------------------------
stopifnot(is.data.frame(W_std), is.data.frame(W_lab))
stopifnot(nrow(W_std) == nrow(W_lab))
stopifnot(all(c("x", "xx") %in% names(W_std)))
stopifnot(all(c("x", "xx", "l1", "l2", "l3", "l4") %in% names(W_lab)))

len_x  <- vapply(W_std$x,  length, integer(1))
len_xx <- vapply(W_std$xx, length, integer(1))
if (any(len_xx != HORIZON)) stop("FAIL: Some xx horizons are not length HORIZON in windows_std.")
cat("PASS: windows_std has expected horizon length.\n")

finite_list <- function(lst) all(vapply(lst, function(v) all(is.finite(as.numeric(v))), logical(1)))
if (!finite_list(W_std$x) || !finite_list(W_std$xx)) stop("FAIL: Non-finite values in windows_std (x/xx).")
cat("PASS: windows_std x/xx are finite.\n")

# Labels must be in {0,1,2}
labels_ok <- function(v) all(v %in% c(0L, 1L, 2L))
if (!labels_ok(W_lab$l1) || !labels_ok(W_lab$l2) || !labels_ok(W_lab$l3) || !labels_ok(W_lab$l4)) {
  stop("FAIL: Labels contain values outside {0,1,2}.")
}
cat("PASS: Labels are integers in {0,1,2}.\n")

# ---------------------------------------------------------------------
# 2) Recompute z1..z4 and labels, check match (implementation verification)
# ---------------------------------------------------------------------
compute_z_four <- function(x, xx) {
  w <- as.numeric(x)
  h <- as.numeric(xx)
  
  H      <- length(h)
  w_last <- tail(w, 1)
  w_tail <- tail(w, H)
  
  S_mean   <- mean(h)
  S_median <- median(h)
  
  B_last      <- w_last
  B_mean_tail <- mean(w_tail)
  
  z1 <- S_mean   - B_last
  z2 <- S_median - B_last
  z3 <- S_mean   - B_mean_tail
  z4 <- S_median - B_mean_tail
  
  c(z1 = z1, z2 = z2, z3 = z3, z4 = z4)
}

z_mat <- t(mapply(compute_z_four, x = W_std$x, xx = W_std$xx))
z1 <- z_mat[, "z1"]; z2 <- z_mat[, "z2"]; z3 <- z_mat[, "z3"]; z4 <- z_mat[, "z4"]

l1_r <- label_from_z_int(z1, c_train)
l2_r <- label_from_z_int(z2, c_train)
l3_r <- label_from_z_int(z3, c_train)
l4_r <- label_from_z_int(z4, c_train)

# Exact match checks
if (!identical(as.integer(W_lab$l1), as.integer(l1_r))) stop("FAIL: l1 does not match recomputed labels.")
if (!identical(as.integer(W_lab$l2), as.integer(l2_r))) stop("FAIL: l2 does not match recomputed labels.")
if (!identical(as.integer(W_lab$l3), as.integer(l3_r))) stop("FAIL: l3 does not match recomputed labels.")
if (!identical(as.integer(W_lab$l4), as.integer(l4_r))) stop("FAIL: l4 does not match recomputed labels.")
cat("PASS: l1–l4 exactly match recomputed labels from z1–z4 and c_train.\n")

# ---------------------------------------------------------------------
# 3) No leakage (pragmatic test): labels are reproducible from artefacts only
#    This is effectively proven by the recomputation match above.
#    Add one more check: no unexpected dependence on other columns.
# ---------------------------------------------------------------------
extra_cols <- setdiff(names(W_lab), c(names(W_std), "l1","l2","l3","l4"))
if (length(extra_cols) > 0) {
  cat("INFO: labeled_file contains extra columns (ok if metadata): ", paste(extra_cols, collapse=", "), "\n")
}
cat("PASS: Labels reproducible from windows_std_file + c_train only (no hidden leakage path).\n")

# ---------------------------------------------------------------------
# 4) Learnability sanity checks
# ---------------------------------------------------------------------
print_dist <- function(lbl, name) {
  tab <- prop.table(table(factor(lbl, levels = c(0,1,2))))
  cat("\nLabel distribution ", name, ":\n", sep = "")
  print(tab)
  invisible(tab)
}

d1 <- print_dist(W_lab$l1, "l1")
d2 <- print_dist(W_lab$l2, "l2")
d3 <- print_dist(W_lab$l3, "l3")
d4 <- print_dist(W_lab$l4, "l4")

# Heuristic: avoid total collapse
collapse_check <- function(tab, nm) {
  if (max(tab) > 0.98) stop("FAIL: ", nm, " collapses (max class prevalence > 0.98).")
}
collapse_check(d1, "l1"); collapse_check(d2, "l2"); collapse_check(d3, "l3"); collapse_check(d4, "l4")
cat("\nPASS: No label variant collapses to a single class.\n")

# Directional consistency sanity: sign(z) should align with Up/Down rates
# (Not a strict requirement, but if violated massively it signals a bug.)
dir_check <- function(z, lbl, nm) {
  up_rate_pos   <- mean(lbl == 2L & z > 0, na.rm = TRUE)
  down_rate_neg <- mean(lbl == 0L & z < 0, na.rm = TRUE)
  cat("Directional sanity ", nm, ": P(Up & z>0)=", round(up_rate_pos,4),
      " P(Down & z<0)=", round(down_rate_neg,4), "\n", sep="")
}
dir_check(z1, W_lab$l1, "l1")
dir_check(z2, W_lab$l2, "l2")
dir_check(z3, W_lab$l3, "l3")
dir_check(z4, W_lab$l4, "l4")

cat("\n================ QA 06 RESULT: PASS ====================\n")

