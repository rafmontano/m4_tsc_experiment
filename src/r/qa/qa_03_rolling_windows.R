# =====================================================================
# qa_03_rolling_windows.R
# QA checks for Script 03 (rolling windows)
#
# Objectives:
# 1) No leakage: windows are built from s$x only, and do NOT use s$xx.
# 2) Learnability sanity: dataset is not degenerate (basic signal checks).
# 3) Correct implementation: indexing/counts/lengths match expectations.
#
# Usage:
#   - Run after Script 03 has produced windows_raw_file
#   - Assumes globals from 00_main.R + 0_common.R exist:
#       subset_clean_file, windows_raw_file, WINDOW_SIZE, HORIZON, STRIDE
# =====================================================================

library(dplyr)

# ---------------------------------------------------------------------
# 0) Required globals
# ---------------------------------------------------------------------
req <- c("subset_clean_file", "windows_raw_file", "WINDOW_SIZE", "HORIZON", "STRIDE")
missing <- req[!vapply(req, exists, logical(1))]
if (length(missing) > 0L) stop("Missing required globals: ", paste(missing, collapse = ", "))

if (!file.exists(subset_clean_file)) stop("Missing subset_clean_file: ", subset_clean_file)
if (!file.exists(windows_raw_file)) stop("Missing windows_raw_file: ", windows_raw_file)

# ---------------------------------------------------------------------
# 1) Load artefacts
# ---------------------------------------------------------------------
M4_clean <- readRDS(subset_clean_file)
W <- readRDS(windows_raw_file)

cat("\n================ QA 03: Rolling windows ================\n")
cat("Series:", length(M4_clean), "\n")
cat("Windows:", nrow(W), "\n")
cat("WINDOW_SIZE:", WINDOW_SIZE, " HORIZON:", HORIZON, " STRIDE:", STRIDE, "\n\n")

stopifnot(is.data.frame(W))
stopifnot(all(c("x", "xx", "series_id") %in% names(W)))

# ---------------------------------------------------------------------
# 2) Correctness: lengths of each window and horizon
# ---------------------------------------------------------------------
len_x  <- vapply(W$x,  length, integer(1))
len_xx <- vapply(W$xx, length, integer(1))

if (any(len_x != WINDOW_SIZE)) {
  stop("FAIL: Some x windows are not WINDOW_SIZE. Bad rows: ", paste(which(len_x != WINDOW_SIZE)[1:10], collapse = ", "))
}
if (any(len_xx != HORIZON)) {
  stop("FAIL: Some xx horizons are not HORIZON. Bad rows: ", paste(which(len_xx != HORIZON)[1:10], collapse = ", "))
}
cat("PASS: All x windows have length WINDOW_SIZE and all xx horizons have length HORIZON.\n")

# ---------------------------------------------------------------------
# 3) Correctness: expected window count per series (given stride)
#    max_start = n - WINDOW_SIZE - HORIZON + 1
#    starts = seq(1, max_start, by=STRIDE)
# ---------------------------------------------------------------------
calc_expected_windows <- function(n, window_size, horizon, stride) {
  max_start <- n - window_size - horizon + 1L
  if (max_start < 1L) return(0L)
  length(seq.int(1L, max_start, by = as.integer(stride)))
}

n_by_series <- vapply(M4_clean, function(s) length(as.numeric(s$x)), integer(1))
expected_by_series <- vapply(n_by_series, calc_expected_windows, integer(1),
                             window_size = WINDOW_SIZE, horizon = HORIZON, stride = STRIDE)

observed_by_series <- W %>%
  count(series_id, name = "n_obs") %>%
  right_join(tibble(series_id = seq_along(M4_clean), n_exp = expected_by_series), by = "series_id") %>%
  mutate(n_obs = ifelse(is.na(n_obs), 0L, n_obs))

bad_counts <- observed_by_series %>% filter(n_obs != n_exp)

if (nrow(bad_counts) > 0L) {
  print(head(bad_counts, 20))
  stop("FAIL: Window counts per series do not match expected (stride/indexing bug).")
}
cat("PASS: Window counts per series match expected given stride.\n")

# ---------------------------------------------------------------------
# 4) No leakage test A: ensure xx is NOT pulling from s$xx (real future)
#    For each series, check whether ANY produced horizon equals
#    the first HORIZON values of the true future s$xx.
#
# If your QA dataset has no s$xx (synthetic may), we still check safely.
# ---------------------------------------------------------------------
has_real_xx <- all(vapply(M4_clean, function(s) !is.null(s$xx) && length(s$xx) >= HORIZON, logical(1)))

if (has_real_xx) {
  leakage_hits <- 0L
  checked <- 0L
  
  for (sid in unique(W$series_id)) {
    s <- M4_clean[[sid]]
    fut <- as.numeric(s$xx)
    if (length(fut) < HORIZON) next
    
    target <- fut[1:HORIZON]
    w_sid <- W %>% filter(series_id == sid)
    
    # Compare each horizon sequence to real future prefix
    hit <- any(vapply(w_sid$xx, function(hh) identical(as.numeric(hh), target), logical(1)))
    checked <- checked + 1L
    if (hit) leakage_hits <- leakage_hits + 1L
  }
  
  if (leakage_hits > 0L) {
    stop("FAIL: Potential leakage detected: for some series, a produced horizon equals real future s$xx prefix.")
  } else {
    cat("PASS: No horizons equal the real future s$xx prefix (no direct s$xx leakage).\n")
  }
} else {
  cat("SKIP: Real-future leakage check (s$xx not available or too short in this QA dataset).\n")
}

# ---------------------------------------------------------------------
# 5) No leakage test B: ensure xx follows x contiguously within s$x
#    For random sample of rows, confirm:
#      tail(x,1) precedes head(xx,1) in the original s$x.
# ---------------------------------------------------------------------
set.seed(123)
n_sample <- min(200L, nrow(W))
idx <- sample.int(nrow(W), n_sample)

check_contiguous <- function(row) {
  sid <- row$series_id[[1]]
  s   <- as.numeric(M4_clean[[sid]]$x)
  
  xw <- as.numeric(row$x[[1]])
  hw <- as.numeric(row$xx[[1]])
  
  Wn <- length(xw)
  Hn <- length(hw)
  
  max_j <- length(s) - (Wn + Hn) + 1L
  if (max_j < 1L) return(FALSE)
  
  for (j in 1:max_j) {
    # Use all.equal to avoid floating-point exact-match issues after tsclean()
    if (isTRUE(all.equal(s[j:(j + Wn - 1L)], xw, tolerance = 1e-12))) {
      return(isTRUE(all.equal(s[(j + Wn):(j + Wn + Hn - 1L)], hw, tolerance = 1e-12)))
    }
  }
  
  FALSE
}

contig_ok <- vapply(seq_along(idx), function(k) check_contiguous(W[idx[k], ]), logical(1))

if (!all(contig_ok)) {
  bad <- idx[which(!contig_ok)][1:min(10, sum(!contig_ok))]
  stop("FAIL: Some (x, xx) pairs are not contiguous slices within s$x. Bad rows: ", paste(bad, collapse = ", "))
}
cat("PASS: Sampled (x, xx) pairs are contiguous within s$x (xx comes right after x in training history).\n")

# ---------------------------------------------------------------------
# 6) Learnability sanity: confirm the dataset isn't degenerate
#    Use a crude proxy: horizon mean change vs last window value.
#    If all horizons are identical / all deltas ~0, learning will be impossible.
# ---------------------------------------------------------------------
delta1 <- vapply(seq_len(nrow(W)), function(i) {
  xw <- as.numeric(W$x[[i]])
  hw <- as.numeric(W$xx[[i]])
  (mean(hw) - tail(xw, 1))
}, numeric(1))

if (!any(is.finite(delta1))) stop("FAIL: All delta proxies are non-finite.")
if (sd(delta1, na.rm = TRUE) < 1e-10) {
  stop("FAIL: Horizon-vs-last-value proxy has ~zero variance; dataset likely degenerate.")
}

cat("PASS: Non-degenerate horizon signal proxy (sd(delta) =", signif(sd(delta1, na.rm = TRUE), 4), ").\n")

cat("\n================ QA 03 RESULT: PASS ====================\n")
