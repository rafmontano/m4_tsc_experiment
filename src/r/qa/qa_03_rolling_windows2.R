# =====================================================================
# QA_LEAK_NEW_WINDOWS.R
# Confirms NO leakage in NEW all_windows generated from s$x only
#
# EXPECTED NEW BEHAVIOUR:
# - Rolling windows are created only from s$x (training history)
# - Therefore, for each series (for the LAST rolling window):
#     - x_last  == s$x[(n_train - WINDOW_SIZE - HORIZON + 1):(n_train - HORIZON)]
#     - xx_last == tail(s$x, HORIZON)       (NOT s$xx)
#     - xx_last != s$xx                     (unless accidental equality)
#
# REQUIREMENTS:
# - windows_raw_file points to the NEW all_windows_raw_*.rds created by new Script 03
# - subset_clean_file points to M4_subset_clean_*.rds (each element has $x and $xx)
# - WINDOW_SIZE and HORIZON match what was used to create all_windows
#
# OUTPUT:
# - Prints per-series proof for one selected series_id (default: 1)
# - Also runs a small sample scan across multiple series_ids for confidence
# =====================================================================

library(dplyr)

# -----------------------------
# 0) Required globals / inputs
# -----------------------------
req <- c("windows_raw_file", "subset_clean_file", "WINDOW_SIZE", "HORIZON")
missing <- req[!vapply(req, exists, logical(1))]
if (length(missing) > 0L) {
  stop("Missing required globals: ", paste(missing, collapse = ", "))
}

if (!file.exists(windows_raw_file)) stop("Missing windows_raw_file: ", windows_raw_file)
if (!file.exists(subset_clean_file)) stop("Missing subset_clean_file: ", subset_clean_file)

cat("\n================ QA: Leakage check on NEW all_windows ================\n")
cat("windows_raw_file :", windows_raw_file, "\n")
cat("subset_clean_file:", subset_clean_file, "\n")
cat("WINDOW_SIZE      :", WINDOW_SIZE, "\n")
cat("HORIZON          :", HORIZON, "\n\n")

# -----------------------------
# 1) Load artefacts
# -----------------------------
W <- readRDS(windows_raw_file)
M4 <- readRDS(subset_clean_file)

stopifnot(is.data.frame(W))
stopifnot(is.list(M4))
stopifnot(all(c("x", "xx", "series_id") %in% names(W)))

n_series_W <- length(unique(W$series_id))
cat("Loaded all_windows rows:", nrow(W), "\n")
cat("Unique series_id in windows:", n_series_W, "\n")
cat("Series in M4_subset_clean   :", length(M4), "\n\n")

if (max(W$series_id, na.rm = TRUE) > length(M4)) {
  stop("FAIL: windows contain series_id > length(M4_subset_clean). Index mismatch.")
}

# =====================================================================
# PART A: Indisputable proof for ONE series (default: series_id = 1)
# =====================================================================

SERIES_ID_TO_PROVE <- 1L

s <- M4[[SERIES_ID_TO_PROVE]]
n_train  <- length(s$x)
n_future <- length(s$xx)

cat("\n================ ONE-SERIES PROOF (NEW) ================\n")
cat("series_id:", SERIES_ID_TO_PROVE, "\n")
cat("length(s$x)  =", n_train, "\n")
cat("length(s$xx) =", n_future, "\n\n")

W1 <- W %>% filter(series_id == SERIES_ID_TO_PROVE)

cat("Windows in windows_raw_file for this series:", nrow(W1), "\n")

# Last window for the series in the windows_raw_file
last_row <- W1 %>% slice_tail(n = 1)

x_last  <- unlist(last_row$x[[1]])
xx_last <- unlist(last_row$xx[[1]])

cat("\n--- Last window row ---\n")
cat("x_last length :", length(x_last), "\n")
cat("xx_last length:", length(xx_last), "\n\n")

# Expected references under NEW code (training-only windows)
x_ref  <- as.numeric(s$x)[(n_train - WINDOW_SIZE - HORIZON + 1):(n_train - HORIZON)]
xx_ref <- as.numeric(s$x)[(n_train - HORIZON + 1):n_train]

cat("Check x_last == expected last window slice of s$x:\n")
print(all.equal(x_last, x_ref))

cat("\nCheck xx_last == tail(s$x, HORIZON):\n")
print(all.equal(xx_last, xx_ref))

cat("\nCheck xx_last == s$xx (should be FALSE; this would indicate leakage):\n")
print(isTRUE(all.equal(xx_last, as.numeric(s$xx))))

cat("\nPrint xx_last (window horizon):\n")
print(xx_last)

cat("\nPrint tail(s$x, HORIZON) (expected under NEW code):\n")
print(xx_ref)

cat("\nPrint s$xx (true future holdout; must not appear in windows_raw_file horizons):\n")
print(as.numeric(s$xx))

cat("\n================ END ONE-SERIES PROOF ================\n\n")

# =====================================================================
# PART B: Scan multiple series_ids (spot-check)
# - Confirms x_last and xx_last match NEW expectations
# - Confirms xx_last != s$xx (no leakage)
# =====================================================================

set.seed(123)
N_SCAN <- 50L
scan_ids <- sample(seq_along(M4), size = min(N_SCAN, length(M4)), replace = FALSE)

scan_res <- lapply(scan_ids, function(sid) {
  s  <- M4[[sid]]
  Wk <- W %>% filter(series_id == sid)
  if (nrow(Wk) == 0L) return(NULL)
  
  last_row <- Wk %>% slice_tail(n = 1)
  
  x_last  <- unlist(last_row$x[[1]])
  xx_last <- unlist(last_row$xx[[1]])
  
  n_train <- length(s$x)
  
  # NEW expected references (match Part A)
  x_ref  <- as.numeric(s$x)[(n_train - WINDOW_SIZE - HORIZON + 1):(n_train - HORIZON)]
  xx_ref <- as.numeric(s$x)[(n_train - HORIZON + 1):n_train]
  
  x_ok  <- isTRUE(all.equal(x_last, x_ref))
  xx_ok <- isTRUE(all.equal(xx_last, xx_ref))
  
  # Leakage indicator: horizon equals true future
  leak_xx <- isTRUE(all.equal(xx_last, as.numeric(s$xx)))
  
  data.frame(
    series_id = sid,
    x_ok = x_ok,
    xx_ok = xx_ok,
    leak_xx_equals_future = leak_xx
  )
})

scan_df <- bind_rows(scan_res)

cat("\n================ SCAN SUMMARY (NEW) ================\n")
cat("Series scanned:", nrow(scan_df), "\n")
cat("x_ok  TRUE count :", sum(scan_df$x_ok,  na.rm = TRUE), "\n")
cat("xx_ok TRUE count :", sum(scan_df$xx_ok, na.rm = TRUE), "\n")
cat("leak_xx_equals_future TRUE count (should be 0):", sum(scan_df$leak_xx_equals_future, na.rm = TRUE), "\n\n")

if (sum(scan_df$leak_xx_equals_future, na.rm = TRUE) > 0L) {
  cat("RESULT: FAIL (Potential leakage detected in scanned series)\n")
  print(scan_df %>% filter(leak_xx_equals_future))
} else {
  cat("RESULT: PASS (No leakage detected in scanned series)\n")
}

cat("\n================ END QA: Leakage check (NEW) =================\n")