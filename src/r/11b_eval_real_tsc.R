# =====================================================================
# 11b_eval_real_tsc.R
# Build TSC-friendly evaluation dataset:
#   - one row per M4 series
#   - columns: st, true_label, x_1, ..., x_WINDOW_SIZE
#
# Uses the SAME labelling logic and scaling as training:
#   - scale_pair_minmax_std(x_win, xx)
#   - compute_z_generic + label_from_z_int
#
# Assumes in 00_main.R / 0_common.R:
#   - LABEL_ID
#   - WINDOW_SIZE
#   - TAG               (e.g. "y", "q", "m", "w", "d", "h")
#   - subset_clean_file (e.g. "data/M4_subset_clean_q.rds")
#   - threshold_file_train (e.g. "data/label_threshold_c_train_q.rds")
#
# Output:
#   data/export/real_eval_tsc_l{LABEL_ID}_{TAG}.csv
# =====================================================================

library(dplyr)
library(tibble)

source("src/r/utils.R")    # scale_pair_minmax_std, compute_z_generic, label_from_z_int, etc.

# ---------------------------------------------------------------------
# 0) Required globals from 00_main.R / 0_common.R
# ---------------------------------------------------------------------

if (!exists("LABEL_ID")) {
  stop("LABEL_ID not defined. Please set it in 00_main.R before running this script.")
}
if (!exists("WINDOW_SIZE")) {
  stop("WINDOW_SIZE not defined. Please set it in 0_common.R before running this script.")
}
if (!exists("TAG")) {
  stop("TAG not defined. Ensure freq_tag(TARGET_PERIOD) ran in 0_common.R.")
}
if (!exists("subset_clean_file")) {
  stop("subset_clean_file not defined in 0_common.R.")
}
if (!exists("threshold_file_train")) {
  stop("threshold_file_train not defined in 0_common.R.")
}

# Hard requirement: this must match training logic
if (!exists("compute_z_generic", mode = "function")) {
  stop(
    "compute_z_generic() is not available.\n",
    "Script 11b requires it to compute z for LABEL_ID.\n",
    "Ensure compute_z_generic is defined in src/r/utils.R."
  )
}

label_col   <- paste0("l", LABEL_ID)
window_size <- as.integer(WINDOW_SIZE)

if (!file.exists(subset_clean_file)) {
  stop("Clean M4 subset not found at: ", subset_clean_file)
}
if (!file.exists(threshold_file_train)) {
  stop("Threshold file not found at: ", threshold_file_train)
}

M4_clean <- readRDS(subset_clean_file)
c_value  <- readRDS(threshold_file_train)

cat("11b: Loaded", length(M4_clean), "M4 series from", subset_clean_file, "\n")
cat("11b: Using label_col =", label_col, " | window_size =", window_size, " | TAG =", TAG, "\n")

# ---------------------------------------------------------------------
# 1) Build TSC evaluation dataset: last window per series
# ---------------------------------------------------------------------

n_series <- length(M4_clean)

rows <- lapply(seq_len(n_series), function(i) {
  s <- M4_clean[[i]]
  
  x_full <- as.numeric(s$x)   # cleaned historical
  xx     <- as.numeric(s$xx)  # TRUE future horizon
  
  if (length(x_full) < window_size) return(NULL)
  
  # last WINDOW_SIZE points as window
  x_win <- tail(x_full, window_size)
  
  # joint scaling of window + future horizon (must match training)
  scaled <- scale_pair_minmax_std(x_win, xx)
  x_std  <- scaled$x_std
  xx_std <- scaled$xx_std
  
  # true label (must match training)
  z_val    <- compute_z_generic(x_std, xx_std, LABEL_ID)
  true_lbl <- label_from_z_int(z_val, c_value)
  
  # put x_std as columns x_1 ... x_WINDOW_SIZE
  x_cols <- as.list(as.numeric(x_std))
  names(x_cols) <- paste0("x_", seq_along(x_std))
  
  tibble(
    st         = s$st,
    true_label = as.integer(true_lbl),
    !!!x_cols
  )
})

real_eval_tsc <- bind_rows(rows)

cat("Built TSC eval dataset with", nrow(real_eval_tsc), "rows and",
    ncol(real_eval_tsc), "columns.\n")

# ---------------------------------------------------------------------
# 2) Export CSV for Python / sktime
# ---------------------------------------------------------------------

export_dir <- file.path("data", "export")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

out_path <- file.path(
  export_dir,
  sprintf("real_eval_tsc_l%d_%s.csv", LABEL_ID, TAG)
)

write.csv(real_eval_tsc, out_path, row.names = FALSE)

cat("Saved TSC-friendly eval data to:\n  ", out_path, "\n", sep = "")