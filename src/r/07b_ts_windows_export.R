# =====================================================================
# 07b_ts_windows_export.R
# Export TSC-ready windows for Python (ROCKET / InceptionTime / HC2)
#
# Assumes:
#   - Scripts 1–7 have already run
#   - labeled_file exists (e.g. "data/all_windows_labeled_q.rds")
#   - LABEL_ID (1..4) and TAG are defined in 00_main.R
#
# This script:
#   - DOES NOT modify existing core objects
#   - Only creates new, TSC-ready CSVs
#   - Writes: data/export/windows_tsc_l{LABEL_ID}_{TAG}.csv
# =====================================================================

# ---------------------------------------------------------------------
# 0. Check that LABEL_ID exists (defined upstream)
# ---------------------------------------------------------------------
if (!exists("LABEL_ID")) {
  stop("LABEL_ID is not defined. Run 00_main.R (up to this step) first.")
}

if (!exists("TAG")) {
  stop("TAG is not defined. Ensure freq_tag(TARGET_PERIOD) was computed in 00_main.R.")
}

# ---------------------------------------------------------------------
# 1. Load all_windows_labeled (but into a new name)
# ---------------------------------------------------------------------
if (!file.exists(labeled_file)) {
  stop(
    "File not found: ", labeled_file,
    ". Run scripts up to 06_labels.R first."
  )
}

tsc_windows_obj <- readRDS(labeled_file)
cat(
  "07b: Loaded", nrow(tsc_windows_obj),
  "windows from", labeled_file, "\n"
)

# ---------------------------------------------------------------------
# 2. Determine the label column from LABEL_ID (l1 / l2 / l3 / l4)
# ---------------------------------------------------------------------
tsc_label_col <- paste0("l", LABEL_ID)

if (!tsc_label_col %in% names(tsc_windows_obj)) {
  stop(
    "Label column not found: ", tsc_label_col,
    "\nAvailable columns are: ", paste(names(tsc_windows_obj), collapse = ", ")
  )
}

tsc_labels_raw <- tsc_windows_obj[[tsc_label_col]]

# Do NOT modify the original labels in the pipeline; just make an integer copy
if (is.factor(tsc_labels_raw)) {
  tsc_label_levels <- levels(tsc_labels_raw)
  tsc_labels_int   <- as.integer(tsc_labels_raw)
} else {
  tsc_label_levels <- sort(unique(tsc_labels_raw))
  tsc_labels_int   <- as.integer(tsc_labels_raw)
}

cat(
  "07b: Using label column", tsc_label_col,
  "with", length(unique(tsc_labels_raw)), "classes.\n"
)

# ---------------------------------------------------------------------
# 3. Extract the time-series window from 'x'
# ---------------------------------------------------------------------
if (!"x" %in% names(tsc_windows_obj)) {
  stop("Column 'x' not found in labeled windows; cannot export TSC windows.")
}

# 'x' is the window per row; we assume it is already numeric and same length
tsc_window_length <- length(tsc_windows_obj$x[[1]])
cat("07b: Detected window length:", tsc_window_length, "time steps.\n")

# Flatten list-column x into an n_windows x W matrix
tsc_X_mat <- matrix(
  unlist(tsc_windows_obj$x),
  nrow = nrow(tsc_windows_obj),
  ncol = tsc_window_length,
  byrow = TRUE
)

# ---------------------------------------------------------------------
# 4. Build the export data.frame: label, t1..tW
# ---------------------------------------------------------------------
tsc_df <- data.frame(
  label = tsc_labels_int,
  tsc_X_mat,
  check.names = FALSE
)

colnames(tsc_df) <- c("label", paste0("t", seq_len(tsc_window_length)))

cat(
  "07b: Final TSC data frame →",
  nrow(tsc_df), "rows,",
  ncol(tsc_df), "columns (1 label +", tsc_window_length, "timesteps)\n"
)

# ---------------------------------------------------------------------
# 5. Write CSV (no impact on the R pipeline objects)
#     Use TAG in filename to avoid overwriting across frequencies.
# ---------------------------------------------------------------------
export_dir <- file.path("data", "export")
if (!dir.exists(export_dir)) dir.create(export_dir, recursive = TRUE)

tsc_csv_path <- file.path(
  export_dir,
  sprintf("windows_tsc_l%d_%s.csv", LABEL_ID, TAG)
)

write.csv(tsc_df, tsc_csv_path, row.names = FALSE)

cat("07b: Saved TSC-ready windows →", normalizePath(tsc_csv_path), "\n")
