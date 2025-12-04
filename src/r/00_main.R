# =====================================================================
# 00_main.R
# Master script that runs the pipeline end-to-end for ONE frequency
#
# You will run this script once per frequency, e.g.:
#   - Yearly
#   - Quarterly
#   - Monthly
#   - Weekly
#   - Daily
# =====================================================================

source("src/r/utils.R")

cat("\n====================\n")
cat("   M4-XGB Pipeline\n")
cat("====================\n\n")

# ---------------------------------------------------------------------
# PARAMETERS (EDIT AS NEEDED BEFORE EACH RUN)
# ---------------------------------------------------------------------

N_KEEP        <- 100        # How many M4 series to keep in 01_load_m4_subset.R
LABEL_ID      <- 3          # Label to evaluate in scripts 10 + 11
RUN_PARALLEL  <- TRUE       # Use parallel version of tsfeatures
FORCE_RERUN   <- TRUE      # If TRUE, recompute even if files exist
TARGET_PERIOD <- "Quarterly" #"Yearly" #"Quarterly"  # Change manually per run ("Yearly", "Monthly", ...)

HORIZON       <- get_m4_horizon(TARGET_PERIOD)
WINDOW_SIZE   <- get_window_size_from_h(TARGET_PERIOD)
TAG           <- freq_tag(TARGET_PERIOD)

cat("Parameters:\n")
cat("  TARGET_PERIOD =", TARGET_PERIOD, "\n")
cat("  TAG           =", TAG, "\n")
cat("  N_KEEP        =", N_KEEP, "\n")
cat("  WINDOW_SIZE   =", WINDOW_SIZE, "\n")
cat("  HORIZON       =", HORIZON, "\n")
cat("  LABEL_ID      =", LABEL_ID, "\n")
cat("  RUN_PARALLEL  =", RUN_PARALLEL, "\n")
cat("  FORCE_RERUN   =", FORCE_RERUN, "\n\n")

# ---------------------------------------------------------------------
# Frequency-tagged file names (per period)
# ---------------------------------------------------------------------

subset_file       <- file.path("data", paste0("M4_subset_", TAG, ".rds"))
subset_clean_file <- file.path("data", paste0("M4_subset_clean_", TAG, ".rds"))
windows_raw_file  <- file.path("data", paste0("all_windows_raw_", TAG, ".rds"))
windows_std_file  <- file.path("data", paste0("all_windows_std_", TAG, ".rds"))
threshold_file    <- file.path("data", paste0("label_threshold_c_", TAG, ".rds"))
labeled_file      <- file.path("data", paste0("all_windows_labeled_", TAG, ".rds"))
features_file     <- file.path("data", paste0("all_windows_with_features_", TAG, ".rds"))

# NOTE:
#  - 01_load_m4_subset.R will write to subset_file
#  - 02_clean_m4.R will write to subset_clean_file
#  - 03_rolling_windows.R will write to windows_raw_file
#  - 04_transformations.R will write to windows_std_file
#  - 05_compute_c.R will write to threshold_file
#  - 06_labels.R will write to labeled_file
#  - 07_ts_features.R will write to features_file

# ---------------------------------------------------------------------
# 01: Load M4 subset
# ---------------------------------------------------------------------

if (!file.exists(subset_file) || FORCE_RERUN) {
  cat("\n[01] Loading M4 subset...\n")
  source("src/r/01_load_m4_subset.R")
} else {
  cat("[01] Skipped (", subset_file, " already exists)\n", sep = "")
}

#gc(full = TRUE)

# ---------------------------------------------------------------------
# 02: Clean M4 (tsclean)
# ---------------------------------------------------------------------

if (!file.exists(subset_clean_file) || FORCE_RERUN) {
  cat("\n[02] Cleaning M4 series...\n")
  source("src/r/02_clean_m4.R")
} else {
  cat("[02] Skipped (clean file exists: ", subset_clean_file, ")\n", sep = "")
}

#gc(full = TRUE)
# ---------------------------------------------------------------------
# 03: Rolling windows
# ---------------------------------------------------------------------

if (!file.exists(windows_raw_file) || FORCE_RERUN) {
  cat("\n[03] Creating rolling windows...\n")
  source("src/r/03_rolling_windows.R")
} else {
  cat("[03] Skipped (", windows_raw_file, " already exists)\n", sep = "")
}

#gc(full = TRUE)

# ---------------------------------------------------------------------
# 04: Min-max and standardisation
# ---------------------------------------------------------------------

if (!file.exists(windows_std_file) || FORCE_RERUN) {
  cat("\n[04] Applying transformations...\n")
  source("src/r/04_transformations.R")
} else {
  cat("[04] Skipped (", windows_std_file, " already exists)\n", sep = "")
}

# ---------------------------------------------------------------------
# 05: Compute z and c
# ---------------------------------------------------------------------

if (!file.exists(threshold_file) || FORCE_RERUN) {
  cat("\n[05] Computing threshold c...\n")
  source("src/r/05_compute_c.R")
} else {
  cat("[05] Skipped (", threshold_file, " already exists)\n", sep = "")
}

#gc(full = TRUE)

# ---------------------------------------------------------------------
# 06: Labels l1–l4
# ---------------------------------------------------------------------

if (!file.exists(labeled_file) || FORCE_RERUN) {
  cat("\n[06] Creating labels...\n")
  source("src/r/06_labels.R")
} else {
  cat("[06] Skipped (", labeled_file, " already exists)\n", sep = "")
}

#gc(full = TRUE)
# ---------------------------------------------------------------------
# 07+08: Compute tsfeatures (parallel optional)
# ---------------------------------------------------------------------

if (!file.exists(features_file) || FORCE_RERUN) {
  cat("\n[07] Computing TS features (parallel = ", RUN_PARALLEL, ")...\n", sep = "")
  source("src/r/07_ts_features.R")
  source("src/r/07b_ts_windows_export.R")
} else {
  cat("[07] Skipped (", features_file, " already exists)\n", sep = "")
}

#gc(full = TRUE)
# ---------------------------------------------------------------------
# 09: XGBoost hyperparameter tuning + model training
# ---------------------------------------------------------------------

cat("\n[09] Training XGBoost model...\n")
source("src/r/09_hyper_xgb.R")

# ---------------------------------------------------------------------
# 10: Evaluate on internal test split
# ---------------------------------------------------------------------

cat("\n[10] Evaluating on held-out windows...\n")
source("src/r/10_eval_xgb.R")

# ---------------------------------------------------------------------
# 11: Evaluate on REAL M4 last-window data
# ---------------------------------------------------------------------

cat("\n[11] Evaluating REAL last-window performance...\n")
source("src/r/11_eval_real_xgb.R")

cat("\n[11b] Exporting TSC real sktime-friendly...\n")
source("src/r/11b_eval_real_tsc.R")

# ---------------------------------------------------------------------
# 12: Baseline SMYL and FFORMA
# ---------------------------------------------------------------------

cat("\n[12] Baseline SMYL and FFORMA...\n")
source("src/r/12_baseline_fforma_smyl.R")

# ---------------------------------------------------------------------
# Done!
# ---------------------------------------------------------------------

cat("\n=====================\n")
cat("  PIPELINE COMPLETE\n")
cat("=====================\n")