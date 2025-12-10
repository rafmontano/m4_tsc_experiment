# =====================================================================
# 00_main.R
# Master script that runs the pipeline end-to-end for ONE frequency
#
# You will run this script once per frequency, e.g.:
#   - Yearly - 23K - Done
#   - Quarterly - 24k - Done
#   - Monthly - 48k - TBA
#   - Weekly -  359 - Done
#   - Daily - 4k  - TBA
#   - Hourly 414 - Done
#    TARGET_PERIOD <- "Monthly"  #"Yearly" #"Quarterly"  # Change manually per run ("Yearly", "Monthly"
# =====================================================================

# =====================================================================
# 00_main.R
# Master script that runs the pipeline end-to-end for ONE frequency
#
# You will run this script once per frequency, e.g.:
#   - Yearly - 23K - Done
#   - Quarterly - 24k - Done
#   - Monthly - 48k - TBA
#   - Weekly -  359 - Done
#   - Daily - 4k  - TBA
#   - Hourly 414 - Done
# =====================================================================

source("src/r/utils.R")

cat("\n====================\n")
cat("   M4-XGB Pipeline\n")
cat("====================\n\n")

# ---------------------------------------------------------------------
# PARAMETERS (EDIT AS NEEDED BEFORE EACH RUN)
# ---------------------------------------------------------------------

N_KEEP        <- 100000       # How many M4 series to keep in 01_load_m4_subset.R
LABEL_ID      <- 3         # Label to evaluate in scripts 10 + 11
RUN_PARALLEL  <- TRUE      # Use parallel version of tsfeatures
FORCE_RERUN   <- FALSE   # If TRUE, recompute even if files exist
TARGET_PERIOD <- "Daily" # "Yearly" / "Quarterly" / "Monthly" / ...

# ---------------------------------------------------------------------
# Load common derived objects and file names
# (HORIZON, WINDOW_SIZE, TAG, subset_file, etc.)
# ---------------------------------------------------------------------

source("src/r/0_common.R")

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
# 01: Load M4 subset
# ---------------------------------------------------------------------

if (!file.exists(subset_file) || FORCE_RERUN) {
  cat("\n[01] Loading M4 subset...\n")
  source("src/r/01_load_m4_subset.R")
} else {
  cat("[01] Skipped (", subset_file, " already exists)\n", sep = "")
}
cleanup_step()

# ---------------------------------------------------------------------
# 02: Clean M4 (tsclean)
# ---------------------------------------------------------------------

if (!file.exists(subset_clean_file) || FORCE_RERUN) {
  cat("\n[02] Cleaning M4 series...\n")
  source("src/r/02_clean_m4.R")
} else {
  cat("[02] Skipped (clean file exists: ", subset_clean_file, ")\n", sep = "")
}
cleanup_step()

# ---------------------------------------------------------------------
# 03: Rolling windows
# ---------------------------------------------------------------------

if (!file.exists(windows_raw_file) || FORCE_RERUN) {
  cat("\n[03] Creating rolling windows...\n")
  source("src/r/03_rolling_windows.R")
} else {
  cat("[03] Skipped (", windows_raw_file, " already exists)\n", sep = "")
}
cleanup_step()

# ---------------------------------------------------------------------
# 04: Min-max and standardisation
# ---------------------------------------------------------------------

if (!file.exists(windows_std_file) || FORCE_RERUN) {
  cat("\n[04] Applying transformations...\n")
  source("src/r/04_transformations.R")
} else {
  cat("[04] Skipped (", windows_std_file, " already exists)\n", sep = "")
}
cleanup_step()

# ---------------------------------------------------------------------
# 05: Compute z and c
# ---------------------------------------------------------------------

if (!file.exists(threshold_file) || FORCE_RERUN) {
  cat("\n[05] Computing threshold c...\n")
  source("src/r/05_compute_c.R")
} else {
  cat("[05] Skipped (", threshold_file, " already exists)\n", sep = "")
}
cleanup_step()

# ---------------------------------------------------------------------
# 06: Labels l1–l4
# ---------------------------------------------------------------------

if (!file.exists(labeled_file) || FORCE_RERUN) {
  cat("\n[06] Creating labels...\n")
  source("src/r/06_labels.R")
} else {
  cat("[06] Skipped (", labeled_file, " already exists)\n", sep = "")
}
cleanup_step()

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
cleanup_step()

# ---------------------------------------------------------------------
# 09: XGBoost hyperparameter tuning + model training
# ---------------------------------------------------------------------

if (!file.exists(model_path) || FORCE_RERUN) {
  cat("\n[09] Training XGBoost model...\n")
  source("src/r/09_hyper_xgb.R")
} else {
  cat("[09] Skipped (", features_file, " already exists)\n", sep = "")
}
cleanup_step()

# ---------------------------------------------------------------------
# 10: Evaluate on internal test split
# ---------------------------------------------------------------------

cat("\n[10] Evaluating on held-out windows...\n")
source("src/r/10_eval_xgb.R")
cleanup_step()

# ---------------------------------------------------------------------
# 11: Evaluate on REAL M4 last-window data
# ---------------------------------------------------------------------

if (FORCE_RERUN) {
  cat("\n[11] Evaluating REAL last-window performance...\n")
  source("src/r/11_eval_real_xgb.R")
  cat("\n[11b] Exporting TSC real sktime-friendly...\n")
  source("src/r/11b_eval_real_tsc.R")
} else {
  cat("[11] Skipped \n")
  cat("\n[11b] \n")
}
cleanup_step()

# ---------------------------------------------------------------------
# 12: Baseline SMYL and FFORMA
# ---------------------------------------------------------------------

cat("\n[12] Baseline SMYL and FFORMA...\n")
source("src/r/12_baseline_fforma_smyl.R")
cleanup_step()

# ---------------------------------------------------------------------
# Done!
# ---------------------------------------------------------------------

cat("\n=====================\n")
cat("  PIPELINE COMPLETE\n")
cat("=====================\n")
