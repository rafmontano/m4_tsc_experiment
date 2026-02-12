# =====================================================================
# 0_common.R
# Shared derived objects and utilities for the M4-XGB pipeline
# ALL non-main configuration + derived values live here.
# =====================================================================

# ---------------------------------------------------------------------
# Guard: variables that must be defined in 00_main.R before sourcing here
# ---------------------------------------------------------------------

required_main <- c("N_KEEP", "LABEL_ID", "RUN_PARALLEL", "FORCE_RERUN", "TARGET_PERIOD")
missing_main <- required_main[!vapply(required_main, exists, logical(1))]
if (length(missing_main) > 0L) {
  stop(
    "0_common.R requires these variables to be defined in 00_main.R first: ",
    paste(missing_main, collapse = ", ")
  )
}

# ---------------------------------------------------------------------
# Derived parameters based on TARGET_PERIOD
# ---------------------------------------------------------------------

HORIZON     <- get_m4_horizon(TARGET_PERIOD)
WINDOW_SIZE <- get_window_size_from_h(TARGET_PERIOD)

# Stride policy for rolling windows
# STRIDE <- WINDOW_SIZE + HORIZON   # non-overlapping
# STRIDE <- WINDOW_SIZE             # partial overlap
STRIDE <- 1L                        # fully overlapping (not recommended)

TAG <- freq_tag(TARGET_PERIOD)

# ---------------------------------------------------------------------
# Label threshold configuration
# ---------------------------------------------------------------------

Q            <- 0.20   # (i.e. q=0.6 => Down = 30%, Neutral 40%, Up 30%)
C_SEED       <- 123
C_TRAIN_FRAC <- 0.70
C_BOOT_B     <- 1000L

# ---------------------------------------------------------------------
# Train/test split configuration (series-id split)
# ---------------------------------------------------------------------

SPLIT_SEED <- 123
TRAIN_FRAC <- 0.80

USE_CLASS_WEIGHTS <- FALSE  # set TRUE to restore weighting later

# ---------------------------------------------------------------------
# Frequency-tagged file names (per period)
# ---------------------------------------------------------------------

subset_file          <- file.path("data", paste0("M4_subset_", TAG, ".rds"))
subset_clean_file    <- file.path("data", paste0("M4_subset_clean_", TAG, ".rds"))
windows_raw_file     <- file.path("data", paste0("all_windows_raw_", TAG, ".rds"))
windows_std_file     <- file.path("data", paste0("all_windows_std_", TAG, ".rds"))
threshold_file_train <- file.path("data", paste0("label_threshold_c_train_", TAG, ".rds"))
threshold_file_all   <- file.path("data", paste0("label_threshold_c_all_", TAG, ".rds"))
labeled_file         <- file.path("data", paste0("all_windows_labeled_", TAG, ".rds"))
features_file        <- file.path("data", paste0("all_windows_with_features_", TAG, ".rds"))

# ---------------------------------------------------------------------
# XGBoost model paths (single source of truth)
# NOTE: use .ubj to avoid the "Unknown file format: model" warning.
# ---------------------------------------------------------------------

model_dir <- file.path("models", "xgb")
if (!dir.exists(model_dir)) dir.create(model_dir, recursive = TRUE)

model_path <- file.path(model_dir, sprintf("xgb_l%d_%s.ubj", LABEL_ID, TAG))
meta_path  <- file.path(model_dir, sprintf("xgb_l%d_%s_meta.rds", LABEL_ID, TAG))

# ---------------------------------------------------------------------
# Helper: cleanup after each step
#   - Keep configuration + file paths
#   - Keep all functions
#   - Drop large data objects from the global environment
# ---------------------------------------------------------------------

cleanup_step <- function() {
  
  keep_explicit <- c(
    # main-owned
    "N_KEEP", "LABEL_ID", "RUN_PARALLEL", "FORCE_RERUN", "TARGET_PERIOD",
    
    # derived/config
    "HORIZON", "WINDOW_SIZE", "STRIDE", "TAG",
    "Q", "C_SEED", "C_TRAIN_FRAC", "C_BOOT_B",
    "SPLIT_SEED", "TRAIN_FRAC", "USE_CLASS_WEIGHTS",
    
    # file paths
    "subset_file", "subset_clean_file",
    "windows_raw_file", "windows_std_file",
    "threshold_file_train", "threshold_file_all",
    "labeled_file", "features_file",
    
    # model paths
    "model_dir", "model_path", "meta_path",
    
    # function itself
    "cleanup_step"
  )
  
  all_objs <- ls(envir = .GlobalEnv)
  
  is_fun <- vapply(
    all_objs,
    function(x) is.function(get(x, envir = .GlobalEnv)),
    logical(1)
  )
  
  non_fun_objs <- all_objs[!is_fun]
  to_remove <- setdiff(non_fun_objs, keep_explicit)
  
  if (length(to_remove) > 0L) {
    rm(list = to_remove, envir = .GlobalEnv)
  }
  
  gc()
}