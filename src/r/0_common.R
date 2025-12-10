# =====================================================================
# 0_common.R
# Shared derived objects and utilities for the M4-XGB pipeline
# =====================================================================

# ---------------------------------------------------------------------
# Derived parameters based on TARGET_PERIOD
# ---------------------------------------------------------------------

HORIZON     <- get_m4_horizon(TARGET_PERIOD)
WINDOW_SIZE <- get_window_size_from_h(TARGET_PERIOD)
TAG         <- freq_tag(TARGET_PERIOD)

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


# ---------------------------------------------------------------------
# XGBoost model paths (shared by 00_main.R and 09_hyper_xgb.R)
# ---------------------------------------------------------------------

model_dir <- file.path("models", "xgb")
if (!dir.exists(model_dir)) dir.create(model_dir, recursive = TRUE)

model_path <- file.path(model_dir, sprintf("xgb_l%d_%s.model", LABEL_ID, TAG))
meta_path  <- file.path(model_dir, sprintf("xgb_l%d_%s_meta.rds", LABEL_ID, TAG))



# ---------------------------------------------------------------------
# Helper: cleanup after each step
#   - Keep configuration + file paths
#   - Keep all functions (from utils.R and scripts)
#   - Drop large data objects from the global environment
# ---------------------------------------------------------------------

cleanup_step <- function() {
  # Names we explicitly want to keep
  keep_explicit <- c(
    "N_KEEP", "LABEL_ID", "RUN_PARALLEL", "FORCE_RERUN",
    "TARGET_PERIOD", "HORIZON", "WINDOW_SIZE", "TAG",
    "subset_file", "subset_clean_file", "windows_raw_file",
    "windows_std_file", "threshold_file", "labeled_file",
    "features_file",
    "cleanup_step", "model_dir", "meta_path", "model_path"
  )
  
  all_objs <- ls(envir = .GlobalEnv)
  
  # Identify which of those are functions (we keep all functions)
  is_fun <- vapply(
    all_objs,
    function(x) is.function(get(x, envir = .GlobalEnv)),
    logical(1)
  )
  
  # Objects that are NOT functions
  non_fun_objs <- all_objs[!is_fun]
  
  # Remove all non-function objects except the explicit keep set
  to_remove <- setdiff(non_fun_objs, keep_explicit)
  
  if (length(to_remove) > 0L) {
    rm(list = to_remove, envir = .GlobalEnv)
  }
  
  gc()
}
