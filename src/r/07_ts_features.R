# =====================================================================
# 07_ts_features.R
# Extract tsfeatures (FFORMA-style) for each window
#
# Input : labeled_file        (e.g. "data/all_windows_labeled_q.rds")
#         windows_std_file    (e.g. "data/all_windows_std_q.rds")
#         subset_clean_file   (e.g. "data/M4_subset_clean_q.rds")
# Output: features_file       (e.g. "data/all_windows_with_features_q.rds")
#
# Assumes the following are defined in 00_main.R:
#   - labeled_file
#   - windows_std_file
#   - subset_clean_file
#   - features_file
#   - RUN_PARALLEL
# And in src/r/utils.R:
#   - infer_frequency(period)
#
# Uses src/r/features.R for:
#   - heterogeneity_tsfeat_workaround
#   - hw_parameters_tsfeat_workaround
#   - (FFORMA-style feature definitions)
# =====================================================================

library(dplyr)
library(tsfeatures)
library(forecast)
library(tibble)

library(parallel)
library(foreach)
library(doParallel)

source("src/r/features.R")

# ---------------------------------------------------------------------
# 1. Load previous step output (labeled windows)
# ---------------------------------------------------------------------

if (!file.exists(labeled_file)) {
  stop(
    "File not found: ", labeled_file,
    ". Run 06_labels.R first."
  )
}

all_windows_labeled <- readRDS(labeled_file)
cat("Loaded", nrow(all_windows_labeled), "windows with labels from", labeled_file, "\n")

# ---------------------------------------------------------------------
# 2. Load standardized windows (needed for ts construction)
# ---------------------------------------------------------------------

if (!file.exists(windows_std_file)) {
  stop(
    "File not found: ", windows_std_file,
    ". Run 04_transformations.R first."
  )
}

all_windows_std <- readRDS(windows_std_file)

# ---------------------------------------------------------------------
# 3. Infer frequency from cleaned M4 subset
# ---------------------------------------------------------------------

if (!file.exists(subset_clean_file)) {
  stop(
    "Missing cleaned subset: ", subset_clean_file,
    ". Run 02_clean_m4.R first."
  )
}

M4_subset_clean <- readRDS(subset_clean_file)
period <- M4_subset_clean[[1]]$period

FREQ <- infer_frequency(period)
cat("Detected period:", period, "→ using ts frequency =", FREQ, "\n")

# ---------------------------------------------------------------------
# 4. Define the FFORMA feature set (aligned with calc_features logic)
# ---------------------------------------------------------------------

fforma_features <- c(
  "acf_features",
  "arch_stat",
  "crossing_points",
  "entropy",
  "flat_spots",
  heterogeneity_tsfeat_workaround,
  "holt_parameters",
  "hurst",
  "lumpiness",
  "nonlinearity",
  "pacf_features",
  "stl_features",
  "stability",
  hw_parameters_tsfeat_workaround,
  "unitroot_kpss",
  "unitroot_pp"
)

# ---------------------------------------------------------------------
# 5. Prepare parallel backend (foreach) – NO tsfeatures internal parallel
# ---------------------------------------------------------------------

use_parallel <- isTRUE(RUN_PARALLEL)  # from 00_main.R
cl <- NULL

if (use_parallel) {
  # You can later swap this for a "physical cores" helper if desired
  n_cores <- max(1L, parallel::detectCores(logical = FALSE) - 1L)
  
  cat("[TSFEATURES] Using foreach with", n_cores, "workers...\n")
  
  cl <- parallel::makeCluster(n_cores, type = "PSOCK")
  doParallel::registerDoParallel(cl)
  
  # Ensure workers have required packages and feature definitions
  parallel::clusterEvalQ(cl, {
    library(tsfeatures)
    library(forecast)
    library(tibble)
  })
  # If features.R defines additional helpers, source it on workers too
  parallel::clusterEvalQ(cl, {
    source("src/r/features.R")
  })
} else {
  cat("[TSFEATURES] Running SEQUENTIALLY (no parallel backend)...\n")
  foreach::registerDoSEQ()
}

# ---------------------------------------------------------------------
# 6. Compute tsfeatures per window using foreach
# ---------------------------------------------------------------------

n_windows <- length(all_windows_std$x)
cat("Extracting FFORMA features for", n_windows, "windows...\n")

if (use_parallel) {
  # Parallel version: build ts + compute features inside foreach
  fforma_feature_matrix <- foreach::foreach(
    vec = all_windows_std$x,
    .combine  = rbind,
    .packages = c("tsfeatures", "forecast", "tibble", "stats")
  ) %dopar% {
    ts_obj <- stats::ts(vec, frequency = FREQ)
    
    # tsfeatures expects a list of ts objects
    tsfeatures::tsfeatures(
      tslist     = list(ts_obj),
      features   = fforma_features,
      scale      = TRUE,
      trim       = FALSE,
      parallel   = FALSE,      # IMPORTANT: no internal tsfeatures parallel
      na.action  = na.pass
    )
  }
  
} else {
  # Sequential version using lapply + bind_rows
  feature_list <- lapply(all_windows_std$x, function(vec) {
    ts_obj <- stats::ts(vec, frequency = FREQ)
    
    tsfeatures::tsfeatures(
      tslist     = list(ts_obj),
      features   = fforma_features,
      scale      = TRUE,
      trim       = FALSE,
      parallel   = FALSE,      # IMPORTANT: sequential inside tsfeatures
      na.action  = na.pass
    )
  })
  
  fforma_feature_matrix <- dplyr::bind_rows(feature_list)
}

# Stop cluster if used
if (!is.null(cl)) {
  parallel::stopCluster(cl)
  foreach::registerDoSEQ()
}

# ---------------------------------------------------------------------
# 7. Post-processing: series_length, seasonal padding, NA → 0
# ---------------------------------------------------------------------

cat("Post-processing features (series_length, seasonal padding, NA → 0)...\n")

series_lengths <- vapply(all_windows_std$x, length, integer(1))
fforma_feature_matrix <- tibble::add_column(
  fforma_feature_matrix,
  series_length = series_lengths
)

# Replace NAs with 0
fforma_feature_matrix[is.na(fforma_feature_matrix)] <- 0

# Ensure seasonal columns exist; add with 0 if missing
seasonal_cols <- c("seas_acf1", "seas_pacf", "seasonal_strength", "peak", "trough")
missing_seasonal <- setdiff(seasonal_cols, names(fforma_feature_matrix))

for (col in missing_seasonal) {
  fforma_feature_matrix[[col]] <- 0
}

# ---------------------------------------------------------------------
# 8. Sanity check and bind with labels
# ---------------------------------------------------------------------

if (nrow(fforma_feature_matrix) != nrow(all_windows_labeled)) {
  stop(
    "ERROR: Feature matrix (", nrow(fforma_feature_matrix),
    " rows) and labeled windows (", nrow(all_windows_labeled),
    " rows) do not match!"
  )
}

all_windows_with_features <- dplyr::bind_cols(
  all_windows_labeled,
  fforma_feature_matrix
)

cat(
  "Final dataset dimensions:",
  nrow(all_windows_with_features), "rows x",
  ncol(all_windows_with_features), "columns.\n"
)

# ---------------------------------------------------------------------
# 9. Save final dataset
# ---------------------------------------------------------------------

saveRDS(all_windows_with_features, features_file)

cat("Saved →", features_file, "\n")
