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
library(future)
library(furrr)


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
# 4. Build ts list for all windows
# ---------------------------------------------------------------------

cat("Converting", length(all_windows_std$x), "windows to ts objects...\n")

ts_list <- lapply(all_windows_std$x, function(vec) ts(vec, frequency = FREQ))

# ---------------------------------------------------------------------
# 5. Define the FFORMA feature set (aligned with calc_features logic)
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
# 6. Choose parallel or sequential mode for tsfeatures
# ---------------------------------------------------------------------

# Use fewer workers on Windows to avoid 'error writing to connection'
n_cores <- future::availableCores()
n_workers <- min(8L, max(1L, n_cores - 1L))

message("[TSFEATURES] Using ", n_workers, " workers for parallel computation.")

future::plan(multisession, workers = n_workers)

use_parallel <- RUN_PARALLEL  # from 00_main.R

if (use_parallel) {
  cat("Running tsfeatures() in PARALLEL via future::multisession...\n")
  n_cores <- max(1, detectCores(logical = FALSE))
  future::plan(multisession, workers = n_cores )
} else {
  cat("Running tsfeatures() SEQUENTIALLY...\n")
  n_cores<-1
  future::plan(sequential)
}

# ---------------------------------------------------------------------
# 7. Single call to tsfeatures() over ALL windows
# ---------------------------------------------------------------------


cat("Extracting FFORMA features for", length(ts_list), "windows...\n")

fforma_feature_matrix <- tsfeatures::tsfeatures(
  tslist      = ts_list,
  features    = fforma_features,
  # match calc_features() defaults: scale=TRUE, trim=FALSE, na.action=na.pass
  scale       = TRUE,
  trim        = FALSE,
  parallel    = use_parallel,
  multiprocess = future::multisession,
  na.action   = na.pass
)

# ---------------------------------------------------------------------
# 8. Post-processing: series_length, seasonal padding, NA → 0
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
# 9. Sanity check and bind with labels
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
# 10. Save final dataset
# ---------------------------------------------------------------------

saveRDS(all_windows_with_features, features_file)

cat("Saved →", features_file, "\n")

