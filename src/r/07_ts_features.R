# =====================================================================
# 07_ts_features.R
# Extract tsfeatures (FFORMA-style) for each window, with chunking
# and temporary saves to avoid losing work on large frequencies.
#
# Input : labeled_file
#         windows_std_file
#         subset_clean_file
# Output: features_file
#
# Assumes the following are defined in 00_main.R:
#   - labeled_file
#   - windows_std_file
#   - subset_clean_file
#   - features_file
#   - RUN_PARALLEL
#
# Uses src/r/features.R for:
#   - heterogeneity_tsfeat_workaround
#   - hw_parameters_tsfeat_workaround
#   - infer_frequency(period)
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
# 0. Chunking parameters and temp directory
# ---------------------------------------------------------------------

# You can override this via options(TSFEATURES_CHUNK_SIZE = 1000L) in 00_main.R
CHUNK_SIZE <- getOption("TSFEATURES_CHUNK_SIZE", 100000L)

# Temporary directory to store chunked feature results
features_dir <- dirname(features_file)
features_tmp_dir <- file.path(features_dir, paste0(basename(features_file), "_tmp"))

if (!dir.exists(features_tmp_dir)) {
  dir.create(features_tmp_dir, recursive = TRUE, showWarnings = FALSE)
}

cat("[TSFEATURES] Using chunk size =", CHUNK_SIZE, "temporary dir =", features_tmp_dir, "\n")

# ---------------------------------------------------------------------
# 1. Load previous step output (labeled windows)
# ---------------------------------------------------------------------

if (!file.exists(labeled_file)) {
  stop("File not found: ", labeled_file, ". Run 06_labels.R first.")
}

all_windows_labeled <- readRDS(labeled_file)
cat("Loaded", nrow(all_windows_labeled), "windows with labels from", labeled_file, "\n")

# ---------------------------------------------------------------------
# 2. Load standardized windows (needed for ts construction)
# ---------------------------------------------------------------------

if (!file.exists(windows_std_file)) {
  stop("File not found: ", windows_std_file, ". Run 04_transformations.R first.")
}

all_windows_std <- readRDS(windows_std_file)

# ---------------------------------------------------------------------
# 3. Infer frequency from cleaned M4 subset
# ---------------------------------------------------------------------

if (!file.exists(subset_clean_file)) {
  stop("Missing cleaned subset: ", subset_clean_file, ". Run 02_clean_m4.R first.")
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

use_parallel <- isTRUE(RUN_PARALLEL)
cl <- NULL

if (use_parallel) {
  # Cap to physical cores and a maximum (e.g. 16) to avoid oversubscription
  n_cores <- min(16L, parallel::detectCores(logical = FALSE))
  n_cores <- max(1L, n_cores)
  
  cat("[TSFEATURES] Using foreach with", n_cores, "workers...\n")
  
  cl <- parallel::makeCluster(n_cores, type = "PSOCK")
  doParallel::registerDoParallel(cl)
  
  parallel::clusterEvalQ(cl, {
    library(tsfeatures)
    library(forecast)
    library(tibble)
    library(stats)
    source("src/r/features.R")
  })
} else {
  cat("[TSFEATURES] Running SEQUENTIALLY (no parallel backend)...\n")
  foreach::registerDoSEQ()
}

# ---------------------------------------------------------------------
# 6. Chunked computation of tsfeatures
# ---------------------------------------------------------------------

n_windows <- length(all_windows_std$x)
cat("Extracting FFORMA features for", n_windows, "windows...\n")

# Define chunk index sets
all_indices <- seq_len(n_windows)
chunk_ids <- ceiling(all_indices / CHUNK_SIZE)
n_chunks <- max(chunk_ids)

cat("[TSFEATURES] Total chunks:", n_chunks, "\n")

chunk_feature_files <- character(n_chunks)

for (chunk_id in seq_len(n_chunks)) {
  idx <- which(chunk_ids == chunk_id)
  chunk_file <- file.path(features_tmp_dir, sprintf("chunk_%03d.rds", chunk_id))
  chunk_feature_files[chunk_id] <- chunk_file
  
  if (file.exists(chunk_file)) {
    cat("[TSFEATURES] Chunk", chunk_id, "already exists, skipping computation.\n")
    next
  }
  
  cat("[TSFEATURES] Processing chunk", chunk_id, "with", length(idx), "windows...\n")
  
  x_chunk <- all_windows_std$x[idx]
  
  if (use_parallel) {
    # Parallel version within chunk
    chunk_matrix <- foreach::foreach(
      vec = x_chunk,
      .combine  = rbind,
      .packages = c("tsfeatures", "forecast", "tibble", "stats")
    ) %dopar% {
      ts_obj <- stats::ts(vec, frequency = FREQ)
      
      tsfeatures::tsfeatures(
        tslist    = list(ts_obj),
        features  = fforma_features,
        scale     = FALSE,
        trim      = FALSE,
        parallel  = FALSE,
        na.action = na.pass
      )
    }
  } else {
    # Sequential version within chunk
    feature_list <- lapply(x_chunk, function(vec) {
      ts_obj <- stats::ts(vec, frequency = FREQ)
      
      tsfeatures::tsfeatures(
        tslist    = list(ts_obj),
        features  = fforma_features,
        scale     = FALSE,
        trim      = FALSE,
        parallel  = FALSE,
        na.action = na.pass
      )
    })
    
    chunk_matrix <- dplyr::bind_rows(feature_list)
  }
  
  saveRDS(chunk_matrix, chunk_file)
  cat("[TSFEATURES] Saved chunk", chunk_id, "→", chunk_file, "\n")
  cat(sprintf("[TSFEATURES] Chunk %d/%d (%.2f%% complete)\n",
              chunk_id, n_chunks, 100 * chunk_id / n_chunks))
  
  
  rm(x_chunk, chunk_matrix)
  gc()
}

# Stop cluster if used
if (!is.null(cl)) {
  parallel::stopCluster(cl)
  foreach::registerDoSEQ()
}

# ---------------------------------------------------------------------
# 7. Reassemble all chunks into a single feature matrix
# ---------------------------------------------------------------------

cat("[TSFEATURES] Reassembling all chunks...\n")

chunk_mats <- lapply(seq_len(n_chunks), function(chunk_id) {
  chunk_file <- chunk_feature_files[chunk_id]
  if (!file.exists(chunk_file)) {
    stop("Missing expected chunk file: ", chunk_file)
  }
  readRDS(chunk_file)
})

fforma_feature_matrix <- dplyr::bind_rows(chunk_mats)

if (nrow(fforma_feature_matrix) != n_windows) {
  stop(
    "ERROR: After reassembly, feature matrix has ",
    nrow(fforma_feature_matrix), " rows, expected ", n_windows
  )
}

# ---------------------------------------------------------------------
# 8. Post-processing: series_length, seasonal padding, NA → 0
# ---------------------------------------------------------------------

cat("Post-processing features (series_length, seasonal padding, NA → 0)...\n")

series_lengths <- vapply(all_windows_std$x, length, integer(1))
fforma_feature_matrix <- tibble::add_column(
  fforma_feature_matrix,
  series_length = series_lengths
)

fforma_feature_matrix[is.na(fforma_feature_matrix)] <- 0

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
cat("Temporary chunk files remain in:", features_tmp_dir, "\n")
