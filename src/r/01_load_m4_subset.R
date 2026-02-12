# =====================================================================
# 01_load_m4_subset.R
# Load M4 dataset and extract a subset for the selected period
#
# Assumes the following globals are defined in 00_main.R:
#   - TARGET_PERIOD  (e.g. "Quarterly", "Monthly", "Yearly", ...)
#   - N_KEEP         (number of series to keep; Inf = all)
#   - TAG            (file tag derived from TARGET_PERIOD via freq_tag())
#   - subset_file    (full path, e.g. "data/M4_subset_q.rds")
# =====================================================================

#library(M4comp2018)

# ---------------------------------------------------------------------
# 1. Load full M4 dataset
# ---------------------------------------------------------------------
#data(M4)
M4 <- readRDS("data/qa/M4_qa.rds")

cat("Total M4 series:", length(M4), "\n")

# ---------------------------------------------------------------------
# 2. Select subset by period (Quarterly, Monthly, etc.)
# ---------------------------------------------------------------------

M4_subset <- Filter(function(s) s$period == TARGET_PERIOD, M4)

cat("Series matching period =", TARGET_PERIOD, ":", length(M4_subset), "\n")

# ---------------------------------------------------------------------
# 3. Validate horizon (h)
# ---------------------------------------------------------------------

H_values <- sapply(M4_subset, function(s) s$h)
cat("Unique forecast horizons in subset:", unique(H_values), "\n")

if (length(unique(H_values)) != 1) {
  stop("ERROR: Mixed horizons detected. This script assumes fixed h.")
}

H <- unique(H_values)
cat("Using horizon h =", H, "\n")

# ---------------------------------------------------------------------
# 4. Optional limit for initial proof-of-concept (N_KEEP)
# ---------------------------------------------------------------------

if (is.finite(N_KEEP) && N_KEEP < length(M4_subset)) {
  M4_subset <- M4_subset[1:N_KEEP]
}

cat("Keeping first", length(M4_subset), "series.\n")

# ---------------------------------------------------------------------
# 5. Length diagnostics
# ---------------------------------------------------------------------

train_lengths <- sapply(M4_subset, function(s) length(s$x))

min_length <- min(train_lengths)
max_length <- max(train_lengths)

cat("Train lengths (x): min =", min_length, "max =", max_length, "\n")

# ---------------------------------------------------------------------
# 6. SAVE OUTPUT FOR NEXT SCRIPTS
# ---------------------------------------------------------------------

# subset_file is defined in 00_main.R as:
#   subset_file <- file.path("data", paste0("M4_subset_", TAG, ".rds"))

saveRDS(M4_subset, file = subset_file)

cat("Saved subset into:", subset_file, "\n")