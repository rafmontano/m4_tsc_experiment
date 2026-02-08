# =====================================================================
# 04_transformations.R
# Apply min–max scaling and standardisation to rolling windows
#
# Input  : windows_raw_file   (e.g. "data/all_windows_raw_q.rds")
# Output : windows_std_file   (e.g. "data/all_windows_std_q.rds")
#
# Assumes the following are defined in 00_main.R:
#   - windows_raw_file
#   - windows_std_file
#   - scale_pair_minmax_std() available via src/r/utils.R
# =====================================================================

library(dplyr)

# ---------------------------------------------------------------------
# 0) Sanity checks
# ---------------------------------------------------------------------

if (!exists("scale_pair_minmax_std")) {
  stop("scale_pair_minmax_std() not found. Ensure src/r/utils.R is sourced before running 04_transformations.R.")
}

if (!file.exists(windows_raw_file)) {
  stop(
    "File not found: ", windows_raw_file,
    ". Run 03_rolling_windows.R first."
  )
}

# ---------------------------------------------------------------------
# 1) Load rolling windows from script 03
# ---------------------------------------------------------------------

all_windows <- readRDS(windows_raw_file)
cat("Loaded", nrow(all_windows), "rolling windows from", windows_raw_file, "\n")

# ---------------------------------------------------------------------
# 2) Apply joint min–max + standardisation per row
#    scale_pair_minmax_std(x, xx) returns list(x_std=..., xx_std=...)
# ---------------------------------------------------------------------

all_windows_std <- all_windows %>%
  mutate(
    scaled = Map(scale_pair_minmax_std, x, xx),
    x  = lapply(scaled, `[[`, "x_std"),
    xx = lapply(scaled, `[[`, "xx_std")
  ) %>%
  select(-scaled)

cat("Applied joint min–max + standardisation.\n")

# ---------------------------------------------------------------------
# 3) Save transformed dataset for downstream scripts
# ---------------------------------------------------------------------

saveRDS(all_windows_std, file = windows_std_file)
cat("Saved transformed windows to:", windows_std_file, "\n")