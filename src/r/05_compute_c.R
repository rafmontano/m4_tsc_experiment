# =====================================================================
# 05_compute_c.R
# Compute z-values and derive threshold c from standardized windows
#
# Input  : windows_std_file      (e.g. "data/all_windows_std_q.rds")
# Output : threshold_file        (e.g. "data/label_threshold_c_q.rds")
#
# Assumes the following are defined in 00_main.R:
#   - windows_std_file
#   - threshold_file
# And in src/r/utils.R:
#   - compute_z(all_windows)
#   - compute_c(z, q = 0.40)
# =====================================================================

# ---------------------------------------------------------------------
# 1. Load standardized windows
# ---------------------------------------------------------------------

if (!file.exists(windows_std_file)) {
  stop(
    "File not found: ", windows_std_file,
    ". Run 04_transformations.R first."
  )
}

all_windows_std <- readRDS(windows_std_file)
cat(
  "Loaded", nrow(all_windows_std),
  "standardised windows from", windows_std_file, "\n"
)

if (!all(c("x", "xx") %in% names(all_windows_std))) {
  stop("ERROR: Expected columns x and xx not found in all_windows_std.")
}

# ---------------------------------------------------------------------
# 2. Compute z-values (uses compute_z() from utils.R)
# ---------------------------------------------------------------------

z_all <- compute_z(all_windows_std)
cat("Computed", length(z_all), "z-values.\n")

# ---------------------------------------------------------------------
# 3. Compute threshold c (uses compute_c() from utils.R)
#    We fix q = 0.40 based on prior analysis.
# ---------------------------------------------------------------------

c <- compute_c(z_all, q = 0.40)

cat("Computed threshold c =", c, "\n")
cat("Summary of z:\n")
print(summary(z_all))

# ---------------------------------------------------------------------
# 4. Save c for downstream scripts
# ---------------------------------------------------------------------

saveRDS(c, file = threshold_file)

cat("Saved threshold c →", threshold_file, "\n")
