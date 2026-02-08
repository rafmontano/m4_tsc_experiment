# =====================================================================
# 06_labels.R
# Compute 4 label variants (l1–l4) for each window+horizon pair
#
# Input  : windows_std_file        (e.g. "data/all_windows_std_q.rds")
#          threshold_file_train    (e.g. "data/label_threshold_c_train_q.rds")
# Output : labeled_file            (e.g. "data/all_windows_labeled_q.rds")
#
# Assumes the following are defined in 00_main.R:
#   - windows_std_file
#   - threshold_file_train
#   - labeled_file
# And in src/r/utils.R:
#   - label_from_z_int(z, c)
# =====================================================================

library(dplyr)

# ---------------------------------------------------------------------
# 1. Load standardized windows and leakage-safe threshold c_train
# ---------------------------------------------------------------------

if (!file.exists(windows_std_file)) {
  stop(
    "File not found: ", windows_std_file,
    ". Run 04_transformations.R and 05_compute_c.R first."
  )
}
if (!file.exists(threshold_file_train)) {
  stop(
    "File not found: ", threshold_file_train,
    ". Run 05_compute_c.R first to compute and save c_train."
  )
}

all_windows_std <- readRDS(windows_std_file)
c_value         <- readRDS(threshold_file_train)

cat(
  "Loaded", nrow(all_windows_std),
  "standardised windows from", windows_std_file, "\n"
)
cat("Loaded threshold c_train =", c_value, "\n")

if (!all(c("x", "xx") %in% names(all_windows_std))) {
  stop("ERROR: Expected columns x and xx not found in all_windows_std.")
}

# ---------------------------------------------------------------------
# 2. Helper: compute the 4 z’s for one window + horizon
# ---------------------------------------------------------------------

compute_z_four <- function(x, xx) {
  w <- as.numeric(x)
  h <- as.numeric(xx)
  
  H      <- length(h)
  w_last <- tail(w, 1)
  w_tail <- tail(w, H)
  
  S_mean   <- mean(h)
  S_median <- median(h)
  
  B_last      <- w_last
  B_mean_tail <- mean(w_tail)
  
  z1 <- S_mean   - B_last
  z2 <- S_median - B_last
  z3 <- S_mean   - B_mean_tail
  z4 <- S_median - B_mean_tail
  
  c(z1 = z1, z2 = z2, z3 = z3, z4 = z4)
}

# ---------------------------------------------------------------------
# 3. Main function: add l1–l4 to all_windows_std
# ---------------------------------------------------------------------

add_labels_four <- function(all_windows_std, c_value) {
  z_mat <- t(
    mapply(
      compute_z_four,
      x  = all_windows_std$x,
      xx = all_windows_std$xx
    )
  )
  
  z1 <- z_mat[, "z1"]
  z2 <- z_mat[, "z2"]
  z3 <- z_mat[, "z3"]
  z4 <- z_mat[, "z4"]
  
  all_windows_std %>%
    mutate(
      l1 = label_from_z_int(z1, c_value),
      l2 = label_from_z_int(z2, c_value),
      l3 = label_from_z_int(z3, c_value),
      l4 = label_from_z_int(z4, c_value)
    )
}

all_windows_labeled <- add_labels_four(all_windows_std, c_value)

cat("Labels l1–l4 added.\n")
cat("Example distribution (l3):\n")
print(table(all_windows_labeled$l3))

# ---------------------------------------------------------------------
# 4. Save labeled windows
# ---------------------------------------------------------------------

saveRDS(all_windows_labeled, labeled_file)
cat("Saved labeled windows to", labeled_file, "\n")