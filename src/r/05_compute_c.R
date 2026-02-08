# =====================================================================
# 05_compute_c.R
# Compute z-values and derive thresholds:
#   - c_all   : computed from ALL windows (diagnostic/oracle only)
#   - c_train : computed from TRAIN subset only using bootstrap (Option C)
#
# Input  : windows_std_file       (e.g. "data/all_windows_std_q.rds")
# Outputs:
#   - threshold_file_train        (RDS) c_train  [used by pipeline]
#   - threshold_file_all          (RDS) c_all    [diagnostic]
#
# Requires in 00_main.R:
#   - windows_std_file
#   - threshold_file_train
#   - threshold_file_all
#   - C_TRAIN_FRAC, C_BOOT_B, Q, C_SEED
#
# Requires in src/r/utils.R:
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

n <- nrow(all_windows_std)
if (n < 2L) {
  stop("Not enough rows to compute thresholds. nrow(all_windows_std) = ", n)
}

# ---------------------------------------------------------------------
# 2. Compute z on ALL windows and c_all (diagnostic/oracle)
# ---------------------------------------------------------------------

z_all <- compute_z(all_windows_std)
cat("Computed", length(z_all), "z-values on ALL windows.\n")

c_all <- compute_c(z_all, q = Q)
cat("Computed c_all (ALL windows) =", c_all, "\n")

# ---------------------------------------------------------------------
# 3. Define TRAIN subset (for c_train estimation only) and compute z_train
#    Note: labels do not exist yet at this stage, so stratification is
#    not possible here. This split is purely to prevent c estimation
#    from using the held-out portion of windows.
# ---------------------------------------------------------------------

set.seed(C_SEED)

n_train <- max(1L, floor(C_TRAIN_FRAC * n))
train_idx <- sample.int(n, size = n_train, replace = FALSE)

train_windows_std <- all_windows_std[train_idx, , drop = FALSE]

z_train <- compute_z(train_windows_std)
cat("Computed", length(z_train), "z-values on TRAIN subset for c_train.\n")

# ---------------------------------------------------------------------
# 4. Option C: Bootstrap-stabilised c_train from TRAIN subset only
# ---------------------------------------------------------------------

B <- as.integer(C_BOOT_B)
if (!is.finite(B) || B < 1L) stop("C_BOOT_B must be a positive integer.")

c_boot <- numeric(B)

for (b in seq_len(B)) {
  z_b <- sample(z_train, size = length(z_train), replace = TRUE)
  c_boot[b] <- compute_c(z_b, q = Q)
}

c_train <- median(c_boot, na.rm = TRUE)

cat("Computed c_train (bootstrap median) =", c_train, "\n")
cat("Bootstrap summary (c_boot):\n")
print(summary(c_boot))

# ---------------------------------------------------------------------
# 5. Save thresholds
#   - threshold_file     : c_train (used downstream by pipeline)
#   - threshold_file_all : c_all   (diagnostic)
# ---------------------------------------------------------------------

# Oracle threshold (full data)
saveRDS(c_all, file = threshold_file_all)
cat("Saved c_all →", threshold_file_all, "\n")

# Leakage-safe threshold (training only)
saveRDS(c_train, file = threshold_file_train)
cat("Saved c_train →", threshold_file_train, "\n")

