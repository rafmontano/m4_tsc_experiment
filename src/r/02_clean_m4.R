# =====================================================================
# 02_clean_m4.R
# Clean only the training part (x) of the M4 subset using tsclean()
# Leaves future part (xx) unchanged.
#
# Assumes the following are defined in 00_main.R:
#   - subset_file        (e.g. "data/M4_subset_q.rds")
#   - subset_clean_file  (e.g. "data/M4_subset_clean_q.rds")
# =====================================================================

library(forecast)

# ---------------------------------------------------------------------
# 1. Load subset created by 01_load_m4_subset.R
# ---------------------------------------------------------------------

if (!file.exists(subset_file)) {
  stop("File not found: ", subset_file, ". Run 01_load_m4_subset.R first.")
}

M4_subset <- readRDS(subset_file)
cat("Loaded M4 subset with", length(M4_subset), "series.\n")

# ---------------------------------------------------------------------
# 2. Determine ts frequency from period (shared helper in utils.R)
# ---------------------------------------------------------------------

target_period <- M4_subset[[1]]$period
freq <- period_to_freq(target_period)

cat("Detected period:", target_period, "→ ts frequency =", freq, "\n")

# ---------------------------------------------------------------------
# 3. Clean only s$x, leave s$xx unchanged
#    (candidate for utils.R if reused elsewhere)
# ---------------------------------------------------------------------

clean_m4_train_only <- function(M4_list, freq) {
  lapply(M4_list, function(s) {
    x_num <- as.numeric(s$x)
    x_ts  <- ts(x_num, frequency = freq)
    
    x_clean <- tsclean(x_ts)
    
    s$x <- as.numeric(x_clean)
    s   # return modified series
  })
}

M4_subset_clean <- clean_m4_train_only(M4_subset, freq = freq)

cat("Cleaned training parts (x) for", length(M4_subset_clean), "series.\n")

# ---------------------------------------------------------------------
# 4. Save cleaned subset for downstream scripts
# ---------------------------------------------------------------------

saveRDS(M4_subset_clean, file = subset_clean_file)

cat("Saved cleaned subset to:", subset_clean_file, "\n")
