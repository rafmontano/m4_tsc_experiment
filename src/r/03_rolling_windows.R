# =====================================================================
# 03_rolling_windows.R
# Create rolling windows (x, xx) from cleaned M4 subset
#
# IMPORTANT:
#   - Windows are constructed from s$x ONLY (no s$xx leakage).
#   - STRIDE controls overlap:
#       STRIDE = 1              -> fully overlapping (max windows)
#       STRIDE = WINDOW_SIZE+HORIZON -> non-overlapping (recommended)
#
# Assumes the following globals are defined in 00_main.R / 0_common.R:
#   - subset_clean_file
#   - windows_raw_file
#   - WINDOW_SIZE
#   - HORIZON
#   - STRIDE
# =====================================================================

library(tibble)
library(dplyr)

M4_subset_clean <- readRDS(subset_clean_file)
cat("Loaded cleaned M4 subset with", length(M4_subset_clean), "series.\n")

cat("Using window_size =", WINDOW_SIZE, "horizon =", HORIZON, "stride =", STRIDE, "\n")

make_rolling_windows <- function(x, window_size, horizon, stride) {
  x <- as.numeric(x)
  n <- length(x)
  
  max_start <- n - window_size - horizon + 1L
  if (max_start < 1L) return(NULL)
  
  starts <- seq.int(1L, max_start, by = as.integer(stride))
  
  windows  <- vector("list", length(starts))
  horizons <- vector("list", length(starts))
  
  for (k in seq_along(starts)) {
    i <- starts[k]
    windows[[k]]  <- x[i:(i + window_size - 1L)]
    horizons[[k]] <- x[(i + window_size):(i + window_size + horizon - 1L)]
  }
  
  tibble(x = windows, xx = horizons)
}

make_all_rolling_windows <- function(M4_list, window_size, horizon, stride) {
  rows <- lapply(seq_along(M4_list), function(i) {
    s <- M4_list[[i]]
    
    tbl <- make_rolling_windows(s$x, window_size, horizon, stride)
    if (is.null(tbl)) return(NULL)
    
    tbl %>% mutate(series_id = i, series_name = s$name)
  })
  
  dplyr::bind_rows(rows)
}

all_windows <- make_all_rolling_windows(
  M4_list     = M4_subset_clean,
  window_size = WINDOW_SIZE,
  horizon     = HORIZON,
  stride      = STRIDE
)

cat("Created", nrow(all_windows), "windows.\n")

saveRDS(all_windows, windows_raw_file)
cat("Saved rolling windows to", windows_raw_file, "\n")