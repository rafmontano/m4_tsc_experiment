# =====================================================================
# 03_rolling_windows.R
# Create rolling windows (x, xx) from cleaned M4 subset
#
# Assumes the following are defined in 00_main.R:
#   - subset_clean_file   (e.g. "data/M4_subset_clean_q.rds")
#   - windows_raw_file    (e.g. "data/all_windows_raw_q.rds")
#   - WINDOW_SIZE         (rolling window size)
#   - HORIZON             (forecast horizon)
# =====================================================================

library(tibble)
library(dplyr)

# ---------------------------------------------------------------------
# 1. Load cleaned M4 subset
# ---------------------------------------------------------------------

if (!file.exists(subset_clean_file)) {
  stop(
    "File not found: ", subset_clean_file,
    ". Run 01_load_m4_subset.R and 02_clean_m4.R first."
  )
}

M4_subset_clean <- readRDS(subset_clean_file)
cat("Loaded cleaned M4 subset with", length(M4_subset_clean), "series.\n")

if (length(M4_subset_clean) == 0) {
  stop("ERROR: cleaned subset is empty.")
}

cat("Using window_size =", WINDOW_SIZE, "and horizon =", HORIZON, "\n")

# ---------------------------------------------------------------------
# 2. Function: Create windows for ONE numeric vector
#    (candidate for utils.R if needed elsewhere)
# ---------------------------------------------------------------------

make_rolling_windows <- function(x, window_size, horizon) {
  x <- as.numeric(x)
  n <- length(x)
  
  max_start <- n - window_size - horizon + 1
  if (max_start < 1) {
    stop(
      "Series too short for window_size = ", window_size,
      " and horizon = ", horizon
    )
  }
  
  windows  <- vector("list", max_start)
  horizons <- vector("list", max_start)
  
  for (i in seq_len(max_start)) {
    windows[[i]]  <- x[i:(i + window_size - 1)]
    horizons[[i]] <- x[(i + window_size):(i + window_size + horizon - 1)]
  }
  
  tibble(
    x  = windows,
    xx = horizons
  )
}

# ---------------------------------------------------------------------
# 3. Function: Apply rolling windows to all series
# ---------------------------------------------------------------------

make_all_rolling_windows <- function(M4_list, window_size, horizon) {
  rows <- lapply(seq_along(M4_list), function(i) {
    s <- M4_list[[i]]
    
    # full series = cleaned training + real future horizon
    full_ts <- c(s$x, s$xx)
    
    tbl <- make_rolling_windows(full_ts, window_size, horizon)
    
    tbl %>%
      mutate(
        series_id   = i,
        series_name = s$name
      )
  })
  
  bind_rows(rows)
}

# ---------------------------------------------------------------------
# 4. Create rolling windows
# ---------------------------------------------------------------------

all_windows <- make_all_rolling_windows(
  M4_list     = M4_subset_clean,
  window_size = WINDOW_SIZE,
  horizon     = HORIZON
)

cat("Created", nrow(all_windows), "windows.\n")

# ---------------------------------------------------------------------
# 5. Save for next scripts
# ---------------------------------------------------------------------

saveRDS(all_windows, windows_raw_file)

cat("Saved rolling windows to", windows_raw_file, "\n")
