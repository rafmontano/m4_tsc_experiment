# =====================================================================
# 06a_compute_class_proportion.R
# Purpose:
#   Build class_proportion.csv summarising TRUE labels for each frequency.
#
# Input (from evaluation stage):
#   results/real_metrics_l{LABEL_ID}_{freq_tag}.rds
#
# Output:
#   data/export/class_proportion.csv
#
# Expected globals (from 00_main.R):
#   - LABEL_ID
# =====================================================================

library(tidyverse)
source("src/r/utils.R")   # for freq_tag()

# --------------------------------------------------------------------
# 0. Check LABEL_ID
# --------------------------------------------------------------------

if (!exists("LABEL_ID")) {
  stop("LABEL_ID is not defined. Please run 00_main.R first.")
}

# --------------------------------------------------------------------
# 1. Frequency list matching the pipeline order
# --------------------------------------------------------------------

freq_list <- c("Yearly", "Quarterly", "Monthly",
               "Weekly", "Daily", "Hourly")

# --------------------------------------------------------------------
# 2. Helper: read real metrics for each frequency
# --------------------------------------------------------------------

read_class_counts <- function(period) {
  
  tag <- freq_tag(period)
  
  metrics_path <- file.path(
    "results",
    sprintf("real_metrics_l%d_%s.rds", LABEL_ID, tag)
  )
  
  if (!file.exists(metrics_path)) {
    message("WARNING: Missing metrics for ", period, " at ", metrics_path)
    return(NULL)
  }
  
  df <- readRDS(metrics_path)
  
  # df contains per-class metrics + overall fields
  # Need only class-level counts: 0=Down, 1=Neutral, 2=Up
  
  class_df <- df %>%
    select(class) %>%
    mutate(
      frequency = period,
      class_label = case_when(
        class == 0 ~ "Down",
        class == 1 ~ "Neutral",
        class == 2 ~ "Up",
        TRUE ~ "Unknown"
      )
    )
  
  # The RDS does NOT contain raw counts — reconstruct using proportional values:
  # For class proportion charts we instead need true counts (n).
  # Use original real_eval_df (produced in script 11) instead.
  
  real_df_path <- file.path(
    "data", "export",
    sprintf("real_eval_l%d_data_%s.csv", LABEL_ID, tag)
  )
  
  if (!file.exists(real_df_path)) {
    stop("Missing real_eval file: ", real_df_path)
  }
  
  real_df <- readr::read_csv(real_df_path, show_col_types = FALSE)
  
  # Count true_label distribution
  counts <- real_df %>%
    count(true_label, name = "n") %>%
    mutate(
      class_label = case_when(
        true_label == 0 ~ "Down",
        true_label == 1 ~ "Neutral",
        true_label == 2 ~ "Up",
        TRUE ~ "Unknown"
      ),
      frequency = period
    ) %>%
    select(frequency, class_label, n)
  
  return(counts)
}

# --------------------------------------------------------------------
# 3. Compute proportions for all frequencies
# --------------------------------------------------------------------

all_counts <- purrr::map_dfr(freq_list, read_class_counts)

if (nrow(all_counts) == 0) {
  stop("No class counts could be produced. Check preceding scripts.")
}

# --------------------------------------------------------------------
# 4. Output path
# --------------------------------------------------------------------

out_dir <- file.path("data", "export")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_csv <- file.path(out_dir, "class_proportion.csv")

readr::write_csv(all_counts, out_csv)

cat("Class proportion CSV saved to:\n  ", out_csv, "\n", sep = "")