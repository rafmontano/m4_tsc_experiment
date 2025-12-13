# =====================================================================
# 06a_compute_class_proportion.R
# Purpose:
#   Build class_proportion.csv summarising TRUE label distribution
#   for each frequency (counts + proportions).
#
# Input (historical variants supported):
#   data/export/real_eval_l{LABEL_ID}_data_{freq_tag}.csv
#   data/export/real_eval_l{LABEL_ID}_{freq_tag}_data.csv
#   data/export/real_eval_{freq_tag}_l{LABEL_ID}_data.csv
#   data/export/real_eval_{freq_tag}_data_l{LABEL_ID}.csv
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

LABEL_ID      <- 3 

# --------------------------------------------------------------------
# 1. Frequencies
# --------------------------------------------------------------------

freq_list <- c("Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly")

# --------------------------------------------------------------------
# 2. Helper: locate real_eval file (supports historical naming)
# --------------------------------------------------------------------

find_real_eval_file <- function(label_id, tag, export_dir = file.path("data", "export")) {
  
  # Preferred + historical candidate names
  candidates <- c(
    file.path(export_dir, sprintf("real_eval_l%d_data_%s.csv", label_id, tag)),  # new
    file.path(export_dir, sprintf("real_eval_l%d_%s_data.csv", label_id, tag)),  # old (your weekly example)
    file.path(export_dir, sprintf("real_eval_%s_l%d_data.csv", tag, label_id)),  # variant
    file.path(export_dir, sprintf("real_eval_%s_data_l%d.csv", tag, label_id))   # variant
  )
  
  existing <- candidates[file.exists(candidates)]
  if (length(existing) > 0) {
    return(existing[1])
  }
  
  # Fallback: search any CSV in export_dir that contains:
  # - "real_eval"
  # - "l{label_id}"
  # - the tag as a token (e.g., "_w_" or "_w." etc.)
  if (!dir.exists(export_dir)) {
    return(NA_character_)
  }
  
  files <- list.files(export_dir, pattern = "\\.csv$", full.names = TRUE)
  
  # Regex: real_eval.*l3.*(w as token)
  # Accept separators "_" or "-" and ensure tag is bounded by separator/dot/end.
  tag_re <- sprintf("(^|[_\\-])%s([_\\-\\.]|$)", tag)
  lbl_re <- sprintf("l%d", label_id)
  
  hits <- files[
    str_detect(basename(files), regex("real_eval", ignore_case = TRUE)) &
      str_detect(basename(files), fixed(lbl_re)) &
      str_detect(basename(files), regex(tag_re, ignore_case = TRUE))
  ]
  
  if (length(hits) > 0) {
    return(hits[1])
  }
  
  NA_character_
}

# --------------------------------------------------------------------
# 3. Helper: read real eval CSV and compute counts + proportions
# --------------------------------------------------------------------

read_class_counts <- function(period) {
  
  tag <- freq_tag(period)
  
  export_dir <- file.path("data", "export")
  real_df_path <- find_real_eval_file(LABEL_ID, tag, export_dir = export_dir)
  
  if (is.na(real_df_path) || !file.exists(real_df_path)) {
    message("[06a] WARNING: Missing real_eval file for ", period, " (tag=", tag, ") in ", export_dir)
    return(NULL)
  }
  
  message("[06a] Using real_eval file for ", period, ": ", real_df_path)
  
  real_df <- readr::read_csv(real_df_path, show_col_types = FALSE)
  
  if (!("true_label" %in% names(real_df))) {
    stop(
      "[06a] Column 'true_label' not found in: ", real_df_path,
      "\n[06a] Available columns: ", paste(names(real_df), collapse = ", ")
    )
  }
  
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
    group_by(frequency) %>%
    mutate(p = n / sum(n)) %>%
    ungroup() %>%
    select(frequency, class_label, n, p)
  
  counts
}

# --------------------------------------------------------------------
# 4. Compute for all frequencies
# --------------------------------------------------------------------

all_counts <- purrr::map_dfr(freq_list, read_class_counts)

if (nrow(all_counts) == 0) {
  stop("[06a] No class counts produced. Check that real_eval CSVs exist in data/export/.")
}

# --------------------------------------------------------------------
# 5. Output
# --------------------------------------------------------------------

out_dir <- file.path("data", "export")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_csv <- file.path(out_dir, "class_proportion.csv")
readr::write_csv(all_counts, out_csv)

cat("[06a] Class proportion CSV saved to:\n  ", out_csv, "\n", sep = "")
