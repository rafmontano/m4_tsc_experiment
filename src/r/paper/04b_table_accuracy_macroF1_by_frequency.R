# =====================================================================
# 04b_table_accuracy_macroF1_by_frequency.R
#
# Purpose:
#   Build the paper table:
#     Model × (Accuracy by frequency + W.Avg) × (Macro F1 by frequency + W.Avg)
#
# Inputs:
#   - XGBoost REAL metrics:   results/xgb/real_metrics_l{LABEL_ID}_{TAG}.csv  (script 11)
#   - Baseline metrics:       results/baseline/baseline_metrics_l{LABEL_ID}_{TAG}.csv (script 12)
#   - Other models (optional): currently filled with 0.000 placeholders
#
# Outputs:
#   - results/paper/tables/model_accuracy_macroF1_by_frequency.csv
#   - results/paper/tables/model_accuracy_macroF1_by_frequency_display.csv
# =====================================================================

library(tidyverse)

# ---------------------------------------------------------------------
# 0) CONFIG
# ---------------------------------------------------------------------

LABEL_ID <- 3

# Frequency ordering (matches your paper ordering)
freq_levels <- c("Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly")

# Sample sizes used for weighted average (edit if your counts differ)
freq_sizes <- tibble::tribble(
  ~dataset,     ~n,
  "Yearly",     23000,
  "Quarterly",  24000,
  "Monthly",    48000,
  "Weekly",       359,
  "Daily",       4000,
  "Hourly",       414
)

stopifnot(all(freq_levels %in% freq_sizes$dataset))

# Map dataset -> tag used by your pipelines
freq_map <- tibble::tribble(
  ~dataset,     ~tag,
  "Yearly",     "y",
  "Quarterly",  "q",
  "Monthly",    "m",
  "Weekly",     "w",
  "Daily",      "d",
  "Hourly",     "h"
)

# Models in the table (keep order as displayed)
model_levels <- c(
  "Benchmark (1NN-DTW)",
  "Rotation Forest",
  "XGBoost",
  "InceptionTime",
  "ROCKET",
  "SMYL",
  "FFORMA"
)

# ---------------------------------------------------------------------
# 1) Output paths
# ---------------------------------------------------------------------

out_dir <- file.path("results", "paper", "tables")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_csv         <- file.path(out_dir, "model_accuracy_macroF1_by_frequency.csv")
out_csv_display <- file.path(out_dir, "model_accuracy_macroF1_by_frequency_display.csv")

# ---------------------------------------------------------------------
# 2) Readers (REAL metrics)
# ---------------------------------------------------------------------

read_xgb_metrics <- function(tag, label_id) {
  path <- file.path("results", "xgb", sprintf("real_metrics_l%d_%s.csv", label_id, tag))
  if (!file.exists(path)) return(NULL)
  
  df <- readr::read_csv(path, show_col_types = FALSE)
  
  # One row per class; overall metrics repeat
  tibble(
    model    = "XGBoost",
    accuracy = as.numeric(first(df$overall_accuracy)),
    macro_f1 = as.numeric(first(df$macro_f1))
  )
}

read_baseline_metrics <- function(tag, label_id) {
  path <- file.path("results", "baseline", sprintf("baseline_metrics_l%d_%s.csv", label_id, tag))
  if (!file.exists(path)) return(NULL)
  
  df <- readr::read_csv(path, show_col_types = FALSE)
  
  # One row per class; overall metrics repeat per method
  df %>%
    group_by(method) %>%
    summarise(
      accuracy = as.numeric(first(overall_accuracy)),
      macro_f1 = as.numeric(first(macro_f1)),
      .groups  = "drop"
    ) %>%
    mutate(
      model = case_when(
        tolower(method) == "smyl"   ~ "SMYL",
        tolower(method) == "fforma" ~ "FFORMA",
        TRUE ~ as.character(method)
      )
    ) %>%
    select(model, accuracy, macro_f1)
}

# Placeholder provider (matches your current example: 0.000)
placeholder_metrics <- function(model_name) {
  tibble(model = model_name, accuracy = 0.000, macro_f1 = 0.000)
}

# ---------------------------------------------------------------------
# 3) Build long table: dataset × model × (accuracy, macro_f1)
# ---------------------------------------------------------------------

rows <- purrr::pmap_dfr(
  freq_map,
  function(dataset, tag) {
    parts <- list(
      read_xgb_metrics(tag, LABEL_ID),
      read_baseline_metrics(tag, LABEL_ID),
      
      # Placeholders (wire real files later)
      placeholder_metrics("Benchmark (1NN-DTW)"),
      placeholder_metrics("Rotation Forest"),
      placeholder_metrics("InceptionTime"),
      placeholder_metrics("ROCKET")
    )
    
    bind_rows(parts) %>%
      mutate(dataset = dataset) %>%
      select(dataset, model, accuracy, macro_f1)
  }
)

# Deduplicate if placeholders + real produce the same (dataset, model)
# Rule: prefer non-zero, otherwise keep the first
rows <- rows %>%
  group_by(dataset, model) %>%
  summarise(
    accuracy = max(accuracy, na.rm = TRUE),
    macro_f1 = max(macro_f1, na.rm = TRUE),
    .groups  = "drop"
  )

# Enforce factor order
rows <- rows %>%
  mutate(
    dataset = factor(dataset, levels = freq_levels),
    model   = factor(model,   levels = model_levels)
  ) %>%
  arrange(model, dataset)

# ---------------------------------------------------------------------
# 4) Weighted averages (W. Avg.) for each model
# ---------------------------------------------------------------------

rows_w <- rows %>%
  left_join(freq_sizes, by = c("dataset")) %>%
  group_by(model) %>%
  summarise(
    dataset  = "W. Avg.",
    accuracy = sum(accuracy * n, na.rm = TRUE) / sum(n),
    macro_f1 = sum(macro_f1 * n, na.rm = TRUE) / sum(n),
    .groups  = "drop"
  ) %>%
  mutate(
    dataset = factor(dataset, levels = c(freq_levels, "W. Avg.")),
    model   = factor(model, levels = model_levels)
  )

rows_all <- bind_rows(
  rows %>% mutate(dataset = factor(dataset, levels = c(freq_levels, "W. Avg."))),
  rows_w
) %>%
  arrange(model, dataset)

# ---------------------------------------------------------------------
# 5) Wide table (two blocks: Accuracy then Macro F1)
# ---------------------------------------------------------------------

wide_acc <- rows_all %>%
  select(model, dataset, accuracy) %>%
  tidyr::pivot_wider(names_from = dataset, values_from = accuracy)

wide_f1 <- rows_all %>%
  select(model, dataset, macro_f1) %>%
  tidyr::pivot_wider(names_from = dataset, values_from = macro_f1)

# Join side-by-side with clear column names
tbl_wide <- wide_acc %>%
  left_join(wide_f1, by = "model", suffix = c("_acc", "_f1"))

# Save numeric (machine-friendly)
readr::write_csv(tbl_wide, out_csv)

# ---------------------------------------------------------------------
# 6) Display-friendly version (3 decimals)
# ---------------------------------------------------------------------

fmt3 <- function(x) ifelse(is.na(x), "", sprintf("%.3f", x))

tbl_display <- tbl_wide
for (j in seq_along(tbl_display)) {
  if (is.numeric(tbl_display[[j]])) {
    tbl_display[[j]] <- fmt3(tbl_display[[j]])
  }
}

readr::write_csv(tbl_display, out_csv_display)

message("Saved numeric table to:  ", out_csv)
message("Saved display table to:  ", out_csv_display)

print(tbl_display)
