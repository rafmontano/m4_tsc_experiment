# =====================================================================
# 04a_build_model_accuracy_by_frequency.R
#
# Purpose:
#   Consolidate experiment outputs into the input required by:
#     04_cd_diagram_models.R
#
# Metric (TSC Bake Off convention):
#   Use overall accuracy directly.
#
# Output:
#   results/paper/tables/model_accuracy_by_frequency.csv
#     columns: dataset, model, accuracy
# =====================================================================

library(tidyverse)

# ---------------------------------------------------------------------
# 0) CONFIG
# ---------------------------------------------------------------------

LABEL_ID <- 3

freq_map <- tibble::tribble(
  ~dataset,     ~tag,
  "Hourly",     "h",
  "Daily",      "d",
  "Weekly",     "w",
  "Monthly",    "m",
  "Quarterly",  "q",
  "Yearly",     "y"
)

ALLOW_MISSING_MODELS <- TRUE

# ---------------------------------------------------------------------
# 1) Output path
# ---------------------------------------------------------------------

out_dir <- file.path("results", "paper", "tables")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_csv <- file.path(out_dir, "model_accuracy_by_frequency.csv")
out_rds <- file.path(out_dir, "model_accuracy_by_frequency.rds")

# ---------------------------------------------------------------------
# 2) Readers
# ---------------------------------------------------------------------

read_xgb_accuracy <- function(tag, label_id) {
  path <- file.path("results", "xgb",
                    sprintf("real_metrics_l%d_%s.csv", label_id, tag))
  if (!file.exists(path)) return(NULL)
  
  df <- readr::read_csv(path, show_col_types = FALSE)
  
  tibble(
    model    = "XGBoost",
    accuracy = first(df$overall_accuracy)
  )
}

read_baseline_accuracy <- function(tag, label_id) {
  path <- file.path("results", "baseline",
                    sprintf("baseline_metrics_l%d_%s.csv", label_id, tag))
  if (!file.exists(path)) return(NULL)
  
  df <- readr::read_csv(path, show_col_types = FALSE)
  
  df %>%
    group_by(method) %>%
    summarise(accuracy = first(overall_accuracy), .groups = "drop") %>%
    mutate(
      model = case_when(
        tolower(method) == "smyl"   ~ "SMYL",
        tolower(method) == "fforma" ~ "FFORMA",
        TRUE ~ method
      )
    ) %>%
    select(model, accuracy)
}

# Placeholder for ROCKET / InceptionTime
read_deep_accuracy <- function(dataset, model_name) {
  NULL
}

# ---------------------------------------------------------------------
# 3) Consolidation
# ---------------------------------------------------------------------

rows <- purrr::pmap_dfr(
  freq_map,
  function(dataset, tag) {
    
    parts <- list(
      read_baseline_accuracy(tag, LABEL_ID),
      read_xgb_accuracy(tag, LABEL_ID),
      read_deep_accuracy(dataset, "ROCKET"),
      read_deep_accuracy(dataset, "InceptionTime")
    )
    
    part_df <- bind_rows(parts)
    if (nrow(part_df) == 0) return(tibble())
    
    part_df %>%
      mutate(dataset = dataset) %>%
      select(dataset, model, accuracy)
  }
)

# ---------------------------------------------------------------------
# 4) Validation & ordering
# ---------------------------------------------------------------------

if (nrow(rows) == 0) {
  stop("No accuracy inputs found. Run scripts 11 and/or 12 first.")
}

rows <- rows %>%
  mutate(
    dataset = factor(dataset,
                     levels = c("Hourly","Daily","Weekly","Monthly","Quarterly","Yearly")),
    model   = factor(model,
                     levels = c("FFORMA","SMYL","XGBoost","InceptionTime","ROCKET"))
  ) %>%
  arrange(dataset, model)

# ---------------------------------------------------------------------
# 5) Save
# ---------------------------------------------------------------------

rows_out <- rows %>%
  mutate(dataset = as.character(dataset),
         model   = as.character(model))

readr::write_csv(rows_out, out_csv)
saveRDS(rows_out, out_rds)

message("Built: ", out_csv)
print(rows_out)
