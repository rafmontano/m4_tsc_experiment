# =====================================================================
# 04_cd_diagram_models.R
# Purpose:
#   1) Read model × dataset accuracy results from CSV.
#   2) Save a wide Frequency × Model table to CSV.
#   3) Produce a Critical Difference diagram (Demsar-style).
#
# Expected input:
#   results/paper/tables/model_accuracy_by_frequency.csv
# with columns: dataset, model, accuracy (higher = better)
#
# Outputs:
#   results/paper/tables/model_accuracy_by_frequency_table_cd.csv
#   results/paper/figures/cd_diagram_models.pdf
# =====================================================================

library(tidyverse)

# scmamp is used for CD plots
if (!requireNamespace("scmamp", quietly = TRUE)) {
  stop(
    "Package 'scmamp' is required. Install it with:\n",
    "install.packages('scmamp')"
  )
}
library(scmamp)

# -------------------------------------------------------------------
# 1) Paths (paper structure)
# -------------------------------------------------------------------

input_csv <- file.path("results", "paper", "tables", "model_accuracy_by_frequency.csv")

output_table_csv <- file.path(
  "results", "paper", "tables", "model_accuracy_by_frequency_table_cd.csv"
)

fig_dir  <- file.path("results", "paper", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
fig_path <- file.path(fig_dir, "cd_diagram_models.pdf")

# -------------------------------------------------------------------
# 2) Read results
# -------------------------------------------------------------------

df <- readr::read_csv(input_csv, show_col_types = FALSE)

# Expect: dataset, model, accuracy
req_cols <- c("dataset", "model", "accuracy")
missing_cols <- setdiff(req_cols, names(df))
if (length(missing_cols) > 0) {
  stop("Input is missing required columns: ", paste(missing_cols, collapse = ", "))
}

df <- df %>%
  mutate(
    dataset = factor(
      dataset,
      levels = c("Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly")
    ),
    model = as.character(model)
  )

# -------------------------------------------------------------------
# 3) Create and save Frequency × Model table
# -------------------------------------------------------------------
# NOTE: CD diagrams require a complete matrix (no NA).
# We therefore restrict to models that are present for ALL datasets.

models_complete <- df %>%
  group_by(model) %>%
  summarise(n_datasets = n_distinct(dataset), .groups = "drop") %>%
  filter(n_datasets == n_distinct(df$dataset)) %>%
  pull(model)

if (length(models_complete) < 2) {
  stop(
    "Not enough models with complete results across all datasets.\n",
    "Models with complete coverage: ", paste(models_complete, collapse = ", ")
  )
}

df_cd <- df %>%
  filter(model %in% models_complete) %>%
  mutate(
    model = factor(
      model,
      levels = c("FFORMA", "SMYL", "XGBoost", "InceptionTime", "ROCKET", "RotF")
    )
  )

freq_model_table <- df_cd %>%
  select(dataset, model, accuracy) %>%
  tidyr::pivot_wider(
    names_from  = model,
    values_from = accuracy
  ) %>%
  arrange(dataset)

# Fail fast if any NA remains
if (any(is.na(freq_model_table))) {
  stop(
    "Wide table contains NA values. CD diagram requires complete data.\n",
    "Check that each (dataset, model) pair has an accuracy value."
  )
}

readr::write_csv(freq_model_table, output_table_csv)

message("Frequency × Model table saved to: ", output_table_csv)
print(freq_model_table)

# -------------------------------------------------------------------
# 4) Convert to matrix for CD diagram
# -------------------------------------------------------------------

results_mat <- freq_model_table
dataset_names <- results_mat$dataset
results_mat <- as.matrix(results_mat[, -1, drop = FALSE])
rownames(results_mat) <- as.character(dataset_names)

message("Results matrix (rows = datasets, columns = models):")
print(results_mat)

# -------------------------------------------------------------------
# 5) Critical Difference diagram
# -------------------------------------------------------------------
# For accuracy (higher = better), keep reverse = TRUE so best rank is on the right.

pdf(fig_path, width = 8, height = 4)
plotCD(
  results.matrix = results_mat,
  alpha          = 0.05,
  cex            = 0.75,
  
)
dev.off()

message("Critical Difference diagram saved to: ", fig_path)
