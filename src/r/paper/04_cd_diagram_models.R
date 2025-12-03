# =====================================================================
# 04_cd_diagram_models.R
# Purpose:
#   1) Read model × dataset error results from CSV.
#   2) Save a wide Frequency × Model table to CSV.
#   3) Produce a Critical Difference diagram (Demsar-style).
#
# Expected input:
#   output/tables/model_error_by_frequency.csv
# with columns: dataset, model, error (lower = better)
#
# Outputs:
#   output/tables/model_error_by_frequency_table.csv
#   output/figures/cd_diagram_models.pdf
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
# 1. Paths (aligned with new structure)
# -------------------------------------------------------------------

input_csv <- file.path("output", "tables", "model_error_by_frequency.csv")

output_table_csv <- file.path(
  "output", "tables", "model_error_by_frequency_table.csv"
)

fig_dir  <- file.path("output", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
fig_path <- file.path(fig_dir, "cd_diagram_models.pdf")

# -------------------------------------------------------------------
# 2. Read results
# -------------------------------------------------------------------

df <- readr::read_csv(input_csv, show_col_types = FALSE)
# Expected columns: dataset, model, error (lower is better)

df <- df %>%
  mutate(
    dataset = factor(
      dataset,
      levels = c("Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly")
    ),
    model = factor(
      model,
      levels = c("FFORMA", "SMYL", "XGBoost", "InceptionTime", "ROCKET")
    )
  )

# -------------------------------------------------------------------
# 3. Create and save Frequency × Model table
# -------------------------------------------------------------------

freq_model_table <- df %>%
  select(dataset, model, error) %>%
  tidyr::pivot_wider(
    names_from  = model,
    values_from = error
  ) %>%
  arrange(dataset)

readr::write_csv(freq_model_table, output_table_csv)

message("Frequency × Model table saved to: ", output_table_csv)
print(freq_model_table)

# -------------------------------------------------------------------
# 4. Convert to matrix for CD diagram
# -------------------------------------------------------------------

results_mat    <- freq_model_table
dataset_names  <- results_mat$dataset
results_mat    <- as.matrix(results_mat[, -1, drop = FALSE])
rownames(results_mat) <- as.character(dataset_names)

message("Results matrix (rows = datasets, columns = models):")
print(results_mat)

# -------------------------------------------------------------------
# 5. Critical Difference diagram
# -------------------------------------------------------------------

pdf(fig_path, width = 8, height = 4)
plotCD(
  results.matrix = results_mat,
  alpha          = 0.05,
  cex            = 1.0,
  reverse        = TRUE   # Rank 1 on the right
)
dev.off()

message("Critical Difference diagram saved to: ", fig_path)
