# =====================================================================
# 01_m4_pivot_period_type.R
# Pivot: rows = period, columns = type, values = number of series
# Output: results/paper/tables/m4_pivot_period_type.{csv,rds}
# =====================================================================

library(M4comp2018)
library(tidyverse)

# ---------------------------------------------------------------------
# 1) Output folders
# ---------------------------------------------------------------------

output_dir <- file.path("results", "paper", "tables")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

output_csv <- file.path(output_dir, "m4_pivot_period_type.csv")
output_rds <- file.path(output_dir, "m4_pivot_period_type.rds")

# ---------------------------------------------------------------------
# 2) Load M4 data
# ---------------------------------------------------------------------

data("M4")

# ---------------------------------------------------------------------
# 3) Extract M4 metadata
# ---------------------------------------------------------------------

m4_meta <- purrr::map_dfr(
  M4,
  function(s) {
    tibble(
      id     = s$st,
      type   = s$type,
      period = s$period,
      length = s$n
    )
  }
)

stopifnot(nrow(m4_meta) == length(M4))

# ---------------------------------------------------------------------
# 4) Pivot table: period (rows) x type (columns)
# ---------------------------------------------------------------------

pivot_tbl <- m4_meta %>%
  count(period, type, name = "n_series") %>%
  tidyr::pivot_wider(
    names_from  = type,
    values_from = n_series,
    values_fill = 0
  ) %>%
  arrange(period)

# ---------------------------------------------------------------------
# 5) Save outputs
# ---------------------------------------------------------------------

readr::write_csv(pivot_tbl, output_csv)
saveRDS(pivot_tbl, output_rds)

# ---------------------------------------------------------------------
# 6) Preview
# ---------------------------------------------------------------------

print(pivot_tbl)

message("Pivot table saved to: ", output_csv)
message("RDS object saved to:  ", output_rds)
