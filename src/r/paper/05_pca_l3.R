# =====================================================================
# 05_pca_l3.R
# Purpose:
#   PCA on train_l{LABEL_ID}_{freq_tag}.csv used for XGBoost,
#   with plots restricted to top contributing variables.
#
# Expected globals (from 00_main.R):
#   - TARGET_PERIOD  (e.g. "Quarterly")
#   - LABEL_ID       (e.g. 3)
#
# Input:
#   data/export/train_l{LABEL_ID}_{freq_tag}.csv
#
# Output:
#   output/figures/pca_*.pdf (tagged by LABEL_ID and freq_tag)
# =====================================================================

library(tidyverse)
library(factoextra)

source("src/r/utils.R")

# --------------------------------------------------------------------
# 0. Check globals and derive freq_tag
# --------------------------------------------------------------------

if (!exists("TARGET_PERIOD")) {
  stop("TARGET_PERIOD not defined. Please run 00_main.R first.")
}
if (!exists("LABEL_ID")) {
  stop("LABEL_ID not defined. Please set it in 00_main.R (e.g. LABEL_ID <- 3).")
}

freq_tag <- freq_tag(TARGET_PERIOD)

# --------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------

input_csv <- file.path(
  "data", "export",
  sprintf("train_l%d_%s.csv", LABEL_ID, freq_tag)
)

cat("Reading training data for PCA from:", input_csv, "\n")

df <- readr::read_csv(input_csv, show_col_types = FALSE)

# --------------------------------------------------------------------
# 2. Prepare PCA matrix
# --------------------------------------------------------------------

# Remove label column if present, keep only numeric predictors
if ("label" %in% names(df)) {
  df_numeric <- df %>% select(-label, where(is.numeric))
} else {
  df_numeric <- df %>% select(where(is.numeric))
}

if (ncol(df_numeric) < 2) {
  stop("Not enough numeric columns for PCA in: ", input_csv)
}

# --------------------------------------------------------------------
# 3. Run PCA
# --------------------------------------------------------------------

pca <- prcomp(df_numeric, center = TRUE, scale. = TRUE)

# Variable contributions per component
pca_var <- get_pca_var(pca)
contrib <- pca_var$contrib   # matrix with Dim.1, Dim.2, ...

# Number of top variables to show
top_n <- 10

# Overall importance on first two PCs (for biplot selection)
contrib12 <- rowSums(contrib[, c("Dim.1", "Dim.2"), drop = FALSE])
top_idx   <- order(contrib12, decreasing = TRUE)[1:top_n]
top_names <- names(contrib12)[top_idx]

# --------------------------------------------------------------------
# 4. Output directory
# --------------------------------------------------------------------

fig_dir <- file.path("output", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# Helper to build filenames with label + freq tag
fn <- function(suffix) {
  file.path(
    fig_dir,
    sprintf("pca_%s_l%d_%s.pdf", suffix, LABEL_ID, freq_tag)
  )
}

# --------------------------------------------------------------------
# 5. Scree plot: variance explained
# --------------------------------------------------------------------

pdf(fn("scree"), width = 12, height = 8)
print(
  fviz_eig(
    pca,
    addlabels = TRUE,
    barfill   = "grey70",
    barcolor  = "grey20",
    linecolor = "black"
  ) +
    theme_minimal(base_size = 11)
)
dev.off()

# --------------------------------------------------------------------
# 6. Variable contributions (top N only)
# --------------------------------------------------------------------

pdf(fn("contrib_pc1_top"), width = 12, height = 8)
print(
  fviz_contrib(
    pca,
    choice = "var",
    axes   = 1,
    top    = top_n
  ) +
    theme_minimal(base_size = 11)
)
dev.off()

pdf(fn("contrib_pc2_top"), width = 12, height = 8)
print(
  fviz_contrib(
    pca,
    choice = "var",
    axes   = 2,
    top    = top_n
  ) +
    theme_minimal(base_size = 11)
)
dev.off()

# --------------------------------------------------------------------
# 7. Biplot with only top N variables (by contribution to Dim1+Dim2)
# --------------------------------------------------------------------

pdf(fn("biplot_top"), width = 24, height = 16)
print(
  fviz_pca_biplot(
    pca,
    repel      = TRUE,
    select.var = list(name = top_names),
    col.var    = "black",
    col.ind    = "grey80",
    pointshape = 19
  ) +
    theme_minimal(base_size = 11) +
    geom_point(alpha = 0.2, size = 0.6)
)
dev.off()

message(
  "PCA analysis complete. Figures saved to '",
  fig_dir,
  "' with LABEL_ID=", LABEL_ID,
  " and freq_tag='", freq_tag, "'."
)
