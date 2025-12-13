# =====================================================================
# 05_pca_l3.R
# Purpose:
#   PCA on train_l{LABEL_ID}_{freq_tag}.csv used for XGBoost,
#   looping across six frequencies (y, q, m, w, d, h),
#   with plots restricted to top contributing variables.
#
# Expected globals (from 00_main.R):
#   - LABEL_ID       (e.g. 3)
#
# Inputs (per frequency):
#   data/export/train_l{LABEL_ID}_{freq_tag}.csv
#
# Outputs (per frequency):
#   results/paper/figures/pca_*.pdf (tagged by LABEL_ID and freq_tag)
# =====================================================================

library(tidyverse)
library(factoextra)

source("src/r/utils.R")

# --------------------------------------------------------------------
# 0. Check globals
# --------------------------------------------------------------------

LABEL_ID      <- 3 


# --------------------------------------------------------------------
# 1. Frequencies to process (six M4 frequencies)
# --------------------------------------------------------------------

target_periods <- c("Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly")

# Number of top variables to show
top_n <- 10

# --------------------------------------------------------------------
# 2. Output directory
# --------------------------------------------------------------------

fig_dir <- file.path("results", "paper", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# --------------------------------------------------------------------
# 3. Helper: build filenames with label + freq tag
# --------------------------------------------------------------------

fn <- function(suffix, freq_tag) {
  file.path(fig_dir, sprintf("pca_%s_l%d_%s.pdf", suffix, LABEL_ID, freq_tag))
}

# --------------------------------------------------------------------
# 4. Loop across frequencies
# --------------------------------------------------------------------

for (tp in target_periods) {
  
  freq_tag_i <- freq_tag(tp)
  
  input_csv <- file.path(
    "data", "export",
    sprintf("train_l%d_%s.csv", LABEL_ID, freq_tag_i)
  )
  
  if (!file.exists(input_csv)) {
    message("[05] Skipping ", tp, " (missing file): ", input_csv)
    next
  }
  
  message("\n[05] PCA for ", tp, " (", freq_tag_i, ")")
  message("     Input: ", input_csv)
  
  df <- readr::read_csv(input_csv, show_col_types = FALSE)
  
  # Remove label column if present, keep only numeric predictors
  if ("label" %in% names(df)) {
    df_numeric <- df %>% select(-label, where(is.numeric))
  } else {
    df_numeric <- df %>% select(where(is.numeric))
  }
  
  if (ncol(df_numeric) < 2) {
    message("[05] Skipping ", tp, " (not enough numeric columns for PCA): ", input_csv)
    next
  }
  
  # Run PCA
  pca <- prcomp(df_numeric, center = TRUE, scale. = TRUE)
  
  # Variable contributions per component
  pca_var <- get_pca_var(pca)
  contrib <- pca_var$contrib
  
  if (!all(c("Dim.1", "Dim.2") %in% colnames(contrib))) {
    message("[05] Skipping ", tp, " (PCA contrib missing Dim.1/Dim.2): ", input_csv)
    next
  }
  
  # Top variables by combined contribution to Dim1 + Dim2
  contrib12 <- rowSums(contrib[, c("Dim.1", "Dim.2"), drop = FALSE])
  top_idx <- order(contrib12, decreasing = TRUE)[1:min(top_n, length(contrib12))]
  top_names <- names(contrib12)[top_idx]
  
  # Scree plot
  pdf(fn("scree", freq_tag_i), width = 12, height = 8)
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
  
  # Contribution plots (top N)
  pdf(fn("contrib_pc1_top", freq_tag_i), width = 12, height = 8)
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
  
  pdf(fn("contrib_pc2_top", freq_tag_i), width = 12, height = 8)
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
  
  # Biplot (top N vars)
 # pdf(fn("biplot_top", freq_tag_i), width = 24, height = 16)
  
 # p_biplot <- fviz_pca_biplot(
#    pca,
#    repel      = requireNamespace("ggrepel", quietly = TRUE),
#    select.var = list(name = top_names),
#    col.var    = "black",
#    col.ind    = "grey80",
#    pointshape = 19,
#    alpha.ind  = 0.2,
#    pointsize  = 0.6
#  ) +
#    theme_minimal(base_size = 11)
  
#  print(p_biplot)
#  dev.off()
  
  message(
    "[05] Completed ", tp,
    ". Figures saved to: ", fig_dir,
    " (LABEL_ID=", LABEL_ID, ", freq_tag=", freq_tag_i, ")"
  )
}

message("\n[05] All PCA runs completed.")