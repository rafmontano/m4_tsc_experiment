# =====================================================================
# 05b_pca_clusters_train_l3.R
# Purpose:
#   PCA on train_l{LABEL_ID}_{freq_tag}.csv with k-means clustering
#   in PC1–PC2, plotted with density contours, looping across
#   six frequencies (y, q, m, w, d, h).
#
# Expected globals (from 00_main.R):
#   - LABEL_ID       (e.g. 3)
#
# Input (per frequency):
#   data/export/train_l{LABEL_ID}_{freq_tag}.csv
#
# Output (per frequency):
#   results/paper/figures/pca_pc1_pc2_clusters_train_l{LABEL_ID}_{freq_tag}.pdf
# =====================================================================

library(tidyverse)

source("src/r/utils.R")

# ---------------------------------------------------------------------
# 0. Check required globals
# ---------------------------------------------------------------------

LABEL_ID      <- 3 

# ---------------------------------------------------------------------
# 1. Frequencies to process (six M4 frequencies)
# ---------------------------------------------------------------------

target_periods <- c("Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly")

# ---------------------------------------------------------------------
# 2. Clustering + plotting parameters
# ---------------------------------------------------------------------

set.seed(123)
k_clusters <- 3
max_points <- 5000

# ---------------------------------------------------------------------
# 3. Output directory
# ---------------------------------------------------------------------

fig_dir <- file.path("results", "paper", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------
# 4. Loop across frequencies
# ---------------------------------------------------------------------

for (tp in target_periods) {
  
  freq_tag_i <- freq_tag(tp)
  
  input_csv <- file.path(
    "data", "export",
    sprintf("train_l%d_%s.csv", LABEL_ID, freq_tag_i)
  )
  
  if (!file.exists(input_csv)) {
    message("[05b] Skipping ", tp, " (missing file): ", input_csv)
    next
  }
  
  message("\n[05b] PCA clusters for ", tp, " (", freq_tag_i, ")")
  message("      Input: ", input_csv)
  
  df <- readr::read_csv(input_csv, show_col_types = FALSE)
  
  # Keep only numeric columns, and remove label if present
  df_numeric <- df %>% select(where(is.numeric))
  if ("label" %in% names(df_numeric)) {
    df_numeric <- df_numeric %>% select(-label)
  }
  
  if (ncol(df_numeric) < 2) {
    message("[05b] Skipping ", tp, " (not enough numeric columns for PCA): ", input_csv)
    next
  }
  
  # -------------------------------------------------------------------
  # PCA
  # -------------------------------------------------------------------
  
  pca <- prcomp(df_numeric, center = TRUE, scale. = TRUE)
  
  scores <- as_tibble(pca$x[, 1:2, drop = FALSE])
  colnames(scores) <- c("PC1", "PC2")
  
  # -------------------------------------------------------------------
  # k-means clustering on PC1–PC2
  # -------------------------------------------------------------------
  
  set.seed(123)
  km <- kmeans(scores, centers = k_clusters, nstart = 20)
  
  scores <- scores %>%
    mutate(cluster = factor(km$cluster))
  
  # -------------------------------------------------------------------
  # Optional downsampling for clearer plotting
  # -------------------------------------------------------------------
  
  set.seed(123)
  scores_plot <- if (nrow(scores) > max_points) {
    dplyr::sample_n(scores, max_points)
  } else {
    scores
  }
  
  # -------------------------------------------------------------------
  # Output file
  # -------------------------------------------------------------------
  
  fig_path <- file.path(
    fig_dir,
    sprintf("pca_pc1_pc2_clusters_train_l%d_%s.pdf", LABEL_ID, freq_tag_i)
  )
  
  # -------------------------------------------------------------------
  # Plot PCA clusters + density contours
  # -------------------------------------------------------------------
  
  p <- ggplot(scores_plot, aes(PC1, PC2)) +
    stat_density_2d(
      colour    = "grey60",
      linewidth = 0.3,
      bins      = 6
    ) +
    geom_point(
      aes(colour = cluster),
      size  = 0.7,
      alpha = 0.7
    ) +
    scale_colour_brewer(
      palette = "Dark2",
      name    = "Cluster"
    ) +
    labs(
      x = "PC1",
      y = "PC2"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title      = element_blank(),
      legend.position = "right"
    )
  
  ggsave(fig_path, p, width = 7.0, height = 5.0)
  
  message("[05b] Clustered PCA plot saved to: ", fig_path)
}

message("\n[05b] All PCA cluster runs completed.")