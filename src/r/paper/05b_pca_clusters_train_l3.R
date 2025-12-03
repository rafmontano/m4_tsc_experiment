# =====================================================================
# 05b_pca_clusters_train_l3.R
# Purpose:
#   PCA on train_l{LABEL_ID}_{freq_tag}.csv with k-means clustering
#   in PC1–PC2, plotted with density contours.
#
# Expected globals (from 00_main.R):
#   - TARGET_PERIOD  (e.g. "Quarterly")
#   - LABEL_ID       (e.g. 3)
#
# Input:
#   output/data_export/train_l{LABEL_ID}_{freq_tag}.csv
#
# Output:
#   output/figures/pca_pc1_pc2_clusters_train_l{LABEL_ID}_{freq_tag}.pdf
# =====================================================================

library(tidyverse)
library(factoextra)

source("src/r/utils.R")

# ---------------------------------------------------------------------
# 0. Check required globals and derive freq_tag
# ---------------------------------------------------------------------

if (!exists("TARGET_PERIOD")) {
  stop("TARGET_PERIOD not defined. Please run 00_main.R first.")
}
if (!exists("LABEL_ID")) {
  stop("LABEL_ID not defined. Please set it in 00_main.R (e.g. LABEL_ID <- 3).")
}

freq_tag <- freq_tag(TARGET_PERIOD)

# ---------------------------------------------------------------------
# 1. Load training dataset (label LABEL_ID, specific frequency)
# ---------------------------------------------------------------------

input_csv <- file.path(
  "data", "export",
  sprintf("train_l%d_%s.csv", LABEL_ID, freq_tag)
)

cat("Reading training data from:", input_csv, "\n")

df <- readr::read_csv(input_csv, show_col_types = FALSE)

# Keep only numeric columns (label + features). PCA can handle label column
# but it just shifts mean; usually we remove label, but we stay simple here:
df_numeric <- df %>% select(where(is.numeric))

if (ncol(df_numeric) < 2) {
  stop("Not enough numeric columns for PCA in: ", input_csv)
}

# ---------------------------------------------------------------------
# 2. PCA
# ---------------------------------------------------------------------

pca <- prcomp(df_numeric, center = TRUE, scale. = TRUE)

scores <- as_tibble(pca$x[, 1:2])
colnames(scores) <- c("PC1", "PC2")

# ---------------------------------------------------------------------
# 3. k-means clustering on PC1–PC2
# ---------------------------------------------------------------------

set.seed(123)
k_clusters <- 3

km <- kmeans(scores, centers = k_clusters, nstart = 20)

scores <- scores %>%
  mutate(cluster = factor(km$cluster))

# ---------------------------------------------------------------------
# 4. Optional downsampling for clearer plotting
# ---------------------------------------------------------------------

max_points <- 5000
set.seed(123)

scores_plot <- if (nrow(scores) > max_points) {
  dplyr::sample_n(scores, max_points)
} else {
  scores
}

# ---------------------------------------------------------------------
# 5. Output directory
# ---------------------------------------------------------------------

fig_dir <- file.path("output", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

fig_path <- file.path(
  fig_dir,
  sprintf("pca_pc1_pc2_clusters_train_l%d_%s.pdf", LABEL_ID, freq_tag)
)

# ---------------------------------------------------------------------
# 6. Plot PCA clusters + density contours
# ---------------------------------------------------------------------

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

message("Clustered PCA plot saved to: ", fig_path)
