# =====================================================================
# run_all.R
# Minimal sequential runner for all paper scripts.
# Runs each script one after the other, end-to-end.
#
# Assumes:
#   - You run this from the project root.
#   - LABEL_ID, FREQ_TAG etc. already match the experiment.
# =====================================================================

source("src/r/paper/01_m4_pivot_period_type.R")
source("src/r/paper/02_m4_period_bar.R")
source("src/r/paper/03_m4_length_box_by_period_facet.R")
source("src/r/paper/04_cd_diagram_models.R")
source("src/r/paper/05_pca_l3.R")
source("src/r/paper/05b_pca_clusters_train_l3.R")
source("src/r/paper/06a_compute_class_proportion.R")
source("src/r/paper/06_class_imbalance_bars.R")

cat("\nAll paper scripts executed successfully.\n")