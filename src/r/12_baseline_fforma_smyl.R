# =====================================================================
# 12_baseline_fforma_smyl.R
# Baseline classification using M4 competition forecasts (Smyl, FFORMA)
# Evaluated on REAL last-window data (aligned with 11_eval_real_xgb.R)
#
# Assumes in 00_main.R:
#   - LABEL_ID          (1..4)
#   - WINDOW_SIZE
#   - TAG               (e.g. "y", "q", "m", "w", "d", "h")
#   - subset_clean_file (e.g. "data/M4_subset_clean_q.rds")
#   - threshold_file    (e.g. "data/label_threshold_c_q.rds")
#
# pt_ff[1, ] = Smyl (winner), pt_ff[2, ] = FFORMA (runner-up)
# =====================================================================

library(dplyr)
library(tibble)
library(caret)

source("src/r/utils.R")     # scale_pair_minmax_std(), label_from_z_int(), etc.
source("src/r/features.R")  # not used here, but keeps environment consistent

# ---------------------------------------------------------------------
# 0) LABEL_ID, WINDOW_SIZE, TAG from 00_main.R
# ---------------------------------------------------------------------

if (!exists("LABEL_ID"))
  stop("LABEL_ID not defined. Set in 00_main.R first.")

if (!exists("WINDOW_SIZE"))
  stop("WINDOW_SIZE not defined. Set in 00_main.R first.")

if (!exists("TAG"))
  stop("TAG not defined. Ensure freq_tag(TARGET_PERIOD) was computed in 00_main.R.")

if (!exists("subset_clean_file"))
  stop("subset_clean_file not defined in 00_main.R.")

if (!exists("threshold_file"))
  stop("threshold_file not defined in 00_main.R.")

label_col   <- paste0("l", LABEL_ID)
window_size <- WINDOW_SIZE

cat("Running baseline Smyl/FFORMA for label:", label_col, " (TAG =", TAG, ")\n")

# ---------------------------------------------------------------------
# 1) Load cleaned M4 subset + threshold c
# ---------------------------------------------------------------------

if (!file.exists(subset_clean_file)) stop("Missing file:", subset_clean_file)
if (!file.exists(threshold_file))    stop("Missing file:", threshold_file)

M4_clean <- readRDS(subset_clean_file)
c_value  <- readRDS(threshold_file)

cat("Loaded", length(M4_clean), "M4 series from", subset_clean_file, "\n")

# ---------------------------------------------------------------------
# 2) Label definition helper (same as 06_labels.R & 11_eval_real_xgb.R)
# ---------------------------------------------------------------------


# label_from_z_int(z, c) is taken from utils.R

# ---------------------------------------------------------------------
# 3) Build evaluation dataset: REAL labels + Smyl/FFORMA labels
# ---------------------------------------------------------------------

n_series <- length(M4_clean)

rows <- lapply(seq_len(n_series), function(i) {
  s <- M4_clean[[i]]
  
  x_full  <- as.numeric(s$x)
  xx_true <- as.numeric(s$xx)
  pt_ff   <- s$pt_ff
  
  if (length(x_full) < window_size) return(NULL)
  if (is.null(pt_ff)) return(NULL)
  
  # Ensure (#methods x horizon)
  if (is.vector(pt_ff)) pt_ff <- matrix(pt_ff, nrow = 1)
  if (nrow(pt_ff) < 2) return(NULL)  # need Smyl + FFORMA
  
  # Last window
  x_win <- tail(x_full, window_size)
  
  # ---------- TRUE horizon: joint scaling ----------
  scaled_true   <- scale_pair_minmax_std(x_win, xx_true)
  x_true_std    <- scaled_true$x_std
  xx_true_std   <- scaled_true$xx_std
  z_true        <- compute_z_generic(x_true_std, xx_true_std, LABEL_ID)
  true_label    <- label_from_z_int(z_true, c_value)
  
  # ---------- SMYL horizon: joint scaling ----------
  xx_smyl       <- as.numeric(pt_ff[1, ])
  scaled_smyl   <- scale_pair_minmax_std(x_win, xx_smyl)
  x_smyl_std    <- scaled_smyl$x_std
  xx_smyl_std   <- scaled_smyl$xx_std
  z_smyl        <- compute_z_generic(x_smyl_std, xx_smyl_std, LABEL_ID)
  smyl_label    <- label_from_z_int(z_smyl, c_value)
  
  # ---------- FFORMA horizon: joint scaling ----------
  xx_fforma     <- as.numeric(pt_ff[2, ])
  scaled_ff     <- scale_pair_minmax_std(x_win, xx_fforma)
  x_ff_std      <- scaled_ff$x_std
  xx_ff_std     <- scaled_ff$xx_std
  z_fforma      <- compute_z_generic(x_ff_std, xx_ff_std, LABEL_ID)
  fforma_label  <- label_from_z_int(z_fforma, c_value)
  
  tibble(
    st           = if (!is.null(s$st)) s$st else s$name,
    true_label   = as.integer(true_label),
    smyl_label   = as.integer(smyl_label),
    fforma_label = as.integer(fforma_label)
  )
})

baseline_df <- bind_rows(rows)

cat("Baseline evaluation dataset contains", nrow(baseline_df), "rows\n")

# ---------------------------------------------------------------------
# 4) Convert to factors with fixed levels (0,1,2)
# ---------------------------------------------------------------------

y_true   <- factor(baseline_df$true_label,   levels = c(0, 1, 2))
y_smyl   <- factor(baseline_df$smyl_label,   levels = c(0, 1, 2))
y_fforma <- factor(baseline_df$fforma_label, levels = c(0, 1, 2))

# ---------------------------------------------------------------------
# 5) caret confusion matrices (aligned with script 11)
# ---------------------------------------------------------------------

cm_smyl   <- confusionMatrix(y_smyl,   y_true)
cm_fforma <- confusionMatrix(y_fforma, y_true)

cat("\n=== Smyl Confusion Matrix ===\n")
print(cm_smyl$table)

cat("\n=== FFORMA Confusion Matrix ===\n")
print(cm_fforma$table)

# ---------------------------------------------------------------------
# 6) Extract metrics from caret
# ---------------------------------------------------------------------

extract_metrics <- function(cm, method_name) {
  
  by_class <- cm$byClass
  if (is.null(dim(by_class))) {
    by_class <- matrix(by_class, nrow = 1)
    rownames(by_class) <- c("0","1","2")
  }
  
  precision <- by_class[, "Precision"]
  recall    <- by_class[, "Recall"]
  
  if ("F1" %in% colnames(by_class)) {
    f1 <- by_class[, "F1"]
  } else {
    f1 <- 2 * precision * recall / (precision + recall)
  }
  
  data.frame(
    class           = c(0, 1, 2),
    precision       = as.numeric(precision),
    recall          = as.numeric(recall),
    f1              = as.numeric(f1),
    overall_accuracy = as.numeric(cm$overall["Accuracy"]),
    kappa            = as.numeric(cm$overall["Kappa"]),
    macro_f1         = mean(f1, na.rm = TRUE),
    method           = method_name
  )
}

metrics_smyl   <- extract_metrics(cm_smyl,   "Smyl")
metrics_fforma <- extract_metrics(cm_fforma, "FFORMA")

metrics_all <- bind_rows(metrics_smyl, metrics_fforma)

cat("\n=== Baseline Metrics (Caret) ===\n")
print(metrics_all)

# ---------------------------------------------------------------------
# 7) Save outputs (TAG-aware, aligned with other scripts)
# ---------------------------------------------------------------------

baseline_dir <- file.path("results", "baseline")
if (!dir.exists(baseline_dir)) dir.create(baseline_dir, recursive = TRUE)

# 7a) Baseline dataset
baseline_path_rds <- file.path(
  baseline_dir,
  sprintf("baseline_l%d_%s_smyl_fforma.rds", LABEL_ID, TAG)
)
baseline_path_csv <- file.path(
  baseline_dir,
  sprintf("baseline_l%d_%s_smyl_fforma.csv", LABEL_ID, TAG)
)

saveRDS(baseline_df, baseline_path_rds)
write.csv(baseline_df, baseline_path_csv, row.names = FALSE)

# 7b) Confusion matrices
saveRDS(
  cm_smyl$table,
  file.path(baseline_dir, sprintf("baseline_conf_smyl_l%d_%s.rds", LABEL_ID, TAG))
)
saveRDS(
  cm_fforma$table,
  file.path(baseline_dir, sprintf("baseline_conf_fforma_l%d_%s.rds", LABEL_ID, TAG))
)

# 7c) Metrics
metrics_path_rds <- file.path(
  baseline_dir,
  sprintf("baseline_metrics_l%d_%s.rds", LABEL_ID, TAG)
)
metrics_path_csv <- file.path(
  baseline_dir,
  sprintf("baseline_metrics_l%d_%s.csv", LABEL_ID, TAG)
)

saveRDS(metrics_all, metrics_path_rds)
write.csv(metrics_all, metrics_path_csv, row.names = FALSE)

cat("\nSaved baseline results for label", label_col, "(TAG =", TAG, "):\n")
cat("  Baseline RDS   :", baseline_path_rds, "\n")
cat("  Baseline CSV   :", baseline_path_csv, "\n")
cat("  Metrics RDS    :", metrics_path_rds, "\n")
cat("  Metrics CSV    :", metrics_path_csv, "\n")

