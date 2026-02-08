# =====================================================================
# 12_baseline_fforma_smyl.R
# Baseline classification using M4 competition point forecasts (Smyl, FFORMA)
# Evaluated on REAL last-window data (aligned with 11_eval_real_xgb.R)
#
# Alignment principles:
#   - Uses c_train from threshold_file_train (05_compute_c.R output)
#   - Uses scaling + z + labeling functions consistent with training (utils.R)
#   - Uses compute_z_generic(x_std, xx_std, LABEL_ID) to match label definitions l1..l4
#
# pt_ff[1, ] = Smyl (winner), pt_ff[2, ] = FFORMA (runner-up)
# =====================================================================

library(dplyr)
library(tibble)
library(caret)

source("src/r/utils.R")  # scale_pair_minmax_std(), compute_z_generic(), label_from_z_int()

# ---------------------------------------------------------------------
# 0) Required globals from 00_main.R / 0_common.R
# ---------------------------------------------------------------------

req_objs <- c("LABEL_ID", "WINDOW_SIZE", "TAG", "subset_clean_file", "threshold_file_train")
missing_objs <- req_objs[!vapply(req_objs, exists, logical(1))]
if (length(missing_objs) > 0L) {
  stop("Missing required globals. Run 00_main.R first. Missing: ", paste(missing_objs, collapse = ", "))
}

req_fns <- c("compute_z_generic", "scale_pair_minmax_std", "label_from_z_int")
missing_fns <- req_fns[!vapply(req_fns, exists, logical(1), mode = "function")]
if (length(missing_fns) > 0L) {
  stop("Missing required functions after sourcing utils.R: ", paste(missing_fns, collapse = ", "))
}

label_col   <- paste0("l", LABEL_ID)
window_size <- as.integer(WINDOW_SIZE)

cat("Running baseline Smyl/FFORMA for label:", label_col, " (TAG =", TAG, ")\n")

# ---------------------------------------------------------------------
# 1) Load cleaned M4 subset + threshold c_train
# ---------------------------------------------------------------------

if (!file.exists(subset_clean_file)) stop("Missing file: ", subset_clean_file)
if (!file.exists(threshold_file_train)) stop("Missing file: ", threshold_file_train)

M4_clean <- readRDS(subset_clean_file)
c_value  <- readRDS(threshold_file_train)

cat("Loaded", length(M4_clean), "M4 series from", subset_clean_file, "\n")

# ---------------------------------------------------------------------
# 2) Build evaluation dataset: TRUE labels + Smyl/FFORMA implied labels
# ---------------------------------------------------------------------

n_series <- length(M4_clean)

rows <- lapply(seq_len(n_series), function(i) {
  s <- M4_clean[[i]]
  
  x_full  <- as.numeric(s$x)
  xx_true <- as.numeric(s$xx)
  pt_ff   <- s$pt_ff
  
  if (length(x_full) < window_size) return(NULL)
  if (length(xx_true) == 0L) return(NULL)
  if (!all(is.finite(xx_true))) return(NULL)
  if (is.null(pt_ff)) return(NULL)
  
  # Ensure pt_ff is a matrix (#methods x horizon)
  if (is.vector(pt_ff)) pt_ff <- matrix(pt_ff, nrow = 1)
  if (!is.matrix(pt_ff)) return(NULL)
  if (nrow(pt_ff) < 2) return(NULL)  # need Smyl + FFORMA
  
  H <- length(xx_true)
  
  # Strict horizon match (do not truncate/reshape competitor forecasts)
  if (ncol(pt_ff) != H) return(NULL)
  if (any(!is.finite(pt_ff))) return(NULL)
  
  # Last window
  x_win <- tail(x_full, window_size)
  
  # ---------- TRUE horizon ----------
  scaled_true <- scale_pair_minmax_std(x_win, xx_true)
  z_true      <- compute_z_generic(scaled_true$x_std, scaled_true$xx_std, LABEL_ID)
  true_label  <- label_from_z_int(z_true, c_value)
  
  # ---------- SMYL ----------
  xx_smyl     <- as.numeric(pt_ff[1, ])
  scaled_smyl <- scale_pair_minmax_std(x_win, xx_smyl)
  z_smyl      <- compute_z_generic(scaled_smyl$x_std, scaled_smyl$xx_std, LABEL_ID)
  smyl_label  <- label_from_z_int(z_smyl, c_value)
  
  # ---------- FFORMA ----------
  xx_fforma     <- as.numeric(pt_ff[2, ])
  scaled_fforma <- scale_pair_minmax_std(x_win, xx_fforma)
  z_fforma      <- compute_z_generic(scaled_fforma$x_std, scaled_fforma$xx_std, LABEL_ID)
  fforma_label  <- label_from_z_int(z_fforma, c_value)
  
  tibble(
    st           = s$st,
    true_label   = as.integer(true_label),
    smyl_label   = as.integer(smyl_label),
    fforma_label = as.integer(fforma_label)
  )
})

baseline_df <- bind_rows(rows)

cat("Baseline evaluation dataset contains", nrow(baseline_df), "rows\n")

if (nrow(baseline_df) == 0L) {
  stop("Baseline dataset is empty. Check pt_ff availability and horizon integrity.")
}

# ---------------------------------------------------------------------
# 3) caret confusion matrices (Pred first, Truth second)
# ---------------------------------------------------------------------

levels_all <- c(0, 1, 2)

y_true   <- factor(baseline_df$true_label,   levels = levels_all)
y_smyl   <- factor(baseline_df$smyl_label,   levels = levels_all)
y_fforma <- factor(baseline_df$fforma_label, levels = levels_all)

cm_smyl   <- confusionMatrix(y_smyl,   y_true)
cm_fforma <- confusionMatrix(y_fforma, y_true)

cat("\n=== Smyl Confusion Matrix ===\n")
print(cm_smyl$table)

cat("\n=== FFORMA Confusion Matrix ===\n")
print(cm_fforma$table)

# ---------------------------------------------------------------------
# 4) Extract metrics from caret (per-class + overall)
# ---------------------------------------------------------------------

extract_metrics <- function(cm, method_name) {
  
  by_class <- cm$byClass
  if (is.null(dim(by_class))) {
    by_class <- matrix(by_class, nrow = 1)
    rownames(by_class) <- as.character(levels_all)
  }
  
  precision <- by_class[, "Precision"]
  recall    <- by_class[, "Recall"]
  
  f1 <- if ("F1" %in% colnames(by_class)) {
    by_class[, "F1"]
  } else {
    2 * precision * recall / (precision + recall)
  }
  
  data.frame(
    class            = levels_all,
    precision        = as.numeric(precision),
    recall           = as.numeric(recall),
    f1               = as.numeric(f1),
    overall_accuracy = as.numeric(cm$overall["Accuracy"]),
    kappa            = as.numeric(cm$overall["Kappa"]),
    macro_f1         = mean(as.numeric(f1), na.rm = TRUE),
    method           = method_name
  )
}

metrics_all <- bind_rows(
  extract_metrics(cm_smyl,   "Smyl"),
  extract_metrics(cm_fforma, "FFORMA")
)

cat("\n=== Baseline Metrics (Caret) ===\n")
print(metrics_all)

# ---------------------------------------------------------------------
# 5) Save outputs (TAG-aware)
# ---------------------------------------------------------------------

baseline_dir <- file.path("results", "baseline")
if (!dir.exists(baseline_dir)) dir.create(baseline_dir, recursive = TRUE)

baseline_path_rds <- file.path(baseline_dir, sprintf("baseline_l%d_%s_smyl_fforma.rds", LABEL_ID, TAG))
baseline_path_csv <- file.path(baseline_dir, sprintf("baseline_l%d_%s_smyl_fforma.csv", LABEL_ID, TAG))

saveRDS(baseline_df, baseline_path_rds)
write.csv(baseline_df, baseline_path_csv, row.names = FALSE)

saveRDS(cm_smyl$table,   file.path(baseline_dir, sprintf("baseline_conf_smyl_l%d_%s.rds", LABEL_ID, TAG)))
saveRDS(cm_fforma$table, file.path(baseline_dir, sprintf("baseline_conf_fforma_l%d_%s.rds", LABEL_ID, TAG)))

metrics_path_rds <- file.path(baseline_dir, sprintf("baseline_metrics_l%d_%s.rds", LABEL_ID, TAG))
metrics_path_csv <- file.path(baseline_dir, sprintf("baseline_metrics_l%d_%s.csv", LABEL_ID, TAG))

saveRDS(metrics_all, metrics_path_rds)
write.csv(metrics_all, metrics_path_csv, row.names = FALSE)

cat("\nSaved baseline results for label", label_col, "(TAG =", TAG, "):\n")
cat("  Baseline RDS   :", baseline_path_rds, "\n")
cat("  Baseline CSV   :", baseline_path_csv, "\n")
cat("  Metrics RDS    :", metrics_path_rds, "\n")
cat("  Metrics CSV    :", metrics_path_csv, "\n")