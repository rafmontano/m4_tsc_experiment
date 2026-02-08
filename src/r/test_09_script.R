cat("\n--- CHECK: artefacts exist ---\n")
print(file.exists(features_file))
print(file.exists(file.path("data","splits", sprintf("train_l%d_%s.rds", LABEL_ID, TAG))))
print(file.exists(file.path("data","splits", sprintf("test_l%d_%s.rds",  LABEL_ID, TAG))))
print(file.exists(file.path("data","splits", sprintf("split_meta_l%d_%s.rds", LABEL_ID, TAG))))

cat("\n--- CHECK: label balance in split ---\n")
train_df <- readRDS(file.path("data","splits", sprintf("train_l%d_%s.rds", LABEL_ID, TAG)))
test_df  <- readRDS(file.path("data","splits", sprintf("test_l%d_%s.rds",  LABEL_ID, TAG)))
label_col <- paste0("l", LABEL_ID)

print(table(train_df[[label_col]]))
print(prop.table(table(train_df[[label_col]])))

print(table(test_df[[label_col]]))
print(prop.table(table(test_df[[label_col]])))


###

cat("\n--- DIAG A: sizes ---\n")
cat("X_train dim:", paste(dim(X_train), collapse=" x "), "\n")
cat("X_test  dim:", paste(dim(X_test), collapse=" x "), "\n")
cat("y_train:", length(y_train), " y_test:", length(y_test), "\n")

cat("\n--- DIAG B: label sanity ---\n")
print(table(y_train))
print(prop.table(table(y_train)))
print(table(y_test))
print(prop.table(table(y_test)))
cat("min/max y_train:", min(y_train), max(y_train), "\n")
cat("min/max y_test :", min(y_test), max(y_test), "\n")

cat("\n--- DIAG C: missingness / non-finite ---\n")
cat("NA rate X_train:", mean(is.na(X_train)), "\n")
cat("NA rate X_test :", mean(is.na(X_test)), "\n")
cat("Any Inf train:", any(is.infinite(X_train), na.rm=TRUE), "\n")
cat("Any Inf test :", any(is.infinite(X_test),  na.rm=TRUE), "\n")

cat("\n--- DIAG D: constant cols removed ---\n")
cat("constant_cols:", length(constant_cols), "\n")
if (length(constant_cols) > 0) print(head(constant_cols, 30))

cat("\n--- DIAG E: potential leakage columns still present? ---\n")
suspects <- grep("(series|id|st$|name|period|freq|tag|horizon|window)", predictor_cols, ignore.case = TRUE, value = TRUE)
print(head(suspects, 100))
cat("num suspect cols:", length(suspects), "\n")

##

cat("\n--- BASELINE 1: No-information rate (TEST) ---\n")
tab_test <- table(y_test)
print(tab_test)
nir_class <- as.integer(names(which.max(tab_test)))
nir_acc <- max(tab_test) / sum(tab_test)
cat("NIR class:", nir_class, " NIR accuracy:", round(nir_acc, 4), "\n")

cat("\n--- BASELINE 2: Random guess matching prevalence (TEST) ---\n")
set.seed(1)
p <- as.numeric(prop.table(tab_test))
rand_pred <- sample(as.integer(names(tab_test)), size = length(y_test), replace = TRUE, prob = p)
cat("Random-prevalence accuracy:", round(mean(rand_pred == y_test), 4), "\n")
###

##################


cat("\n--- XGB QUICK RUN: compare weights vs no-weights (TEST) ---\n")
dtr_nowt <- xgb.DMatrix(X_train, label = y_train)
dtr_wt   <- xgb.DMatrix(X_train, label = y_train, weight = w_train)
dte      <- xgb.DMatrix(X_test,  label = y_test)

params_q <- list(
  booster="gbtree",
  objective="multi:softprob",
  eval_metric="merror",
  num_class=length(unique(y_train)),
  max_depth=8,
  eta=0.1,
  subsample=0.9,
  colsample_bytree=0.8,
  nthread=8
)

m_nowt <- xgb.train(params=params_q, data=dtr_nowt, nrounds=200, verbose=0)
m_wt   <- xgb.train(params=params_q, data=dtr_wt,   nrounds=200, verbose=0)

pred_nowt <- max.col(matrix(predict(m_nowt, dte), ncol=params_q$num_class, byrow=TRUE)) - 1L
pred_wt   <- max.col(matrix(predict(m_wt,   dte), ncol=params_q$num_class, byrow=TRUE)) - 1L

acc_nowt <- mean(pred_nowt == y_test)
acc_wt   <- mean(pred_wt   == y_test)

cat("Accuracy no-weights:", round(acc_nowt, 4), "\n")
cat("Accuracy with-weights:", round(acc_wt, 4), "\n")

cat("\nPred distribution no-weights:\n")
print(prop.table(table(pred_nowt)))
cat("\nPred distribution with-weights:\n")
print(prop.table(table(pred_wt)))


####

cat("\n--- SIGNAL CHECK: univariate AUC-like separability via ANOVA p-values ---\n")
pvals <- sapply(seq_len(ncol(X_train)), function(j) {
  x <- X_train[, j]
  if (sd(x) == 0) return(NA_real_)
  summary(aov(x ~ as.factor(y_train)))[[1]][["Pr(>F)"]][1]
})
names(pvals) <- colnames(X_train)
pvals <- sort(pvals)
print(head(pvals, 15))
cat("Num features with p < 1e-6:", sum(pvals < 1e-6, na.rm=TRUE), "\n")

