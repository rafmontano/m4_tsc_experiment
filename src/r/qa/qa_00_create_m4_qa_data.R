# =====================================================================
# qa_00_create_m4_qa_data.R
# Create a minimal synthetic dataset that mimics the M4comp2018 "M4" object
# Output: data/qa/M4_qa.rds
# =====================================================================

# ---------------------------------------------------------------------
# 0) Config (keep minimal and explicit)
# ---------------------------------------------------------------------

QA_PERIOD <- "Weekly"
QA_FREQ   <- 52
QA_H      <- 13
QA_W      <- 200
QA_N_SERIES <- 30

set.seed(123)

qa_dir <- file.path("data", "qa")
if (!dir.exists(qa_dir)) dir.create(qa_dir, recursive = TRUE)

qa_path <- file.path(qa_dir, "M4_qa.rds")

# ---------------------------------------------------------------------
# 1) Synthetic generator (simple but learnable patterns)
# ---------------------------------------------------------------------

gen_one_series <- function(st_id, w, h, freq, period) {
  # Regimes:
  #   - Up-trend
  #   - Down-trend
  #   - Flat (neutral)
  regime <- sample(c("UP", "DOWN", "FLAT"), size = 1, prob = c(0.34, 0.33, 0.33))
  
  # Base level and noise
  level <- runif(1, 50, 150)
  noise_sd <- runif(1, 0.5, 2.0)
  
  # Deterministic slope controls label separability
  slope <- switch(
    regime,
    "UP"   = runif(1, 0.4, 1.2),
    "DOWN" = -runif(1, 0.4, 1.2),
    "FLAT" = runif(1, -0.05, 0.05)
  )
  
  # Optional seasonal component (kept small)
  season_amp <- runif(1, 0.0, 2.5)
  phase <- runif(1, 0, 2 * pi)
  
  t_all <- seq_len(w + h)
  
  seasonal <- season_amp * sin(2 * pi * t_all / freq + phase)
  trend    <- slope * t_all
  
  y_all <- level + trend + seasonal + rnorm(w + h, mean = 0, sd = noise_sd)
  
  x  <- ts(y_all[1:w], frequency = freq)
  xx <- as.numeric(y_all[(w + 1):(w + h)])
  
  list(
    st     = paste0("QA_", period, "_", sprintf("%05d", st_id)),
    x      = x,
    n      = length(x),
    type   = "QA",
    h      = h,
    period = period,
    xx     = xx
  )
}

# ---------------------------------------------------------------------
# 2) Build QA dataset (list-of-lists like M4)
# ---------------------------------------------------------------------

M4_qa <- lapply(seq_len(QA_N_SERIES), function(i) {
  gen_one_series(
    st_id  = i,
    w      = QA_W,
    h      = QA_H,
    freq   = QA_FREQ,
    period = QA_PERIOD
  )
})

# Quick structure sanity checks
stopifnot(is.list(M4_qa), length(M4_qa) > 0)
req_names <- c("st", "x", "n", "type", "h", "period", "xx")
stopifnot(all(req_names %in% names(M4_qa[[1]])))
stopifnot(all(sapply(M4_qa, function(s) length(s$xx) == QA_H)))
stopifnot(all(sapply(M4_qa, function(s) length(s$x)  == QA_W)))
stopifnot(all(sapply(M4_qa, function(s) frequency(s$x) == QA_FREQ)))
stopifnot(all(sapply(M4_qa, function(s) s$period == QA_PERIOD)))

saveRDS(M4_qa, qa_path)

cat("Saved QA dataset →", qa_path, "\n")
cat("QA dataset size  →", length(M4_qa), "series\n")
cat("QA period        →", QA_PERIOD, "\n")
cat("QA window (x)    →", QA_W, "\n")
cat("QA horizon (xx)  →", QA_H, "\n")

# ---------------------------------------------------------------------
# 3) Optional: print fields like M4comp2018 example
# ---------------------------------------------------------------------

cat("\nNames of first element (M4_qa[[1]]):\n")
print(names(M4_qa[[1]]))

# ---------------------------------------------------------------------
# 4) M4-style plot for visual inspection (1 series)
# ---------------------------------------------------------------------

i_plot <- min(1, length(M4_qa))
s <- M4_qa[[i_plot]]

plot(
  ts(
    c(as.numeric(s$x), as.numeric(s$xx)),
    start = start(s$x),
    frequency = frequency(s$x)
  ),
  col = "red",
  type = "l",
  ylab = "",
  main = paste0("QA series: ", s$st, " (", s$period, ")")
)
lines(s$x, col = "black")


####

M4 <- readRDS("data/qa/M4_qa.rds")
names(M4[[1]])
plot(ts(c(M4[[1]]$x, M4[[1]]$xx),
        start=start(M4[[1]]$x), frequency = frequency(M4[[1]]$x)),
     col="red", type="l", ylab="")
lines(M4[[1]]$x, col="black")
