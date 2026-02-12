# src/r/utils.R
# Common utility functions for the M4-XGB pipeline:
# - Frequency and horizon helpers
# - Scaling helpers for windows and horizons

# ------------------------------------------------------------
# Frequency helpers
# ------------------------------------------------------------

infer_frequency <- function(period) {
  p <- as.character(period)
  
  switch(
    p,
    "Yearly"    = 1,
    "Quarterly" = 4,
    "Monthly"   = 12,
    "Weekly"    = 52,
    "Daily"     = 7,
    "Hourly"    = 24,
    1  # default fallback
  )
}

get_m4_horizon <- function(period) {
  p <- as.character(period)
  
  switch(
    p,
    "Yearly"    = 6L,
    "Quarterly" = 8L,
    "Monthly"   = 18L,
    "Weekly"    = 13L,
    "Daily"     = 14L,
    "Hourly"    = 48L,
    stop("Unknown M4 period: ", p)
  )
}

get_window_size_from_h <- function(period) {
  p <- as.character(period)
  H <- get_m4_horizon(p)
  
  if (p %in% c("Yearly", "Quarterly", "Monthly")) {
    return(2L * H)
  } else if (p %in% c("Weekly", "Daily")) {
    return(4L * H)
  } else if (p == "Hourly") {
    return(2L * H)
  } else {
    stop("No rule defined for period: ", p)
  }
}

# ------------------------------------------------------------
# Scaling helpers
# ------------------------------------------------------------

# Min-max scale a numeric vector to [0, 1]
minmax_vec <- function(v) {
  v <- as.numeric(v)
  r <- range(v, na.rm = TRUE)
  
  if (r[1] == r[2]) {
    return(rep(0, length(v)))
  }
  
  (v - r[1]) / (r[2] - r[1])
}

# Standardise a numeric vector (mean 0, sd 1)
standardise_vec <- function(v) {
  v <- as.numeric(v)
  m <- mean(v, na.rm = TRUE)
  s <- sd(v, na.rm = TRUE)
  
  if (s == 0 || is.na(s)) {
    return(rep(0, length(v)))
  }
  
  (v - m) / s
}

# Joint scaling for a window/horizon pair (SAFE):
# - Min–max stats computed from x only
# - Standardisation stats computed from x_mm only
# - Returns xx_std only if xx is provided (otherwise NULL)
scale_pair_minmax_std <- function(x, xx = NULL) {
  x  <- as.numeric(x)
  xx <- if (!is.null(xx)) as.numeric(xx) else NULL
  
  # Min–max using x only
  min_x   <- min(x, na.rm = TRUE)
  max_x   <- max(x, na.rm = TRUE)
  range_x <- max_x - min_x
  
  if (!is.finite(range_x) || range_x == 0) {
    x_mm  <- rep(0, length(x))
    xx_mm <- if (!is.null(xx)) rep(0, length(xx)) else NULL
  } else {
    x_mm  <- (x - min_x) / range_x
    xx_mm <- if (!is.null(xx)) (xx - min_x) / range_x else NULL
  }
  
  # Standardise using x_mm only
  mean_x <- mean(x_mm, na.rm = TRUE)
  sd_x   <- sd(x_mm, na.rm = TRUE)
  
  if (!is.finite(sd_x) || sd_x == 0) {
    x_std  <- rep(0, length(x_mm))
    xx_std <- if (!is.null(xx_mm)) rep(0, length(xx_mm)) else NULL
  } else {
    x_std  <- (x_mm - mean_x) / sd_x
    xx_std <- if (!is.null(xx_mm)) (xx_mm - mean_x) / sd_x else NULL
  }
  
  list(x_std = x_std, xx_std = xx_std)
}



# ------------------------------------------------------------
# Frequency → file tag
# ------------------------------------------------------------

freq_tag <- function(period) {
  p <- as.character(period)
  
  switch(
    p,
    "Yearly"    = "y",
    "Quarterly" = "q",
    "Monthly"   = "m",
    "Weekly"    = "w",
    "Daily"     = "d",
    "Hourly"    = "h",
    stop("Unknown period: ", p, call. = FALSE)
  )
}


period_to_freq <- function(period) {
  # M4 uses: "YEARLY", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"
  # but names can vary in case; we normalise.
  p <- tolower(period)
  if (p == "yearly")   return(1)
  if (p == "quarterly")return(4)
  if (p == "monthly")  return(12)
  if (p == "weekly")   return(52)
  if (p == "daily")    return(1)   # no strong seasonality in M4 daily
  if (p == "hourly")   return(24)
  # fallback
  return(1)
}

# ---------------------------------------------------------------------
# 3. Compute threshold c from distribution of |z|
# ---------------------------------------------------------------------

#compute_c <- function(z, q = 0.40) {
#  z <- z[is.finite(z)]  # remove NA and infinite
#  
#  if (length(z) == 0L) {
#    stop("No valid z-values available for computing c.")
#  }
#  
#  abs_z <- abs(z)
#  unname(quantile(abs_z, probs = q))
#}

compute_c <- function(z, q = 0.40) {
  z <- z[is.finite(z)]
  abs_z <- abs(z)
  unname(quantile(abs_z, probs = 1 - q))
}

# ---------------------------------------------------------------------
# 2. Compute z-values for each window (using standardised data)
# ---------------------------------------------------------------------

compute_z <- function(all_windows) {
  # horizon lengths (usually fixed)
  Hs <- sapply(all_windows$xx, length)
  
  # compute z(i) = mean(xx_i) - mean(tail(x_i, H_i))
  z <- mapply(
    FUN = function(w, h, H) {
      w <- as.numeric(w)
      h <- as.numeric(h)
      
      B <- mean(tail(w, H))  # baseline
      S <- mean(h)           # future summary
      
      S - B                  # standardized delta
    },
    w = all_windows$x,
    h = all_windows$xx,
    H = Hs
  )
  
  z
}


# ------------------------------------------------------------
# Map z -> label in {0,1,2}
# Down = 0, Neutral = 1, Up = 2
# ------------------------------------------------------------

label_from_z_int <- function(z, c) {
  ifelse(z >=  c, 2L,
         ifelse(z <= -c, 0L, 1L))
}

# ---------------------------------------------------------------------
# Generic z function for REAL evaluation (label IDs 1..4)
# Used in 11_eval_real_xgb.R, 12_baseline_fforma_smyl.R, 11b_eval_real_tsc.R
# ---------------------------------------------------------------------

compute_z_generic <- function(x_std, xx_std, label_id) {
  w <- as.numeric(x_std)
  h <- as.numeric(xx_std)
  
  H       <- length(h)
  w_tail  <- tail(w, H)
  w_last  <- tail(w, 1)
  S_mean  <- mean(h)
  S_med   <- median(h)
  B_last  <- w_last
  B_meanH <- mean(w_tail)
  
  z1 <- S_mean - B_last
  z2 <- S_med  - B_last
  z3 <- S_mean - B_meanH
  z4 <- S_med  - B_meanH
  
  c(z1, z2, z3, z4)[label_id]
}
