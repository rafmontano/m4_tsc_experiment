# ============================================================
#  FFORMA-Compatible Feature Extraction (Talagala–Hyndman–Athanaspoulos)
#  Clean version for your PoC – identical feature names to FFORMA
# ============================================================

library(tsfeatures)
library(forecast)
library(tibble)


# ------------------------------------------------------------
# Heterogeneity wrapper (FFORMA used a patched tsfeatures)
# ------------------------------------------------------------
heterogeneity_tsfeat_workaround <- function(x) {
  output <- c(
    arch_acf  = 0,
    garch_acf = 0,
    arch_r2   = 0,
    garch_r2  = 0
  )
  try(output <- tsfeatures::heterogeneity(x), silent = TRUE)
  output
}


# ------------------------------------------------------------
# Holt–Winters parameters wrapper (stable version)
# ------------------------------------------------------------
hw_parameters_tsfeat_workaround <- function(x) {
  # initialise a 3-length vector of NAs
  pars <- c(NA, NA, NA)
  
  try({
    hw_fit <- forecast::ets(x, model = "AAA")
    p <- hw_fit$par
    
    # copy up to 3 parameters into pars
    len <- min(3L, length(p))
    if (len > 0) {
      pars[1:len] <- as.numeric(p[1:len])
    }
  }, silent = TRUE)
  
  names(pars) <- c("hw_alpha", "hw_beta", "hw_gamma")
  pars
}


# ------------------------------------------------------------
# Main FFORMA feature extractor for ONE time series
# Input: list(x = numeric vector)
# Output: same list with $features added
# ------------------------------------------------------------
calc_features <- function(seriesentry) {
  series <- seriesentry$x
  
  # tsfeatures(): must pass a ts object OR numeric vector – FFORMA used numeric
  featrow <- tsfeatures(
    series,
    features = c(
      "acf_features",
      "arch_stat",
      "crossing_points",
      "entropy",
      "flat_spots",
      heterogeneity_tsfeat_workaround,
      "holt_parameters",
      "hurst",
      "lumpiness",
      "nonlinearity",
      "pacf_features",
      "stl_features",
      "stability",
      hw_parameters_tsfeat_workaround,
      "unitroot_kpss",
      "unitroot_pp"
    )
  )
  
  # ---- Additional: series length (FFORMA feature) ----
  series_length <- length(series)
  featrow <- tibble::add_column(featrow, series_length = series_length)
  
  # Replace NAs with 0 (same as FFORMA codebase)
  featrow[is.na(featrow)] <- 0
  
  # ---- FFORMA dummy padding for non-seasonal series ----
  # They expected 40+ features when seasonal; fewer when non-seasonal.
  # If missing, add dummy values to preserve feature vector shape.
  if (length(featrow) == 37) {
    featrow <- tibble::add_column(featrow, seas_acf1 = 0, .before = 7)
    featrow <- tibble::add_column(featrow, seas_pacf = 0, .before = 24)
    featrow <- tibble::add_column(
      featrow,
      seasonal_strength = 0,
      peak = 0,
      trough = 0,
      .before = 33
    )
  }
  
  seriesentry$features <- featrow
  seriesentry
}



