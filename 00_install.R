# 00_install.R
# Install required R packages and create the project folder structure

# ------------------------------
# Helper: install CRAN packages
# ------------------------------
install_if_missing <- function(pkgs) {
  installed <- rownames(installed.packages())
  to_install <- setdiff(pkgs, installed)
  if (length(to_install) > 0) {
    message("[INSTALL] Installing CRAN packages: ",
            paste(to_install, collapse = ", "))
    install.packages(to_install)
  } else {
    message("[INSTALL] All requested CRAN packages already installed.")
  }
}

# ------------------------------------------
# Install M4comp2018 from specific tar.gz
# ------------------------------------------
install_M4comp2018 <- function() {
  pkg_name <- "M4comp2018"
  tar_url <- "https://github.com/carlanetto/M4comp2018/releases/download/0.2.0/M4comp2018_0.2.0.tar.gz"
  
  if (pkg_name %in% rownames(installed.packages())) {
    message("[M4comp2018] Package already installed. Attempting clean reinstall.")
    # Optional aggressive removal of package + its dependencies, guarded in try
    try({
      deps <- tools::package_dependencies(pkg_name, recursive = TRUE)
      deps_vec <- unique(c(pkg_name, deps[[pkg_name]]))
      deps_vec <- deps_vec[deps_vec %in% rownames(installed.packages())]
      if (length(deps_vec) > 0) {
        message("[M4comp2018] Removing existing package and dependencies: ",
                paste(deps_vec, collapse = ", "))
        remove.packages(deps_vec)
      }
    }, silent = TRUE)
  } else {
    message("[M4comp2018] Package not found. Fresh install will be performed.")
  }
  
  message("[M4comp2018] Installing from tar.gz: ", tar_url)
  install.packages(tar_url, repos = NULL, type = "source", clean = TRUE)
  
  if (!requireNamespace("M4comp2018", quietly = TRUE)) {
    stop("[M4comp2018] Installation appears to have failed. Please check the output.")
  } else {
    message("[M4comp2018] Installation successful.")
  }
}

# ------------------------------------------
# Install scmamp from GitHub
# ------------------------------------------
install_scmamp <- function() {
  if (!requireNamespace("devtools", quietly = TRUE)) {
    message("[INSTALL] devtools not found. Installing devtools from CRAN.")
    install.packages("devtools")
  }
  library(devtools)
  
  if (!requireNamespace("scmamp", quietly = TRUE)) {
    message("[INSTALL] Installing scmamp from GitHub: b0rxa/scmamp")
    devtools::install_github("b0rxa/scmamp")
  } else {
    message("[INSTALL] scmamp already installed.")
  }
}

# ------------------------------------------
# Create project folder structure
# ------------------------------------------
create_project_structure <- function() {
  message("[STRUCTURE] Creating project folder structure under: ", getwd())
  
  labels <- paste0("label_", 1:4)
  freqs  <- c("yearly", "quarterly", "monthly", "weekly", "daily")
  
  # Top-level directories
  dirs_top <- c(
    "config",
    "data/raw/m4",
    "data/processed/features",
    "data/metadata",
    "models",
    "src/r",
    "src/python",
    "scripts",
    "results/intermediate",
    "results/consolidated",
    "results/logs/r",
    "results/logs/python",
    "figures/cd_diagrams",
    "figures/other_plots",
    "docs/paper/tex/sections",
    "docs/paper/figures",
    "docs/paper/notes",
    "docs/lab_notebooks"
  )
  
  for (d in dirs_top) {
    dir.create(d, recursive = TRUE, showWarnings = FALSE)
  }
  
  # Label-specific directories for processed labels, models, and intermediate results
  for (lab in labels) {
    # Labelled data
    dir.create(file.path("data", "processed", "labels", lab),
               recursive = TRUE, showWarnings = FALSE)
    
    # Models: models/label_k/frequency
    for (fr in freqs) {
      dir.create(file.path("models", lab, fr),
                 recursive = TRUE, showWarnings = FALSE)
    }
    
    # Results intermediate: results/intermediate/label_k/frequency
    for (fr in freqs) {
      dir.create(file.path("results", "intermediate", lab, fr),
                 recursive = TRUE, showWarnings = FALSE)
    }
  }
  
  message("[STRUCTURE] Folder structure created (or already exists).")
}

# ------------------------------------------
# Main bootstrap function
# ------------------------------------------
run_bootstrap <- function() {
  message("[BOOTSTRAP] Starting environment setup...")
  
  # 1. Install required CRAN packages
  cran_pkgs <- c(
    "forecast",
    "tidyverse",
    "tsfeatures",
    "xgboost",
    "rBayesianOptimization",
    "caret",
    "factoextra"
  )
  install_if_missing(cran_pkgs)
  
  # 2. Install M4comp2018 from GitHub release tar.gz
  install_M4comp2018()
  
  # 3. Install scmamp from GitHub
  install_scmamp()
  
  # 4. Create project folder structure
  create_project_structure()
  
  message("[BOOTSTRAP] Setup completed successfully.")
}

# ------------------------------------------
# Execute when script is sourced
# ------------------------------------------
run_bootstrap()
