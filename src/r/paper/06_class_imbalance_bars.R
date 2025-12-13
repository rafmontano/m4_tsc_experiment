# File: src/r/paper/06_class_imbalance_bars.R
# Purpose:
#   Visualise class imbalance by frequency (Up / Neutral / Down),
#   including percentage labels within each stacked segment.

library(tidyverse)

# --------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------

input_csv <- file.path("data", "export", "class_proportion.csv")
if (!file.exists(input_csv)) {
  stop(
    "Missing input file: ", input_csv,
    "\nRun 06a_compute_class_proportion.R first."
  )
}

df_counts <- readr::read_csv(input_csv, show_col_types = FALSE)

required_cols <- c("frequency", "class_label", "n")
missing_cols  <- setdiff(required_cols, names(df_counts))
if (length(missing_cols) > 0) {
  stop(
    "Missing required columns in input CSV: ",
    paste(missing_cols, collapse = ", ")
  )
}

# --------------------------------------------------------------------
# 2. Prepare factors and proportions
# --------------------------------------------------------------------
# Ordering trick:
# After coord_flip(), rows appear top → bottom as:
# Yearly, Quarterly, Monthly, Weekly, Daily, Hourly

freq_levels  <- c("Hourly", "Daily", "Weekly",
                  "Monthly", "Quarterly", "Yearly")
class_levels <- c("Up", "Neutral", "Down")

df_counts <- df_counts %>%
  mutate(
    frequency   = as.character(frequency),
    class_label = as.character(class_label)
  ) %>%
  mutate(
    class_label = case_when(
      tolower(class_label) == "up"      ~ "Up",
      tolower(class_label) == "neutral" ~ "Neutral",
      tolower(class_label) == "down"    ~ "Down",
      TRUE ~ class_label
    )
  ) %>%
  tidyr::complete(
    frequency   = freq_levels,
    class_label = class_levels,
    fill        = list(n = 0)
  ) %>%
  mutate(
    frequency   = factor(frequency, levels = freq_levels),
    class_label = factor(class_label, levels = class_levels)
  )

df_props <- df_counts %>%
  group_by(frequency) %>%
  mutate(
    total_n = sum(n),
    prop    = ifelse(total_n > 0, n / total_n, 0),
    pct_lab = ifelse(prop >= 0.02,
                     scales::percent(prop, accuracy = 1),
                     "")
  ) %>%
  ungroup()

# --------------------------------------------------------------------
# 3. Output directory
# --------------------------------------------------------------------

fig_dir <- file.path("results", "paper", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

fig_path <- file.path(fig_dir, "class_imbalance_by_frequency_bar.pdf")

# --------------------------------------------------------------------
# 4. Horizontal stacked bar chart + in-bar percentage labels
# --------------------------------------------------------------------

p <- ggplot(df_props, aes(x = frequency, y = prop, fill = class_label)) +
  geom_col(color = "grey20", linewidth = 0.2) +
  geom_text(
    aes(label = pct_lab, color = class_label),
    position = position_stack(vjust = 0.5),
    size = 3,
    show.legend = FALSE
  ) +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_fill_grey(start = 0.8, end = 0.2, name = "Class") +
  scale_color_manual(
    values = c(
      "Up"      = "black",
      "Neutral" = "black",
      "Down"    = "white"
    )
  ) +
  labs(
    x = "Frequency",
    y = "Class proportion",
    title = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x      = element_text(),
    axis.text.y      = element_text()
  )

ggsave(fig_path, p, width = 6.3, height = 3.8)

message("Class imbalance bar chart saved to: ", fig_path)
