# File: src/r/paper/06_class_imbalance_bars.R
# Purpose: Visualise class imbalance by frequency (Up / Neutral / Down).

library(tidyverse)

# --------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------

input_csv <- file.path("data", "export", "class_proportion.csv")

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
# NOTE: use the same trick as Figure 5(a) so that, AFTER coord_flip(),
# the rows appear (top→bottom): Yearly, Quarterly, Monthly, Weekly, Daily, Hourly

freq_levels <- c("Hourly", "Daily", "Weekly",
                 "Monthly", "Quarterly", "Yearly")

df_counts <- df_counts %>%
  mutate(
    frequency   = factor(frequency, levels = freq_levels),
    class_label = factor(class_label, levels = c("Up", "Neutral", "Down"))
  )

df_props <- df_counts %>%
  group_by(frequency) %>%
  mutate(
    total_n = sum(n),
    prop    = n / total_n
  ) %>%
  ungroup()

# --------------------------------------------------------------------
# 3. Output directory
# --------------------------------------------------------------------

fig_dir  <- file.path("output", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

fig_path <- file.path(fig_dir, "class_imbalance_by_frequency_bar.pdf")

# --------------------------------------------------------------------
# 4. Horizontal stacked bar chart (proportions)
# --------------------------------------------------------------------

p <- ggplot(df_props, aes(x = frequency, y = prop, fill = class_label)) +
  geom_col(color = "grey20") +
  coord_flip() +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_fill_grey(start = 0.8, end = 0.2, name = "Class") +
  labs(
    x = "Frequency",
    y = "Class proportion",
    title = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x      = element_text(),   # no rotation needed after flip
    axis.text.y      = element_text()
  )

ggsave(fig_path, p, width = 6.3, height = 3.8)

message("Class imbalance bar chart saved to: ", fig_path)
