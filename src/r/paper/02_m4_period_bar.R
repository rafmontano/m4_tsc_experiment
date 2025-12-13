# =====================================================================
# 02_m4_period_bar.R
# Horizontal bar chart of M4 series counts per period
# Output: results/paper/figures/m4_period_bar.pdf
# =====================================================================

library(M4comp2018)
library(tidyverse)
library(scales)

# ---------------------------------------------------------------------
# 1) Output directory
# ---------------------------------------------------------------------


fig_dir  <- file.path("results", "paper", "figures")
fig_path <- file.path(fig_dir, "m4_period_bar.pdf")

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------
# 2) Load M4 metadata
# ---------------------------------------------------------------------

data("M4")

m4_meta <- purrr::map_dfr(
  M4,
  function(s) tibble(period = s$period)
)

# ---------------------------------------------------------------------
# 3) Counts & percentages (M4 table order)
# ---------------------------------------------------------------------

# Desired chronological order, matching M4 Table 1
#period_levels <- c("Yearly", "Quarterly", "Monthly",
#                   "Weekly", "Daily", "Hourly")

period_levels <- c("Hourly", "Daily", "Weekly",  
                    "Monthly", "Quarterly", "Yearly")



period_tbl <- m4_meta %>%
  count(period, name = "n_series") %>%
  mutate(
    total_series = sum(n_series),
    pct          = n_series / total_series,
    n_fmt        = format(n_series, big.mark = ",", trim = TRUE),
    pct_label    = sprintf("%.1f%%", 100 * pct),
    label_text   = paste0(n_fmt, " (", pct_label, ")")
  ) %>%
  mutate(period = factor(period, levels = period_levels)) %>%
  arrange(match(period, period_levels))

# ---------------------------------------------------------------------
# 4) Plot
# ---------------------------------------------------------------------

p <- ggplot(period_tbl, aes(x = period, y = n_series)) +
  geom_col(width = 0.7, fill = "grey20") +
  coord_flip() +
  geom_text(
    aes(label = label_text),
    hjust = -0.10,
    size  = 3.5
  ) +
  scale_y_continuous(
    labels = comma,
    expand = expansion(mult = c(0, 0.25))
  ) +
  labs(
    x = "Frequency",
    y = "Number of series"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title      = element_blank(),
    legend.position = "none",
    axis.title.y    = element_text(margin = margin(r = 6)),
    axis.title.x    = element_text(margin = margin(t = 4)),
    axis.text       = element_text(size = 10)
  )

# ---------------------------------------------------------------------
# 5) Save
# ---------------------------------------------------------------------

ggsave(
  filename = fig_path,
  plot     = p,
  width    = 6.3,
  height   = 3.8
)

message("Bar chart saved to: ", fig_path)

