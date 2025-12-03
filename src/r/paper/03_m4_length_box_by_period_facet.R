# =====================================================================
# 03_m4_length_box_by_period_facet.R
# Faceted boxplots of M4 series length by period, with median labels
# Output: output/figures/m4_length_box_by_period_facet.pdf
# =====================================================================

library(M4comp2018)
library(tidyverse)

# ---------------------------------------------------------------------
# 1) Output
# ---------------------------------------------------------------------

fig_dir  <- file.path("output", "figures")
fig_path <- file.path(fig_dir, "m4_length_box_by_period_facet.pdf")

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------
# 2) Load M4 and metadata
# ---------------------------------------------------------------------

data("M4")

m4_meta <- purrr::map_dfr(
  M4,
  function(s) {
    tibble(
      period = s$period,
      length = as.numeric(s$n)
    )
  }
) %>%
  mutate(
    period = factor(
      period,
      levels = c("Yearly", "Quarterly", "Monthly",
                 "Weekly", "Daily", "Hourly")
    )
  )

# ---------------------------------------------------------------------
# 3) Summary for medians
# ---------------------------------------------------------------------

length_summary <- m4_meta %>%
  group_by(period) %>%
  summarise(
    median_len = median(length),
    min_len    = min(length),
    .groups    = "drop"
  )

print(length_summary)

# ---------------------------------------------------------------------
# 4) Faceted boxplots + median labels
# ---------------------------------------------------------------------

p <- ggplot(m4_meta, aes(x = length, y = 1)) +
  geom_boxplot(
    fill          = "grey80",
    colour        = "grey20",
    outlier.size  = 0.5,
    outlier.alpha = 0.6
  ) +
  facet_wrap(
    ~ period,
    ncol   = 1,
    scales = "free_x"
  ) +
  geom_text(
    data = length_summary,
    aes(
      x     = min_len,
      y     = 1,
      label = paste0("median = ", median_len)
    ),
    inherit.aes = FALSE,
    hjust       = 1.1,
    size        = 3
  ) +
  labs(
    x = "Series length (number of observations)",
    y = NULL
  ) +
  coord_cartesian(clip = "off") +
  theme_minimal(base_size = 11) +
  theme(
    plot.title   = element_blank(),
    strip.text   = element_text(face = "bold"),
    axis.title.y = element_blank(),
    axis.text.y  = element_blank(),
    axis.ticks.y = element_blank(),
    plot.margin  = margin(t = 5, r = 10, b = 5, l = 60)
  )

# ---------------------------------------------------------------------
# 5) Save
# ---------------------------------------------------------------------

ggsave(
  filename = fig_path,
  plot     = p,
  width    = 6.3,
  height   = 7.0
)

message("Faceted boxplot of series length by period saved to: ", fig_path)
