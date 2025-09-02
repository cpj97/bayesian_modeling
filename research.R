# Install relevant packages ----------------------------------------------------
library(pacman)
pacman::p_load(tidyverse, ggplot2, cmdstanr, posterior, purrr, readxl, bayestplot, tidybayes, knitr, kableExtra, gridExtra, grid)

# Load data --------------------------------------------------------------------
data1 <- read_excel("Datos-ICM-2023.xlsx", 
                   sheet = "BD_ICM_Municipal", 
                   range = "C5:GQ14591") %>%
  select(c(1:3, 4, 6:11, "CV-01-1")) %>%
  #select(c(1:3, 4, 6:9, 11, "CV-03-16", 196)) %>%
  filter(!if_any(everything(), is.na)) %>%
  rename(
    department = Departamento,
    municipality = Municipio,
    muni_id = Divipola,
    year = AÑO,
    EIS = `EIS-00-0`,
    CTI = `CTI-00-0`,
    PCC = `PCC-00-0`,
    GPI = `GPI-00-0`,
    SEG = `SEG-00-0`,
    SOS = `SOS-00-0`,
    Population = `CV-01-1`
  )

# Standardize continuous predictors --------------------------------------------
data <- data1 %>%
  mutate(across(c("EIS", "CTI", "PCC", "GPI", "SEG", "SOS"), 
                ~ scale(.)[, 1]))

# Generate identifiers for grouping --------------------------------------------
data <- data %>%
  mutate(
    muni_id = as.integer(as.factor(muni_id)),
    dept_id = as.integer(as.factor(department)),
    year_id = as.integer(as.factor(year)),
    muni_year = paste0(muni_id, "_", year_id),
    muni_year_id = as.integer(as.factor(muni_year))
  )

# Count unique group levels ----------------------------------------------------
data$muni_id %>% unique() %>% length()      # Number of municipalities (J)
data$dept_id %>% unique() %>% length()      # Number of departments (K)
data$year_id %>% unique() %>% length()      # Number of years (L)
data$muni_year_id %>% unique() %>% length() # Number of municipality-years (M)

# Link municipality to department (like j_id to c_id) --------------------------
municipality_level_data <- data %>%
  distinct(muni_id, .keep_all = TRUE) %>%
  select(muni_id, dept_id) %>%
  arrange(muni_id)

# Define covariates to include -------------------------------------------------
X_vars <- c("EIS", "CTI", "PCC", "GPI", "SOS")  # Example covariates
X <- as.matrix(data %>% select(all_of(X_vars)))
P <- ncol(X)

# Prepare Stan data list ------------------------------------------------------
data_stan <- list(
  N = nrow(data),
  J = length(unique(data$muni_id)),
  K = length(unique(data$dept_id)),
  L = length(unique(data$year_id)),
  M = length(unique(data$muni_year_id)),
  P = P,
  jj = data$muni_id,
  kk = municipality_level_data$dept_id,  # Only one per municipality
  ll = data$year_id,
  X = X,
  y = data$SEG
)

# Save data for Stan -----------------------------------------------------------
nested_slopes <- "
// Nested Hierarchical Bayesian Model for Homicide Rates
// Nested structure: Year nested in Municipality nested in Department
// Varying intercepts by department, municipality, and year
// Varying slopes by department

data {
  int<lower=1> N;                    // number of observations
  int<lower=1> J;                    // number of municipalities
  int<lower=1> K;                    // number of departments
  int<lower=1> L;                    // number of years
  int<lower=1> P;                    // number of covariates

  array[N] int<lower=1, upper=J> jj;   // municipality ID
  array[J] int<lower=1, upper=K> kk;   // department ID for each municipality
  array[N] int<lower=1, upper=L> ll;   // year ID

  matrix[N, P] X;                    // predictor matrix
  vector[N] y;                       // outcome: Security Index
}

parameters {
  vector[J] alpha_muni;              // municipality intercepts
  vector[K] eta_dept;                // department intercepts
  vector[L] gamma_year;              // year intercepts

  matrix[K, P] beta_dept;            // varying slopes by department
  row_vector[P] mu_beta;             // global slope means

  real mu_eta;                       // global mean for department intercepts

  real<lower=0> sigma_alpha;         // SD municipality intercepts
  real<lower=0> sigma_eta;           // SD department intercepts
  real<lower=0> sigma_gamma;         // SD year intercepts
  vector<lower=0>[P] sigma_beta;     // SDs varying slopes
  real<lower=0> sigma;               // residual SD
}

model {
  // Hyperpriors
  mu_beta ~ normal(0, 1);              // Global slopes
  mu_eta ~ normal(0, 1);               // Global department mean

  sigma_alpha ~ student_t(3, 0, 1);          // SD of municipality intercepts
  sigma_eta ~ student_t(3, 0, 1);            // SD of department intercepts
  sigma_gamma ~ student_t(3, 0, 1);          // SD of year intercepts
  sigma_beta ~ student_t(3, 0, 1);           // SDs of varying slopes
  sigma ~ student_t(3, 0, 1);                // Residual SD

  // Hierarchical structure priors
  eta_dept ~ student_t(3, mu_eta, sigma_eta);
  alpha_muni ~ student_t(3, eta_dept[kk], sigma_alpha);
  gamma_year ~ student_t(3, 0, sigma_gamma);

  for (k in 1:K)
    beta_dept[k] ~ normal(mu_beta, sigma_beta);

  // Likelihood
  for (n in 1:N) {
    vector[P] beta = to_vector(beta_dept[kk[jj[n]]]);
    y[n] ~ normal(
      alpha_muni[jj[n]] +
      gamma_year[ll[n]] +
      dot_product(X[n], beta),
      sigma);
  }
}
generated quantities {
  vector[N] y_hat;  // Predicted values
  real var_y_hat;
  real var_res;
  real bayes_R2;

  for (n in 1:N) {
    vector[P] beta = to_vector(beta_dept[kk[jj[n]]]);
    y_hat[n] = alpha_muni[jj[n]] +
               gamma_year[ll[n]] +
               dot_product(X[n], beta);
  }

  var_y_hat = variance(y_hat);
  var_res = sigma^2;

  bayes_R2 = var_y_hat / (var_y_hat + var_res);
}
"

# save as .stan file --------------------------------------------------------------------
write(nested_slopes, "nested_slopes.stan")

# compile model -------------------------------------------------------------------------
nested_slopes_compiled <- cmdstan_model("nested_slopes.stan")

# Fit model -------------------------------------------------------------------------------
fit <- nested_slopes_compiled$sample(
  data = data_stan,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  max_treedepth = 12,       # helps avoid tree-depth warnings
  adapt_delta = 0.95,
  refresh = 100,
  output_dir = "nested_slopes",
  output_basename = "nested_slopes_compiled"
)

draws <- fit$draws()
variables(draws)
# Global slopes ---------------------------------------------------------------
# Define variable labels
var_labels <- tibble(
  parameter = paste0("mu_beta[", 1:length(X_vars), "]"),
  label = X_vars
)


# Plot with names
sds_x <- sapply(data1[X_vars], sd, na.rm = TRUE)
sd_y <- sd(data1$SEG, na.rm = TRUE)

draws %>%
  as_draws_df() %>%
  select(starts_with("mu_beta")) %>%
  pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
  mutate(
    covariate = as.integer(str_extract(parameter, "\\d+")),
    label = X_vars[covariate],
    rescaled_value = value * sd_y / sds_x[covariate]
  ) %>%
  ggplot(aes(x = rescaled_value, y = fct_rev(label))) +
  stat_halfeye(
    .width = 0.95,
    fill = "skyblue",
    color = "skyblue4",
    slab_alpha = 0.6,
    point_interval = median_qi
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  theme_minimal(base_size = 13) +
  theme(
    axis.title.y = element_text(margin = margin(r = 10)),
    axis.title.x = element_text(margin = margin(t = 10)),
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    axis.text = element_text(size = 11)
  ) +
  labs(
    title = "Posterior Distributions of Global Slopes",
    subtitle = "Median and 95% credible intervals",
    x = "Effect on SEG",
    y = "Covariate"
  )

# Department-specific slopes ---------------------------------------------------
dept_labels <- data %>%
  distinct(dept_id, department) %>%
  arrange(dept_id)

draws %>%
  spread_draws(beta_dept[dept, covariate]) %>%
  filter(covariate == 1) %>% 
  mutate(
    beta_rescaled = beta_dept * sd_y / sds_x[[1]]
  ) %>%
  median_qi(beta_rescaled, .width = 0.95) %>%
  left_join(dept_labels, by = c("dept" = "dept_id")) %>%
  ggplot(aes(x = beta_rescaled, y = fct_reorder(department, beta_rescaled))) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray60") +  # reference line
  geom_errorbarh(aes(xmin = .lower, xmax = .upper), height = 0.25, color = "skyblue4") +
  geom_point(size = 2.5, color = "skyblue4") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title.y = element_text(margin = margin(r = 10)),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.text = element_text(size = 11)
  ) +
  labs(
    x = "Department-Specific Slope for EIS",
    y = "Department", 
    title = "Varying Slopes for Equity and Social Inclusion Index (EIS)",
    subtitle = "Posterior medians with 95% credible intervals"
  )

draws %>%
  spread_draws(beta_dept[dept, covariate]) %>%
  filter(covariate == 2) %>%
  mutate(
    beta_rescaled = beta_dept * sd_y / sds_x[[2]]
  ) %>%
  median_qi(beta_rescaled, .width = 0.95) %>%
  left_join(dept_labels, by = c("dept" = "dept_id")) %>%
  ggplot(aes(x = beta_rescaled, y = fct_reorder(department, beta_rescaled))) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray60") +  # reference line
  geom_errorbarh(aes(xmin = .lower, xmax = .upper), height = 0.25, color = "skyblue4") +
  geom_point(size = 2.5, color = "skyblue4") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title.y = element_text(margin = margin(r = 10)),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.text = element_text(size = 11)
  ) +
  labs(
    x = "Department-Specific Slope for CTI",
    y = "Department", 
    title = "Varying Slopes for Science, Technology and Innovation Index (CTI)",
    subtitle = "Posterior medians with 95% credible intervals"
  )

draws %>%
  spread_draws(beta_dept[dept, covariate]) %>%
  filter(covariate == 3) %>%
  mutate(
    beta_rescaled = beta_dept * sd_y / sds_x[[3]]
  ) %>%
  median_qi(beta_rescaled, .width = 0.95) %>%
  left_join(dept_labels, by = c("dept" = "dept_id")) %>%
  ggplot(aes(x = beta_rescaled, y = fct_reorder(department, beta_rescaled))) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray60") +  # reference line
  geom_errorbarh(aes(xmin = .lower, xmax = .upper), height = 0.25, color = "skyblue4") +
  geom_point(size = 2.5, color = "skyblue4") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title.y = element_text(margin = margin(r = 10)),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.text = element_text(size = 11)
  ) +
  labs(
    x = "Department-Specific Slope for PCC",
    y = "Department", 
    title = "Varying Slopes for Productivity, Competitiveness and \nEconomic Complementarity Index (PCC)",
    subtitle = "Posterior medians with 95% credible intervals"
  )

draws %>%
  spread_draws(beta_dept[dept, covariate]) %>%
  filter(covariate == 4) %>%
  mutate(
    beta_rescaled = beta_dept * sd_y / sds_x[[4]]
  ) %>%
  median_qi(beta_rescaled, .width = 0.95) %>%
  left_join(dept_labels, by = c("dept" = "dept_id")) %>%
  ggplot(aes(x = beta_rescaled, y = fct_reorder(department, beta_rescaled))) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray60") +  # reference line
  geom_errorbarh(aes(xmin = .lower, xmax = .upper), height = 0.25, color = "skyblue4") +
  geom_point(size = 2.5, color = "skyblue4") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title.y = element_text(margin = margin(r = 10)),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.text = element_text(size = 11)
  ) +
  labs(
    x = "Department-Specific Slope for GPI",
    y = "Department", 
    title = "Varying Slopes for Governance, Participation and \nInstitutions Index (GPI)",
    subtitle = "Posterior medians with 95% credible intervals"
  )

draws %>%
  spread_draws(beta_dept[dept, covariate]) %>%
  filter(covariate == 5) %>%
  mutate(
    beta_rescaled = beta_dept * sd_y / sds_x[[4]]
  ) %>%
  median_qi(beta_rescaled, .width = 0.95) %>%
  left_join(dept_labels, by = c("dept" = "dept_id")) %>%
  ggplot(aes(x = beta_rescaled, y = fct_reorder(department, beta_rescaled))) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray60") +  # reference line
  geom_errorbarh(aes(xmin = .lower, xmax = .upper), height = 0.25, color = "skyblue4") +
  geom_point(size = 2.5, color = "skyblue4") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title.y = element_text(margin = margin(r = 10)),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.text = element_text(size = 11)
  ) +
  labs(
    x = "Department-Specific Slope for SOS",
    y = "Department", 
    title = "Varying Slopes for Sustainability Index (SOS)",
    subtitle = "Posterior medians with 95% credible intervals"
  )


# R squared ---------------------------------------------------------------
# Extract draws of bayes_R2
bayes_r2 <- as_draws_df(draws)$bayes_R2

ggplot(data.frame(bayes_r2), aes(x = bayes_r2)) +
  geom_density(fill = "skyblue", color = "skyblue4", alpha = 0.8) +
  geom_vline(aes(xintercept = median(bayes_r2)), 
             linetype = "dashed", color = "gray30", linewidth = 0.8) +
  annotate("text", 
           x = median(bayes_r2), y = Inf, 
           label = paste0("Median = ", round(median(bayes_r2), 2)),
           vjust = -0.5, hjust = 0.5,
           size = 4, color = "gray30") +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(size = 12),
    plot.margin = margin(10, 20, 10, 10)
  ) +
  labs(
    title = "Posterior Distribution of Bayesian R²",
    x = expression(Bayesian ~ R^2),
    y = "Posterior Sample Frequency",
    caption = paste0("RMSE = ", round(rmse_rescaled, 3), 
                     " | Bayesian R² = ", round(bayes_r2_median, 3))
  ) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0.5, size = 12, face = "italic", color = "gray30")
  )

# Posterior predictive checks ---------------------------------------------------
# Extract y_hat (each row is a posterior draw, each column is an observation)
y_hat_draws <- fit$draws(variables = "y_hat") %>%
  posterior::as_draws_matrix()

mean_y <- mean(data1$SEG, na.rm = TRUE)
sd_y <- sd(data1$SEG, na.rm = TRUE)
y_hat_mean_rescaled <- y_hat_mean * sd_y + mean_y
y_obs_rescaled <- data_stan$y * sd_y + mean_y

# Recalculate RMSE on the original scale
rmse_rescaled <- sqrt(mean((y_obs_rescaled - y_hat_mean_rescaled)^2))

ggplot(data.frame(y_obs = y_obs_rescaled, y_pred = y_hat_mean_rescaled), 
       aes(x = y_pred, y = y_obs)) +
  geom_point(alpha = 0.3, color = "steelblue", size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "darkred", linewidth = 0.5) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Observed vs Predicted SEG",
    x = "Predicted SEG",
    y = "Observed SEG",
    caption = paste0("RMSE = ", round(rmse_rescaled, 3), 
                     " | Bayesian R² = ", round(bayes_r2_median, 3))
  ) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title = element_text(face = "bold"),
    plot.caption = element_text(hjust = 0.5, size = 12, face = "italic", color = "gray30")
  )

# Heatmap of department-specific slopes ------------------------------------------
# Ensure sd_y and sds_x are defined
sds_x <- sapply(data1[X_vars], sd, na.rm = TRUE)
sd_y <- sd(data1$SEG, na.rm = TRUE)

# Rescale slopes in the heatmap
heat_df <- draws %>%
  spread_draws(beta_dept[dept, covariate]) %>%
  group_by(dept, covariate) %>%
  summarise(mean_slope = mean(beta_dept), .groups = "drop") %>%
  mutate(
    rescaled_slope = mean_slope * sd_y / sds_x[covariate]
  ) %>%
  left_join(dept_labels, by = c("dept" = "dept_id")) %>%
  mutate(covariate_label = X_vars[covariate])

# Plot rescaled slopes
ggplot(heat_df, aes(x = covariate_label, y = fct_reorder(department, dept), fill = rescaled_slope)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "darkred", mid = "white", high = "darkblue", midpoint = 0) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Rescaled Mean Department Slopes by Covariate",
    x = "Covariate",
    y = "Department",
    fill = "Effect on SEG"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Table summary of global slopes -------------------------------------------------------------------------
# Standard deviations
sds_x <- sapply(data1[X_vars], sd, na.rm = TRUE)
sd_y <- sd(data1$SEG, na.rm = TRUE)

# Create rescaled summary table
summary_table <- draws %>%
  as_draws_df() %>%
  select(starts_with("mu_beta")) %>%
  pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
  mutate(covariate = as.integer(str_extract(parameter, "\\d+"))) %>%
  group_by(covariate) %>%
  summarise(
    mean = mean(value),
    median = median(value),
    lower = quantile(value, 0.025),
    upper = quantile(value, 0.975),
    .groups = "drop"
  ) %>%
  mutate(
    label = X_vars[covariate],
    rescale_factor = sd_y / sds_x[covariate],
    mean = mean * rescale_factor,
    median = median * rescale_factor,
    lower = lower * rescale_factor,
    upper = upper * rescale_factor
  ) %>%
  select(label, mean, median, lower, upper)

# Create a table grob
table_grob <- tableGrob(
  summary_table,
  rows = NULL
)

# Create title grob
title <- textGrob(
  "Rescaled Posterior Summaries for Global Slopes (Effect on SEG)",
  gp = gpar(fontsize = 16, fontface = "bold")
)

# Plot the title and table in the Plots panel
grid.newpage()
grid.arrange(title, table_grob, nrow = 2, heights = c(0.1, 1))


# Table summary of department-specific variables ---------------------------------------------------------
all_vars <- c("EIS-00-0", "CTI-00-0", "PCC-00-0", "GPI-00-0", "SEG-00-0", "SOS-00-0")
 
summary_table <- data1 %>%
  group_by(department) %>%
  summarise(across(all_of(all_vars), list(mean = ~mean(.x, na.rm = TRUE),
                                            median = ~median(.x, na.rm = TRUE))))


