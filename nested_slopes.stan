
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

