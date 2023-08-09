{% include "chunks/copyright.stan" %}

data {
{%- block data -%}
  /* data declarations */
  int<lower=0> N;
  array[N] int<lower=1, upper=2> j;
{%- endblock data -%}
}

transformed data {
{%- block transformed_data -%}
{%- endblock transformed_data -%}
}

parameters {
{%- block parameters %}
  /* raw model parameters */
  vector[2] mu;
  vector<lower=0>[2] sigma;
{%- endblock parameters -%}
}

transformed parameters {
{%- block transformed_parameters %}
  /* parameter transformations */
  // nb: no change of variables adjustments are made
  // to these parameters
  vector[N] mu_j = mu[j];
  vector<lower=0>[N] sigma_j = sigma[j];
{%- endblock transformed_parameters -%}
}

model {
{%- block model %}
{%- block priors %}
  /* priors */
  mu ~ {{ priors.mu }};
  sigma ~ {{ priors.sigma }};
{%- endblock priors -%}
{%- block likelihood -%}
  /* likelihood statement */
{%- endblock likelihood -%}
{%- endblock model -%}
}

generated quantities {
{%- block generated_quantities -%}
{%- block declarations %}
  /* declarations */
  real mu_diff;
  real sigma_diff;
{%- endblock declarations -%}
{%- block computations %}
  /* computations */
  mu_diff = mu[1] - mu[2];
  sigma_diff = sigma[1] - sigma[2];
{%- endblock computations -%}
{%- endblock generated_quantities -%}
}
