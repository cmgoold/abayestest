{% include "chunks/copyright.stan" %}

data {
{%- block data -%}
  /* data declarations */
  int<lower=0> N;
  array[N] int<lower=1, upper=2> j;
{%- endblock data -%}
}

transformed data {
{%- block transformed_data %}
  /* data transformations */
{%- endblock transformed_data %}
}

parameters {
{%- block parameters %}
  /* raw model parameters */
  vector[2] mu;
{%- endblock parameters %}
}

transformed parameters {
{%- block transformed_parameters %}
  /* parameter transformations */
  // nb: no change of variables adjustments are made
  // to these parameters
  vector[N] mu_j = mu[j];
{%- endblock transformed_parameters %}
}

model {
{%- block model %}
{%- block priors %}
  /* priors */
  mu ~ {{ priors.mu }};
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
{%- endblock declarations -%}
{%- block computations %}
  /* computations */
  mu_diff = mu[1] - mu[2];
{%- endblock computations -%}
{%- endblock generated_quantities -%}
}
