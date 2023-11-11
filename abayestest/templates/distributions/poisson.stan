{% extends "base.stan" %}

{% block data %}
  {{ super() }}
  array[N] int<lower=0> y;
{% endblock %}

{% block transformed_parameters %}
{% block tpar_declarations %}
  {{ super() }}
{% endblock tpar_declarations %}
{% block likelihood %}
  {{ super() }}
  for(nn in 1:N)
    lp[nn] = poisson_log_lpmf(y[nn] | mu_star_j[nn]);
{% endblock likelihood %}
{% endblock transformed_parameters %}


{% block model %}
{% block priors %}
  {{ super() }}
{% endblock priors %}
{% block log_density %}
  {{ super() }}
{% endblock log_density %}
{% endblock %}

{% block generated_quantities %}
{% block declarations %}
  {{ super() }}
  vector<lower=0>[2] mu = exp(mu_star);
  vector[N] mu_j = exp(mu_star_j);
  real mu_diff = mu[1] - mu[2];
  array[N] int<lower=0> y_rep;
{% endblock %}
{% block computations %}
  {{ super() }}
  for(n in 1:N)
    y_rep[n] = poisson_log_rng(mu_star_j[n]);
{% endblock %}
{% endblock %}
  
