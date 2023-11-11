{% extends "base.stan" %}

{% block data %}
  {{ super() }}
  array[N] int<lower=0> y;
  array[N] int<lower=0> n;
{% endblock %}

{% block transformed_parameters %}
{% block tpar_declarations %}
  {{ super() }}
{% endblock tpar_declarations %}
{% block likelihood %}
  for(nn in 1:N)
    lp[nn] = binomial_logit_lpmf(y[nn] | n[nn], mu_star_j[nn]);
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
  vector<lower=0, upper=1>[N] mu_j = inv_logit(mu_star_j);
  vector<lower=0, upper=1>[2] mu = inv_logit(mu_star);
  real mu_diff = mu[1] - mu[2];
  array[N] int<lower=0> y_rep;
{% endblock %}
{% block computations %}
  {{ super() }}
  for(nn in 1:N)
    y_rep[nn] = binomial_rng(n[nn], mu_j[nn]);
{% endblock %}
{% endblock %}
  
