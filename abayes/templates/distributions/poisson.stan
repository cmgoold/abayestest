{% extends "base.stan" %}

{% block data %}
  {{ super() }}
  array[N] int<lower=0> y;
{% endblock %}

{% block model %}
{% block priors %}
  {{ super() }}
{% endblock priors %}
{% block likelihood %}
  {{ super() }}
  y ~ poisson_log(mu_star_j);
{% endblock likelihood %}
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
  
