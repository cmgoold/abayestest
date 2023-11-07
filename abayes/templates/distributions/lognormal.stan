{% extends "base.stan" %}

{% block data %}
  {{ super() }}
  vector<lower=0>[N] y;
{% endblock data %}

{% block parameters %}
  {{ super() }}
  vector[2] sigma_star;
{% endblock parameters %}

{% block transformed_parameters %}
  {{ super() }}
  vector[N] sigma_star_j = sigma_star[j];
{% endblock transformed_parameters %}

{% block model %}
{% block priors %}
  {{ super() }}
  sigma_star ~ {{ priors.sigma_star }};
{% endblock priors %}
{% block likelihood %}
  {{ super() }}
  y ~ lognormal(mu_star_j, exp(sigma_star_j));
{% endblock likelihood %}
{% endblock model %}

{% block generated_quantities %}
{% block declarations %}
  {{ super() }}
  vector[2] mu = mu_star;
  vector[N] mu_j = mu_star_j;
  real mu_diff = mu[1] - mu[2];
  vector[2] sigma = exp(sigma_star);
  vector[N] sigma_j = exp(sigma_star_j);
  real sigma_star_diff = sigma_star[1] - sigma_star[2];
  real sigma_diff = sigma[1] - sigma[2];
  vector[N] y_rep;
{% endblock declarations %}
{% block computations %}
  {{ super() }}
  for(n in 1:N)
    y_rep[n] = lognormal_rng(mu_star_j[n], sigma_j[n]);
{% endblock computations %}
{% endblock generated_quantities %}
  
  
  
